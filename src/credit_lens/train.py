"""Treino: regressão logística (baseline) e LightGBM (Fase 3).

Pipeline:
  1. Carrega ``data/processed/dataset.parquet``.
  2. Separa treino / validação / teste via ``split.py``.
  3. Treina regressão logística com pipeline (imputação + escala).
  4. Treina LightGBM com validação cruzada estratificada e RandomizedSearchCV.
  5. Salva ``models/modelo_<nome>_v<version>.joblib`` e ``models/metricas.json``.

Uso:
    python -m credit_lens.train
    python -m credit_lens.train --modelo lgbm --versao 2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_lens.calibracao import calibrar, ece
from credit_lens.evaluate import resumo_metricas
from credit_lens.split import split_estratificado

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"

_RANDOM_STATE = 42
_CV_FOLDS = 5

# Colunas que não são features
_COLS_EXCLUIR = {"SK_ID_CURR", "TARGET"}


def _carregar_dataset() -> pd.DataFrame:
    parquet = PROCESSED / "dataset.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {parquet}. Execute `make features` primeiro."
        )
    df = pd.read_parquet(parquet)
    # Substitui valores infinitos decorrentes de divisões por zero por NaN
    return df.replace([np.inf, -np.inf], np.nan)


def _separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cols_features = [c for c in df.columns if c not in _COLS_EXCLUIR]
    return df[cols_features], df["TARGET"]


def _pipeline_logistica(X: pd.DataFrame) -> Pipeline:
    """Pipeline: imputa mediana + escala para regressão logística."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
    ]
    if cat_cols:
        from sklearn.preprocessing import OrdinalEncoder
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols,
        ))

    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=_RANDOM_STATE,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])


def _pipeline_lgbm(X: pd.DataFrame) -> Pipeline:
    """Pipeline LightGBM (lida nativamente com NaN e categóricas numéricas)."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm não instalado. Execute: pip install lightgbm>=4.3"
        ) from exc

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers: list = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        from sklearn.preprocessing import OrdinalEncoder
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols,
        ))

    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", LGBMClassifier(
            class_weight="balanced",
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


def treinar_logistica(
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict]:
    logger.info("Treinando regressão logística (baseline) …")
    pipe = _pipeline_logistica(X_treino)
    pipe.fit(X_treino, y_treino)
    y_prob = pipe.predict_proba(X_val)[:, 1]
    metricas = {"modelo": "logistica", "conjunto": "validacao", **resumo_metricas(y_val, y_prob)}
    logger.info("Logística | validação: %s", metricas)
    return pipe, metricas


def treinar_lgbm(
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[object, dict]:
    logger.info("Treinando LightGBM com RandomizedSearchCV …")
    pipe = _pipeline_lgbm(X_treino)

    param_dist = {
        "clf__n_estimators": [300, 500, 800],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__num_leaves": [31, 63, 127],
        "clf__min_child_samples": [20, 50, 100],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_treino, y_treino)

    best = search.best_estimator_
    y_prob = best.predict_proba(X_val)[:, 1]
    metricas = {
        "modelo": "lgbm",
        "conjunto": "validacao",
        "cv_auc_mean": float(search.cv_results_["mean_test_score"][search.best_index_]),
        "cv_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "best_params": search.best_params_,
        **resumo_metricas(y_val, y_prob),
    }
    logger.info("LGBM | validação: %s", {k: v for k, v in metricas.items() if k != "best_params"})
    return best, metricas


def registrar_metricas(arquivo: str, versao: int, metricas: dict) -> None:
    """Grava/atualiza a entrada do modelo em ``models/metricas.json`` (uma por arquivo)."""
    metricas_path = MODELS / "metricas.json"
    historico: list = []
    if metricas_path.exists():
        historico = json.loads(metricas_path.read_text())

    def _py(e: dict) -> dict:
        return {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in e.items()}

    entrada = _py({"versao": versao, "arquivo": arquivo, **metricas})
    historico = [_py(e) for e in historico if e.get("arquivo") != arquivo]
    historico.append(entrada)
    metricas_path.write_text(json.dumps(historico, indent=2, ensure_ascii=False))
    logger.info("Métricas salvas em %s", metricas_path)


def main(modelo: str = "lgbm", versao: int = 1, calibrar_probabilidades: bool = True) -> None:
    """Ponto de entrada: treina, calibra (padrão) e salva o modelo escolhido."""
    MODELS.mkdir(parents=True, exist_ok=True)

    df = _carregar_dataset()
    df_treino, df_val, df_teste = split_estratificado(df)

    X_tr, y_tr = _separar_xy(df_treino)
    X_val, y_val = _separar_xy(df_val)
    X_teste, y_teste = _separar_xy(df_teste)

    if modelo == "logistica":
        estimador, metricas_val = treinar_logistica(X_tr, y_tr, X_val, y_val)
    elif modelo == "lgbm":
        estimador, metricas_val = treinar_lgbm(X_tr, y_tr, X_val, y_val)
    else:
        raise ValueError(f"Modelo desconhecido: {modelo!r}. Use 'logistica' ou 'lgbm'.")

    extras: dict = {"calibrado": False}
    if calibrar_probabilidades:
        brier_antes = resumo_metricas(y_teste, estimador.predict_proba(X_teste)[:, 1])["brier"]
        estimador = calibrar(estimador, X_val, y_val)
        extras = {
            "calibrado": True,
            "calibracao": "isotonica (ajustada na validação)",
            "teste_brier_antes_calibracao": brier_antes,
        }

    # Avaliação no conjunto de teste holdout
    y_prob_teste = estimador.predict_proba(X_teste)[:, 1]
    metricas_teste = resumo_metricas(y_teste, y_prob_teste)
    logger.info("%s | teste final: %s", modelo, metricas_teste)

    metricas = {
        **metricas_val,
        "teste_roc_auc": metricas_teste["roc_auc"],
        "teste_gini": metricas_teste["gini"],
        "teste_ks": metricas_teste["ks"],
        "teste_pr_auc": metricas_teste["pr_auc"],
        "teste_brier": metricas_teste["brier"],
        "teste_ece": ece(y_teste, y_prob_teste),
        **extras,
    }

    # Salva o modelo versionado (fora do Git via .gitignore)
    modelo_path = MODELS / f"modelo_{modelo}_v{versao}.joblib"
    joblib.dump(estimador, modelo_path)
    logger.info("Modelo salvo em %s", modelo_path)

    registrar_metricas(modelo_path.name, versao, metricas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Treina modelo de risco de crédito.")
    parser.add_argument("--modelo", default="lgbm", choices=["logistica", "lgbm"])
    parser.add_argument("--versao", type=int, default=1)
    parser.add_argument(
        "--sem-calibracao", action="store_true", help="não calibrar as probabilidades"
    )
    args = parser.parse_args()
    main(modelo=args.modelo, versao=args.versao, calibrar_probabilidades=not args.sem_calibracao)
