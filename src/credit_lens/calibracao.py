"""Calibração de probabilidades (pós-treino).

Os modelos são treinados com pesos de classe para lidar com o desbalanceamento (~8% de
inadimplentes), o que ordena bem os clientes (AUC/KS) mas infla o score: ele **não** é
uma probabilidade. A calibração isotônica, ajustada no conjunto de **validação**, ajusta o
score para que ele represente a taxa observada de inadimplência. O teste holdout continua
intocado e serve para medir o efeito (Brier e ECE antes x depois).

Uso (calibra um modelo já treinado, sem retreinar):
    python -m credit_lens.calibracao --modelo lgbm --origem 1 --destino 2
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calibrar(estimador, X_val, y_val):
    """Retorna o estimador envolvido por calibração isotônica ajustada na validação."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    cal = CalibratedClassifierCV(FrozenEstimator(estimador), method="isotonic")
    cal.fit(X_val, y_val)
    return cal


def curva_calibracao(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Probabilidade média prevista x taxa observada por faixa (quantis de score)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    d["faixa"] = pd.qcut(d["p"].rank(method="first"), q=n_bins, labels=False)
    return (
        d.groupby("faixa")
        .agg(clientes=("y", "count"), prob_media=("p", "mean"), taxa_observada=("y", "mean"))
        .reset_index()
    )


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error (faixas por quantil, ponderado por tamanho)."""
    c = curva_calibracao(y_true, y_prob, n_bins)
    pesos = c["clientes"] / c["clientes"].sum()
    return float((pesos * (c["prob_media"] - c["taxa_observada"]).abs()).sum())


def calibrar_existente(modelo: str = "lgbm", origem: int = 1, destino: int = 2) -> None:
    """Calibra ``modelo_<modelo>_v<origem>.joblib`` e salva como ``v<destino>``."""
    # Import tardio: train importa este módulo.
    from credit_lens.evaluate import resumo_metricas
    from credit_lens.split import split_estratificado
    from credit_lens.train import MODELS, _carregar_dataset, _separar_xy, registrar_metricas

    df = _carregar_dataset()
    _, df_val, df_teste = split_estratificado(df)
    X_val, y_val = _separar_xy(df_val)
    X_te, y_te = _separar_xy(df_teste)

    base = joblib.load(MODELS / f"modelo_{modelo}_v{origem}.joblib")
    antes = base.predict_proba(X_te)[:, 1]
    cal = calibrar(base, X_val, y_val)
    depois = cal.predict_proba(X_te)[:, 1]

    m = resumo_metricas(y_te, depois)
    metricas = {
        "modelo": modelo,
        "conjunto": "validacao",
        "calibrado": True,
        "calibracao": "isotonica (ajustada na validação)",
        "teste_roc_auc": m["roc_auc"],
        "teste_gini": m["gini"],
        "teste_ks": m["ks"],
        "teste_pr_auc": m["pr_auc"],
        "teste_brier": m["brier"],
        "teste_ece": ece(y_te, depois),
        "teste_brier_antes_calibracao": resumo_metricas(y_te, antes)["brier"],
        "teste_ece_antes_calibracao": ece(y_te, antes),
        "teste_prob_media": float(depois.mean()),
        "teste_taxa_observada": float(np.asarray(y_te).mean()),
    }
    arquivo = f"modelo_{modelo}_v{destino}.joblib"
    joblib.dump(cal, MODELS / arquivo)
    registrar_metricas(arquivo, destino, metricas)
    logger.info("Calibrado %s: %s", arquivo, metricas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Calibra um modelo treinado (isotônica).")
    p.add_argument("--modelo", default="lgbm", choices=["logistica", "lgbm"])
    p.add_argument("--origem", type=int, default=1)
    p.add_argument("--destino", type=int, default=2)
    a = p.parse_args()
    calibrar_existente(a.modelo, a.origem, a.destino)
