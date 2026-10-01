"""Exporta resultados agregados do Credit-Lens para Tableau Public / Power BI.

Gera em ``bi/`` (ignorado pelo Git; regenerável) apenas dados **agregados**,
sem identificadores de clientes, adequados para publicação no Tableau Public:

    bi_metricas.csv   modelo, conjunto, metrica, valor   (de models/metricas.json)
    bi_decis.csv      tabela de decis do conjunto de teste, por modelo
    bi_roc.csv        pontos da curva ROC (teste), por modelo
    bi_corte.csv      varredura de ponto de corte (teste): aprovação x inadimplência
    bi_segmentos.csv  taxa real de inadimplência e score médio por faixa etária e de crédito
    bi_psi.csv        PSI por variável (de reports/monitoramento.csv) com flag de proxy

Uso (com dados processados e modelos treinados):
    python scripts/exportar_para_bi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from credit_lens.evaluate import tabela_decis  # noqa: E402

BI_DIR = RAIZ / "bi"
MODELOS = {"Regressão logística": "logistica", "LightGBM": "lgbm"}
NOMES_MODELO = {"logistica": "Regressão logística", "lgbm": "LightGBM"}
# A partição simulada de monitoramento usa DAYS_BIRTH como proxy temporal:
# o PSI dessas variáveis não mede drift real (ver docs/model_card.md).
VARIAVEIS_PROXY = {"DAYS_BIRTH"}


def modelo_mais_recente(models_dir: Path, chave: str) -> Path | None:
    """Maior versão de ``modelo_<chave>_v<N>.joblib`` (a calibrada, se existir)."""
    def versao(p: Path) -> int:
        return int(p.stem.rsplit("_v", 1)[1])

    candidatos = sorted(models_dir.glob(f"modelo_{chave}_v*.joblib"), key=versao)
    return candidatos[-1] if candidatos else None


def metricas_longas(historico: list[dict]) -> pd.DataFrame:
    """Converte ``metricas.json`` em formato longo (modelo, conjunto, metrica, valor)."""
    linhas = []
    for e in historico:
        modelo = NOMES_MODELO.get(e.get("modelo"), e.get("modelo"))
        if e.get("calibrado"):
            modelo += " (calibrado)"
        for chave in ("roc_auc", "gini", "ks", "pr_auc", "brier", "ece"):
            if chave in e:
                linhas.append((modelo, "validação", chave, float(e[chave])))
            if f"teste_{chave}" in e:
                linhas.append((modelo, "teste", chave, float(e[f"teste_{chave}"])))
        if "cv_auc_mean" in e:
            linhas.append((modelo, "cv_5_folds", "roc_auc", float(e["cv_auc_mean"])))
    return pd.DataFrame(linhas, columns=["modelo", "conjunto", "metrica", "valor"])


def curva_roc(y_true, y_score, max_pontos: int = 200) -> pd.DataFrame:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) > max_pontos:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_pontos).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def varredura_corte(y_true, y_score, n: int = 50) -> pd.DataFrame:
    """Para cada corte de score, aprova quem está abaixo dele.

    Retorna % de aprovados e taxa de inadimplência entre aprovados.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    linhas = []
    for corte in np.quantile(s, np.linspace(0.02, 1.0, n)):
        aprovados = s <= corte
        if aprovados.sum() == 0:
            continue
        linhas.append(
            {
                "corte_score": float(corte),
                "pct_aprovados": float(aprovados.mean()),
                "taxa_inadimplencia_aprovados": float(y[aprovados].mean()),
                "pct_maus_barrados": float(1 - y[aprovados].sum() / y.sum()),
            }
        )
    return pd.DataFrame(linhas)


def segmentos(df: pd.DataFrame, score) -> pd.DataFrame:
    """Taxa real de inadimplência e score médio por faixa etária e faixa de crédito."""
    d = df[["TARGET", "DAYS_BIRTH", "AMT_CREDIT"]].copy()
    d["score"] = np.asarray(score)
    d["idade"] = -d["DAYS_BIRTH"] / 365.25
    d["faixa_etaria"] = pd.cut(
        d["idade"], [0, 30, 40, 50, 60, 200], right=False,
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
    )
    d["faixa_credito"] = pd.qcut(d["AMT_CREDIT"], 4, labels=["Q1 (menor)", "Q2", "Q3", "Q4 (maior)"])
    saidas = []
    for dimensao, col in (("faixa etária", "faixa_etaria"), ("faixa de crédito", "faixa_credito")):
        g = d.groupby(col, observed=True).agg(
            clientes=("TARGET", "count"),
            taxa_inadimplencia=("TARGET", "mean"),
            score_medio=("score", "mean"),
        ).reset_index().rename(columns={col: "faixa"})
        g.insert(0, "dimensao", dimensao)
        g["faixa"] = g["faixa"].astype(str)
        saidas.append(g)
    return pd.concat(saidas, ignore_index=True)


def psi_com_flag(monitoramento: pd.DataFrame) -> pd.DataFrame:
    out = monitoramento.copy()
    out["artefato_do_proxy"] = out["variavel"].isin(VARIAVEIS_PROXY)
    return out


def exportar(saida: Path = BI_DIR) -> dict[str, Path]:
    import joblib

    from credit_lens.split import split_estratificado
    from credit_lens.train import _carregar_dataset, _separar_xy

    saida.mkdir(parents=True, exist_ok=True)
    destinos: dict[str, Path] = {}

    def gravar(nome: str, df: pd.DataFrame) -> None:
        caminho = saida / f"bi_{nome}.csv"
        df.to_csv(caminho, index=False, encoding="utf-8")
        destinos[nome] = caminho

    historico = json.loads((RAIZ / "models" / "metricas.json").read_text())
    gravar("metricas", metricas_longas(historico))

    mon = RAIZ / "reports" / "monitoramento.csv"
    if mon.exists():
        gravar("psi", psi_com_flag(pd.read_csv(mon)))

    df = _carregar_dataset()
    _, _, df_teste = split_estratificado(df)
    X, y = _separar_xy(df_teste)

    decis, rocs, cortes, segs = [], [], [], []
    for nome, chave in MODELOS.items():
        caminho = modelo_mais_recente(RAIZ / "models", chave)
        if caminho is None:
            print(f"[aviso] modelo ausente, ignorado: {nome}")
            continue
        print(f"[info] {nome}: usando {caminho.name}")
        score = joblib.load(caminho).predict_proba(X)[:, 1]
        t = tabela_decis(y, score)
        t.insert(0, "modelo", nome)
        decis.append(t)
        r = curva_roc(y, score)
        r.insert(0, "modelo", nome)
        rocs.append(r)
        c = varredura_corte(y, score)
        c.insert(0, "modelo", nome)
        cortes.append(c)
        if nome == "LightGBM":
            segs.append(segmentos(df_teste, score))

    if not decis:
        raise SystemExit("Nenhum modelo encontrado em models/. Rode `make train` antes.")
    gravar("decis", pd.concat(decis, ignore_index=True))
    gravar("roc", pd.concat(rocs, ignore_index=True))
    gravar("corte", pd.concat(cortes, ignore_index=True))
    if segs:
        gravar("segmentos", segs[0])
    return destinos


def main() -> None:
    for nome, caminho in exportar().items():
        print(f"{nome:10s} -> {caminho}")


if __name__ == "__main__":
    main()
