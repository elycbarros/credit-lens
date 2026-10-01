"""Métricas de avaliação de modelos de crédito."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def gini(y_true, y_score) -> float:
    """Gini = 2 * AUC - 1."""
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """KS = maior distância entre as CDFs de maus e bons ordenadas por score."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    ordem = np.argsort(s)
    y = y[ordem]
    n_maus = y.sum()
    n_bons = len(y) - n_maus
    if n_maus == 0 or n_bons == 0:
        raise ValueError("KS exige as duas classes presentes.")
    cdf_maus = np.cumsum(y) / n_maus
    cdf_bons = np.cumsum(1 - y) / n_bons
    # Com empates, as CDFs só valem no último elemento de cada score distinto.
    fim_do_grupo = np.append(s[ordem][1:] != s[ordem][:-1], True)
    return float(np.max(np.abs(cdf_maus - cdf_bons)[fim_do_grupo]))


def resumo_metricas(y_true, y_score) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "gini": float(gini(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def tabela_decis(y_true, y_score, n_bins: int = 10):
    """Gera tabela de ordenação por decis de risco.

    Decil 1 = maior probabilidade de inadimplência (maior risco).
    Decil 10 = menor probabilidade de inadimplência (menor risco).
    """
    import pandas as pd

    y = np.asarray(y_true)
    s = np.asarray(y_score)
    df = pd.DataFrame({"y": y, "score": s})
    df["decil"] = pd.qcut(
        df["score"].rank(method="first", ascending=False),
        q=n_bins,
        labels=range(1, n_bins + 1),
    )

    tabela = (
        df.groupby("decil", observed=False)
        .agg(
            total_clientes=("y", "count"),
            maus=("y", "sum"),
        )
        .reset_index()
    )

    tabela["bons"] = tabela["total_clientes"] - tabela["maus"]
    tabela["taxa_maus"] = tabela["maus"] / tabela["total_clientes"]

    total_maus = tabela["maus"].sum()
    total_bons = tabela["bons"].sum()

    tabela["acum_maus"] = (tabela["maus"].cumsum() / total_maus) if total_maus > 0 else 0.0
    tabela["acum_bons"] = (tabela["bons"].cumsum() / total_bons) if total_bons > 0 else 0.0
    tabela["ks"] = (tabela["acum_maus"] - tabela["acum_bons"]).abs()

    return tabela

