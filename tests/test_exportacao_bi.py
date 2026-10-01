"""Testes das funções puras de exportação para BI (não exigem dados nem modelos)."""

import numpy as np
import pandas as pd
import pytest

from scripts.exportar_para_bi import (
    curva_roc,
    metricas_longas,
    psi_com_flag,
    segmentos,
    varredura_corte,
)


def test_metricas_longas_separa_conjuntos():
    hist = [
        {"modelo": "lgbm", "roc_auc": 0.70, "teste_roc_auc": 0.71, "teste_ks": 0.31,
         "cv_auc_mean": 0.70},
    ]
    m = metricas_longas(hist)
    assert set(m["conjunto"]) == {"validação", "teste", "cv_5_folds"}
    linha = m[(m["conjunto"] == "teste") & (m["metrica"] == "roc_auc")].iloc[0]
    assert linha["modelo"] == "LightGBM"
    assert linha["valor"] == pytest.approx(0.71)


def test_curva_roc_limites():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    r = curva_roc(y, rng.random(1000), max_pontos=50)
    assert len(r) <= 50
    assert r["fpr"].iloc[0] == 0 and r["tpr"].iloc[-1] == pytest.approx(1.0)


def test_varredura_corte_monotonica():
    rng = np.random.default_rng(1)
    s = rng.random(2000)
    y = (rng.random(2000) < s * 0.3).astype(int)  # score informativo
    c = varredura_corte(y, s)
    assert c["pct_aprovados"].is_monotonic_increasing
    # com score informativo, aprovar menos reduz inadimplência
    assert c["taxa_inadimplencia_aprovados"].iloc[0] < c["taxa_inadimplencia_aprovados"].iloc[-1]


def test_segmentos_so_agregados():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({
        "TARGET": rng.integers(0, 2, n),
        "DAYS_BIRTH": -rng.integers(20 * 365, 70 * 365, n),
        "AMT_CREDIT": rng.uniform(50_000, 900_000, n),
    })
    seg = segmentos(df, rng.random(n))
    assert set(seg["dimensao"]) == {"faixa etária", "faixa de crédito"}
    assert seg.groupby("dimensao")["clientes"].sum().eq(n).all()
    assert "SK_ID_CURR" not in seg.columns


def test_psi_flag_proxy():
    mon = pd.DataFrame({"variavel": ["DAYS_BIRTH", "AMT_CREDIT"], "psi": [12.4, 0.02],
                        "classificacao": ["acao", "estavel"]})
    out = psi_com_flag(mon).set_index("variavel")["artefato_do_proxy"]
    assert bool(out["DAYS_BIRTH"]) and not bool(out["AMT_CREDIT"])
