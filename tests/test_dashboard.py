"""Testes para carregamento de dados e funções auxiliares do Dashboard."""

from __future__ import annotations

import pandas as pd

from credit_lens.dashboard import carregar_dados_bi, carregar_modelo_campeao


def test_carregar_dados_bi():
    dados = carregar_dados_bi()
    assert isinstance(dados, dict)
    for chave in ["metricas", "decis", "corte", "segmentos", "psi", "roc"]:
        assert chave in dados
        assert isinstance(dados[chave], pd.DataFrame)
        assert not dados[chave].empty


def test_carregar_modelo_campeao():
    modelo, nome = carregar_modelo_campeao()
    assert modelo is not None
    assert "modelo_" in nome
    assert hasattr(modelo, "predict_proba")
