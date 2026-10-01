"""Testes unitários para credit_lens.split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_lens.split import split_estratificado, split_temporal_simulado


@pytest.fixture()
def df_sintetico() -> pd.DataFrame:
    """Dataset sintético com TARGET desbalanceado (~8% de maus)."""
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame({
        "SK_ID_CURR": range(n),
        "TARGET": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
        "feature_a": rng.normal(size=n),
        "feature_b": rng.uniform(size=n),
        "dias": np.arange(n),
    })


# ── split_estratificado ───────────────────────────────────────────────────────

class TestSplitEstratificado:
    def test_proporcoes_aproximadas(self, df_sintetico):
        tr, val, te = split_estratificado(df_sintetico)
        n = len(df_sintetico)
        assert 0.68 <= len(tr) / n <= 0.72, "Treino deve ser ~70%"
        assert 0.13 <= len(val) / n <= 0.17, "Validação deve ser ~15%"
        assert 0.13 <= len(te) / n <= 0.17, "Teste deve ser ~15%"

    def test_soma_total(self, df_sintetico):
        tr, val, te = split_estratificado(df_sintetico)
        assert len(tr) + len(val) + len(te) == len(df_sintetico)

    def test_sem_sobreposicao(self, df_sintetico):
        tr, val, te = split_estratificado(df_sintetico)
        ids_tr = set(tr["SK_ID_CURR"])
        ids_val = set(val["SK_ID_CURR"])
        ids_te = set(te["SK_ID_CURR"])
        assert ids_tr.isdisjoint(ids_val), "Treino e validação não devem se sobrepor"
        assert ids_tr.isdisjoint(ids_te), "Treino e teste não devem se sobrepor"
        assert ids_val.isdisjoint(ids_te), "Validação e teste não devem se sobrepor"

    def test_estratificacao_preserva_taxa(self, df_sintetico):
        """Taxa de inadimplência em cada split deve ser próxima da original."""
        taxa_original = df_sintetico["TARGET"].mean()
        tr, val, te = split_estratificado(df_sintetico)
        for nome, subset in [("treino", tr), ("validacao", val), ("teste", te)]:
            taxa = subset["TARGET"].mean()
            assert abs(taxa - taxa_original) < 0.02, (
                f"Taxa de inadimplência no {nome} ({taxa:.3f}) distante "
                f"da original ({taxa_original:.3f})"
            )

    def test_erro_proporcao_invalida(self, df_sintetico):
        with pytest.raises(ValueError, match="deve ser < 1.0"):
            split_estratificado(df_sintetico, train_ratio=0.7, val_ratio=0.4)


# ── split_temporal_simulado ───────────────────────────────────────────────────

class TestSplitTemporalSimulado:
    def test_ordem_preservada(self, df_sintetico):
        """Treino deve conter os menores valores da coluna de ordenação."""
        tr, val, te = split_temporal_simulado(df_sintetico, sort_col="dias")
        assert tr["dias"].max() < val["dias"].min(), "Treino deve anteceder validação"
        assert val["dias"].max() < te["dias"].min(), "Validação deve anteceder teste"

    def test_soma_total(self, df_sintetico):
        tr, val, te = split_temporal_simulado(df_sintetico, sort_col="dias")
        assert len(tr) + len(val) + len(te) == len(df_sintetico)

    def test_erro_proporcao_invalida(self, df_sintetico):
        with pytest.raises(ValueError, match="deve ser < 1.0"):
            split_temporal_simulado(df_sintetico, sort_col="dias", train_ratio=0.6, val_ratio=0.5)
