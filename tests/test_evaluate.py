import pytest

from credit_lens.evaluate import gini, ks_statistic


def test_separacao_perfeita():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert gini(y, s) == pytest.approx(1.0)
    assert ks_statistic(y, s) == pytest.approx(1.0)


def test_score_sem_informacao():
    y = [0, 1, 0, 1]
    s = [0.5, 0.5, 0.5, 0.5]
    assert gini(y, s) == pytest.approx(0.0)
    assert ks_statistic(y, s) == pytest.approx(0.0)


def test_ks_exige_duas_classes():
    with pytest.raises(ValueError):
        ks_statistic([0, 0, 0], [0.1, 0.2, 0.3])


def test_resumo_metricas():
    from credit_lens.evaluate import resumo_metricas
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    res = resumo_metricas(y, s)
    assert res["roc_auc"] == 1.0
    assert res["gini"] == 1.0
    assert "brier" in res


def test_tabela_decis():
    from credit_lens.evaluate import tabela_decis
    y = [1] * 20 + [0] * 80
    s = list(range(100, 0, -1))  # higher score for index 0..19 (the 1s)
    tab = tabela_decis(y, s, n_bins=10)
    assert len(tab) == 10
    # Decile 1 must capture the high risk customers
    assert tab.iloc[0]["maus"] > tab.iloc[-1]["maus"]
    assert "ks" in tab.columns
    assert "acum_maus" in tab.columns

