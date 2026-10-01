import numpy as np
import pytest

from credit_lens.monitor import classificar_psi, psi


def test_psi_zero_para_distribuicoes_iguais():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_detecta_deslocamento():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    atual = rng.normal(1.0, 1, 5000)
    assert psi(ref, atual) > 0.25


def test_classificacao():
    assert classificar_psi(0.05) == "estavel"
    assert classificar_psi(0.15) == "atencao"
    assert classificar_psi(0.40) == "acao"


def test_psi_por_variavel():
    import pandas as pd

    from credit_lens.monitor import psi_por_variavel

    rng = np.random.default_rng(42)
    df_ref = pd.DataFrame({
        "var_a": rng.normal(0, 1, 1000),
        "var_b": rng.normal(5, 2, 1000),
    })
    df_atual = pd.DataFrame({
        "var_a": rng.normal(0, 1, 1000),
        "var_b": rng.normal(7, 2, 1000),  # shift
    })
    res = psi_por_variavel(df_ref, df_atual, colunas=["var_a", "var_b"])
    assert len(res) == 2
    assert res[0]["variavel"] == "var_b"  # sorted by highest PSI
    assert res[0]["classificacao"] in ("atencao", "acao")
    assert res[1]["variavel"] == "var_a"
    assert res[1]["classificacao"] == "estavel"


def test_gerar_relatorio(tmp_path):
    import pandas as pd

    from credit_lens.monitor import gerar_relatorio

    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "SK_ID_CURR": range(200),
        "TARGET": [0] * 180 + [1] * 20,
        "DAYS_BIRTH": np.linspace(-20000, -8000, 200),
        "AMT_CREDIT": rng.uniform(10000, 500000, 200),
    })
    dados_parquet = tmp_path / "mock_dataset.parquet"
    data.to_parquet(dados_parquet, index=False)

    saida_csv = tmp_path / "monitoramento.csv"
    gerar_relatorio(caminho_dados=dados_parquet, saida=saida_csv)

    assert saida_csv.exists()
    relatorio = pd.read_csv(saida_csv)
    assert "variavel" in relatorio.columns
    assert "AMT_CREDIT" in relatorio["variavel"].values
