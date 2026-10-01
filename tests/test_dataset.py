"""Testes de integridade do dataset processado (Fase 1)."""

from pathlib import Path

import pandas as pd
import pytest

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset.parquet"


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset.parquet não gerado")
def test_dataset_integridade():
    """Valida integridade estrutural, chave primária e target."""
    df = pd.read_parquet(DATASET_PATH)

    # 1. Unicidade de chave primária
    assert df["SK_ID_CURR"].nunique() == len(df), "SK_ID_CURR contém valores duplicados"

    # 2. Total de registros do application_train
    assert len(df) == 307511, f"Esperado 307.511 linhas, obtido {len(df)}"

    # 3. Target válido
    assert set(df["TARGET"].unique()).issubset({0, 1}), "TARGET possui valores inválidos"
    assert df["TARGET"].isna().sum() == 0, "TARGET contém valores nulos"

    # 4. Colunas essenciais criadas
    colunas_obrigatorias = [
        "SK_ID_CURR",
        "TARGET",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "DAYS_EMPLOYED_ANOMALO",
        "bureau_n_creditos",
        "prev_n_pedidos",
        "inst_n_parcelas",
    ]
    for col in colunas_obrigatorias:
        assert col in df.columns, f"Coluna esperada ausente: {col}"
