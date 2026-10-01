"""Construção do dataset via DuckDB (Fase 1).

Pipeline:
  1. Carrega os CSVs de ``data/raw/`` diretamente no DuckDB (sem carregar em pandas).
  2. Executa os SQLs de agregação (bureau, pedidos anteriores, parcelas).
  3. Executa o join final e exporta ``data/processed/dataset.parquet``.
  4. Aplica tratamento do valor sentinela DAYS_EMPLOYED = 365243.

Pré-requisito: baixar manualmente os CSVs do Kaggle (Home Credit Default Risk)
e colocar em ``data/raw/``:
  - application_train.csv
  - bureau.csv
  - previous_application.csv
  - installments_payments.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
SQL = RAIZ / "sql"

# Arquivos CSV obrigatórios
_CSV_REQUIRED = [
    "application_train.csv",
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
]

# Valor sentinela do dataset Home Credit (empregado desde 1000 anos = sem emprego)
_DAYS_EMPLOYED_SENTINEL = 365243


def _verificar_dados() -> None:
    """Verifica se todos os CSVs necessários estão presentes."""
    ausentes = [f for f in _CSV_REQUIRED if not (RAW / f).exists()]
    if ausentes:
        raise FileNotFoundError(
            f"Arquivos ausentes em {RAW}:\n  " + "\n  ".join(ausentes)
            + "\n\nBaixe manualmente do Kaggle: "
            "https://www.kaggle.com/c/home-credit-default-risk/data"
        )


def _carregar_sql(nome: str) -> str:
    """Lê o conteúdo de um arquivo SQL."""
    return (SQL / nome).read_text(encoding="utf-8")


def main() -> None:
    """Executa o pipeline completo e salva dataset.parquet."""
    _verificar_dados()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    db_path = PROCESSED / "credit.duckdb"
    logger.info("Conectando ao DuckDB em %s", db_path)

    con = duckdb.connect(str(db_path))

    try:
        # ── 1. Carrega os CSVs como views (sem copiar para memória) ──────────
        logger.info("Registrando tabelas via read_csv_auto …")
        con.execute(f"""
            CREATE OR REPLACE VIEW application_train AS
                SELECT * FROM read_csv_auto('{RAW}/application_train.csv');
            CREATE OR REPLACE VIEW bureau AS
                SELECT * FROM read_csv_auto('{RAW}/bureau.csv');
            CREATE OR REPLACE VIEW previous_application AS
                SELECT * FROM read_csv_auto('{RAW}/previous_application.csv');
            CREATE OR REPLACE VIEW installments_payments AS
                SELECT * FROM read_csv_auto('{RAW}/installments_payments.csv');
        """)

        # ── 2. Agregações por cliente ────────────────────────────────────────
        logger.info("Executando 01_bureau_agg.sql …")
        con.execute(
            f"CREATE OR REPLACE TABLE b_agg AS {_carregar_sql('01_bureau_agg.sql')}"
        )

        logger.info("Executando 02_prev_app_agg.sql …")
        con.execute(
            f"CREATE OR REPLACE TABLE p_agg AS {_carregar_sql('02_prev_app_agg.sql')}"
        )

        logger.info("Executando 03_installments_agg.sql …")
        con.execute(
            f"CREATE OR REPLACE TABLE i_agg AS {_carregar_sql('03_installments_agg.sql')}"
        )

        # ── 3. Dataset final ─────────────────────────────────────────────────
        logger.info("Executando 99_dataset_final.sql …")
        df = con.execute(_carregar_sql("99_dataset_final.sql")).df()

        # ── 4. Tratamento pós-SQL ─────────────────────────────────────────────
        # Valor sentinela: DAYS_EMPLOYED = 365243 → NaN + flag binária
        if "DAYS_EMPLOYED" in df.columns:
            mask = df["DAYS_EMPLOYED"] == _DAYS_EMPLOYED_SENTINEL
            df["DAYS_EMPLOYED_ANOMALO"] = mask.astype("int8")
            df.loc[mask, "DAYS_EMPLOYED"] = float("nan")
            logger.info(
                "DAYS_EMPLOYED sentinela corrigido: %d registros → NaN (flag criada)",
                mask.sum(),
            )

        # ── 5. Exporta Parquet ───────────────────────────────────────────────
        out = PROCESSED / "dataset.parquet"
        df.to_parquet(out, index=False)
        logger.info(
            "Dataset salvo em %s | shape: %s | inadimplência: %.2f%%",
            out,
            df.shape,
            df["TARGET"].mean() * 100 if "TARGET" in df.columns else float("nan"),
        )

    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
