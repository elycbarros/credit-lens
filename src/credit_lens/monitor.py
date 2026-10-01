"""Monitoramento de estabilidade: PSI (Population Stability Index).

Uso:
    python -m credit_lens.monitor          # relatório completo
    python -m credit_lens.monitor --saida reports/monitoramento.csv

Faixas convencionais de PSI (documentar no model card):
    < 0,10  → estável   (sem ação necessária)
    0,10–0,25 → atenção (investigar)
    > 0,25  → ação      (considerar retreino)

Nota: o drift neste projeto é SIMULADO por partições ordenadas (sem datas reais).
Ver docs/dados_e_limitacoes.md.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"


def psi(referencia, atual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index entre duas amostras.

    Faixas convencionais: < 0,10 estável; 0,10-0,25 atenção; > 0,25 ação.
    Os cortes vêm dos quantis da amostra de referência.
    """
    ref = np.asarray(referencia, dtype=float)
    atu = np.asarray(atual, dtype=float)
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, bins=cortes)[0] / len(ref)
    p_atu = np.histogram(atu, bins=cortes)[0] / len(atu)
    p_ref = np.clip(p_ref, eps, None)
    p_atu = np.clip(p_atu, eps, None)
    return float(np.sum((p_atu - p_ref) * np.log(p_atu / p_ref)))


def classificar_psi(valor: float) -> str:
    if valor < 0.10:
        return "estavel"
    if valor <= 0.25:
        return "atencao"
    return "acao"


def psi_por_variavel(
    df_ref: pd.DataFrame,
    df_atual: pd.DataFrame,
    colunas: list[str] | None = None,
    bins: int = 10,
) -> list[dict]:
    """Calcula PSI para cada variável numérica entre referência e atual.

    Parameters
    ----------
    df_ref:
        Dataset de referência (ex.: treino).
    df_atual:
        Dataset atual (ex.: partição simulada de monitoramento).
    colunas:
        Lista de colunas a avaliar. Se None, usa todas as numéricas.
    bins:
        Número de faixas (quantis da referência).

    Returns
    -------
    Lista de dicts com: variavel, psi, classificacao.
    """
    if colunas is None:
        colunas = df_ref.select_dtypes(include="number").columns.tolist()

    resultado = []
    for col in colunas:
        if col not in df_atual.columns:
            logger.warning("Coluna %r ausente no dataset atual; ignorada.", col)
            continue
        ref_vals = df_ref[col].dropna().values
        atu_vals = df_atual[col].dropna().values
        if len(ref_vals) == 0 or len(atu_vals) == 0:
            logger.warning("Coluna %r sem valores; PSI não calculado.", col)
            continue
        valor = psi(ref_vals, atu_vals, bins=bins)
        resultado.append({
            "variavel": col,
            "psi": round(valor, 6),
            "classificacao": classificar_psi(valor),
        })

    resultado.sort(key=lambda x: x["psi"], reverse=True)
    return resultado


def gerar_relatorio(saida: Path | None = None, caminho_dados: Path | None = None) -> None:
    """Gera relatório de PSI a partir do dataset processado (simulado).

    Divide o dataset em duas metades ordenadas por DAYS_BIRTH como proxy temporal.
    """
    parquet = caminho_dados or (PROCESSED / "dataset.parquet")
    if not parquet.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {parquet}. Execute `make features` primeiro."
        )

    df = pd.read_parquet(parquet)

    # Partição simulada: 50% mais antigos (referência) vs 50% mais recentes (atual)
    # Ordena por DAYS_BIRTH (negativo: menor = mais velho)
    if "DAYS_BIRTH" in df.columns:
        df_sorted = df.sort_values("DAYS_BIRTH")
    else:
        logger.warning("DAYS_BIRTH ausente; usando ordem original.")
        df_sorted = df

    meio = len(df_sorted) // 2
    df_ref = df_sorted.iloc[:meio]
    df_atual = df_sorted.iloc[meio:]

    logger.info(
        "Partição simulada: ref=%d linhas, atual=%d linhas", len(df_ref), len(df_atual)
    )

    # Exclui colunas de ID e target
    excluir = {"SK_ID_CURR", "TARGET"}
    colunas = [c for c in df_ref.select_dtypes(include="number").columns if c not in excluir]

    resultado = psi_por_variavel(df_ref, df_atual, colunas=colunas)

    # Exibe no terminal
    n_acao = sum(1 for r in resultado if r["classificacao"] == "acao")
    n_atencao = sum(1 for r in resultado if r["classificacao"] == "atencao")
    logger.info(
        "PSI calculado para %d variáveis | ação: %d | atenção: %d",
        len(resultado), n_acao, n_atencao,
    )
    for r in resultado[:10]:  # top 10 com maior drift
        logger.info("  %-40s PSI=%.4f  [%s]", r["variavel"], r["psi"], r["classificacao"])

    # Exporta CSV
    if saida is None:
        REPORTS.mkdir(parents=True, exist_ok=True)
        saida = REPORTS / "monitoramento.csv"

    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variavel", "psi", "classificacao"])
        writer.writeheader()
        writer.writerows(resultado)

    logger.info("Relatório salvo em %s", saida)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Monitora estabilidade das variáveis (PSI).")
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Caminho de saída do CSV (padrão: reports/monitoramento.csv).",
    )
    args = parser.parse_args()
    gerar_relatorio(saida=args.saida)
