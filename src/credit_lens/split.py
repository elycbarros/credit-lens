"""Split estratificado treino/validação/teste (Fase 3).

Estratégia:
  - Padrão: split estratificado por TARGET (preserva taxa de inadimplência).
  - Temporal simulado: ordena por `sort_col` antes de dividir (sem embaralhamento).
  - Teste ficaINTOCADO até a fase de avaliação final.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# Proporções padrão: 70 % treino · 15 % validação · 15 % teste
_TRAIN_RATIO = 0.70
_VAL_RATIO = 0.15
_RANDOM_STATE = 42


def split_estratificado(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    train_ratio: float = _TRAIN_RATIO,
    val_ratio: float = _VAL_RATIO,
    random_state: int = _RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o dataframe em treino / validação / teste de forma estratificada.

    Parameters
    ----------
    df:
        Dataset completo com a coluna alvo já presente.
    target_col:
        Nome da coluna alvo (0/1).
    train_ratio:
        Proporção destinada ao treino (padrão 0,70).
    val_ratio:
        Proporção destinada à validação (padrão 0,15); o restante vai para teste.
    random_state:
        Semente para reprodutibilidade.

    Returns
    -------
    tuple[df_treino, df_val, df_teste]
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio deve ser < 1.0")

    y = df[target_col]

    # 1ª divisão: separa teste do restante
    df_temp, df_teste = train_test_split(
        df,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # 2ª divisão: treino / validação dentro do restante
    val_ratio_ajustado = val_ratio / (train_ratio + val_ratio)
    df_treino, df_val = train_test_split(
        df_temp,
        test_size=val_ratio_ajustado,
        stratify=df_temp[target_col],
        random_state=random_state,
    )

    return df_treino, df_val, df_teste


def split_temporal_simulado(
    df: pd.DataFrame,
    sort_col: str,
    target_col: str = "TARGET",
    train_ratio: float = _TRAIN_RATIO,
    val_ratio: float = _VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal simulado: ordena por `sort_col` e divide sequencialmente.

    Usado quando se quer simular a validação out-of-time sem datas reais.
    Documentar no model card que o drift é simulado (seção de limitações).

    Parameters
    ----------
    df:
        Dataset completo.
    sort_col:
        Coluna usada como proxy temporal (ex.: ``DAYS_BIRTH``, ``DAYS_DECISION``).
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio deve ser < 1.0")

    df_sorted = df.sort_values(sort_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    df_treino = df_sorted.iloc[:n_train]
    df_val = df_sorted.iloc[n_train : n_train + n_val]
    df_teste = df_sorted.iloc[n_train + n_val :]

    return df_treino, df_val, df_teste
