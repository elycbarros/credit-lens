-- Agregações do bureau por cliente (1 linha por SK_ID_CURR).
-- Pré-requisito: tabela `bureau` carregada no DuckDB (ver features.py).
-- Features: volume, atividade, atraso e utilização de crédito.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                        AS bureau_n_creditos,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)      AS bureau_n_ativos,
    ROUND(
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1.0 ELSE 0.0 END)
        / NULLIF(COUNT(*), 0), 4
    )                                                               AS bureau_prop_ativos,
    MAX(CREDIT_DAY_OVERDUE)                                         AS bureau_atraso_max_dias,
    AVG(CREDIT_DAY_OVERDUE)                                         AS bureau_atraso_medio_dias,
    SUM(AMT_CREDIT_SUM)                                             AS bureau_credito_total,
    SUM(AMT_CREDIT_SUM_DEBT)                                        AS bureau_divida_total,
    ROUND(
        SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0), 4
    )                                                               AS bureau_utilizacao_media,
    SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)        AS bureau_n_com_atraso
FROM bureau
GROUP BY SK_ID_CURR;
