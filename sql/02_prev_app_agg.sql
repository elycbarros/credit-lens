-- Pedidos anteriores por cliente.
-- Features: histórico de aprovação, recusa e cancelamento.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                                AS prev_n_pedidos,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved'  THEN 1 ELSE 0 END)    AS prev_n_aprovados,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'   THEN 1 ELSE 0 END)    AS prev_n_recusados,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled'  THEN 1 ELSE 0 END)    AS prev_n_cancelados,
    ROUND(
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1.0 ELSE 0.0 END), 4
    )                                                                       AS prev_taxa_aprovacao,
    ROUND(
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1.0 ELSE 0.0 END), 4
    )                                                                       AS prev_taxa_recusa,
    ROUND(
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1.0 ELSE 0.0 END), 4
    )                                                                       AS prev_taxa_cancelamento,
    AVG(AMT_CREDIT)                                                         AS prev_credito_medio,
    AVG(AMT_APPLICATION)                                                    AS prev_valor_pedido_medio,
    -- Razão entre valor aprovado e solicitado (indicador de capacidade de crédito)
    ROUND(AVG(AMT_CREDIT) / NULLIF(AVG(AMT_APPLICATION), 0), 4)            AS prev_ratio_aprovado_pedido
FROM previous_application
GROUP BY SK_ID_CURR;
