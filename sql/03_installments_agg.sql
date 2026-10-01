-- Comportamento de pagamento de parcelas por cliente.
-- ⚠️  ATENÇÃO a vazamento: usar apenas parcelas de contratos anteriores.
--     (installments_payments refere-se a contratos passados, não ao atual.)
-- Features comportamentais: pontualidade e suficiência dos pagamentos.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                                AS inst_n_parcelas,
    -- Atraso: positivo = atrasado, negativo = adiantado
    AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)                              AS inst_atraso_medio_dias,
    MAX(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))                 AS inst_atraso_max_dias,
    STDDEV(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)                           AS inst_atraso_desvpad,
    -- Parcelas pagas em dia ou antes (indicador de boa conduta)
    ROUND(
        AVG(CASE WHEN DAYS_ENTRY_PAYMENT <= DAYS_INSTALMENT THEN 1.0 ELSE 0.0 END), 4
    )                                                                       AS inst_taxa_em_dia,
    -- Parcelas pagas com valor inferior ao devido
    ROUND(
        AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0 ELSE 0.0 END), 4
    )                                                                       AS inst_taxa_pagto_menor,
    -- Razão média entre o que foi pago e o que era devido
    ROUND(AVG(AMT_PAYMENT) / NULLIF(AVG(AMT_INSTALMENT), 0), 4)            AS inst_ratio_pagto_devido
FROM installments_payments
GROUP BY SK_ID_CURR;
