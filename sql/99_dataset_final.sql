-- Dataset final: 1 linha por cliente de application_train.
SELECT
    a.SK_ID_CURR,
    a.TARGET,
    a.AMT_INCOME_TOTAL,
    a.AMT_CREDIT,
    a.AMT_ANNUITY,
    a.DAYS_BIRTH,
    a.DAYS_EMPLOYED,
    b.* EXCLUDE (SK_ID_CURR),
    p.* EXCLUDE (SK_ID_CURR),
    i.* EXCLUDE (SK_ID_CURR)
FROM application_train a
LEFT JOIN b_agg b ON a.SK_ID_CURR = b.SK_ID_CURR
LEFT JOIN p_agg p ON a.SK_ID_CURR = p.SK_ID_CURR
LEFT JOIN i_agg i ON a.SK_ID_CURR = i.SK_ID_CURR;
