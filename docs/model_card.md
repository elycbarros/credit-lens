# Model Card — Credit-Lens (v1.0.0)

## Resumo do Modelo

O **Credit-Lens** é uma solução de credit scoring ponta a ponta projetada para prever o risco de inadimplência em operações de crédito ao consumidor com base no dataset [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data).

- **Data da versão**: Outubro/2026
- **Tipo de modelo**: Classificação binária supervisionada (Supervised Binary Classification)
- **Algoritmo campeão**: LightGBM Classifier com busca estocástica de hiperparâmetros (5-fold Stratified CV)
- **Baseline de comparação**: Regressão Logística com pré-processamento padronizado (`StandardScaler` + `SimpleImputer`)

---

## Métricas de Performance (Resultados Reais)

Avaliação realizada com partição estratificada 70% treino / 15% validação / 15% teste holdout (`random_state=42`):

| Métrica | Regressão Logística (Baseline) | LightGBM (Campeão) | Variação (Delta) |
|---|---|---|---|
| **ROC-AUC (Validação)** | 0.6775 | **0.7028** | +2.53 p.p. |
| **ROC-AUC (Teste Holdout)** | 0.6811 | **0.7093** | **+2.82 p.p.** |
| **Gini (Teste Holdout)** | 0.3621 | **0.4187** | **+5.66 p.p.** |
| **KS Statistic (Teste Holdout)** | 0.2667 | **0.3097** | **+4.30 p.p.** |
| **PR-AUC (Teste Holdout)** | 0.1601 | **0.1871** | +2.70 p.p. |
| **Brier Score (Calibração)** | 0.2256 | **0.2031** | -0.0225 (melhor) |
| **CV ROC-AUC Médio (5 folds)** | — | **0.7002 ± 0.0036** | Altamente estável |

### Hiperparâmetros Selecionados (LightGBM)
- `n_estimators`: 800
- `learning_rate`: 0.02
- `num_leaves`: 31
- `min_child_samples`: 100
- `subsample`: 0.7
- `colsample_bytree`: 0.7
- `class_weight`: balanced

---

## Variáveis Mais Importantes & Features Criadas

O pipeline DuckDB processou 307.511 clientes e construiu 32 preditores numéricos a partir de 4 tabelas relacionais:

1. **Birô Externo (`bureau.csv`)**:
   - `bureau_divida_total`, `bureau_credito_total`, `bureau_utilizacao_media`, `bureau_atraso_max_dias`
2. **Histórico de Aplicações Anteriores (`previous_application.csv`)**:
   - `prev_taxa_recusa`, `prev_n_recusados`, `prev_ratio_aprovado_pedido`
3. **Comportamento de Pagamento de Parcelas (`installments_payments.csv`)**:
   - `inst_taxa_em_dia`, `inst_atraso_medio_dias`, `inst_ratio_pagto_devido`
4. **Tratamento de Dados Cadastrais**:
   - Correção do valor sentinela `DAYS_EMPLOYED = 365243` com conversão para `NaN` e criação da flag binária `DAYS_EMPLOYED_ANOMALO`.

---

## Monitoramento e Estabilidade (PSI)

A estabilidade populacional entre a safra histórica e a safra recente (simulada via proxy temporal em `DAYS_BIRTH`) registrou:
- Variáveis cadastrais de idade/tempo de emprego com drift esperado (`DAYS_BIRTH` PSI alto como esperado no proxy).
- Variáveis transacionais e de birô estáveis (`bureau_utilizacao_media` PSI = 0.052, `prev_valor_pedido_medio` PSI = 0.093).
- Relatório automatizado disponível em `reports/monitoramento.csv`.

---

## Limitações e Uso Ético

- **Finalidade Exclusiva**: Projeto demonstrativo de engenharia de dados, machine learning e MLOps para portfólio.
- **Não Aplicabilidade**: Não deve ser utilizado para decisões reais de concessão de crédito a pessoas físicas sem auditoria de viés, conformidade regulatória (LGPD / BACEN) e validação em dados de produção.
