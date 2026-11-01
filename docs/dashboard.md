# Dashboard & Visualização — Credit-Lens

Camada analítica e de visualização do Credit-Lens. Os dados de entrada são gerados de forma reproduzível pelo código do projeto via `make bi` (`scripts/exportar_para_bi.py`), garantindo que todos os números coincidam estritamente com `models/metricas.json` e `reports/monitoramento.csv`.

---

## 1. Opção A: Cockpit Interativo em Python (Streamlit + Plotly)

A solução nativa integrada ao repositório oferece interatividade completa, versionamento em código e deploy direto no Streamlit Community Cloud.

> 🌐 **Deploy Público (Cockpit Online):** **[credit-lens-ecb.streamlit.app](https://credit-lens-ecb.streamlit.app)**

### Execução Local

```bash
make bi          # Gera ou atualiza os CSVs agregados em bi/
make dashboard   # Inicia a aplicação Streamlit em http://localhost:8501
```

### Funcionalidades do Cockpit

1. **Performance & Curva ROC:**
   - KPI cards de ROC-AUC, KS, Gini, Brier Score e ECE.
   - Curva ROC interativa comparando LightGBM (AUC 0.7093) vs. Regressão Logística (AUC 0.6811).
   - Tabela de métricas com validação cruzada 5-fold e calibração isotônica.
2. **Ordenação por Decis (Scorecard):**
   - Gráfico de barras da taxa de inadimplência (21.9% no Decil 1 até 1.9% no Decil 10).
   - Curva de Kolmogorov-Smirnov (KS Máximo = 30.97% no Decil 4).
   - Concentração de inadimplentes nos decis de maior risco (44.7% nos Decis 1 e 2).
3. **Simulador de Cutoff (Políticas de Concessão):**
   - Slider dinâmico de corte de probabilidade máxima.
   - Projeção imediata de: taxa de aprovação, inadimplência esperada na safra aprovada, inadimplentes evitados e volume financeiro concedido.
4. **Monitoramento Populacional (PSI):**
   - Gráfico de barras horizontais com cores semafóricas (estável, atenção, ação).
   - Filtro para ressalva técnica do proxy temporal de `DAYS_BIRTH`.
   - Protocolo formal de regras de retreino.
5. **Segmentos & Calibração:**
   - Avaliação por faixa etária e quartis de valor de crédito financiado (`AMT_CREDIT`).
   - Comparação da taxa de inadimplência real vs. score médio calibrado.
6. **Calculadora de Scoring Individual:**
   - Inferência em tempo real consumindo o modelo treinado (`modelo_lgbm_v2.joblib`).
   - Medidor (gauge) de probabilidade e recomendação de esteira (Aprovação Automática, Análise Manual, Recusa).

---

## 2. Opção B: Tableau Public (BI Tradicional)

Para candidatos e organizações que exigem visualização em Tableau, os CSVs de `bi/` são 100% compatíveis com o Tableau Desktop e Tableau Public.

> [!WARNING]
> O Tableau Public publica o workbook **abertamente**. Os CSVs em `bi/` contêm apenas dados **agregados**
> (sem `SK_ID_CURR` nem dados pessoais de clientes). Não adicione tabelas por cliente ao workbook.

### Fontes de Dados (`bi/`)

| Arquivo (`bi/`) | Granularidade | Colunas |
|---|---|---|
| `bi_metricas.csv` | modelo x conjunto x métrica | `modelo, conjunto, metrica, valor` |
| `bi_decis.csv` | modelo x decil (teste) | `decil, total_clientes, maus, bons, taxa_maus, acum_maus, acum_bons, ks` |
| `bi_roc.csv` | pontos da curva ROC (teste) | `modelo, fpr, tpr` |
| `bi_corte.csv` | varredura de ponto de corte (teste) | `corte_score, pct_aprovados, taxa_inadimplencia_aprovados, pct_maus_barrados` |
| `bi_segmentos.csv` | faixa etária e faixa de crédito (LightGBM, teste) | `dimensao, faixa, clientes, taxa_inadimplencia, score_medio` |
| `bi_psi.csv` | variável | `variavel, psi, classificacao, artefato_do_proxy` |

### Valores de Conferência (Teste Holdout)

| Modelo | ROC-AUC | Gini | KS | Brier (Calibrado) |
|---|---|---|---|---|
| LightGBM | 0,7093 | 0,4187 | 0,3097 | 0,2031 |
| LightGBM (calibrado) | 0,7085 | 0,4170 | 0,3086 | 0,0705 |
| Regressão logística | 0,6811 | 0,3621 | 0,2667 | 0,2256 |

---

## 3. Cuidados de Interpretação & Governança

- O pipeline prioriza o modelo **calibrado** (`modelo_lgbm_v2.joblib`), assegurando que o score médio previsto reflita a probabilidade real de inadimplência observada (~8.07%).
- O PSI elevado em `DAYS_BIRTH` decorre da simulação de corte temporal e está devidamente identificado como artefato de proxy metodológico (ver [`docs/model_card.md`](model_card.md)).
