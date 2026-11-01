# Roadmap — Credit-Lens

Guia de execução e controle de entregas do projeto.

**Legenda:** ✅ pronto · ⬜ a fazer · 🎯 entrega da fase · ⚠️ cuidado

---

## Ambiente no VS Code (uma vez)

```bash
cd /Volumes/KINGSTON/DEV2026/MAC/credit-lens
python3 -m venv .venv && source .venv/bin/activate
make install
make test            # 36 testes passando (88.46% coverage)
```

- Interpretador: `Cmd+Shift+P` → *Python: Select Interpreter* → `.venv`.
- Testes: ícone de frasco (Testing) → pytest já configurado via `pytest.ini`.

---

## ✅ Fase 0 — Setup
- ✅ Estrutura de pastas, `Makefile`, `requirements*.txt`, `pytest.ini`, `Dockerfile`, CI.
- ✅ `evaluate.py` (Gini, KS, AUC, PR-AUC, Brier), `monitor.py` (PSI), `/health`.
- ✅ CI verde no GitHub Actions.

---

## ✅ Fase 1 — Dados e SQL
**Objetivo:** 1 linha por cliente, sem duplicidade e sem vazamento.
- ✅ Ingestão analítica DuckDB em `src/credit_lens/features.py`.
- ✅ Agregações SQL multi-tabelas (`sql/01_bureau_agg.sql`, `sql/02_prev_app_agg.sql`, `sql/03_installments_agg.sql`).
- ✅ Dataset final consolidado em Parquet (307.511 registros, 32 preditores).
- ✅ Testes de integridade e unicidade de chave primária em `tests/test_dataset.py`.

🎯 **Entrega:** `make features` gera o parquet; testes de unicidade passam.

---

## ✅ Fase 2 — EDA Enxuta
- ✅ `notebooks/01_eda.ipynb` com análises que justificam decisões:
  - desbalanceamento do alvo (~8%);
  - nulos por coluna e estratégias de imputação;
  - sentinela `DAYS_EMPLOYED = 365243` tratado com flag anômala;
  - distribuições de renda e crédito;
  - inadimplência por faixas de idade e bureau.

🎯 **Entrega:** Conclusões e decisões documentadas.

---

## ✅ Fase 3 — Modelagem
- ✅ `split.py`: Split estratificado treino/validação/teste (70/15/15) com seed fixa.
- ✅ Pré-processamento em pipeline com imputação e encoders.
- ✅ Baseline de Regressão Logística com pesos balanceados (`modelo_logistica_v1.joblib`).
- ✅ LightGBM com tuning via RandomizedSearchCV 5-fold CV (`modelo_lgbm_v1.joblib`).
- ✅ Artefatos versionados e métricas salvas em `models/metricas.json`.

🎯 **Entrega:** `make train` e `make train-logistica` reprodutíveis.

---

## ✅ Fase 4 — Avaliação, Validação e Calibração
- ✅ Métricas consolidadas: ROC-AUC, KS, Gini, PR-AUC e Brier Score.
- ✅ Tabela de decis de score com taxa de inadimplência e lift acumulado.
- ✅ **Calibração Isotônica:** Probabilidades alinhadas à taxa real observada (Brier 0.2031 → 0.0705, ECE 0.0026).
- ✅ Varredura de pontos de corte com tradeoff de aprovação vs. inadimplência.
- ✅ Documentação completa em `docs/model_card.md` com números reais.

🎯 **Entrega:** Model Card completo e modelos calibrados (`modelo_lgbm_v2.joblib`).

---

## ✅ Fase 5 — Monitoramento (PSI)
- ✅ Cálculo do Population Stability Index (PSI) em `src/credit_lens/monitor.py`.
- ✅ Partições temporais simuladas para análise de drift.
- ✅ Regras e convenção de mercado (<0.10 estável, 0.10–0.25 atenção, >0.25 ação).
- ✅ Relatório em `reports/monitoramento.csv` e suíte de testes em `tests/test_monitor.py`.

🎯 **Entrega:** `make monitor`.

---

## ✅ Fase 6 — API REST & Docker
- ✅ API FastAPI em `src/credit_lens/api.py`: `POST /predict`, `GET /health`, `GET /model-info`.
- ✅ Validação de schemas Pydantic e categorização em faixas de risco (baixo, médio, alto).
- ✅ Startup com lifespan carregando automaticamente o modelo calibrado.
- ✅ Container `Dockerfile` com suporte a OpenMP (`libgomp1`) para LightGBM.
- ✅ Testes de contrato e smoke test do container no CI.

🎯 **Entrega:** API respondendo localmente (`make api`) e em container.

---

## ✅ Fase 7 — Cockpit Interativo em Streamlit & BI
- ✅ Exportador `scripts/exportar_para_bi.py` gerando dados agregados em `bi/`.
- ✅ Aplicação interativa em `src/credit_lens/dashboard.py` com 6 visões analíticas:
  1. Performance & Curva ROC Comparativa.
  2. Ordenação por Decis (Scorecard) e KS Máximo.
  3. Simulador de Política de Concessão (Cutoff Slider & Impacto Financeiro).
  4. Monitoramento Populacional (PSI) com semáforo de estabilidade.
  5. Segmentos & Calibração (faixa etária e faixas de crédito).
  6. Calculadora de Scoring Online com inferência em tempo real.
- ✅ Deploy público ativo no Streamlit Community Cloud: [`credit-lens-ecb.streamlit.app`](https://credit-lens-ecb.streamlit.app).

🎯 **Entrega:** Cockpit no ar e documentado em `docs/dashboard.md`.

---

## ✅ Fase 8 — Vitrine e Release v1.0.0
- ✅ `README.md` completo com resultados reais, diagrama arquitetural, preview do dashboard e badges.
- ✅ Limitações éticas e de conformidade declaradas em `docs/dados_e_limitacoes.md`.
- ✅ 36 testes automatizados passando (100%) com 88.46% de cobertura.
- ✅ Tag de versão `v1.0.0` registrada no Git.

---

## Critérios de aceite (v1.0) — Todos Aprovados
- [x] `make install && make features && make train && make test` roda do zero.
- [x] Números do README = números gerados pelo código.
- [x] Testes: chave única, KS/Gini, PSI, contrato da API, carregamento do dashboard.
- [x] CI verde (4/4); imagem Docker constrói e passa no smoke test.
- [x] Model card e limitações presentes; dashboard publicado.
