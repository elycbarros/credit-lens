# Roteiro B — **Credit-Lens**: Risco de Crédito de Ponta a Ponta

> Do dado bruto ao monitoramento: SQL + Python + modelo de classificação + API + dashboard.
> Repositório novo, complementar ao `PROJETO_ML_T3` (KNN x Árvore) e ao `Power-Monitor` (SQL/pipeline/Tableau).

**Autor:** Ely Barros · **Versão do roteiro:** 1.0 · **Data:** 01/10/2026

---

## 1. Por que este projeto

Cobre as lacunas que as vagas de dados/crédito (ex.: Starta, fintechs, cooperativas) mais pedem e que o portfólio atual só toca de leve:

| Lacuna das vagas | Como o Credit-Lens cobre |
|---|---|
| Modelos de classificação/propensão | Regressão logística (baseline) + gradient boosting |
| Avaliação e validação de modelos | AUC, KS, Gini, PR-AUC, Brier, calibração, validação cruzada estratificada |
| Atualização/monitoramento de modelos | PSI, drift de variáveis, regra de retreino |
| Modelo em produção / esteira | API (FastAPI) + Docker + CI + versionamento de artefatos |
| SQL para cruzamento de bases | Feature store em DuckDB/SQLite com joins entre várias tabelas |
| Dashboard (Tableau/Power BI) | Painel de performance e monitoramento no Tableau Public |
| Contexto de crédito | Home Credit Default Risk (multi-tabelas, público) |

**Princípio ético do repositório:** projeto de estudo com dados públicos. Não é motor de concessão real, não usa dados pessoais e declara limitações (seção 9).

---

## 2. Escopo e decisão de dados

- **Base:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk) — `application_train`, `bureau`, `previous_application`, `installments_payments` etc.
- **Alvo:** `TARGET` (1 = dificuldade de pagamento).
- **Licença/uso:** baixar manualmente (exige conta Kaggle) e **não versionar** os dados (`data/raw/` no `.gitignore`).
- **Limitação conhecida:** a base **não tem datas de calendário**, então não há safras (vintages) reais nem validação out-of-time verdadeira. O monitoramento de drift será **simulado** por partições (ex.: ordenar por `DAYS_DECISION`/`DAYS_BIRTH` de forma documentada). Isso deve constar no model card.
- **Alternativa se quiser datas reais:** Lending Club (tem `issue_d`), permitindo validação temporal. Escolher **uma** base e manter o escopo.

---

## 3. Arquitetura

```
credit-lens/
├── data/                # raw/ (ignorado), interim/, processed/
├── sql/                 # features por tabela + joins (DuckDB)
│   ├── 01_bureau_agg.sql
│   ├── 02_prev_app_agg.sql
│   ├── 03_installments_agg.sql
│   └── 99_dataset_final.sql
├── src/credit_lens/
│   ├── features.py      # execução do SQL + tratamento
│   ├── split.py         # split estratificado/temporal simulado
│   ├── train.py         # logística + GBM, CV, salva modelo
│   ├── evaluate.py      # AUC, KS, Gini, PR-AUC, Brier, calibração
│   ├── monitor.py       # PSI, drift, relatório
│   └── api.py           # FastAPI /predict /health
├── notebooks/           # EDA e narrativa (opcional, saída limpa)
├── tests/               # pytest (métricas, PSI, contrato da API)
├── docs/                # model_card.md, dados_e_limitacoes.md, dashboard.md
├── bi/                  # CSVs exportados para Tableau
├── Dockerfile
├── Makefile
├── pyproject.toml / requirements*.txt
└── .github/workflows/ci.yml
```

**Stack:** Python 3.10+, pandas, DuckDB, scikit-learn, LightGBM (ou HistGradientBoosting para menos dependências), FastAPI, pytest, Docker, GitHub Actions, Tableau Public.

---

## 4. Fases e tarefas

### Fase 0 — Setup (0,5 dia)
- [ ] Criar repo `credit-lens`, licença MIT, `.gitignore` (dados, `.venv`, `._*`, `*.db`, `bi/*.csv` se grandes).
- [ ] `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `Makefile` (`make data`, `make train`, `make test`, `make api`).
- [ ] CI mínima: instalar deps + `pytest`.
- **Entrega:** repo roda `make test` verde (mesmo com testes triviais).

### Fase 1 — Dados e SQL (1–2 dias)  → *lacuna: SQL de cruzamento*
- [ ] Carregar CSVs no DuckDB (sem carregar tudo em memória do pandas).
- [ ] Agregações por cliente (`SK_ID_CURR`): bureau (nº créditos, atraso máximo, utilização), pedidos anteriores (taxa de aprovação/recusa), parcelas (atraso médio, pagamento a menor).
- [ ] Join final 1 linha por cliente; checar duplicidade de chave e vazamento (colunas que usam informação posterior ao alvo).
- [ ] Teste: 1 linha por `SK_ID_CURR`; nº de linhas = `application_train`.
- **Entrega:** `sql/*.sql` versionado + `docs/dados_e_limitacoes.md`.

### Fase 2 — EDA enxuta e qualidade (1 dia)
- [ ] Desbalanceamento do alvo (~8% de inadimplência), nulos, outliers (ex.: `DAYS_EMPLOYED = 365243`).
- [ ] 5–6 gráficos que sustentam decisões de tratamento, não galeria.
- **Entrega:** notebook curto com conclusões escritas.

### Fase 3 — Modelagem (2–3 dias)  → *lacuna: classificação/propensão*
- [ ] Split estratificado treino/validação/teste (teste intocado até o fim).
- [ ] **Baseline:** regressão logística com pipeline (imputação, escala, codificação) — interpretável.
- [ ] **Modelo principal:** gradient boosting com validação cruzada estratificada e busca enxuta de hiperparâmetros.
- [ ] Tratar desbalanceamento (pesos de classe) e comparar com/sem.
- [ ] Importância de variáveis + SHAP (opcional) e leitura de negócio.
- **Entrega:** `train.py` reprodutível (seed fixa) salvando modelo + métricas em JSON.

### Fase 4 — Avaliação e validação (1–2 dias)  → *lacuna: avaliação/validação*
- [ ] Métricas: **ROC-AUC, Gini (=2·AUC−1), KS, PR-AUC, Brier**, curva de calibração.
- [ ] Tabela por **decis de score** (taxa de inadimplência por faixa, lift acumulado).
- [ ] Comparar logística x GBM no teste; discutir ganho vs. interpretabilidade.
- [ ] Ponto de corte: curva custo-benefício com premissas explícitas e fictícias (custo de perda x margem), claramente rotuladas como hipótese.
- **Entrega:** `evaluate.py` + testes unitários das métricas (ex.: KS e Gini em casos conhecidos).

### Fase 5 — Monitoramento (1–2 dias)  → *lacuna: atualização de modelos*
- [ ] Implementar **PSI** por variável e do score entre referência (treino) e partições simuladas.
- [ ] Regras de alerta: PSI < 0,10 estável · 0,10–0,25 atenção · > 0,25 ação (convenção de mercado; documentar).
- [ ] Relatório de monitoramento (CSV + HTML estático).
- [ ] Regra de retreino documentada (gatilho: PSI do score > 0,25 ou queda de KS/AUC além de tolerância).
- **Entrega:** `monitor.py` + testes (PSI = 0 para distribuições idênticas; > 0 quando deslocadas).
- **Honestidade:** deixar claro que o drift é simulado (seção 2).

### Fase 6 — Serviço e esteira (1–2 dias)  → *lacuna: modelo em produção*
- [ ] API FastAPI: `POST /predict` (valida entrada com pydantic, devolve probabilidade e faixa de risco), `GET /health`, `GET /model-info` (versão, data, métricas).
- [ ] Testes de contrato da API (`TestClient`).
- [ ] `Dockerfile` enxuto; `docker run` documentado.
- [ ] CI: lint (ruff), pytest com cobertura, build da imagem.
- [ ] Versionar artefato do modelo (nome com versão/hash) fora do Git (release ou pasta ignorada).
- **Entrega:** `make api` sobe localmente e responde.

### Fase 7 — Dashboard no Tableau Public (1–2 dias)  → *lacuna: BI*
- [ ] Exportar CSVs (`bi/`): score por decil, métricas, PSI por variável, importância.
- [ ] Páginas: **Performance** (ROC/KS/decis), **Risco por segmento**, **Monitoramento** (PSI), **Qualidade dos dados**.
- [ ] Parâmetro de ponto de corte para simular aprovação x inadimplência esperada.
- [ ] Publicar e linkar no README. Só dados públicos e agregados.
- **Entrega:** `docs/dashboard.md` + link público.

### Fase 8 — Documentação e vitrine (1 dia)
- [ ] `README` com: pergunta de negócio, dados, resultados (números reais do seu run), como reproduzir, limitações.
- [ ] `docs/model_card.md` (uso pretendido, dados, métricas, limitações, riscos de viés).
- [ ] Seção de **vieses e uso responsável** (variáveis sensíveis como gênero/idade: avaliar impacto, não usar para decisão real).
- [ ] Badge de CI, GIF/print do dashboard.

**Estimativa total:** ~10–14 dias de trabalho dedicado (ritmo parcial: 3–5 semanas).

---

## 5. Critérios de aceite (definição de pronto)

- [ ] `make data && make train && make test` roda do zero em máquina limpa.
- [ ] Métricas do README **iguais** às geradas pelo código (sem números "de cabeça").
- [ ] Teste cobrindo: unicidade de chave, KS/Gini, PSI, contrato da API.
- [ ] CI verde no GitHub; imagem Docker constrói.
- [ ] Model card e seção de limitações presentes.
- [ ] Dashboard publicado com link no README.

---

## 6. Resultado esperado (faixas de referência — não prometer, medir)

Em Home Credit, modelos bem ajustados costumam ficar em torno de **AUC 0,76–0,79** com GBM e **0,73–0,75** com logística após feature engineering multi-tabelas. Usar apenas como sanidade: se aparecer AUC > 0,85, suspeitar de **vazamento**.

---

## 7. Como apresentar em candidatura (sem inflar)

**Frases seguras (se feitas de fato):**
- "Construí pipeline de risco de crédito com SQL (DuckDB), regressão logística e gradient boosting, avaliado por AUC, KS e Gini."
- "Implementei monitoramento de estabilidade (PSI) e API de scoring containerizada com CI."
- "Dashboard de performance e monitoramento do modelo em Tableau Public."

**Não afirmar:** experiência em instituição financeira, modelos em produção real, nem uso de dados de clientes. É projeto de portfólio com dados públicos.

**Mapeamento por tipo de vaga:**

| Vaga | Destacar |
|---|---|
| Cientista/Analista de dados (crédito) | Fases 3–5 |
| Analista de BI | Fases 1 e 7 |
| Analista de risco | Fases 4–5 + model card |
| ML/MLOps júnior | Fase 6 + CI/Docker |

---

## 8. Riscos do projeto e mitigação

| Risco | Mitigação |
|---|---|
| Escopo crescer demais | Fechar fases em ordem; Fase 6 e 7 podem ser "v1.1" |
| Vazamento de alvo | Revisar cada feature; teste de AUC suspeito; split antes de agregações dependentes do alvo |
| Dados grandes na máquina | DuckDB + Parquet; amostrar para desenvolvimento |
| Resultados "bons demais" | Comparar com baseline e faixa da seção 6 |
| Dados no GitHub | `.gitignore` + checklist antes de cada push |

---

## 9. Limitações a declarar no README

1. Base pública sem datas → sem safras reais; drift simulado.
2. Custos/margens do ponto de corte são hipóteses ilustrativas.
3. Não há política de crédito, regulatória (Res. CMN/BCB) nem explicabilidade formal exigida em produção.
4. Possíveis vieses em variáveis demográficas; análise de impacto é exploratória.

---

## 10. Primeiros passos (hoje)

1. Criar o repo `credit-lens` e baixar a base Kaggle em `data/raw/`.
2. Fase 1: escrever `01_bureau_agg.sql` e o teste de unicidade da chave.
3. Commit pequeno e CI verde antes de modelar.

> **Dica de nomes alternativos:** `Credit-Lens` (lente sobre o risco) · `RiskPath` · `ScoreLab` · `Inadimplo-Scope`.
