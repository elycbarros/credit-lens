# Roadmap — Credit-Lens

Guia de execução para continuar no VS Code. Marque os itens conforme avançar; cada fase termina com um **commit** e **CI verde**.

**Legenda:** ✅ pronto · ⬜ a fazer · 🎯 entrega da fase · ⚠️ cuidado

---

## Ambiente no VS Code (uma vez)

```bash
cd /Volumes/KINGSTON/DEV2026/MAC/credit-lens
python3 -m venv .venv && source .venv/bin/activate
make install
make test            # deve passar 7 testes
git init && git add . && git commit -m "chore: esqueleto Credit-Lens (fase 0)"
```

- Extensões: **Python**, **Pylance**, **Ruff**, **SQLTools** (ou "DuckDB SQL Tools"), **Jupyter**, **Docker**, **GitHub Actions**.
- Interpretador: `Cmd+Shift+P` → *Python: Select Interpreter* → `.venv`.
- Testes: ícone de frasco (Testing) → pytest já configurado via `pytest.ini`.
- ⚠️ Projeto está em volume externo (KINGSTON): se o git reclamar de `index.lock`, ejete/remonte o volume ou mova o repo para o disco interno.
- Criar repo vazio no GitHub e: `git remote add origin ... && git push -u origin main`.

---

## ✅ Fase 0 — Setup
- ✅ Estrutura de pastas, `Makefile`, `requirements*.txt`, `pytest.ini`, `Dockerfile`, CI.
- ✅ `evaluate.py` (Gini, KS, AUC, PR-AUC, Brier), `monitor.py` (PSI), `/health`, 7 testes.
- ⬜ Primeiro push e conferir CI verde no GitHub.

---

## ⬜ Fase 1 — Dados e SQL (1–2 dias)
**Objetivo:** 1 linha por cliente, sem duplicidade e sem vazamento.

1. ⬜ Baixar a base do Kaggle (Home Credit Default Risk) e colocar em `data/raw/`:
   `application_train.csv`, `bureau.csv`, `previous_application.csv`, `installments_payments.csv`.
2. ⬜ Implementar `src/credit_lens/features.py`:
   - criar `data/processed/credit.duckdb`;
   - `read_csv_auto` de cada CSV como tabela;
   - executar `sql/01..03` como `CREATE TABLE b_agg/p_agg/i_agg AS ...`;
   - executar `sql/99_dataset_final.sql` e exportar `data/processed/dataset.parquet`.
3. ⬜ Testar o SQL no VS Code antes de automatizar (rodar cada `.sql` manualmente).
4. ⬜ Escrever `tests/test_dataset.py` (marcado `@pytest.mark.skipif` se não houver dados):
   - `SK_ID_CURR` único;
   - nº de linhas = `application_train`;
   - `TARGET` só 0/1.
5. ⬜ Revisar vazamento: nenhuma feature usa informação posterior à decisão do empréstimo atual (ex.: em `installments_payments`, filtrar parcelas do próprio contrato atual se existirem).
6. ⬜ Completar `docs/dados_e_limitacoes.md` com shape final, % de nulos e taxa de inadimplência.

🎯 **Entrega:** `make features` gera o parquet; testes de unicidade passam. Commit: `feat: dataset via DuckDB`.

---

## ⬜ Fase 2 — EDA enxuta (1 dia)
- ⬜ `notebooks/01_eda.ipynb` com 5–6 gráficos que **justificam decisões**:
  - desbalanceamento do alvo (~8%);
  - nulos por coluna (decidir imputação);
  - `DAYS_EMPLOYED = 365243` (valor sentinela → NaN + flag);
  - distribuição de renda/crédito (cauda longa → log);
  - taxa de inadimplência por faixa de idade e por `bureau_n_creditos`.
- ⬜ Limpar saídas do notebook antes do commit (ou usar `nbstripout`).

🎯 **Entrega:** conclusões escritas em Markdown no notebook. Commit: `docs: EDA`.

---

## ⬜ Fase 3 — Modelagem (2–3 dias)
1. ⬜ `split.py`: split estratificado treino/validação/teste (ex.: 70/15/15), `random_state` fixo; **teste só no final**.
2. ⬜ Pré-processamento em `Pipeline` (imputação, escala para a logística, codificação de categóricas); ajustar **só no treino**.
3. ⬜ **Baseline:** `LogisticRegression` com `class_weight` (comparar com/sem).
4. ⬜ **Modelo principal:** `HistGradientBoostingClassifier` (sem dependência extra) ou LightGBM.
5. ⬜ Validação cruzada estratificada (5 folds) + busca enxuta de hiperparâmetros (`RandomizedSearchCV`, poucas iterações).
6. ⬜ `train.py`: salvar `models/modelo_vX.joblib` + `models/metricas.json` (⚠️ modelo fora do Git).
7. ⬜ Importância de variáveis (permutation importance; SHAP opcional).

🎯 **Entrega:** `make train` reprodutível. Commit: `feat: treino logística e GBM`.

---

## ⬜ Fase 4 — Avaliação e validação (1–2 dias)
- ⬜ Rodar `resumo_metricas` em validação e teste para os dois modelos.
- ⬜ Tabela por **decis de score** (n, inadimplência %, lift acumulado) em `evaluate.py` + teste.
- ✅ Curva de calibração e Brier; calibração isotônica aplicada (Brier 0,2031 → 0,0705).
- ⬜ Ponto de corte: curva aprovação × inadimplência esperada, com custos **fictícios e rotulados como hipótese**.
- ⬜ Sanidade: AUC esperado ~0,73–0,79; **AUC > 0,85 = suspeitar de vazamento**.
- ⬜ Preencher `docs/model_card.md` com números reais.

🎯 **Entrega:** `reports/avaliacao.md` ou seção no README com tabela comparativa. Commit: `feat: avaliação e decis`.

---

## ⬜ Fase 5 — Monitoramento (1–2 dias)
- ⬜ Definir partições simuladas (ex.: ordenar por `DAYS_BIRTH` ou `DAYS_DECISION` em blocos) e **documentar que o drift é simulado**.
- ⬜ PSI por variável e do score (referência = treino), com classificação estável/atenção/ação.
- ⬜ Relatório `reports/monitoramento.csv` (+ HTML estático opcional).
- ⬜ Regra de retreino documentada: PSI do score > 0,25 ou queda de KS/AUC além da tolerância definida.
- ⬜ Testes extras em `test_monitor.py` (variável categórica, bins com poucos casos).

🎯 **Entrega:** `make monitor`. Commit: `feat: monitoramento PSI`.

---

## ⬜ Fase 6 — API, Docker e esteira (1–2 dias)
- ⬜ `api.py`: `POST /predict` (schema pydantic, validação, retorna probabilidade e faixa de risco), `GET /model-info` (versão, data, métricas).
- ⬜ Carregar o modelo no startup; erro claro se ausente.
- ⬜ Testes de contrato (entrada inválida → 422; saída entre 0 e 1).
- ⬜ `docker build -t credit-lens . && docker run -p 8000:8000 credit-lens`; testar em `/docs`.
- ⬜ CI: adicionar passo de build da imagem.

🎯 **Entrega:** API responde localmente e em container. Commit: `feat: API de scoring`.

---

## 🟡 Fase 7 — Dashboard Tableau Public (1–2 dias) — exportação pronta; falta montar e publicar no Tableau
- ✅ `scripts/exportar_para_bi.py` (mesmo padrão do Power-Monitor): decis, métricas, PSI, importância → `bi/`.
- ⬜ Páginas: **Performance** (decis/KS), **Risco por segmento**, **Monitoramento** (PSI), **Qualidade dos dados**.
- ⬜ Parâmetro de ponto de corte no Tableau.
- ⬜ Publicar (⚠️ público: só dados agregados) e linkar no README; completar `docs/dashboard.md` com valores de conferência.

🎯 **Entrega:** link público do dashboard. Commit: `docs: dashboard`.

---

## ⬜ Fase 8 — Vitrine (1 dia)
- ⬜ README final: pergunta de negócio, dados, **resultados reais**, como reproduzir, limitações, link do dashboard.
- ⬜ Model card completo + seção de vieses e uso responsável.
- ⬜ Badges (CI, cobertura), print/GIF do dashboard, tag `v1.0.0` e release.
- ⬜ Atualizar currículo/LinkedIn **só com o que foi de fato entregue**.

---

## Critérios de aceite (v1.0)
- [ ] `make install && make features && make train && make test` roda do zero.
- [ ] Números do README = números gerados pelo código.
- [ ] Testes: chave única, KS/Gini, PSI, contrato da API.
- [ ] CI verde; imagem Docker constrói.
- [ ] Model card e limitações presentes; dashboard publicado.

## Cronograma sugerido (ritmo parcial)

| Semana | Fases |
|---|---|
| 1 | 1 e 2 |
| 2 | 3 e 4 |
| 3 | 5 e 6 |
| 4 | 7 e 8 |

> **Corte de escopo, se apertar:** entregar v1.0 até a Fase 5 + README (já cobre classificação, validação e monitoramento); Fases 6 e 7 viram v1.1.

## Convenções de commit
`feat:` funcionalidade · `fix:` correção · `docs:` documentação · `test:` testes · `chore:` infra. Um commit por tarefa concluída, mensagens curtas.
