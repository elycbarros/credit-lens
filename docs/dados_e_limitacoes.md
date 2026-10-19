# Dados e Limitações do Projeto

## 1. Dados de Origem

- **Fonte**: Competição [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk/data).
- **Download**: Feito manualmente; os arquivos brutos em `data/raw/` não são versionados no Git (listados em `.gitignore`).
- **Dimensões do Dataset Processado**:
  - **Volume total**: 307.511 clientes (1 linha por `SK_ID_CURR`).
  - **Features**: 34 colunas (32 preditores numéricos + chave `SK_ID_CURR` + variável alvo `TARGET`).
  - **Taxa de Inadimplência**: 8,07% (24.825 maus pagadores vs. 282.686 adimplentes).
  - **Armazenamento**: `data/processed/dataset.parquet` (20 MB comprimido com snappy/pyarrow).

---

## 2. Tratamentos Realizados no Pipeline DuckDB

1. **Agregações Relacionais**:
   - `bureau.csv` (162 MB) → consolidação de dívida total, limite de crédito e atrasos.
   - `previous_application.csv` (386 MB) → contagem de pedidos, taxas de aprovação, recusa e cancelamento.
   - `installments_payments.csv` (690 MB) → contagem de parcelas pagas, média de dias de atraso e razão pago/devido.
2. **Tratamento de Anomalia Cadastral**:
   - 55.374 registros com o valor sentinela `DAYS_EMPLOYED = 365243` (~1000 anos, indicando aposentados ou sem vínculo empregatício formal) foram convertidos para `NaN` e sinalizados com a flag booleana `DAYS_EMPLOYED_ANOMALO = 1`.

---

## 3. Limitações Conhecidas

- **Ausência de Datas de Calendário Reais**: O dataset original não fornece timestamps de calendário (usa dias relativos à aplicação). Por isso, o monitoramento populacional (PSI) e a partição temporal utilizam proxies simulados documentados (`DAYS_BIRTH` ou ordem relativa).
- **Hipóteses Financeiras**: Quaisquer parâmetros de taxa de juros, custo de captação ou perda em caso de default (LGD) utilizados na matriz de confusão e no corte de decisão são **hipóteses didáticas e ilustrativas**.
- **Finalidade do Projeto**: Este repositório é um portfólio acadêmico e técnico de engenharia de software e machine learning aplicado. **Não deve ser utilizado para decisões reais de concessão de crédito.**

---

## 4. Atributos pessoais e uso responsável

O modelo usa **`DAYS_BIRTH`** (idade em dias relativos) e **`DAYS_EMPLOYED`** / **`DAYS_EMPLOYED_ANOMALO`** (tempo de emprego formal e indicador de ausência de vínculo) como preditores. A inclusão da idade e do tempo de vínculo profissional atua como proxy do ciclo de vida financeiro e da capacidade produtiva na base histórica. Em contrapartida, variáveis pessoais sensíveis presentes na base original do Kaggle — como **sexo (`CODE_GENDER`)**, **estado civil (`NAME_FAMILY_STATUS`)**, **filhos/dependentes (`CNT_CHILDREN`)** e **escolaridade (`NAME_EDUCATION_TYPE`)** — foram deliberadamente **excluídas** na montagem do dataset (`sql/99_dataset_final.sql`) para mitigar discriminação direta e concentrar a análise em capacidade financeira e histórico comportamental de crédito. Em uso real, seria necessária análise de viés entre grupos, base legal conforme a LGPD, validação independente do modelo e acompanhamento de desempenho por segmento. Este projeto é demonstrativo e não deve ser usado em decisões reais de crédito.

