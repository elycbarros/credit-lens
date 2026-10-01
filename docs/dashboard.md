# Dashboard no Tableau Public

Camada de visualização do Credit-Lens. Os CSVs são gerados pelo próprio código do projeto
(`scripts/exportar_para_bi.py`), então os números do dashboard coincidem com `models/metricas.json`
e `reports/monitoramento.csv`.

> [!WARNING]
> O Tableau Public publica o workbook **abertamente**. Os CSVs abaixo contêm apenas dados **agregados**
> (sem `SK_ID_CURR` nem linhas de clientes). Não adicione tabelas por cliente ao workbook.

## 1. Gerar os dados

```bash
make train            # se ainda não houver modelos em models/
make bi               # = python scripts/exportar_para_bi.py
```

| Arquivo (`bi/`) | Granularidade | Colunas |
|---|---|---|
| `bi_metricas.csv` | modelo x conjunto x métrica | `modelo, conjunto, metrica, valor` |
| `bi_decis.csv` | modelo x decil (teste) | `decil, total_clientes, maus, bons, taxa_maus, acum_maus, acum_bons, ks` |
| `bi_roc.csv` | pontos da curva ROC (teste) | `modelo, fpr, tpr` |
| `bi_corte.csv` | varredura de ponto de corte (teste) | `corte_score, pct_aprovados, taxa_inadimplencia_aprovados, pct_maus_barrados` |
| `bi_segmentos.csv` | faixa etária e faixa de crédito (LightGBM, teste) | `dimensao, faixa, clientes, taxa_inadimplencia, score_medio` |
| `bi_psi.csv` | variável | `variavel, psi, classificacao, artefato_do_proxy` |

## 2. Páginas sugeridas

1. **Performance:** cartões (ROC-AUC, Gini, KS do LightGBM x logística) + curva ROC (`bi_roc`).
2. **Decis de risco:** barras de `taxa_maus` por decil + linha de KS acumulado (`bi_decis`).
3. **Ponto de corte:** parâmetro/slider sobre `bi_corte` (aprovação x inadimplência entre aprovados).
4. **Segmentos:** taxa de inadimplência por faixa etária e de crédito (`bi_segmentos`).
5. **Monitoramento:** barras de PSI com faixas 0,10 / 0,25 (`bi_psi`). **Excluir ou sinalizar** as linhas com `artefato_do_proxy = true`.

## 3. Valores de conferência (teste holdout)

| Modelo | ROC-AUC | Gini | KS |
|---|---|---|---|
| LightGBM | 0,7093 | 0,4187 | 0,3097 |
| LightGBM (calibrado) | 0,7085 | 0,4170 | 0,3086 |
| Regressão logística | 0,6811 | 0,3621 | 0,2667 |

Taxa de inadimplência por decil (LightGBM): decil 1 = 22,2%; decil 10 = 1,8%.

## 4. Cuidados de interpretação

- O exportador usa a versão **calibrada** de cada modelo (maior `_vN`), então `score_medio` ≈ taxa de inadimplência observada (ex.: <30 anos: 11,5% previsto x 11,4% observado). Os modelos brutos continuam em `bi_metricas.csv` para comparação (Brier 0,2031 → 0,0705 no LightGBM). Para comunicar risco, prefira decis e taxas observadas.
- O PSI de `DAYS_BIRTH` é artefato do proxy temporal (ver `docs/model_card.md`): não é alerta de retreino.
- Faixa etária é mostrada só como análise exploratória de segmentos; o projeto não deve ser usado para decisão real de crédito.

## 5. Publicação

1. Tableau Public → Conectar → Arquivo de texto → `bi/bi_decis.csv` (adicionar os demais como fontes separadas).
2. Montar as páginas, publicar e colar o link aqui e no README.

**Link do dashboard:** _a preencher após a publicação_
