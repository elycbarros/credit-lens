"""API de scoring (Fase 6).

Endpoints:
  GET  /health      → status da API e versão.
  GET  /model-info  → metadados do modelo carregado (versão, métricas, data).
  POST /predict     → recebe features, retorna probabilidade de inadimplência e faixa de risco.

Uso local:
    make api          # uvicorn com --reload
    make api-docker   # via container

Exemplo de payload para /predict:
    {
        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 450000,
        "AMT_ANNUITY": 20000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000
    }
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from credit_lens import __version__

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
MODELS = RAIZ / "models"

# ── Estado global (carregado no startup) ──────────────────────────────────────
_state: dict[str, Any] = {"modelo": None, "metricas": None}


def _encontrar_modelo() -> Optional[Path]:
    """Retorna o modelo campeão ou o arquivo .joblib mais recente em models/."""
    arquivos = list(MODELS.glob("modelo_*.joblib"))
    if not arquivos:
        return None
    lgbm_arquivos = [f for f in arquivos if "lgbm" in f.name]
    if lgbm_arquivos:
        return max(lgbm_arquivos, key=lambda f: f.stat().st_mtime)
    return max(arquivos, key=lambda f: f.stat().st_mtime)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Carrega o modelo na inicialização da API."""
    caminho = _encontrar_modelo()
    if caminho:
        logger.info("Carregando modelo: %s", caminho)
        _state["modelo"] = joblib.load(caminho)
        _state["modelo_nome"] = caminho.name

        metricas_path = MODELS / "metricas.json"
        if metricas_path.exists():
            historico = json.loads(metricas_path.read_text())
            # Pega a entrada mais recente com o mesmo arquivo
            for entry in reversed(historico):
                if entry.get("arquivo") == caminho.name:
                    _state["metricas"] = entry
                    break
    else:
        logger.warning("Nenhum modelo encontrado em %s. Execute `make train` primeiro.", MODELS)
    yield
    _state.clear()


app = FastAPI(
    title="Credit-Lens",
    version=__version__,
    description="API de scoring de risco de crédito (portfólio — dados públicos).",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class EntradaPredicao(BaseModel):
    """Features mínimas para o modelo de scoring.

    Todas as colunas do dataset original podem ser passadas; campos ausentes
    serão imputados pelo pipeline do modelo.
    """

    AMT_INCOME_TOTAL: Optional[float] = Field(None, description="Renda anual declarada.")
    AMT_CREDIT: Optional[float] = Field(None, description="Valor do crédito solicitado.")
    AMT_ANNUITY: Optional[float] = Field(None, description="Anuidade (parcela mensal × 12).")
    DAYS_BIRTH: Optional[int] = Field(None, description="Idade em dias (negativo).")
    DAYS_EMPLOYED: Optional[int] = Field(None, description="Tempo de emprego em dias (negativo).")

    model_config = {"extra": "allow"}  # aceita campos extras do dataset completo

    @model_validator(mode="before")
    @classmethod
    def pelo_menos_um_campo(cls, values: dict) -> dict:
        if not any(v is not None for v in values.values()):
            raise ValueError("Ao menos um campo deve ser informado.")
        return values


class SaidaPredicao(BaseModel):
    probabilidade_inadimplencia: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidade calibrada de inadimplência (0–1). Só é uma probabilidade se o modelo foi calibrado (`calibrado: true` em /model-info)."
    )
    faixa_risco: str = Field(..., description="Classificação: baixo | médio | alto.")
    modelo: str = Field(..., description="Nome do arquivo de modelo utilizado.")


# Cortes sobre a probabilidade CALIBRADA (taxa base ~8%). Referência nos decis do teste:
# decis 6-10 têm inadimplência < 6%; decis 1-2 ficam acima de ~14%.
CORTE_RISCO_BAIXO = 0.06
CORTE_RISCO_ALTO = 0.15


def _classificar_risco(prob: float) -> str:
    if prob < CORTE_RISCO_BAIXO:
        return "baixo"
    if prob < CORTE_RISCO_ALTO:
        return "médio"
    return "alto"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
def health() -> dict[str, str]:
    """Verifica se a API está no ar."""
    return {"status": "ok", "versao": __version__}


@app.get("/model-info", tags=["infra"])
def model_info() -> dict[str, Any]:
    """Retorna metadados do modelo carregado (versão, métricas, parâmetros)."""
    if _state.get("modelo") is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute `make train` para treinar um modelo.",
        )
    return {
        "modelo": _state.get("modelo_nome", "desconhecido"),
        "api_versao": __version__,
        "metricas": _state.get("metricas"),
    }


@app.post("/predict", response_model=SaidaPredicao, tags=["scoring"])
def predict(entrada: EntradaPredicao) -> SaidaPredicao:
    """Retorna a probabilidade de inadimplência e a faixa de risco para um cliente."""
    if _state.get("modelo") is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute `make train` para treinar um modelo.",
        )

    # Converte entrada para DataFrame (1 linha) e alinha colunas com o modelo
    dados = pd.DataFrame([entrada.model_dump()])
    modelo = _state["modelo"]
    if hasattr(modelo, "feature_names_in_"):
        dados = dados.reindex(columns=modelo.feature_names_in_, fill_value=float("nan"))

    prob = float(modelo.predict_proba(dados)[0, 1])

    return SaidaPredicao(
        probabilidade_inadimplencia=round(prob, 6),
        faixa_risco=_classificar_risco(prob),
        modelo=_state.get("modelo_nome", "desconhecido"),
    )
