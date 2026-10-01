"""Testes de contrato da API Credit-Lens."""

import pytest
from fastapi.testclient import TestClient

from credit_lens.api import MODELS, app

client = TestClient(app)


def test_health():
    """GET /health deve retornar 200 com status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "versao" in r.json()


def test_model_info_sem_modelo():
    """GET /model-info sem modelo treinado deve retornar 503."""
    r = client.get("/model-info")
    assert r.status_code == 503


def test_predict_sem_modelo():
    """POST /predict sem modelo treinado deve retornar 503."""
    r = client.post("/predict", json={"AMT_INCOME_TOTAL": 150000})
    assert r.status_code == 503


def test_predict_payload_vazio():
    """POST /predict com payload vazio deve retornar 422 (validação Pydantic)."""
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_com_modelo_mockado():
    """POST /predict com modelo mockado retorna 200 com score e risco."""
    from unittest.mock import MagicMock

    import numpy as np

    import credit_lens.api as api_mod

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
    api_mod._state["modelo"] = mock_model
    api_mod._state["modelo_nome"] = "modelo_lgbm_v1.joblib"
    api_mod._state["metricas"] = {"roc_auc": 0.78}

    try:
        payload = {"AMT_INCOME_TOTAL": 150000.0, "AMT_CREDIT": 500000.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["probabilidade_inadimplencia"] == 0.15
        assert data["faixa_risco"] == "médio"
        assert data["modelo"] == "modelo_lgbm_v1.joblib"

        # Test model-info when model is loaded
        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["modelo"] == "modelo_lgbm_v1.joblib"
    finally:
        api_mod._state.clear()


@pytest.mark.skipif(
    not list(MODELS.glob("modelo_lgbm_*.joblib")),
    reason="modelo treinado ausente (models/*.joblib não é versionado); rode `make train`",
)
def test_lifespan_e_predicao_com_modelo_real():
    """Valida que o startup com lifespan carrega o modelo treinado de disco e executa predição real."""
    with TestClient(app) as live_client:
        info = live_client.get("/model-info")
        assert info.status_code == 200
        assert "lgbm" in info.json()["modelo"]

        payload = {
            "AMT_INCOME_TOTAL": 200000.0,
            "AMT_CREDIT": 450000.0,
            "AMT_ANNUITY": 25000.0,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -2000,
        }
        r = live_client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert 0.0 <= data["probabilidade_inadimplencia"] <= 1.0
        assert data["faixa_risco"] in ("baixo", "médio", "alto")


