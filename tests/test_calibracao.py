import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from credit_lens.calibracao import calibrar, curva_calibracao, ece


def _dados(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(X[:, 0] * 1.2 - 3.0)))  # taxa base ~8%
    y = (rng.random(n) < p).astype(int)
    return X, y


def test_ece_zero_quando_perfeito():
    y = np.array([0, 1] * 500)
    assert ece(y, np.full(1000, 0.5) + np.linspace(-1e-6, 1e-6, 1000)) == pytest.approx(0.0, abs=1e-3)


def test_calibracao_reduz_ece_de_modelo_com_pesos_de_classe():
    X, y = _dados()
    Xtr, ytr, Xv, yv, Xte, yte = X[:3000], y[:3000], X[3000:4500], y[3000:4500], X[4500:], y[4500:]
    bruto = LogisticRegression(class_weight="balanced").fit(Xtr, ytr)
    antes = bruto.predict_proba(Xte)[:, 1]
    cal = calibrar(bruto, Xv, yv)
    depois = cal.predict_proba(Xte)[:, 1]
    assert ece(yte, depois) < ece(yte, antes)
    assert depois.mean() == pytest.approx(yte.mean(), abs=0.03)


def test_curva_calibracao_formato():
    X, y = _dados(1000)
    c = curva_calibracao(y, np.random.default_rng(1).random(1000), n_bins=5)
    assert len(c) == 5 and c["clientes"].sum() == 1000
