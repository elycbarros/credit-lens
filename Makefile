PY ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: bi install test lint lint-fix features train monitor api api-docker

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q --cov=src --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests

lint-fix:
	$(PY) -m ruff check --fix src tests

features:
	$(PY) -m credit_lens.features

train:
	$(PY) -m credit_lens.train

train-logistica:
	$(PY) -m credit_lens.train --modelo logistica

monitor:
	$(PY) -m credit_lens.monitor

api:
	$(PY) -m uvicorn credit_lens.api:app --reload --host 0.0.0.0 --port 8000

api-docker:
	docker build -t credit-lens . && docker run --rm -p 8000:8000 credit-lens

bi:
	$(PY) scripts/exportar_para_bi.py
