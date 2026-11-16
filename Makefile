.PHONY: help install data train test run report reset clean lint

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

help:
	@echo "PayBridge - make targets"
	@echo "  make install   Create .venv and install dependencies"
	@echo "  make data      Regenerate the seeded synthetic dataset"
	@echo "  make train     Train and persist the FraudLens model bundle"
	@echo "  make test      Run the automated test suite"
	@echo "  make run       Start the Streamlit application"
	@echo "  make report    Print every headline figure, recomputed from the data"
	@echo "  make reset     Delete generated data/models and rebuild from scratch"

install:
	python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

data:
	$(PY) scripts/generate_data.py

train:
	$(PY) scripts/train_model.py

test:
	$(PY) -m pytest -q

run:
	$(VENV)/bin/streamlit run app.py

report:
	$(PY) scripts/report.py

reset:
	rm -f data/*.parquet data/*.csv data/*.db data/*.db-wal data/*.db-shm \
	      data/dataset_manifest.json data/qa_results.json models/*.joblib models/*.json
	$(MAKE) data
	$(MAKE) train

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
