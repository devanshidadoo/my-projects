# Delivery risk prediction and a margin-optimised intervention policy.
#
#   make setup     install the package and dev dependencies
#   make test      unit + property tests (~1 min)
#   make smoke     tiny end-to-end run; proves the pipeline works
#   make headline  the full 99K-order experiment (~3 min)
#   make results   headline + regenerate docs/RESULTS.md and every figure
#   make schema    print the MySQL DDL for the nine tables
#   make sql       regenerate the committed SQL artefacts in sql/
#   make olist     load the real Olist CSVs from data/raw

PY ?= python3
PYTHONPATH := src
export PYTHONPATH

CONFIG ?= configs/headline.yaml
OUT    ?= results/headline

.PHONY: help setup test test-fast lint smoke headline results schema sql leak-audit olist clean clean-cache

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

setup:
	$(PY) -m pip install -e ".[dev,gbdt]"

test:
	$(PY) -m pytest tests -q

test-fast:
	$(PY) -m pytest tests -q --ignore=tests/test_models_and_calibration.py

lint:
	$(PY) -m ruff check src tests experiments

# --docs-dir keeps a throwaway run from overwriting the committed report and figures.
smoke:
	$(PY) experiments/run_all.py --config configs/smoke.yaml \
		--out-dir results/smoke --docs-dir results/smoke/docs --cache-dir results/smoke/cache

headline:
	$(PY) experiments/run_all.py --config $(CONFIG) --out-dir $(OUT)

results: headline

schema:
	$(PY) -m deliveryrisk.cli schema --dialect mysql

sql:
	$(PY) scripts/regenerate_sql.py

leak-audit:
	$(PY) -m deliveryrisk.cli leak-audit $(CONFIG)

# Requires the Kaggle files in data/raw/ (see configs/olist.yaml).
olist:
	$(PY) -m deliveryrisk.cli ingest-olist --raw-dir data/raw --database-url sqlite:///data/olist.db

clean:
	rm -rf results/* docs/figures/*.png data/*.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .hypothesis

clean-cache:
	rm -rf data/cache
