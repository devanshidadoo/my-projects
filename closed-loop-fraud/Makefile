# Closed-loop fraud detection under selection bias.
#
#   make setup     install the package and dev dependencies
#   make test      unit + property tests (fast)
#   make smoke     tiny end-to-end run; proves the pipeline works
#   make headline  the full 590K x 18-cycle experiment (~35 min)
#   make sweeps    exploration frontier + sensitivity (~50 min)
#   make results   headline + sweeps + regenerate docs/RESULTS.md

PY ?= python3
PYTHONPATH := src
export PYTHONPATH

CONFIG ?= configs/headline.yaml
SWEEP  ?= configs/exploration_frontier.yaml
OUT    ?= results/headline

.PHONY: help setup test test-fast lint smoke headline sweeps results leak-audit ieee clean clean-cache

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

setup:
	$(PY) -m pip install -e ".[dev,gbdt]"

test:
	$(PY) -m pytest tests -q

test-fast:
	$(PY) -m pytest tests -q -m "not slow" --ignore=tests/test_simulator.py

lint:
	$(PY) -m ruff check src tests experiments

# --docs-dir keeps the throwaway run from overwriting the committed report and figures.
smoke:
	$(PY) experiments/run_all.py --config configs/smoke.yaml --sweep-config configs/smoke.yaml \
		--out-dir results/smoke --docs-dir results/smoke/docs

headline:
	$(PY) experiments/run_all.py --config $(CONFIG) --only dataset,leakage,loop --out-dir $(OUT)

sweeps:
	$(PY) experiments/run_all.py --config $(CONFIG) --sweep-config $(SWEEP) \
		--only frontier,sensitivity,ope --out-dir $(OUT)

results: headline sweeps
	$(PY) experiments/run_all.py --config $(CONFIG) --sweep-config $(SWEEP) --only "" --out-dir $(OUT)

leak-audit:
	$(PY) -m clfraud.cli leak-audit $(CONFIG)

# Requires the Kaggle files in data/raw/ (see configs/ieee_cis.yaml).
ieee:
	$(PY) -m clfraud.cli ingest-ieee --raw-dir data/raw --out data/processed/ieee.parquet

clean:
	rm -rf results/* docs/figures/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .hypothesis

clean-cache:
	rm -rf data/cache
