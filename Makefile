PYTHON ?= python3

.PHONY: deps data all test timing clean

deps:
	$(PYTHON) -m pip install -r requirements.txt

data:
	bash data/acquire.sh

all:
	bash run_all.sh

test:
	$(PYTHON) tests/test_rank_auditor.py

timing:
	$(PYTHON) experiments/auditor_latency.py
	$(PYTHON) experiments/duckdb_latency.py
	$(PYTHON) experiments/decision_latency.py

clean:
	rm -rf experiments/__pycache__ src/__pycache__ results/scale_*.db
