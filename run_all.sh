#!/usr/bin/env bash
# Regenerate every result file except the hardware-dependent timings.
# Requires: pip install -r requirements.txt ; bash data/acquire.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "[ 1/14] local k-clean search at k=10"
$PY experiments/kclean_search.py
echo "[ 2/14] naive tracker against each defense (Fig. 2)"
$PY experiments/tracker_ladder.py
$PY experiments/plot_ladder.py
echo "[ 3/14] certificates against each defense (Table I)"
$PY experiments/defense_matrix.py
$PY experiments/export_certificates.py 10
echo "[ 4/14] drill-down and roll-up sessions (Table II)"
$PY experiments/rank_benign.py
echo "[ 5/14] harder benign workload, per session and per account (Table III)"
$PY experiments/hard_benign.py 10
$PY experiments/accumulated_state.py 10
echo "[ 6/14] population workload and full controller (Table IV)"
$PY experiments/population_workload.py
$PY experiments/full_benign.py
echo "[ 7/14] thresholds 5, 10, 20 on the local grid (Table V)"
$PY experiments/kclean_ksweep.py
echo "[ 8/14] cross-prefix search and auditor at k = 5, 10, 20 (Table V, Section VII-C)"
for k in 5 10 20; do $PY experiments/crossprefix.py $k; done
echo "[ 9/14] range filter (Section VII-B)"
$PY experiments/range_filter.py
echo "[10/14] feasibility attacker (Table VI)"
$PY experiments/ip_attacker.py 10
$PY experiments/empty_cell_case.py
echo "[11/14] rounding and Laplace baselines"
$PY experiments/noise_baselines.py
echo "[12/14] refusal pattern"
$PY experiments/refusal_pattern.py
echo "[13/14] refusals by group"
$PY experiments/disparity.py
echo "[14/14] rank auditor latency on a local grid"
$PY experiments/auditor_latency.py

echo "Done. Results are in ${PRIVSHIELD_RESULTS:-results}/."
