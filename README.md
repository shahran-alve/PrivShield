# PrivShield

Code and results for the paper "PrivShield: Measuring Sequence-Aware Inference
Control in Healthcare Analytics" (IEEE BigData 2026, special session on Privacy
and Security of Big Data).

Healthcare dashboards usually withhold any count that covers fewer than `k`
people and apply that rule to one query at a time. The paper shows that an
analyst can issue `COUNT` queries that each cover at least `k` patients and
combine the answers to recover a withheld cell exactly. This repository
contains the search for such attacks, the rank auditor that stops them, the
other defenses it is compared with, and the scripts behind every table and
figure.

## Layout

```
src/            dataset loaders, the patient-level hospital table, and the
                full controller (policy, threshold, risk model, perturbation)
experiments/    one script per result; rank_auditor.py is the auditor
results/        the output of those scripts
config/         fields, roles and policies of the full controller
models/         the trained risk model used by the full controller
data/           download script and checksums (no data is stored here)
```

## Setup

```bash
pip install -r requirements.txt
bash data/acquire.sh     # downloads the three public datasets, builds the hospital tables
bash run_all.sh          # regenerates results/
```

The results were produced with Python 3.13 and the versions in
`requirements.txt`. `PRIVSHIELD_DATA` (default `data/raw`), `PRIVSHIELD_PATIENTS`
(default `data/processed/patients_index.csv`) and `PRIVSHIELD_RESULTS` (default
`results`) change the paths.

## Scripts and results

| Paper | Script | Output in `results/` |
|---|---|---|
| Attack counts at k = 10 (abstract, Section VI) | `kclean_search.py` | `kclean_search.csv` |
| Certificates with coefficients | `export_certificates.py` | `certificates/local_certificates_k10.json` |
| Fig. 2, naive tracker | `tracker_ladder.py`, `plot_ladder.py` | `ladder.csv`, `ladder.png` |
| Table I | `defense_matrix.py` | `kclean_matrix.csv` |
| Table II | `rank_benign.py` | `benign_fp_compare.csv` |
| Table III | `hard_benign.py`, `accumulated_state.py` | `hard_benign_k10.csv`, `accumulated_state_k10.csv` |
| Table IV | `population_workload.py`, `full_benign.py` | `population_workload.csv`, `full_benign_summary.json` |
| Table V, local | `kclean_ksweep.py` | `ksweep.csv` |
| Table V, cross-prefix, and Section VII-C | `crossprefix.py 5`, `10`, `20` | `crossprefix_k*.csv`, `crossprefix_audit_k*.csv` |
| Table VI | `ip_attacker.py`, `empty_cell_case.py` | `ip_attacker_k10.csv` |
| Rounding and Laplace noise (Section VI) | `noise_baselines.py` | `baseline_*.csv`, `laplace_averaging.csv` |
| Refusal channel (Section VI) | `refusal_pattern.py` | `refusal_pattern.csv` |
| Cost (Section VI) | `duckdb_latency.py`, `auditor_latency.py` | `scale.csv`, `auditor_latency.csv` |
| Range filter (Section VII-B) | `range_filter.py` | `range_filter.csv` |
| Patient unit (Section VIII) | `python -m src.patient_table` | `patient_table_report.json` |
| Refusals by group, decision latency (Section IX) | `disparity.py`, `decision_latency.py` | `disparity.csv`, `decision_latency.csv` |

`results/README.md` describes the columns of each file.

## The rank auditor

`experiments/rank_auditor.py` keeps a reduced row-echelon basis of the queries
it has released and, for every protected cell, the cell's indicator reduced
against that basis. A new query is refused when its reduced vector is a
multiple of one of those residuals, which is exactly the case in which
releasing it would put the cell in the released span. Arithmetic is exact:
rationals on a local grid, integers modulo a 61-bit prime on the full
attribute grid, with a second prime as a cross-check.

`tests/test_rank_auditor.py` compares the auditor's decisions with a
brute-force rank computation on random grids.

## Notes

- Exactness. The local search, the certificates and the local auditor use
  rational arithmetic. The cross-prefix span test and audit work modulo two
  primes and require both to agree.
- Determinism. The attack search covers every at-risk target, with no
  sampling. Workload generators and noise draws use fixed seeds.
- Patient unit. All attack and defense experiments use the one-row-per-patient
  hospital table (`data/README.md`).
- Full controller. `full_benign.py`, `population_workload.py` and
  `decision_latency.py` run the controller in `src/` with the model in
  `models/`. That controller has no rank auditor. `rank_benign.py` shows the
  auditor refuses nothing in the drill-down and roll-up sessions, so the
  session count reported for PrivShield-Full is the same with it added.
- Timing. `scale.csv` and `decision_latency.csv` hold the measurements quoted
  in the paper. `auditor_latency.csv` and the `ms_per_query` columns of
  `crossprefix_audit_k*.csv` come from a different machine. All of them depend
  on hardware, and `make timing` reruns them.

## License

The code is released under the MIT license (`LICENSE`). The datasets belong to
their providers and are not redistributed.

## Citation

```
S. R. Alve, H. J. Bhuiyan, L. Khan, and B. Thuraisingham, "PrivShield: Measuring
Sequence-Aware Inference Control in Healthcare Analytics," in Proc. IEEE
International Conference on Big Data (BigData), 2026.
```
