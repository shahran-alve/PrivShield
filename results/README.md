# Results

Output of the scripts in `experiments/`. `bash run_all.sh` regenerates every
file except `scale.csv` and `decision_latency.csv`, which are timing runs
(`make timing`). Counts of targets are people, not cells, unless a column name
says otherwise.

## Attack

| File | Content |
|---|---|
| `kclean_search.csv` | Per dataset at k = 10: at-risk targets, targets with a k-clean certificate, and those whose group is a singleton or has one outcome. |
| `certificates/local_certificates_k10.json` | One entry per protected cell with a certificate: the queries used, the count each returns, the coefficients, and a verification flag. `true_count` is the number of people in the cell. |
| `ksweep.csv` | The local search at k = 5, 10, 20, with the number of attackable targets the rank auditor protects in both query orders. |
| `crossprefix_k{5,10,20}.csv` | Attackable targets when queries may drop a quasi-identifier (`mode` = `cross`) next to the local search (`local`). `grid` is the number of cells of the attribute grid, `n_query_vecs` the number of answerable queries. |
| `range_filter.csv` | NHANES targets attackable with equality filters only and with one range filter on education added. |
| `ip_attacker_k10.csv` | Feasibility attacker. `ip_pinned_*`: targets whose cell has one feasible value. `homog_forced_*`: targets whose group is forced to one outcome. Suffix `patientk` is the threshold alone, `rank_auditor` adds the auditor, `auditorB` also protects empty cells. `groups_forced_*` counts groups instead of people. |

## Defenses

| File | Content |
|---|---|
| `ladder.csv`, `ladder.png` | The naive tracker against each defense: cells reconstructed, targets disclosed, distinct people exposed (Fig. 2). |
| `kclean_matrix.csv` | Certificates answered in full (`succeed`) or stopped (`prevented`) by each defense under the attacker's best query order (Table I). |
| `benign_fp_compare.csv` | Share of drill-down and roll-up sessions with a refused legitimate query, for symmetric containment and the rank auditor (Table II). 200 sessions of each type per dataset. |
| `hard_benign_k10.csv` | Harder benign workload, fresh auditor per session: legitimate queries and the share each auditor refuses (Table III). |
| `accumulated_state_k10.csv` | The same workload with auditor state kept for 1, 10, 50 or 150 sessions per account (Table III). |
| `population_workload.csv` | Queries answered and answered exactly, out of 120, by patient-k, PrivShield-Lite and PrivShield-Full (Table IV). |
| `full_benign_summary.json`, `full_benign_sessions.csv` | PrivShield-Full on the 400 hospital drill-down and roll-up sessions (Table IV). |
| `crossprefix_audit_k{5,10,20}.csv` | The rank auditor on the full cross-prefix query set, broadest query first (`fwd`) and last (`rev`): queries refused, protected cells still computable from the released queries (independent check), agreement of the two primes, and time per query. `targets_computable_no_auditor` counts distinct at-risk cells. |
| `baseline_dp.csv`, `baseline_rounding.csv`, `baseline_utility.csv`, `laplace_averaging.csv` | Laplace noise and rounding against the 88 certificates, and their error on benign cells. |
| `refusal_pattern.csv` | Held-out accuracy of an attacker who sees the threshold's allow/refuse pattern, compared with guessing the majority outcome. |
| `disparity.csv` | Share of outcome-breakdown questions the threshold refuses, per race and age group. |

## Data and timing

| File | Content |
|---|---|
| `patient_table_report.json` | Visits, patients, patients with several visits and with conflicting diagnosis groups, groups below k, patients at risk. |
| `scale.csv` | Query latency with and without the guard on synthetic DuckDB tables of 1M, 10M and 100M rows. |
| `auditor_latency.csv` | Time per request of the rank auditor on a 45-cell grid for histories of 10^3 to 10^5 requests. |
| `decision_latency.csv` | End-to-end decision time of the full controller on samples of 10,000 to 50,000 visits. |
