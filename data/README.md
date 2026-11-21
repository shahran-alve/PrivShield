# Data

No dataset is stored in this repository. `bash data/acquire.sh` downloads the
three public sources into `data/raw/`, builds the processed hospital tables in
`data/processed/`, and checks the files against `data/checksums.sha256`.

| Dataset | Source | Sensitive attribute | Partition attribute |
|---|---|---|---|
| UCI Diabetes 130-US Hospitals | UCI Machine Learning Repository, id 296 | readmission status after the index visit (none, within 30 days, after 30 days) | diagnosis group |
| NHANES 2017-March 2020 | CDC NCHS, files `P_DEMO.xpt` and `P_DIQ.xpt` | self-reported diabetes (item DIQ010) | education level |
| UCI Adult | UCI Machine Learning Repository | income above $50K | occupation |

Each dataset has its own terms of use. Please read them at the source.

## Hospital tables

The UCI file has one row per visit: 101,766 visits by 71,518 patients, and
16,773 patients have more than one visit.

- `data/processed/diabetes_full.csv` is the cleaned visit-level table. It is
  used only to train the risk model of the full controller.
- `data/processed/patients_index.csv` has one row per patient, the index visit
  (smallest `encounter_id`). All attack and defense experiments use this table,
  because the attack subtracts counts across diagnosis groups and that is valid
  only when each person is in one group. `python -m src.patient_table` builds
  it and writes the counts above to `results/patient_table_report.json`.

`patient_nbr` and `encounter_id` are used to build the patient table and are
removed from both processed files.
