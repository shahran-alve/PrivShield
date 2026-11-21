#!/usr/bin/env bash
# Download the three public datasets and build the processed hospital tables.
# Nothing under data/raw or data/processed is committed to the repository.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
mkdir -p data/raw data/processed

# UCI Adult
BASE=https://archive.ics.uci.edu/ml/machine-learning-databases/adult
curl -fsSL "$BASE/adult.data" -o data/raw/adult.data
curl -fsSL "$BASE/adult.test" -o data/raw/adult.test

# NHANES 2017-March 2020 (CDC): demographics and the diabetes questionnaire
NHANES=https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles
curl -fsSL "$NHANES/P_DEMO.xpt" -o data/raw/P_DEMO.xpt
curl -fsSL "$NHANES/P_DIQ.xpt"  -o data/raw/P_DIQ.xpt

# UCI Diabetes 130-US Hospitals (id 296), then the visit-level and patient-level tables
$PY -m src.data_acquisition
$PY -m src.data_preprocessing
$PY -m src.patient_table

sha256sum -c data/checksums.sha256
