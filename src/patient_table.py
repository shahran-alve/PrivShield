"""Build the one-row-per-patient hospital table.

The UCI file has one row per hospital visit: 101,766 visits by 71,518 patients.
The attack subtracts counts across diagnosis groups, which is only valid when
every person sits in exactly one group, so the table is rebuilt with one row
per patient. The row kept is the index visit, the one with the smallest
encounter_id, and the protected fact is that visit's `readmitted` value.

Writes data/processed/patients_index.csv (source identifiers removed) and
results/patient_table_report.json with the counts quoted in the paper.

Run:  python -m src.patient_table
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.config import DATA_DIR, K_THRESHOLD
from src.data_acquisition import RAW_CSV
from src.data_preprocessing import clean

QUASI = ["age_band", "race", "gender", "diagnosis_group"]
TARGET = "readmitted"
PATIENT_CSV = os.path.join(DATA_DIR, "processed", "patients_index.csv")
RESULTS = os.environ.get("PRIVSHIELD_RESULTS", "results")


def build_patient_table(save: bool = True) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(RAW_CSV, dtype=str, low_memory=False)
    enc = clean(raw.copy()).reset_index(drop=True)
    enc["patient_nbr"] = raw["patient_nbr"].reset_index(drop=True).values
    enc["encounter_id"] = pd.to_numeric(raw["encounter_id"], errors="coerce").values

    g = enc.groupby("patient_nbr")
    report = {
        "patients": int(enc.patient_nbr.nunique()),
        "encounters": int(len(enc)),
        "multi_encounter_patients": int((g.size() > 1).sum()),
        "conflicting_diagnosis": int((g["diagnosis_group"].nunique() > 1).sum()),
        "changed_age_band": int((g["age_band"].nunique() > 1).sum()),
        "conflicting_race": int((g["race"].nunique() > 1).sum()),
        "multiple_outcome_categories": int((g[TARGET].nunique() > 1).sum()),
    }

    idx = enc.sort_values("encounter_id").groupby("patient_nbr", as_index=False).first()
    idx = idx.reset_index(drop=True)
    idx.insert(0, "record_id", np.arange(1, len(idx) + 1))
    report["encounters_dropped"] = int(len(enc) - len(idx))

    sizes_e = enc.groupby(QUASI, observed=True).size()
    sizes_p = idx.groupby(QUASI, observed=True).size()
    report["encounter_groups"] = int(len(sizes_e))
    report["patient_groups"] = int(len(sizes_p))
    report["encounter_groups_below_k"] = int((sizes_e < K_THRESHOLD).sum())
    report["patient_groups_below_k"] = int((sizes_p < K_THRESHOLD).sum())
    gs = idx.groupby(QUASI, observed=True)[TARGET].transform("size")
    report["patients_at_risk"] = int((gs < K_THRESHOLD).sum())

    if save:
        os.makedirs(os.path.dirname(PATIENT_CSV), exist_ok=True)
        os.makedirs(RESULTS, exist_ok=True)
        keep = [c for c in idx.columns if c not in ("patient_nbr", "encounter_id")]
        idx[keep].to_csv(PATIENT_CSV, index=False)
        with open(os.path.join(RESULTS, "patient_table_report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
    return idx, report


if __name__ == "__main__":
    _, rep = build_patient_table()
    print(json.dumps(rep, indent=2))
