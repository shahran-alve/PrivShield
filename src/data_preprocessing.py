"""Clean the raw UCI table and write the processed files.

Maps the raw columns to the analytical schema (age band, race, gender,
admission type, diagnosis group from the ICD-9 code, and so on), removes the
source identifiers, and writes data/processed/diabetes_full.csv plus random
samples of 10,000, 25,000 and 50,000 visits.

Run:  python -m src.data_preprocessing
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.config import DATA_DIR

RAW_CSV = os.path.join(DATA_DIR, "raw", "diabetic_data.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SAMPLE_SIZES = [10_000, 25_000, 50_000]

ANALYTICAL_COLUMNS = [
    "record_id", "age_band", "race", "gender", "admission_type",
    "service_department", "diagnosis_group", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "prior_utilization", "number_diagnoses", "A1Cresult", "max_glu_serum",
    "insulin", "diabetesMed", "readmitted",
]

FORBIDDEN_IN_PROCESSED = ["patient_nbr", "encounter_id", "diag_1"]

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "Not Available", 7: "Trauma Center", 8: "Other",
}

SERVICE_DEPARTMENT_MAP = {
    "Cardiology": "Cardiology",
    "Cardiology-Pediatric": "Cardiology",
    "Endocrinology": "Endocrinology",
    "Endocrinology-Metabolism": "Endocrinology",
    "Emergency/Trauma": "Emergency",
    "InternalMedicine": "Internal Medicine",
    "Family/GeneralPractice": "Internal Medicine",
    "Surgery-General": "Surgery",
    "Surgery-Cardiovascular/Thoracic": "Surgery",
    "Surgery-Neuro": "Surgery",
    "Orthopedics": "Surgery",
    "Orthopedics-Reconstructive": "Surgery",
}
SERVICE_DEPARTMENTS = [
    "Cardiology", "Endocrinology", "Emergency",
    "Internal Medicine", "Surgery", "Other", "Unknown",
]


def _clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def map_admission_type(value) -> str:
    try:
        return ADMISSION_TYPE_MAP.get(int(value), "Not Mapped")
    except (ValueError, TypeError):
        return "Not Mapped"


def map_service_department(value) -> str:
    v = str(value).strip()
    if v in ("?", "", "nan", "None"):
        return "Unknown"
    if v in SERVICE_DEPARTMENT_MAP:
        return SERVICE_DEPARTMENT_MAP[v]
    return "Other"


def map_diagnosis_group(code) -> str:
    """Map a diag_1 ICD-9 code to a broad clinical group."""
    v = str(code).strip()
    if v in ("?", "", "nan", "None"):
        return "Missing"
    if v.startswith("V") or v.startswith("E"):
        return "Other"
    if v.startswith("250"):
        return "Diabetes"
    try:
        num = float(v)
    except ValueError:
        return "Other"
    n = int(num)
    if 1 <= n <= 139:
        return "Infectious"
    if 140 <= n <= 239:
        return "Neoplasms"
    if 240 <= n <= 279:
        return "Other"          # endocrine/metabolic (diabetes handled above)
    if 280 <= n <= 289:
        return "Other"
    if 290 <= n <= 319:
        return "Mental"
    if 320 <= n <= 389:
        return "Nervous/Sense"
    if 390 <= n <= 459:
        return "Circulatory"
    if 460 <= n <= 519:
        return "Respiratory"
    if 520 <= n <= 579:
        return "Digestive"
    if 580 <= n <= 629:
        return "Genitourinary"
    if 630 <= n <= 679:
        return "Other"
    if 680 <= n <= 709:
        return "Skin"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 780 <= n <= 799:
        return "Symptoms"
    if 800 <= n <= 999:
        return "Injury"
    return "Other"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Produce the cleaned full analytical frame (before sampling)."""
    df = df.copy()

    cat_source = ["race", "gender", "age", "medical_specialty", "diag_1",
                  "max_glu_serum", "A1Cresult", "insulin", "change",
                  "diabetesMed", "readmitted"]
    for c in cat_source:
        if c in df.columns:
            df[c] = _clean_string_series(df[c])

    out = pd.DataFrame(index=df.index)

    out["age_band"] = df["age"].replace({"?": "Missing", "": "Missing"})

    race = df["race"].replace({"?": "Missing", "Unknown/Invalid": "Missing", "": "Missing"})
    out["race"] = race
    out["gender"] = df["gender"].replace({"Unknown/Invalid": "Missing", "?": "Missing", "": "Missing"})

    out["admission_type"] = df["admission_type_id"].map(map_admission_type)
    out["service_department"] = df["medical_specialty"].map(map_service_department)
    out["diagnosis_group"] = df["diag_1"].map(map_diagnosis_group)

    numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                    "num_medications", "number_diagnoses"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    out["prior_utilization"] = (
        pd.to_numeric(df["number_outpatient"], errors="coerce").fillna(0)
        + pd.to_numeric(df["number_emergency"], errors="coerce").fillna(0)
        + pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0)
    ).astype(int)

    # A1Cresult / max_glu_serum: keep "None" as the explicit "Not measured"
    out["A1Cresult"] = df["A1Cresult"].replace({"None": "Not measured", "?": "Missing", "": "Missing"})
    out["max_glu_serum"] = df["max_glu_serum"].replace({"None": "Not measured", "?": "Missing", "": "Missing"})
    out["insulin"] = df["insulin"].replace({"?": "Missing", "": "Missing"})
    out["diabetesMed"] = df["diabetesMed"].replace({"?": "Missing", "": "Missing"})

    out["readmitted"] = df["readmitted"].replace({
        "<30": "Within 30 days", ">30": "After 30 days", "NO": "No readmission",
    })

    return out


def make_samples(clean_df: pd.DataFrame, sizes=SAMPLE_SIZES) -> dict:
    """Seeded bootstrap samples of the cleaned table (with replacement)."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    samples = {}
    for size in sizes:
        sample = clean_df.sample(n=size, replace=True, random_state=42 + size)
        sample = sample.reset_index(drop=True)
        sample.insert(0, "record_id", np.arange(1, size + 1))
        sample = sample[ANALYTICAL_COLUMNS]
        path = os.path.join(PROCESSED_DIR, f"diabetes_{size}.csv")
        sample.to_csv(path, index=False)
        samples[size] = path
    return samples


def make_full(clean_df: pd.DataFrame) -> str:
    """Write the entire cleaned dataset as the working set (no sampling)."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    full = clean_df.reset_index(drop=True).copy()
    full.insert(0, "record_id", np.arange(1, len(full) + 1))
    full = full[ANALYTICAL_COLUMNS]
    path = os.path.join(PROCESSED_DIR, "diabetes_full.csv")
    full.to_csv(path, index=False)
    return path


def run(all_sizes: bool = True) -> dict:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run `python -m src.data_acquisition` first."
        )
    raw = pd.read_csv(RAW_CSV, dtype=str, low_memory=False)
    cleaned = clean(raw)
    sizes = SAMPLE_SIZES if all_sizes else [25_000]
    paths = make_samples(cleaned, sizes)
    full_path = make_full(cleaned)
    paths["full"] = full_path
    for size, p in paths.items():
        n = len(pd.read_csv(p)) if size == "full" else size
        print(f"[preprocess] wrote {p} ({n} rows)")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UCI data into samples.")
    parser.add_argument("--all-sizes", action="store_true", default=True)
    parser.add_argument("--only-25k", action="store_true")
    args = parser.parse_args()
    run(all_sizes=not args.only_25k)
