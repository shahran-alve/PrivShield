"""Download the UCI Diabetes 130-US Hospitals dataset (UCI id 296).

Tries the ucimlrepo package first and the official UCI zip second, checks the
required columns, and writes data/raw/diabetic_data.csv together with a small
manifest that records the source and the file's SHA-256. A cached copy whose
checksum matches the manifest is reused.

Run:  python -m src.data_acquisition [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import zipfile

import pandas as pd

from src.config import DATA_DIR
from src.utils import sha256_file

RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_CSV = os.path.join(RAW_DIR, "diabetic_data.csv")
MANIFEST = os.path.join(RAW_DIR, "source_manifest.json")

UCI_ID = 296
UCI_DOI = "10.24432/C5230J"
UCI_NAME = "Diabetes 130-US Hospitals for Years 1999-2008"
UCI_LICENSE = "CC BY 4.0"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130+us+hospitals+for+years+1999+2008.zip"
)

REQUIRED_RAW_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age",
    "admission_type_id", "medical_specialty", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "number_diagnoses", "max_glu_serum", "A1Cresult",
    "insulin", "change", "diabetesMed", "readmitted",
]


def acquire(force: bool = False) -> dict:
    os.makedirs(RAW_DIR, exist_ok=True)
    if not force and _cached_and_valid():
        manifest = _load_manifest()
        print(f"[acquire] reusing cached raw dataset ({manifest['raw_rows']} rows)")
        return manifest

    errors = []
    df = _try_ucimlrepo(errors)
    method = "ucimlrepo"
    if df is None:
        df = _try_official_zip(errors)
        method = "official_zip"
    if df is None:
        raise RuntimeError("Could not download UCI dataset 296:\n" + "\n".join(errors))

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {missing}")
    df = df[REQUIRED_RAW_COLUMNS + [c for c in df.columns if c not in REQUIRED_RAW_COLUMNS]]
    df.to_csv(RAW_CSV, index=False)

    manifest = {
        "uci_id": UCI_ID,
        "dataset_name": UCI_NAME,
        "doi": UCI_DOI,
        "license": UCI_LICENSE,
        "acquisition_method": method,
        "acquired_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_rows": int(df.shape[0]),
        "raw_columns": int(df.shape[1]),
        "sha256": sha256_file(RAW_CSV),
    }
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[acquire] wrote {RAW_CSV} ({df.shape[0]} rows, {method})")
    return manifest


def _load_manifest() -> dict:
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cached_and_valid() -> bool:
    if not (os.path.exists(RAW_CSV) and os.path.exists(MANIFEST)):
        return False
    try:
        if _load_manifest().get("sha256") != sha256_file(RAW_CSV):
            return False
        head = pd.read_csv(RAW_CSV, nrows=5)
        return all(col in head.columns for col in REQUIRED_RAW_COLUMNS)
    except Exception:
        return False


def _try_ucimlrepo(errors: list) -> "pd.DataFrame | None":
    try:
        from ucimlrepo import fetch_ucirepo
        return fetch_ucirepo(id=UCI_ID).data.original
    except Exception as exc:
        errors.append(f"ucimlrepo: {type(exc).__name__}: {str(exc)[:180]}")
        return None


def _try_official_zip(errors: list) -> "pd.DataFrame | None":
    try:
        import urllib.request
        req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "PrivShield/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            name = next(n for n in zf.namelist() if n.endswith("diabetic_data.csv"))
            with zf.open(name) as fh:
                return pd.read_csv(fh)
    except Exception as exc:
        errors.append(f"official_zip: {type(exc).__name__}: {str(exc)[:180]}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UCI Diabetes 130-US Hospitals.")
    parser.add_argument("--force", action="store_true", help="download again even if cached")
    acquire(force=parser.parse_args().force)
