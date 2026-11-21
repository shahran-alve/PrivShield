"""Loaders for the three evaluation datasets.

Each loader returns one row per person with the same five columns:

    age_band, race, gender   quasi-identifiers
    partition                the attribute the attack subtracts over
                             (diagnosis group, education level, occupation)
    outcome                  the sensitive attribute

A `record_id` column is added for bookkeeping. The Adult and NHANES files are
read from $PRIVSHIELD_DATA (default data/raw). The hospital table is the
one-row-per-patient file written by `python -m src.patient_table`.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("PRIVSHIELD_DATA", os.path.join("data", "raw"))
PATIENT_CSV = os.environ.get(
    "PRIVSHIELD_PATIENTS", os.path.join("data", "processed", "patients_index.csv"))

DIMS = ["age_band", "race", "gender", "partition", "outcome"]


def load_hospital(path=None) -> pd.DataFrame:
    """UCI Diabetes 130-US Hospitals, one index visit per patient."""
    d = pd.read_csv(path or PATIENT_CSV)
    out = pd.DataFrame({
        "age_band": d["age_band"].astype(str),
        "race": d["race"].astype(str),
        "gender": d["gender"].astype(str),
        "partition": d["diagnosis_group"].astype(str),
        "outcome": d["readmitted"].astype(str),
    })
    out.insert(0, "record_id", np.arange(1, len(out) + 1))
    return out


def load_adult(path=None) -> pd.DataFrame:
    """UCI Adult (adult.data and adult.test), outcome = income bracket."""
    path = path or DATA_DIR
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"]
    frames = []
    for f in ("adult.data", "adult.test"):
        fp = os.path.join(path, f)
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp, header=None, names=cols, skipinitialspace=True,
                        na_values="?", skiprows=1 if f == "adult.test" else 0)
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a = a.dropna(subset=["age", "race", "sex", "occupation", "income"])
    a["age_band"] = pd.cut(a.age.astype(int), bins=range(10, 101, 10),
                           right=False).astype(str)
    out = pd.DataFrame({
        "age_band": a.age_band,
        "race": a.race.astype(str),
        "gender": a.sex.astype(str),
        "partition": a.occupation.astype(str),
        "outcome": a.income.astype(str).str.replace(".", "", regex=False).str.strip(),
    }).reset_index(drop=True)
    out.insert(0, "record_id", np.arange(1, len(out) + 1))
    return out


def load_nhanes(demo=None, diq=None) -> pd.DataFrame:
    """NHANES 2017-March 2020, outcome = self-reported diabetes (DIQ010)."""
    demo = demo or os.path.join(DATA_DIR, "P_DEMO.xpt")
    diq = diq or os.path.join(DATA_DIR, "P_DIQ.xpt")
    d = pd.read_sas(demo, format="xport")
    q = pd.read_sas(diq, format="xport")
    m = d.merge(q[["SEQN", "DIQ010"]], on="SEQN", how="inner")
    m = m[m.DIQ010.isin([1.0, 2.0, 3.0])]
    m = m.dropna(subset=["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2"])
    m = m[m.DMDEDUC2 <= 5]

    race_map = {1: "MexicanAmerican", 2: "OtherHispanic", 3: "White",
                4: "Black", 6: "Asian", 7: "OtherMulti"}
    edu_map = {1: "<9th", 2: "9-11th", 3: "HS/GED", 4: "SomeCollege",
               5: "CollegeGrad"}
    out_map = {1.0: "Diabetes", 2.0: "NoDiabetes", 3.0: "Borderline"}

    res = pd.DataFrame({
        "age_band": pd.cut(m.RIDAGEYR.astype(float),
                           bins=range(0, 101, 10), right=False).astype(str),
        "race": m.RIDRETH3.map(race_map).astype(str),
        "gender": m.RIAGENDR.map({1.0: "Male", 2.0: "Female"}).astype(str),
        "partition": m.DMDEDUC2.map(edu_map).astype(str),
        "outcome": m.DIQ010.map(out_map).astype(str),
    }).dropna().reset_index(drop=True)
    res.insert(0, "record_id", np.arange(1, len(res) + 1))
    return res
