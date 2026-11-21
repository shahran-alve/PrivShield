"""Append-only audit log of controller decisions.

Each entry records who asked what and which action the controller took. Raw
rows, result values and source identifiers are never written.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import threading

import pandas as pd

from src.config import DATA_DIR
from src.utils import canonical_json

AUDIT_DIR = os.path.join(DATA_DIR, "audit")
AUDIT_LOG = os.path.join(AUDIT_DIR, "audit_log.csv")

AUDIT_COLUMNS = [
    "timestamp_utc", "request_id", "user_id", "user_role", "user_department",
    "purpose", "query_summary", "aggregate", "target_field", "group_by",
    "filters_json", "query_fingerprint", "structural_signature",
    "requested_sensitivity", "policy_allowed", "policy_id", "policy_reason",
    "matched_row_count", "min_result_group_size", "repetition_score",
    "narrowing_score", "risk_label", "risk_score", "action",
    "explanation_json", "latency_ms",
]

FORBIDDEN_SUBSTRINGS = ("patient_nbr", "encounter_id", "record_id")

_LOCK = threading.Lock()


def _ensure_file() -> None:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    if not os.path.exists(AUDIT_LOG):
        with open(AUDIT_LOG, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(AUDIT_COLUMNS)


def append_audit(entry: dict) -> None:
    """Append one row. Missing keys are blanked; unknown keys are dropped."""
    _ensure_file()
    row = {c: entry.get(c, "") for c in AUDIT_COLUMNS}

    # Guardrail: no identifier column names may leak into free-text fields.
    for key in ("filters_json", "explanation_json", "query_summary"):
        blob = str(row.get(key, "")).lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in blob:
                row[key] = "[redacted]"

    with _LOCK:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
            writer.writerow(row)
            fh.flush()


def make_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_recent_audit_logs(limit: int = 50) -> pd.DataFrame:
    if not os.path.exists(AUDIT_LOG):
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    try:
        df = pd.read_csv(AUDIT_LOG)
    except Exception:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    return df.tail(limit).reset_index(drop=True)


def load_user_history(user_id: str, since_utc: "dt.datetime | None" = None) -> pd.DataFrame:
    if not os.path.exists(AUDIT_LOG):
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    rows = []
    with open(AUDIT_LOG, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if raw.get("user_id") != user_id:
                continue
            if since_utc is not None:
                try:
                    ts = dt.datetime.fromisoformat(raw["timestamp_utc"])
                    if ts < since_utc:
                        continue
                except (ValueError, KeyError):
                    continue  # malformed row -> skip
            rows.append(raw)
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS) if rows else pd.DataFrame(columns=AUDIT_COLUMNS)


def encode_filters(filters: dict) -> str:
    return canonical_json({k: str(v) for k, v in (filters or {}).items()})
