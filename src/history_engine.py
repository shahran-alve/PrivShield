"""Query-history features for one user.

Looks back over the same user's audit entries in a 30-minute window and scores
how often the new query repeats earlier ones and whether it narrows them.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from src.audit_logger import load_user_history

WINDOW_MINUTES = 30
JACCARD_THRESHOLD = 0.67
SENTINEL_SECONDS = 10 ** 7


@dataclass
class HistoryScores:
    query_repetition_score: int
    sequence_narrowing_score: int
    filter_specificity_score: float
    time_since_similar_seconds: float


def _parse_filters(blob: str) -> dict:
    try:
        return json.loads(blob) if blob else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _is_similar(current: dict, prior: dict) -> bool:
    if (current["aggregate"], current["target_field"], current["group_by"]) != \
       (prior["aggregate"], prior["target_field"], prior["group_by"]):
        return False
    return _jaccard(set(current["filter_fields"]), set(prior["filter_fields"])) >= JACCARD_THRESHOLD


def compute_history_scores(
    user_id: str,
    aggregate: str,
    target_field,
    group_by,
    filters: dict,
    matched_row_count: int,
    total_dataset_rows: int,
    now_utc: "dt.datetime | None" = None,
) -> HistoryScores:
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    window_start = now - dt.timedelta(minutes=WINDOW_MINUTES)

    current = {
        "aggregate": aggregate,
        "target_field": "" if target_field is None else str(target_field),
        "group_by": "" if group_by is None else str(group_by),
        "filter_fields": set((filters or {}).keys()),
    }

    hist = load_user_history(user_id, since_utc=window_start)

    repetition = 0
    narrowing = 0
    most_recent_similar_ts = None

    for _, raw in hist.iterrows():
        try:
            ts = dt.datetime.fromisoformat(str(raw["timestamp_utc"]))
        except (ValueError, TypeError):
            continue
        if ts >= now:
            continue
        prior_filters = _parse_filters(raw.get("filters_json", ""))
        prior = {
            "aggregate": raw.get("aggregate", ""),
            "target_field": "" if str(raw.get("target_field", "")) in ("", "nan") else str(raw.get("target_field")),
            "group_by": "" if str(raw.get("group_by", "")) in ("", "nan") else str(raw.get("group_by")),
            "filter_fields": set(prior_filters.keys()),
        }
        if not _is_similar(current, prior):
            continue

        repetition += 1
        if most_recent_similar_ts is None or ts > most_recent_similar_ts:
            most_recent_similar_ts = ts

        # Narrowing: strict superset of filter fields, or matched rows halved.
        strict_superset = current["filter_fields"] > prior["filter_fields"]
        try:
            prior_matched = float(raw.get("matched_row_count", 0) or 0)
        except (ValueError, TypeError):
            prior_matched = 0.0
        row_narrowed = prior_matched > 0 and matched_row_count <= 0.5 * prior_matched
        if strict_superset or row_narrowed:
            narrowing += 1

    specificity = 0.0
    if total_dataset_rows > 0:
        specificity = 1.0 - (matched_row_count / total_dataset_rows)
    specificity = max(0.0, min(1.0, specificity))

    if most_recent_similar_ts is None:
        time_since = float(SENTINEL_SECONDS)
    else:
        time_since = max(0.0, (now - most_recent_similar_ts).total_seconds())

    return HistoryScores(
        query_repetition_score=min(repetition, 5),
        sequence_narrowing_score=min(narrowing, 5),
        filter_specificity_score=round(specificity, 6),
        time_since_similar_seconds=time_since,
    )
