"""Feature vector for the risk model.

The feature order is fixed because the trained model depends on it. Features
describe the user, the shape of the query, the sizes of the result and the
recent history. Patient rows, the policy decision and the final action are
not features.
"""
from __future__ import annotations

import math

import pandas as pd

from src.config import FieldRegistry, load_field_registry
from src.contracts import QueryRequest, SENSITIVITY_RANK

FEATURE_ORDER = [
    "num_filters",
    "uses_sensitive_field",
    "log_matched_group_size",
    "log_min_result_group_size",
    "has_group_by",
    "role_clearance_score",
    "sensitivity_score",
    "query_repetition_score",
    "filter_specificity_score",
    "sequence_narrowing_score",
    "is_count",
    "is_mean",
    "is_extreme_aggregate",
]

# Anything in here must never appear in the model feature matrix.
FORBIDDEN_FEATURES = {
    "policy_allowed", "action", "label", "final_label", "violates_k_threshold",
    "label_rule_reason", "readmitted", "dominant_proportion", "protected_result",
}


def _requested_fields(request: QueryRequest) -> list:
    fields = []
    if request.target_field:
        fields.append(request.target_field)
    if request.group_by:
        fields.append(request.group_by)
    fields.extend((request.filters or {}).keys())
    return fields


def compute_features(
    request: QueryRequest,
    clearance_level: int,
    matched_row_count: int,
    min_result_group_size: int,
    repetition_score: int,
    specificity_score: float,
    narrowing_score: int,
    field_registry: FieldRegistry | None = None,
) -> dict:
    registry = field_registry or load_field_registry()
    fields = _requested_fields(request)

    if fields:
        max_sens = max(registry.sensitivity_rank(f) for f in fields)
    else:
        max_sens = SENSITIVITY_RANK["Public"]
    uses_sensitive = 1 if max_sens >= SENSITIVITY_RANK["Confidential"] else 0

    return {
        "num_filters": len(request.filters or {}),
        "uses_sensitive_field": uses_sensitive,
        "log_matched_group_size": math.log1p(max(0, matched_row_count)),
        "log_min_result_group_size": math.log1p(max(0, min_result_group_size)),
        "has_group_by": 1 if request.group_by else 0,
        "role_clearance_score": int(clearance_level),
        "sensitivity_score": int(max_sens),
        "query_repetition_score": int(repetition_score),
        "filter_specificity_score": float(specificity_score),
        "sequence_narrowing_score": int(narrowing_score),
        "is_count": 1 if request.aggregate == "COUNT" else 0,
        "is_mean": 1 if request.aggregate == "MEAN" else 0,
        "is_extreme_aggregate": 1 if request.aggregate in ("MIN", "MAX") else 0,
    }


def features_to_row(features: dict) -> pd.DataFrame:
    """Model-ready one-row frame in the fixed feature order, with leakage guard."""
    leaked = FORBIDDEN_FEATURES & set(features.keys())
    if leaked:
        raise ValueError(f"Forbidden leakage features present: {sorted(leaked)}")
    return pd.DataFrame([[features[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)
