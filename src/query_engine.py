"""Bounded aggregate query engine.

COUNT, MEAN, MIN or MAX with at most one group-by and at most four equality
filters on validated categorical fields. No user-supplied string reaches
DataFrame.query, eval or SQL, and record identifiers cannot be selected,
filtered, grouped or returned.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.config import FieldRegistry, load_field_registry
from src.contracts import (
    QueryRequest, QueryStats, QueryValidationError, EmptyResultError,
)
from src.utils import canonical_json, sha256_text

AGGREGATES = {"COUNT", "MEAN", "MIN", "MAX"}
MAX_FILTERS = 4
FILTER_VALUE_MARKER = "*"
NEVER_QUERYABLE = {"record_id"}


def _validate(request: QueryRequest, registry: FieldRegistry) -> None:
    agg = request.aggregate
    if agg not in AGGREGATES:
        raise QueryValidationError(f"Unsupported aggregate: {agg}")

    if agg == "COUNT":
        if request.target_field is not None:
            raise QueryValidationError("COUNT must not have a target field")
    else:
        tf = request.target_field
        if tf is None:
            raise QueryValidationError(f"{agg} requires a numeric target field")
        if tf in NEVER_QUERYABLE:
            raise QueryValidationError(f"{tf} cannot be queried")
        if tf not in registry:
            raise QueryValidationError(f"Unknown target field: {tf}")
        spec = registry.get(tf)
        if not spec.is_numeric_target() or agg.lower() not in [u.lower() for u in spec.allowed_use]:
            raise QueryValidationError(f"{agg} not allowed on {tf}")

    gb = request.group_by
    if gb is not None:
        if gb in NEVER_QUERYABLE:
            raise QueryValidationError(f"{gb} cannot be used as group_by")
        if gb not in registry:
            raise QueryValidationError(f"Unknown group_by field: {gb}")
        if not registry.get(gb).can_group():
            raise QueryValidationError(f"{gb} cannot be grouped")

    filters = request.filters or {}
    if len(filters) > MAX_FILTERS:
        raise QueryValidationError(f"At most {MAX_FILTERS} filters allowed")
    for fname, fval in filters.items():
        if fname in NEVER_QUERYABLE:
            raise QueryValidationError(f"{fname} cannot be filtered")
        if fname not in registry:
            raise QueryValidationError(f"Unknown filter field: {fname}")
        if not registry.get(fname).can_filter():
            raise QueryValidationError(f"{fname} cannot be filtered")
        if fval is None or str(fval).strip() == "":
            raise QueryValidationError(f"Empty filter value for {fname}")
        if gb is not None and fname == gb:
            raise QueryValidationError("A field cannot be both group_by and a filter")


def _apply_filters(data: pd.DataFrame, filters: dict, registry: FieldRegistry) -> pd.DataFrame:
    mask = pd.Series(True, index=data.index)
    for fname, fval in filters.items():
        if fname not in data.columns:
            raise QueryValidationError(f"Filter field {fname} not in dataset")
        valid_values = set(data[fname].astype(str).unique())
        if str(fval) not in valid_values:
            raise QueryValidationError(f"Unknown category '{fval}' for {fname}")
        mask &= (data[fname].astype(str) == str(fval))
    return data.loc[mask]


def build_summary(request: QueryRequest, registry: FieldRegistry) -> str:
    parts = []
    if request.aggregate == "COUNT":
        parts.append("Count of encounters")
    else:
        disp = registry.get(request.target_field).display_name
        parts.append(f"{request.aggregate.title()} of {disp}")
    if request.group_by:
        parts.append(f"grouped by {registry.get(request.group_by).display_name}")
    if request.filters:
        clause = ", ".join(
            f"{registry.get(k).display_name} = {v}" for k, v in sorted(request.filters.items())
        )
        parts.append(f"where {clause}")
    return " ".join(parts)


def _fingerprints(request: QueryRequest) -> Tuple[str, str]:
    base = {
        "aggregate": request.aggregate,
        "target_field": request.target_field,
        "group_by": request.group_by,
        "filters": {k: str(v) for k, v in sorted((request.filters or {}).items())},
    }
    fingerprint = sha256_text(canonical_json(base))
    structural = dict(base)
    structural["filters"] = {k: FILTER_VALUE_MARKER for k in sorted((request.filters or {}))}
    signature = sha256_text(canonical_json(structural))
    return fingerprint, signature


def execute_query(
    data: pd.DataFrame,
    request: QueryRequest,
    field_registry: FieldRegistry | None = None,
) -> Tuple[pd.DataFrame, QueryStats]:
    registry = field_registry or load_field_registry()
    _validate(request, registry)

    subset = _apply_filters(data, request.filters or {}, registry)
    matched = int(subset.shape[0])
    if matched == 0:
        raise EmptyResultError("No rows match the requested filters")

    summary = build_summary(request, registry)
    fingerprint, signature = _fingerprints(request)

    agg = request.aggregate
    gb = request.group_by

    if gb is None:
        if agg == "COUNT":
            value = matched
        else:
            col = subset[request.target_field]
            value = {"MEAN": col.mean, "MIN": col.min, "MAX": col.max}[agg]()
        result = pd.DataFrame({"value": [value], "group_size": [matched]})
        min_group = matched
    else:
        grouped = subset.groupby(gb, observed=True, dropna=True)
        sizes = grouped.size()
        if agg == "COUNT":
            values = sizes
        else:
            values = grouped[request.target_field].agg(
                {"MEAN": "mean", "MIN": "min", "MAX": "max"}[agg]
            )
        frame = pd.DataFrame({"value": values})
        frame["group_size"] = sizes
        frame = frame.dropna(subset=["group_size"])
        if frame.empty:
            raise EmptyResultError("No non-null groups for the requested group_by")
        result = pd.DataFrame({
            gb: frame.index.astype(str),
            "value": frame["value"].to_numpy(),
            "group_size": frame["group_size"].astype(int).to_numpy(),
        }).sort_values(gb).reset_index(drop=True)
        min_group = int(result["group_size"].min())

    stats = QueryStats(
        matched_row_count=matched,
        result_row_count=int(result.shape[0]),
        min_result_group_size=int(min_group),
        query_summary=summary,
        query_fingerprint=fingerprint,
        structural_signature=signature,
    )
    return result.copy(), stats
