"""Generate labelled query episodes for the risk model.

Simulates policy-allowed queries against the processed visit table, executes
each one to obtain the matched and group sizes, attaches a recent history to a
quarter of them, and labels each with an offline rule. The rule may use one
quantity the runtime model never sees: how concentrated the readmission
outcome is within the matched rows.

Run:  python -m src.training_data
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.config import (
    load_field_registry, load_users, DATA_DIR, GLOBAL_SEED,
)
from src.contracts import QueryRequest, EmptyResultError, QueryValidationError
from src.query_engine import execute_query
from src.policy_engine import evaluate_policy
from src.risk_features import compute_features, FEATURE_ORDER
from src.utils import derive_seed

from src.config import WORKING_CSV as PROCESSED_25K
TRAINING_DIR = os.path.join(DATA_DIR, "training")
TRAINING_CSV = os.path.join(TRAINING_DIR, "query_risk_training.csv")

MIN_PER_CLASS = 1500
DEFAULT_EXAMPLES = 6000

SENSITIVE_FIELDS = {
    "age_band", "race", "gender", "diagnosis_group", "A1Cresult",
    "max_glu_serum", "insulin", "diabetesMed", "readmitted",
}
NUMERIC_TARGETS = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                   "num_medications", "prior_utilization", "number_diagnoses"]


def _dominant_readmitted_proportion(subset: pd.DataFrame) -> float:
    if subset.empty:
        return 0.0
    counts = subset["readmitted"].value_counts(normalize=True)
    return float(counts.iloc[0])


def oracle_label(request, matched, min_group, repetition, narrowing,
                 specificity, dominant_prop, uses_sensitive, restricted,
                 num_filters, is_extreme) -> tuple:
    """Return (label, reason). Evaluate High before Medium."""
    if uses_sensitive and min_group < 10:
        return "High Risk", "sensitive_field_min_group_lt_10"
    if restricted and min_group < 20:
        return "High Risk", "restricted_field_min_group_lt_20"
    if repetition >= 3 and narrowing >= 2:
        return "High Risk", "repeated_narrowing"
    if uses_sensitive and matched < 50 and dominant_prop >= 0.90:
        return "High Risk", "concentrated_sensitive_outcome"

    if uses_sensitive:
        return "Medium Risk", "any_sensitive_field"
    if 10 <= min_group <= 29:
        return "Medium Risk", "small_group_10_to_29"
    if num_filters >= 3:
        return "Medium Risk", "three_or_more_filters"
    if repetition >= 2:
        return "Medium Risk", "repetition_ge_2"
    if specificity >= 0.98:
        return "Medium Risk", "high_specificity"
    if is_extreme and num_filters >= 2:
        return "Medium Risk", "extreme_aggregate_with_filters"

    return "Safe", "no_high_or_medium_condition"


NON_SENSITIVE_GROUP = ["admission_type", "service_department"]
NON_SENSITIVE_FILTER = ["admission_type", "service_department"]


def _random_request(rng, registry, data_categories, intent="mixed"):
    """Draw a bounded query.

    intent controls the field pool so the three risk classes all fill in
    reasonable time: 'plain' avoids sensitive fields (feeds Safe/low-Medium),
    'sensitive' leans into them (feeds Medium/High).
    """
    agg = rng.choice(["COUNT", "MEAN", "MIN", "MAX"], p=[0.4, 0.3, 0.15, 0.15])
    target = None if agg == "COUNT" else str(rng.choice(NUMERIC_TARGETS))

    if intent == "plain":
        group_pool = NON_SENSITIVE_GROUP
        filter_pool = NON_SENSITIVE_FILTER
    else:
        group_pool = registry.groupable()
        filter_pool = registry.filterable()

    group_by = None
    if rng.random() < 0.6:
        group_by = str(rng.choice(group_pool))

    filterable = [f for f in filter_pool if f != group_by]
    if intent == "sensitive":
        n_filters = rng.choice([1, 2, 3, 4], p=[0.30, 0.34, 0.22, 0.14])
    elif intent == "plain":
        n_filters = rng.choice([0, 1, 2], p=[0.30, 0.45, 0.25])
    else:
        n_filters = rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.30, 0.28, 0.17, 0.10])
    n_filters = min(int(n_filters), len(filterable))
    chosen = rng.choice(filterable, size=n_filters, replace=False) if n_filters else []
    filters = {f: str(rng.choice(data_categories[f])) for f in chosen}

    return QueryRequest(aggregate=str(agg),
                        target_field=None if target is None else target,
                        group_by=group_by,
                        filters=filters)


def generate(n_examples: int = DEFAULT_EXAMPLES) -> pd.DataFrame:
    registry = load_field_registry()
    users = load_users()
    data = pd.read_csv(PROCESSED_25K)
    total_rows = len(data)

    # Prefer higher-clearance users so that most episodes are policy-allowed and
    # reach the sensitive fields.
    candidate_users = [u for u in users.values() if u.role.value in ("Admin", "Researcher")]

    data_categories = {f: sorted({str(x) for x in data[f].dropna().unique()})
                       for f in registry.filterable()}

    rng = np.random.default_rng(derive_seed(GLOBAL_SEED, "training"))

    rows = []
    seen = set()
    class_counts = {"Safe": 0, "Medium Risk": 0, "High Risk": 0}
    attempts = 0
    max_attempts = 30000
    stagnation = 0
    max_stagnation = 3000  # stop if this many attempts add nothing new
    intents = ["plain", "plain", "sensitive", "sensitive", "mixed"]

    while (len(rows) < n_examples or min(class_counts.values()) < MIN_PER_CLASS) \
            and attempts < max_attempts and stagnation < max_stagnation:
        attempts += 1
        stagnation += 1
        if attempts % 4000 == 0:
            print(f"[training_data] attempts={attempts} kept={len(rows)} "
                  f"counts={class_counts} stagnation={stagnation}", flush=True)
        user = candidate_users[rng.integers(0, len(candidate_users))]
        # Steer toward whichever class is currently starved.
        if class_counts["Safe"] < MIN_PER_CLASS and class_counts["Safe"] <= class_counts["High Risk"]:
            intent = "plain"
        elif class_counts["High Risk"] < MIN_PER_CLASS:
            intent = "sensitive"
        else:
            intent = intents[attempts % len(intents)]
        request = _random_request(rng, registry, data_categories, intent=intent)

        # Optional synthetic history for ~25% of episodes (decided up front so it
        # can be part of the dedup key).
        repetition = narrowing = 0
        if rng.random() < 0.25:
            repetition = int(rng.integers(1, 5))
            narrowing = int(rng.integers(0, repetition + 1))

        # Cheap dedup FIRST, before any policy check or query execution. This is
        # what keeps the starved-class tail fast: duplicates are rejected without
        # touching the 25k-row frame.
        key = (request.aggregate, request.target_field, request.group_by,
               tuple(sorted(request.filters.items())), repetition, narrowing)
        if key in seen:
            continue

        policy = evaluate_policy(user, request, registry)
        if not policy.allowed:
            continue

        try:
            result, stats = execute_query(data, request, registry)
        except (EmptyResultError, QueryValidationError):
            continue

        fields = list(request.filters.keys())
        if request.group_by:
            fields.append(request.group_by)
        if request.target_field:
            fields.append(request.target_field)
        uses_sensitive = any(f in SENSITIVE_FIELDS for f in fields)
        restricted = "readmitted" in fields
        is_extreme = request.aggregate in ("MIN", "MAX")

        # The oracle only needs the concentration peek for one High-Risk rule
        # (sensitive query, small matched population). Compute it lazily so the
        # common case never pays for the extra filtering pass.
        dominant_prop = 0.0
        if uses_sensitive and stats.matched_row_count < 50 and request.filters:
            subset = data
            for f, v in request.filters.items():
                subset = subset[subset[f].astype(str) == str(v)]
            dominant_prop = _dominant_readmitted_proportion(subset)

        feats = compute_features(
            request=request,
            clearance_level=user.clearance_level,
            matched_row_count=stats.matched_row_count,
            min_result_group_size=stats.min_result_group_size,
            repetition_score=repetition,
            specificity_score=1.0 - stats.matched_row_count / total_rows,
            narrowing_score=narrowing,
            field_registry=registry,
        )

        label, reason = oracle_label(
            request, stats.matched_row_count, stats.min_result_group_size,
            repetition, narrowing, feats["filter_specificity_score"],
            dominant_prop, uses_sensitive, restricted,
            len(request.filters), is_extreme,
        )

        # Rejection sampling. Once we already have the requested total, only keep
        # classes still under their floor so the tail fills deficient classes.
        if len(rows) >= n_examples and class_counts[label] >= MIN_PER_CLASS:
            continue

        seen.add(key)

        record = dict(feats)
        record["label"] = label
        record["label_rule_reason"] = reason
        record["user_role"] = user.role.value
        rows.append(record)
        class_counts[label] += 1
        stagnation = 0

    df = pd.DataFrame(rows)
    return df


def run(n_examples: int = DEFAULT_EXAMPLES) -> pd.DataFrame:
    os.makedirs(TRAINING_DIR, exist_ok=True)
    df = generate(n_examples)
    df.to_csv(TRAINING_CSV, index=False)
    counts = df["label"].value_counts().to_dict()
    print(f"[training_data] wrote {TRAINING_CSV} with {len(df)} episodes")
    print(f"[training_data] class counts: {counts}")
    # Leakage guard: feature columns must be exactly FEATURE_ORDER.
    feature_cols = [c for c in df.columns if c in FEATURE_ORDER]
    assert set(feature_cols) == set(FEATURE_ORDER), "feature columns drifted"
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    args = parser.parse_args()
    run(args.examples)
