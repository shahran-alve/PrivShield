"""The controller's single entry point.

`decide` validates a request, applies the access policy, runs the bounded
aggregate, computes history and risk features, asks the risk model for a
label, and maps policy result, threshold and label to one action: allow, allow
with noise, mask small groups, or block. A blocked result carries no data.
"""
from __future__ import annotations

import datetime as dt
import time
import uuid

import pandas as pd

from src.config import load_field_registry, K_THRESHOLD, GLOBAL_SEED
from src.contracts import (
    Action, DecisionResult, PolicyDecision, QueryRequest, RiskLabel,
    UserProfile, QueryValidationError, EmptyResultError,
)
from src.query_engine import execute_query
from src.policy_engine import evaluate_policy
from src.history_engine import compute_history_scores
from src.risk_features import compute_features
from src.risk_model import predict_risk
from src.privacy_actions import (
    apply_allow, apply_noise, apply_mask, numeric_range_for, SUPPRESSED,
)
from src.audit_logger import append_audit, make_timestamp, encode_filters
from src.utils import canonical_json

SENSITIVE_FIELDS = {
    "age_band", "race", "gender", "diagnosis_group", "A1Cresult",
    "max_glu_serum", "insulin", "diabetesMed", "readmitted",
}
DEFAULT_SALT = "privshield-local-salt"


def _requested_fields(request: QueryRequest):
    fields = []
    if request.target_field:
        fields.append(request.target_field)
    if request.group_by:
        fields.append(request.group_by)
    fields.extend((request.filters or {}).keys())
    return fields


def _uses_sensitive(request: QueryRequest) -> bool:
    return any(f in SENSITIVE_FIELDS for f in _requested_fields(request))


def _date_bucket(now: dt.datetime) -> str:
    return now.strftime("%Y-%m-%d")


def decide(
    user: UserProfile,
    request: QueryRequest,
    data: pd.DataFrame,
    now_utc: "dt.datetime | None" = None,
    salt: str = DEFAULT_SALT,
    write_audit: bool = True,
) -> DecisionResult:
    start = time.perf_counter()
    registry = load_field_registry()
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    request_id = uuid.uuid4().hex[:12]

    def _finish(action, policy, risk, privacy_result, explanation, stats, protected,
                rep=0, narrow=0):
        latency = round((time.perf_counter() - start) * 1000, 3)
        if write_audit:
            append_audit({
                "timestamp_utc": now.isoformat(),
                "request_id": request_id,
                "user_id": user.user_id,
                "user_role": user.role.value,
                "user_department": user.department,
                "purpose": user.purpose,
                "query_summary": stats.query_summary if stats else "",
                "aggregate": request.aggregate,
                "target_field": request.target_field or "",
                "group_by": request.group_by or "",
                "filters_json": encode_filters(request.filters),
                "query_fingerprint": stats.query_fingerprint if stats else "",
                "structural_signature": stats.structural_signature if stats else "",
                "requested_sensitivity": policy.requested_sensitivity,
                "policy_allowed": policy.allowed,
                "policy_id": policy.policy_id,
                "policy_reason": policy.reason,
                "matched_row_count": stats.matched_row_count if stats else 0,
                "min_result_group_size": stats.min_result_group_size if stats else 0,
                "repetition_score": rep,
                "narrowing_score": narrow,
                "risk_label": risk.label.value if risk else "",
                "risk_score": risk.risk_score if risk else "",
                "action": action.value,
                "explanation_json": canonical_json(explanation),
                "latency_ms": latency,
            })
        return DecisionResult(
            allowed=action in (Action.ALLOW, Action.ALLOW_WITH_NOISE, Action.MASK_SMALL_GROUPS),
            action=action, policy=policy, risk=risk, privacy_result=privacy_result,
            explanation=explanation, query_stats=stats, protected_result=protected,
            latency_ms=latency,
        )

    # 1. Validation + policy gate (policy denial always wins).
    try:
        policy = evaluate_policy(user, request, registry)
    except QueryValidationError as exc:
        empty_policy = PolicyDecision(False, "ABAC-000", str(exc), 0)
        return _finish(Action.BLOCK_POLICY, empty_policy, None,
                       "Request rejected during validation.",
                       [f"Validation error: {exc}"], None, None)

    if not policy.allowed:
        expl = [
            f"Policy {policy.policy_id}: {policy.reason}",
            "The request was blocked before any data was read.",
        ]
        return _finish(Action.BLOCK_POLICY, policy, None,
                       "Blocked by access policy.", expl, None, None)

    # 2. Execute the allowed request.
    try:
        result, stats = execute_query(data, request, registry)
    except EmptyResultError:
        expl = [f"Policy {policy.policy_id}: {policy.reason}",
                "No encounters match the requested filters."]
        return _finish(Action.ALLOW, policy, None, "No matching population.", expl, None, None)
    except QueryValidationError as exc:
        return _finish(Action.BLOCK_POLICY, policy, None, "Validation error.",
                       [f"Validation error: {exc}"], None, None)

    # 3. History + features + risk.
    hist = compute_history_scores(
        user_id=user.user_id, aggregate=request.aggregate,
        target_field=request.target_field, group_by=request.group_by,
        filters=request.filters, matched_row_count=stats.matched_row_count,
        total_dataset_rows=len(data), now_utc=now,
    )
    feats = compute_features(
        request=request, clearance_level=user.clearance_level,
        matched_row_count=stats.matched_row_count,
        min_result_group_size=stats.min_result_group_size,
        repetition_score=hist.query_repetition_score,
        specificity_score=hist.filter_specificity_score,
        narrowing_score=hist.sequence_narrowing_score,
        field_registry=registry,
    )
    risk = predict_risk(feats)

    sensitive = _uses_sensitive(request)
    min_group = stats.min_result_group_size
    date_bucket = _date_bucket(now)

    base_expl = [
        f"Policy {policy.policy_id}: {policy.reason}",
        f"Requested sensitivity level {policy.requested_sensitivity} "
        f"({'includes a sensitive field' if sensitive else 'non-sensitive fields only'}).",
    ]

    # 4. Precedence table.
    # k violation on a sensitive query -> hard block.
    if sensitive and min_group < K_THRESHOLD:
        expl = base_expl + [
            "A matched subgroup is smaller than the privacy threshold, so the "
            "result is withheld to prevent identifying a small group.",
            f"Model label {risk.label.value} (score {risk.risk_score}).",
        ]
        return _finish(Action.BLOCK_HIGH_RISK, policy, risk,
                       "Blocked: small sensitive subgroup.", expl, stats, None,
                       hist.query_repetition_score, hist.sequence_narrowing_score)

    # k violation on a non-sensitive query -> mask.
    if not sensitive and min_group < K_THRESHOLD:
        masked = apply_mask(request, result, K_THRESHOLD)
        expl = base_expl + [
            f"Matched {stats.matched_row_count} rows; smallest group {min_group} "
            f"is below the threshold, so small groups are suppressed.",
            f"Model label {risk.label.value} (score {risk.risk_score}).",
            "Applied: small-group suppression.",
        ]
        return _finish(Action.MASK_SMALL_GROUPS, policy, risk,
                       "Small groups suppressed.", expl, stats, masked,
                       hist.query_repetition_score, hist.sequence_narrowing_score)

    # k passes. Model-driven branches.
    if risk.label == RiskLabel.HIGH:
        expl = base_expl + [
            "The controller judged this request high risk for inference, so the "
            "result is withheld.",
            f"Model label High Risk (score {risk.risk_score}); "
            f"factors: {', '.join(risk.top_factors) or 'query shape'}.",
        ]
        return _finish(Action.BLOCK_HIGH_RISK, policy, risk,
                       "Blocked: high inference risk.", expl, stats, None,
                       hist.query_repetition_score, hist.sequence_narrowing_score)

    if risk.label == RiskLabel.MEDIUM:
        if request.aggregate in ("COUNT", "MEAN"):
            nr = numeric_range_for(data, request.target_field) if request.aggregate == "MEAN" else None
            protected = apply_noise(request, result, stats.query_fingerprint,
                                    user.user_id, date_bucket, salt, nr)
            expl = base_expl + [
                f"Matched {stats.matched_row_count} rows; smallest group {min_group}.",
                f"Model label Medium Risk (score {risk.risk_score}).",
                "Applied: deterministic small noise (heuristic protection, not "
                "formal differential privacy).",
            ]
            return _finish(Action.ALLOW_WITH_NOISE, policy, risk,
                           "Allowed with noise.", expl, stats, protected,
                           hist.query_repetition_score, hist.sequence_narrowing_score)
        if request.group_by is not None:  # grouped MIN/MAX
            masked = apply_mask(request, result, K_THRESHOLD)
            expl = base_expl + [
                f"Model label Medium Risk (score {risk.risk_score}).",
                "Applied: small-group suppression for a grouped extreme query.",
            ]
            return _finish(Action.MASK_SMALL_GROUPS, policy, risk,
                           "Small groups suppressed.", expl, stats, masked,
                           hist.query_repetition_score, hist.sequence_narrowing_score)
        # ungrouped MIN/MAX medium -> block
        expl = base_expl + [
            f"Model label Medium Risk (score {risk.risk_score}).",
            "An ungrouped extreme value on a medium-risk query is withheld.",
        ]
        return _finish(Action.BLOCK_HIGH_RISK, policy, risk,
                       "Blocked: medium-risk extreme value.", expl, stats, None,
                       hist.query_repetition_score, hist.sequence_narrowing_score)

    # Safe.
    expl = base_expl + [
        f"Matched {stats.matched_row_count} rows; smallest group {min_group}.",
        f"Model label Safe (score {risk.risk_score}).",
        "Applied: none. Result returned unchanged.",
    ]
    return _finish(Action.ALLOW, policy, risk, "Allowed.", expl, stats,
                   apply_allow(result), hist.query_repetition_score,
                   hist.sequence_narrowing_score)
