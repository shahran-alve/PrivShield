"""The population workload and the two defenses it is run against.

The workload is a fixed stream of broad grouped COUNT queries by one analyst
account: group by one of four broad attributes, with no filter or one filter
on another of them. A defense takes (user, request, data, registry) and
returns (result or None, exact, action name).
"""
from __future__ import annotations

import random

from src import decision_engine as de
from src.config import GLOBAL_SEED, K_THRESHOLD
from src.contracts import Action, QueryRequest, Role, UserProfile
from src.query_engine import EmptyResultError, QueryValidationError, execute_query

BROAD_FIELDS = ["age_band", "race", "gender", "admission_type"]


def _raw(data, request, registry):
    try:
        result, stats = execute_query(data, request, registry)
    except (EmptyResultError, QueryValidationError):
        return None, False, None
    return result, True, stats


def defense_k_only(user, request, data, registry):
    """Refuse the whole result when its smallest group is below the threshold."""
    result, ok, stats = _raw(data, request, registry)
    if not ok:
        return None, False, "EMPTY"
    if stats.min_result_group_size < K_THRESHOLD:
        return None, False, "BLOCK_K"
    return result, True, "ALLOW"


def defense_full(user, request, data, registry):
    """The complete controller. The result may be perturbed or have groups masked."""
    try:
        res = de.decide(user, request, data, write_audit=True)
    except Exception:
        return None, False, "ERROR"
    if res.action in (Action.BLOCK_POLICY, Action.BLOCK_HIGH_RISK):
        return None, False, res.action.value
    return res.protected_result, res.action == Action.ALLOW, res.action.value


def requests(data, n=120, seed=GLOBAL_SEED):
    """The request stream as (group_by, filters) pairs."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        gb = rng.choice(BROAD_FIELDS)
        filters = {}
        if rng.random() < 0.5:
            f = rng.choice([c for c in BROAD_FIELDS if c != gb])
            filters = {f: str(rng.choice(sorted(data[f].dropna().unique())))}
        out.append((gb, filters))
    return out


def run(data, registry, defense, n=120, seed=GLOBAL_SEED):
    """Answered and exact rates of a defense on the workload."""
    user = UserProfile(
        user_id="benign_analyst", display_name="Benign Analyst",
        role=Role.RESEARCHER, department="Clinical Research",
        clearance_level=4, purpose="research",
    )
    answered = exact_n = 0
    for gb, filters in requests(data, n, seed):
        request = QueryRequest(aggregate="COUNT", target_field=None, group_by=gb, filters=filters)
        _, ok, _ = _raw(data, request, registry)
        if not ok:
            continue
        result, exact, _ = defense(user, request, data, registry)
        if result is None or result.empty:
            continue
        answered += 1
        exact_n += int(exact)
    return {"answered": answered, "exact": exact_n, "queries": n}
