"""Actions applied to an allowed result: masking of small groups and noise.

The noise is deterministic: the same user asking the same question on the same
day gets the same perturbed value, so repeating a query does not average it
away. It is a heuristic with no epsilon and no privacy budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import Action, QueryRequest
from src.utils import sha256_text

SUPPRESSED = "Suppressed (<10)"
SMALL_MARKER = "<10"


def _seed(user_id: str, fingerprint: str, date_bucket: str, salt: str, extra: str = "") -> int:
    digest = sha256_text("|".join([user_id, fingerprint, date_bucket, salt, extra]))
    return int(digest[:8], 16)


def _count_noise(rng) -> int:
    return int(rng.choice([-2, -1, 0, 1, 2]))


def apply_allow(result: pd.DataFrame) -> pd.DataFrame:
    return result.copy()


def apply_noise(
    request: QueryRequest,
    result: pd.DataFrame,
    fingerprint: str,
    user_id: str,
    date_bucket: str,
    salt: str,
    numeric_range: float | None = None,
) -> pd.DataFrame:
    """ALLOW_WITH_NOISE for COUNT / MEAN. MIN/MAX never reach here."""
    out = result.copy()
    is_grouped = request.group_by is not None and request.group_by in out.columns

    for idx in out.index:
        group_label = str(out.at[idx, request.group_by]) if is_grouped else ""
        rng = np.random.default_rng(_seed(user_id, fingerprint, date_bucket, salt, group_label))
        val = out.at[idx, "value"]
        if request.aggregate == "COUNT":
            noised = max(0, int(val) + _count_noise(rng))
            out.at[idx, "value"] = noised
        elif request.aggregate == "MEAN":
            sigma = 0.01 * (numeric_range if numeric_range else max(1.0, abs(float(val))))
            noised = float(val) + rng.normal(0.0, sigma)
            if numeric_range is not None and numeric_range > 0:
                lo = float(out.at[idx, "value"]) - numeric_range
                hi = float(out.at[idx, "value"]) + numeric_range
                noised = min(hi, max(lo, noised))
            out.at[idx, "value"] = round(float(noised), 2)
    return out


def apply_mask(request: QueryRequest, result: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """MASK_SMALL_GROUPS. Grouped: suppress small groups. Ungrouped: '<10'."""
    out = result.copy()
    if request.group_by is not None and request.group_by in out.columns:
        out["value"] = out["value"].astype(object)
        mask = out["group_size"] < k
        out.loc[mask, "value"] = SUPPRESSED
        return out
    # Ungrouped, non-sensitive, below k.
    out["value"] = out["value"].astype(object)
    out["value"] = SMALL_MARKER
    return out


def numeric_range_for(data: pd.DataFrame, target_field: str) -> float:
    col = pd.to_numeric(data[target_field], errors="coerce").dropna()
    if col.empty:
        return 1.0
    return float(col.max() - col.min()) or 1.0
