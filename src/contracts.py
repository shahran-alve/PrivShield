"""Enums, dataclasses and exceptions shared by the controller modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class Role(str, Enum):
    ADMIN = "Admin"
    RESEARCHER = "Researcher"
    MANAGER = "Manager"
    ANALYST = "Analyst"
    INTERN = "Intern"


class Sensitivity(str, Enum):
    PUBLIC = "Public"
    INTERNAL = "Internal"
    CONFIDENTIAL = "Confidential"
    RESTRICTED = "Restricted"


# Numeric ordering used everywhere a "maximum sensitivity" comparison is needed.
SENSITIVITY_RANK = {
    "Public": 1,
    "Internal": 2,
    "Confidential": 3,
    "Restricted": 4,
}


class RiskLabel(str, Enum):
    SAFE = "Safe"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


class Action(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_NOISE = "ALLOW_WITH_NOISE"
    MASK_SMALL_GROUPS = "MASK_SMALL_GROUPS"
    BLOCK_POLICY = "BLOCK_POLICY"
    BLOCK_HIGH_RISK = "BLOCK_HIGH_RISK"


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    display_name: str
    role: Role
    department: str
    clearance_level: int
    purpose: str


@dataclass(frozen=True)
class QueryRequest:
    aggregate: str
    target_field: Optional[str]
    group_by: Optional[str]
    filters: dict  # dict[str, str]

    def with_filters(self, new_filters: dict) -> "QueryRequest":
        return QueryRequest(
            aggregate=self.aggregate,
            target_field=self.target_field,
            group_by=self.group_by,
            filters=dict(new_filters),
        )


@dataclass
class QueryStats:
    matched_row_count: int
    result_row_count: int
    min_result_group_size: int
    query_summary: str
    query_fingerprint: str
    structural_signature: str


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    policy_id: str
    reason: str
    requested_sensitivity: int


@dataclass(frozen=True)
class RiskPrediction:
    label: RiskLabel
    risk_score: float
    probabilities: dict  # dict[str, float]
    top_factors: list  # list[str]


@dataclass
class DecisionResult:
    allowed: bool
    action: Action
    policy: PolicyDecision
    risk: Optional[RiskPrediction]
    privacy_result: str
    explanation: list
    query_stats: Optional[QueryStats]
    protected_result: Optional[pd.DataFrame]
    latency_ms: float


# ---------------------------------------------------------------------------
# Controlled exceptions
# ---------------------------------------------------------------------------
class PrivShieldError(Exception):
    """Base class for all controlled PrivShield errors."""


class QueryValidationError(PrivShieldError):
    """Raised for an unsupported or internally inconsistent request."""


class PolicyDeniedError(PrivShieldError):
    """Optional internal signal after a PolicyDecision denial."""


class EmptyResultError(PrivShieldError):
    """Valid filters that match zero rows."""


class ArtifactMissingError(PrivShieldError):
    """A dataset, model, config, or evaluation artifact is absent."""


class ArtifactValidationError(PrivShieldError):
    """A schema, model metadata record, or checksum does not match."""
