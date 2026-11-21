"""Attribute-based access policy.

The policy is evaluated before anything else and a denial cannot be overridden
by the risk model. All rules come from config/policies.json.
"""
from __future__ import annotations

from src.config import FieldRegistry, load_field_registry, load_policies
from src.contracts import (
    QueryRequest, UserProfile, PolicyDecision, Role, SENSITIVITY_RANK,
)


def _requested_fields(request: QueryRequest) -> list:
    fields = []
    if request.target_field:
        fields.append(request.target_field)
    if request.group_by:
        fields.append(request.group_by)
    fields.extend((request.filters or {}).keys())
    return fields


def _max_sensitivity(request: QueryRequest, registry: FieldRegistry) -> int:
    fields = _requested_fields(request)
    if not fields:
        return SENSITIVITY_RANK["Public"]
    return max(registry.sensitivity_rank(f) for f in fields)


def _is_broad_manager_query(request: QueryRequest, rule: dict, registry: FieldRegistry) -> bool:
    fields = _requested_fields(request)
    for f in fields:
        if f in rule["broad_query_forbidden_fields"]:
            return False
        if registry.sensitivity_rank(f) > SENSITIVITY_RANK["Internal"]:
            return False
    if len(request.filters or {}) > 1:
        return False
    gb = request.group_by
    if gb not in (None, "admission_type", "service_department"):
        return False
    return True


def evaluate_policy(
    user: UserProfile,
    request: QueryRequest,
    field_registry: FieldRegistry | None = None,
) -> PolicyDecision:
    registry = field_registry or load_field_registry()
    policies = load_policies()
    role = user.role.value
    rule = policies["role_rules"][role]
    sensitivity = _max_sensitivity(request, registry)

    # 1. Purpose allowed for role.
    if user.purpose not in policies["role_purposes"][role]:
        return PolicyDecision(False, "ABAC-001",
                              f"Purpose '{user.purpose}' is not permitted for role {role}.",
                              sensitivity)

    # 2. Clearance vs requested sensitivity.
    if sensitivity > rule["max_sensitivity"]:
        # Role-specific denial identifiers make the audit trail readable.
        if role == "Intern":
            return PolicyDecision(False, "ABAC-003",
                                  "Interns may only run public COUNT queries on admission type.",
                                  sensitivity)
        if role == "Analyst":
            return PolicyDecision(False, "ABAC-004",
                                  "Analysts cannot use demographic, clinical, or outcome fields.",
                                  sensitivity)
        if role == "Manager" and sensitivity >= SENSITIVITY_RANK["Restricted"]:
            return PolicyDecision(False, "ABAC-006",
                                  "Managers can never access the readmission outcome.",
                                  sensitivity)
        return PolicyDecision(False, "ABAC-002",
                              f"Clearance for role {role} is below the requested sensitivity.",
                              sensitivity)

    # 3. Explicitly forbidden fields.
    fields = _requested_fields(request)
    for f in fields:
        if f in rule.get("forbidden_fields", []):
            if role == "Manager":
                return PolicyDecision(False, "ABAC-006",
                                      "Managers can never access the readmission outcome.",
                                      sensitivity)
            if role == "Analyst":
                return PolicyDecision(False, "ABAC-004",
                                      "Analysts cannot use demographic, clinical, or outcome fields.",
                                      sensitivity)
            return PolicyDecision(False, "ABAC-009",
                                  f"Field '{f}' is not permitted for role {role}.",
                                  sensitivity)

    # 4. Aggregate restrictions.
    if request.aggregate not in rule["allowed_aggregates"]:
        if role == "Intern":
            return PolicyDecision(False, "ABAC-003",
                                  "Interns may only run COUNT queries.", sensitivity)
        return PolicyDecision(False, "ABAC-009",
                              f"Aggregate {request.aggregate} not permitted for {role}.",
                              sensitivity)

    # 5. Filter-count ceiling per role.
    if len(request.filters or {}) > rule["max_filters"]:
        return PolicyDecision(False, "ABAC-009",
                              f"Role {role} allows at most {rule['max_filters']} filter(s).",
                              sensitivity)

    # 6. Intern allowed-field whitelist.
    if role == "Intern":
        allowed = set(rule.get("allowed_fields", []))
        for f in fields:
            if f not in allowed:
                return PolicyDecision(False, "ABAC-003",
                                      "Interns may only touch admission type.", sensitivity)

    # 7. Manager department scope.
    if role == "Manager":
        dept_field = rule["department_field"]
        filt = request.filters or {}
        has_own_dept = filt.get(dept_field) == user.department
        if not has_own_dept and not _is_broad_manager_query(request, rule, registry):
            return PolicyDecision(False, "ABAC-005",
                                  f"Managers must scope to their own department "
                                  f"({user.department}) or run a broad non-sensitive query.",
                                  sensitivity)

    # 8. Allowed. Provisional for high-sensitivity Researcher / Admin requests.
    if role == "Researcher" and sensitivity >= SENSITIVITY_RANK["Restricted"]:
        return PolicyDecision(True, "ABAC-007",
                              "Restricted request provisionally allowed; subject to k and risk controls.",
                              sensitivity)
    if role == "Admin" and sensitivity >= SENSITIVITY_RANK["Restricted"]:
        return PolicyDecision(True, "ABAC-008",
                              "Admin request provisionally allowed; subject to k and risk controls.",
                              sensitivity)
    return PolicyDecision(True, "ABAC-009",
                          f"Request permitted under role {role} rules.", sensitivity)
