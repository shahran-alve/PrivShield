"""Configuration loader.

Reads config/fields.json, config/users.json and config/policies.json once,
checks them for inconsistencies (a user with an unknown role, a policy that
names a field that does not exist) and returns typed objects. Also defines the
project paths, the global seed and the threshold k.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from src.contracts import (
    Role,
    UserProfile,
    SENSITIVITY_RANK,
    PrivShieldError,
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_HERE, "config")
DATA_DIR = os.path.join(_HERE, "data")
MODELS_DIR = os.path.join(_HERE, "models")
REPORTS_DIR = os.path.join(_HERE, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

GLOBAL_SEED = 42
K_THRESHOLD = 10

# Visit-level table used to train the risk model.
WORKING_CSV = os.path.join(DATA_DIR, "processed", "diabetes_full.csv")


class ConfigError(PrivShieldError):
    pass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str
    sensitivity: str
    allowed_use: tuple
    display_name: str
    notes: str

    @property
    def sensitivity_rank(self) -> int:
        return SENSITIVITY_RANK[self.sensitivity]

    def can_filter(self) -> bool:
        return "filter" in self.allowed_use

    def can_group(self) -> bool:
        return "group" in self.allowed_use

    def is_numeric_target(self) -> bool:
        return self.type == "numeric"


class FieldRegistry:
    """Read-only view over config/fields.json."""

    def __init__(self, raw: dict):
        self._levels = raw["sensitivity_levels"]
        self._fields = {}
        for name, spec in raw["fields"].items():
            self._fields[name] = FieldSpec(
                name=name,
                type=spec["type"],
                sensitivity=spec["sensitivity"],
                allowed_use=tuple(spec["allowed_use"]),
                display_name=spec["display_name"],
                notes=spec.get("notes", ""),
            )

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def get(self, name: str) -> FieldSpec:
        if name not in self._fields:
            raise ConfigError(f"Unknown field: {name}")
        return self._fields[name]

    def all_fields(self) -> list:
        return list(self._fields.values())

    def categorical_fields(self) -> list:
        return [f for f in self._fields.values() if f.type == "categorical"]

    def numeric_fields(self) -> list:
        return [f for f in self._fields.values() if f.type == "numeric"]

    def filterable(self) -> list:
        return [f.name for f in self._fields.values() if f.can_filter()]

    def groupable(self) -> list:
        return [f.name for f in self._fields.values() if f.can_group()]

    def numeric_targets(self) -> list:
        return [f.name for f in self._fields.values() if f.is_numeric_target()]

    def sensitivity_rank(self, name: str) -> int:
        return self.get(name).sensitivity_rank


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        raise ConfigError(f"Missing configuration file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_field_registry() -> FieldRegistry:
    return FieldRegistry(_read_json(os.path.join(CONFIG_DIR, "fields.json")))


@lru_cache(maxsize=1)
def load_users() -> dict:
    raw = _read_json(os.path.join(CONFIG_DIR, "users.json"))
    profiles = {}
    for u in raw["users"]:
        try:
            role = Role(u["role"])
        except ValueError as exc:
            raise ConfigError(f"User {u['user_id']} has unknown role {u['role']}") from exc
        profiles[u["user_id"]] = UserProfile(
            user_id=u["user_id"],
            display_name=u["display_name"],
            role=role,
            department=u["department"],
            clearance_level=int(u["clearance_level"]),
            purpose=u["purpose"],
        )
    return profiles


@lru_cache(maxsize=1)
def load_policies() -> dict:
    return _read_json(os.path.join(CONFIG_DIR, "policies.json"))


def get_user(user_id: str) -> UserProfile:
    users = load_users()
    if user_id not in users:
        raise ConfigError(f"Unknown user_id: {user_id}")
    return users[user_id]


def validate_all() -> dict:
    """Cross-check the three config files against each other.

    Returns a small report dict; raises ConfigError on the first hard problem.
    """
    registry = load_field_registry()
    users = load_users()
    policies = load_policies()

    known_fields = set(f.name for f in registry.all_fields())

    # Every field referenced by a policy must exist in the registry.
    for role, rule in policies["role_rules"].items():
        for key in ("forbidden_fields", "allowed_fields", "broad_query_forbidden_fields"):
            for fname in rule.get(key, []):
                if fname not in known_fields:
                    raise ConfigError(
                        f"Policy for {role} references unknown field '{fname}'"
                    )

    # Every user role must have a matching set of policy rules and purposes.
    for uid, profile in users.items():
        role_name = profile.role.value
        if role_name not in policies["role_rules"]:
            raise ConfigError(f"No role rule for {role_name} (user {uid})")
        if role_name not in policies["role_purposes"]:
            raise ConfigError(f"No purpose list for {role_name} (user {uid})")

    return {
        "fields": len(known_fields),
        "users": len(users),
        "roles": len(policies["role_rules"]),
        "policy_ids": len(policies["policy_ids"]),
    }


if __name__ == "__main__":
    report = validate_all()
    print("Configuration OK:", json.dumps(report, indent=2))
