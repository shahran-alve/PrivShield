"""PrivShield-Full on the hospital drill-down and roll-up sessions (Table IV).

Regenerates the 200 drill-down and 200 roll-up hospital sessions of
rank_benign.py (same generator and seed) and sends every query through the
complete controller in src/: policy, threshold, history features, learned risk
model and perturbation. Each session uses a fresh researcher account. A
refusal counts against a session only when the refused query itself matches at
least k patients.

The rank auditor refuses no query in these sessions (rank_benign.py), so the
session count is the same with it added.

Output: results/full_benign_sessions.csv and results/full_benign_summary.json

Run:  python experiments/full_benign.py     (needs config/ and models/)
"""
import json
import os
import warnings
from collections import Counter

import pandas as pd

from common import Schema, load_hospital, result_path
from rank_benign import sessions
from src import audit_logger
from src import decision_engine as de
from src.contracts import Action, QueryRequest, Role, UserProfile
from src.datasets import PATIENT_CSV

warnings.filterwarnings("ignore")
K = 10
REFUSED = {Action.BLOCK_POLICY, Action.BLOCK_HIGH_RISK}


if __name__ == "__main__":
    raw = pd.read_csv(PATIENT_CSV)
    schema = Schema(load_hospital(), K)
    if os.path.exists(audit_logger.AUDIT_LOG):
        os.remove(audit_logger.AUDIT_LOG)

    rows = []
    actions = Counter()
    number = 0
    for drill, roll in sessions(schema, 200, seed=1):
        for kind, session in (("drill", drill), ("roll", roll)):
            user = UserProfile(user_id=f"benign_{number:04d}", display_name="Benign",
                               role=Role.RESEARCHER, department="Clinical Research",
                               clearance_level=4, purpose="research")
            row = dict(session=number, kind=kind, legit_queries=0, refused_legit=0,
                       exact=0, noised=0, masked=0)
            for f in session:
                legit = schema.count(f) >= K
                filters = {("diagnosis_group" if col == "partition" else col): str(v)
                           for col, v in f.items()}
                request = QueryRequest(aggregate="COUNT", target_field=None, group_by=None,
                                       filters=filters)
                try:
                    action = de.decide(user, request, raw, write_audit=True).action
                except Exception:
                    action = None
                actions[(action.value if action else "ERROR", legit)] += 1
                if not legit:
                    continue
                row["legit_queries"] += 1
                if action is None or action in REFUSED:
                    row["refused_legit"] += 1
                elif action == Action.ALLOW:
                    row["exact"] += 1
                elif action == Action.ALLOW_WITH_NOISE:
                    row["noised"] += 1
                elif action == Action.MASK_SMALL_GROUPS:
                    row["masked"] += 1
            rows.append(row)
            number += 1

    df = pd.DataFrame(rows)
    df.to_csv(result_path("full_benign_sessions.csv"), index=False)
    clean = df.refused_legit == 0
    summary = dict(sessions=len(df), no_refusal=int(clean.sum()),
                   no_refusal_drill=int((clean & (df.kind == "drill")).sum()),
                   no_refusal_roll=int((clean & (df.kind == "roll")).sum()),
                   legit_queries=int(df.legit_queries.sum()),
                   refused_legit=int(df.refused_legit.sum()),
                   exact=int(df.exact.sum()), noised=int(df.noised.sum()),
                   masked=int(df.masked.sum()))
    with open(result_path("full_benign_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print("actions (action, query has at least k patients):", dict(actions))
