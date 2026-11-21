"""Drill-down and roll-up sessions: rank auditor against symmetric containment (Table II).

For each dataset the script draws 200 cohorts (age band, race, gender,
partition value). A drill-down session adds one filter at a time in the order
age band, gender, race, partition; the roll-up session asks the same four
queries in reverse. Sessions are not filtered to avoid small cells.

Each session is run against two auditors with fresh state. The rank auditor
works on the full attribute grid and protects every cell with a count from 1
to k-1. The symmetric auditor refuses a query whose filter set strictly
contains, or is strictly contained in, an earlier one. A refusal counts as a
false positive only when the refused query itself matches at least k people.

Output: results/benign_fp_compare.csv, the share of sessions with at least one
false positive per auditor and session type.

Run:  python experiments/rank_benign.py
"""
import numpy as np
import pandas as pd

from common import Schema, load_adult, load_hospital, load_nhanes, result_path
from rank_auditor import P61, RankAuditor

K = 10


def sessions(schema, n, seed):
    ages, races, genders, parts, _ = schema.levels
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = rng.choice(ages)
        r = rng.choice(races)
        g = rng.choice(genders)
        p = rng.choice(parts)
        drill = [{"age_band": a}, {"age_band": a, "gender": g},
                 {"age_band": a, "gender": g, "race": r},
                 {"age_band": a, "gender": g, "race": r, "partition": p}]
        yield drill, list(reversed(drill))


def false_positives(schema, session):
    """(rank auditor, symmetric auditor) false positives in one session."""
    aud = RankAuditor(schema.protected, P61)
    prev = []
    fp_rank = fp_sym = 0
    for f in session:
        legit = schema.count(f) >= K
        if legit and not aud.ask(schema.cells(f)):
            fp_rank += 1
        fs = frozenset(f.items())
        if legit and any(pf < fs or fs < pf for pf in prev):
            fp_sym += 1
        prev.append(fs)
    return fp_rank, fp_sym


def run(name, data, n_sessions=200, seed=1):
    schema = Schema(data, K)
    d_rank = d_sym = r_rank = r_sym = 0
    for drill, roll in sessions(schema, n_sessions, seed):
        a, b = false_positives(schema, drill)
        d_rank += int(a > 0)
        d_sym += int(b > 0)
        a, b = false_positives(schema, roll)
        r_rank += int(a > 0)
        r_sym += int(b > 0)
    return dict(dataset=name, sessions=n_sessions,
                drill_fp_sym=round(d_sym / n_sessions, 3), drill_fp_rank=round(d_rank / n_sessions, 3),
                roll_fp_sym=round(r_sym / n_sessions, 3), roll_fp_rank=round(r_rank / n_sessions, 3))


if __name__ == "__main__":
    rows = [run(n, l()) for n, l in [("Hospital", load_hospital), ("Adult", load_adult), ("NHANES", load_nhanes)]]
    df = pd.DataFrame(rows)
    df.to_csv(result_path("benign_fp_compare.csv"), index=False)
    print(df.to_string(index=False))
