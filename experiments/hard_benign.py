"""A harder benign workload for the rank auditor (Table III, per-session columns).

Drill-down sessions never fix the outcome, so they cannot place an outcome cell
in the released span. This workload does: each session mixes outcome-stratified
rate queries, visits the same cohort through several filter orders, and ends
with the outcome breakdown of a small cohort that is still above the threshold.
All of these are ordinary analyst questions, and the last kind can complete a
protected cell.

Only queries whose own count is at least k are scored. The rank auditor works
on the full attribute grid, starts each session with empty state, and protects
every cell with a count from 1 to k-1. The symmetric-containment auditor is
run on the same sessions for comparison.

Output: results/hard_benign_k{k}.csv

Run:  python experiments/hard_benign.py [k]
"""
import sys

import numpy as np
import pandas as pd

from common import Schema, load_adult, load_hospital, load_nhanes, result_path
from rank_auditor import P61, RankAuditor


def session_queries(rng, levels):
    ages, races, genders, parts, outs = levels
    a = rng.choice(ages)
    g = rng.choice(genders)
    r = rng.choice(races)
    ps = list(rng.choice(parts, size=min(2, len(parts)), replace=False))
    p0, p1 = ps[0], ps[-1]
    Q = [{"age_band": a}, {"age_band": a, "partition": p0}]
    Q += [{"age_band": a, "partition": p0, "outcome": o} for o in outs]
    Q.append({"age_band": a, "gender": g})
    Q += [{"age_band": a, "gender": g, "outcome": o} for o in outs]
    Q.append({"age_band": a, "gender": g, "partition": p0})
    Q.append({"age_band": a, "gender": g, "partition": p1})
    Q.append({"age_band": a, "gender": g, "race": r})
    Q.append({"age_band": a, "gender": g, "race": r, "partition": p0})
    # the same cohort again, approached from the partition side
    Q.append({"partition": p0})
    Q += [{"partition": p0, "outcome": o} for o in outs]
    Q.append({"partition": p0, "gender": g})
    Q.append({"age_band": a, "race": r})
    Q += [{"age_band": a, "race": r, "outcome": o} for o in outs]
    return Q


def small_cohort_queries(rng, schema):
    """Outcome breakdown of a fully specified cohort with k to 4k-1 members.

    Such a cohort often has one outcome cell below k, and then the cohort total
    and the other outcome cells determine it.
    """
    ages, races, genders, parts, outs = schema.levels
    k = schema.k
    for _ in range(20):
        a = rng.choice(ages)
        r = rng.choice(races)
        g = rng.choice(genders)
        p = rng.choice(parts)
        cohort = {"age_band": a, "race": r, "gender": g, "partition": p}
        if k <= schema.count(cohort) < 4 * k:
            return [cohort] + [dict(cohort, outcome=o) for o in outs]
    return []


def score(schema, queries, auditor, prev):
    """Run queries through both auditors. Returns (legitimate, rank refusals, symmetric refusals)."""
    legit = rank_fp = sym_fp = 0
    for f in queries:
        if schema.count(f) < schema.k:
            continue
        legit += 1
        if not auditor.ask(schema.cells(f)):
            rank_fp += 1
        fs = frozenset(f.items())
        if any(pf < fs or fs < pf for pf in prev):
            sym_fp += 1
        prev.append(fs)
    return legit, rank_fp, sym_fp


def run(name, data, k, n_sessions=150, seed=1):
    schema = Schema(data, k)
    rng = np.random.default_rng(seed)
    legit = rank_fp = sym_fp = sess_rank = sess_sym = 0
    for _ in range(n_sessions):
        Q = session_queries(rng, schema.levels) + small_cohort_queries(rng, schema)
        n, a, b = score(schema, Q, RankAuditor(schema.protected, P61), [])
        legit += n
        rank_fp += a
        sym_fp += b
        sess_rank += int(a > 0)
        sess_sym += int(b > 0)
    return dict(dataset=name, sessions=n_sessions, legit_queries=legit,
                rank_fp_queries=rank_fp, rank_fp_rate=round(rank_fp / legit, 4),
                sym_fp_queries=sym_fp, sym_fp_rate=round(sym_fp / legit, 4),
                sess_rank_blocked=sess_rank, sess_sym_blocked=sess_sym)


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows = [run(n, l(), K) for n, l in
            [("Hospital", load_hospital), ("NHANES", load_nhanes), ("Adult", load_adult)]]
    df = pd.DataFrame(rows)
    df.to_csv(result_path(f"hard_benign_k{K}.csv"), index=False)
    print(df.to_string(index=False))
