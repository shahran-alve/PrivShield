"""Feasibility (integer-programming) attacker on the local grid (Table VI).

Counts are non-negative integers and a refused count is at most k-1. For every
at-risk target the attacker writes the released answers of the target's local
grid as equalities, the refused queries as upper bounds of k-1 and the cells
as non-negative integers, and solves two integer programs for the smallest and
largest feasible value of the target's cell.

  pinned          the feasible range is a single value
  not in span     pinned, but not an exact linear reconstruction
  homog. forced   no feasible assignment gives the target's group two
                  different outcomes, so every member's outcome is known

Three released sets are compared: patient-k alone (every query with a count of
at least k), patient-k plus the rank auditor protecting cells with a count
from 1 to k-1, and the same auditor protecting empty cells as well. The user
cannot tell an auditor refusal from a threshold refusal, so the attacker is
given the k-1 bound on every query whose true count is below k. This favours
the attacker.

Output: results/ip_attacker_k{k}.csv (counts are individuals)

Run:  python experiments/ip_attacker.py [k]     (needs scipy >= 1.9)
"""
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from common import in_span, load_adult, load_hospital, load_nhanes, result_path
from rank_auditor import RankAuditor

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def rank_auditor_allowed(queries, protected):
    """The (vector, value) queries the rank auditor releases, in the order given."""
    aud = RankAuditor(protected)
    return [(vec, val) for vec, val in queries if aud.ask([i for i, x in enumerate(vec) if x])]


def milp_range(n, eq_rows, eq_vals, ub_rows, ub_vals, tidx, extra=None):
    A = []; lb = []; ub = []
    for r, v in zip(eq_rows, eq_vals): A.append(r); lb.append(v); ub.append(v)
    for r, v in zip(ub_rows, ub_vals): A.append(r); lb.append(0); ub.append(v)
    if extra:
        for r, l, u in extra: A.append(r); lb.append(l); ub.append(u)
    cons = LinearConstraint(np.array(A, dtype=float), np.array(lb, float), np.array(ub, float))
    bnds = Bounds(np.zeros(n), np.full(n, np.inf)); integ = np.ones(n)
    c = np.zeros(n); c[tidx] = 1.0
    lo = milp(c, constraints=cons, integrality=integ, bounds=bnds)
    hi = milp(-c, constraints=cons, integrality=integ, bounds=bnds)
    if not (lo.success and hi.success): return None
    return int(round(lo.fun)), int(round(-hi.fun))

def feasible(n, eq_rows, eq_vals, ub_rows, ub_vals, extra):
    A = []; lb = []; ub = []
    for r, v in zip(eq_rows, eq_vals): A.append(r); lb.append(v); ub.append(v)
    for r, v in zip(ub_rows, ub_vals): A.append(r); lb.append(0); ub.append(v)
    for r, l, u in extra: A.append(r); lb.append(l); ub.append(u)
    cons = LinearConstraint(np.array(A, float), np.array(lb, float), np.array(ub, float))
    res = milp(np.zeros(n), constraints=cons, integrality=np.ones(n),
               bounds=Bounds(np.zeros(n), np.full(n, np.inf)))
    return res.success

def run(name, data):
    for c in ["age_band","race","gender","partition","outcome"]: data[c] = data[c].astype(str)
    parts = sorted(data.partition.unique()); outs = sorted(data.outcome.unique())
    P, O = len(parts), len(outs); n = P * O
    pidx = {p:i for i,p in enumerate(parts)}; oidx = {o:j for j,o in enumerate(outs)}
    cell = lambda p,o: pidx[p]*O + oidx[o]
    g = data.groupby(["age_band","race","gender","partition","outcome"]).size()
    loc = defaultdict(dict)
    for (a,r,gg,p,o),cnt in g.items(): loc[(a,r,gg)][(p,o)] = int(cnt)
    grp = data.groupby(["age_band","race","gender","partition"]).size()

    stats = dict(dataset=name, k=K, at_risk=0, linear_bypass=0,
                 ip_pinned_patientk=0, ip_pinned_not_in_span=0, homog_forced_patientk=0,
                 ip_pinned_rank_auditor=0, homog_forced_rank_auditor=0,
                 ip_pinned_auditorB=0, homog_forced_auditorB=0,
                 released_total=0, refused_A=0, refused_B=0,
                 groups_forced_patientk=0, groups_forced_one_outcome_in_data=0)
    seen_base = {}
    for (a,r,gg,pp), gtot in grp.items():
        if not (0 < gtot < K): continue
        bk = (a,r,gg); cells = loc[bk]
        cnt = lambda p,o: cells.get((p,o), 0)
        # every query of the local grid with its true count
        Q = []
        tot = sum(cells.values()); v = [1]*n; Q.append((v, tot))
        for o in outs:
            v = [0]*n
            for p in parts: v[cell(p,o)] = 1
            Q.append((v, sum(cnt(p,o) for p in parts)))
        for p in parts:
            v = [0]*n
            for o in outs: v[cell(p,o)] = 1
            Q.append((v, sum(cnt(p,o) for o in outs)))
        for p in parts:
            for o in outs:
                v = [0]*n; v[cell(p,o)] = 1; Q.append((v, cnt(p,o)))
        released = [(v,val) for v,val in Q if val >= K]
        refused  = [(v,val) for v,val in Q if val <  K]
        # queries the rank auditor releases, computed once per grid
        if bk not in seen_base:
            protected = [cell(p,o) for p in parts for o in outs if 0 < cnt(p,o) < K]
            protectedZ = [cell(p,o) for p in parts for o in outs if cnt(p,o) < K]
            seen_base[bk] = (rank_auditor_allowed(released, protected),
                             rank_auditor_allowed(released, protectedZ), len(released))
        allowed, allowedZ, nrel = seen_base[bk]
        stats["released_total"] += nrel; stats["refused_A"] += nrel-len(allowed); stats["refused_B"] += nrel-len(allowedZ)
        eqAZ=[v for v,_ in allowedZ]; eqVZ=[val for _,val in allowedZ]
        eqA = [v for v,_ in released]; eqV = [val for _,val in released]
        ubA = [v for v,_ in refused];  ubV = [K-1]*len(refused)
        eqA2 = [v for v,_ in allowed]; eqV2 = [val for _,val in allowed]
        # the attacker keeps the k-1 bound on every query whose true count is below k
        ubA2 = ubA; ubV2 = ubV
        # forced homogeneous: no feasible assignment has two outcomes present in the target's group
        def homog_forced(eqA_, eqV_, ubA_, ubV_):
            for o1, o2 in combinations(outs, 2):
                e1 = [0]*n; e1[cell(pp,o1)] = 1; e2 = [0]*n; e2[cell(pp,o2)] = 1
                if feasible(n, eqA_, eqV_, ubA_, ubV_, [(e1,1,np.inf),(e2,1,np.inf)]): return False
            return True
        hf_pk = homog_forced(eqA, eqV, ubA, ubV)
        if hf_pk:
            stats["groups_forced_patientk"] += 1
            stats["groups_forced_one_outcome_in_data"] += int(sum(1 for o in outs if cnt(pp, o) > 0) == 1)
        hf_ra = homog_forced(eqA2, eqV2, ubA2, ubV2)
        hf_rb = homog_forced(eqAZ, eqVZ, ubA, ubV)
        for o in outs:
            w = cnt(pp,o)
            if w == 0: continue
            stats["at_risk"] += w
            tvec = [0]*n; tvec[cell(pp,o)] = 1
            lin = bool(released) and in_span([v for v,_ in released], tvec)
            if lin: stats["linear_bypass"] += w
            rng = milp_range(n, eqA, eqV, ubA, ubV, cell(pp,o))
            pinned = rng is not None and rng[0] == rng[1]
            if pinned:
                assert rng[0] == w, (rng, w)
                stats["ip_pinned_patientk"] += w
                if not lin: stats["ip_pinned_not_in_span"] += w
            if hf_pk: stats["homog_forced_patientk"] += w
            rng2 = milp_range(n, eqA2, eqV2, ubA2, ubV2, cell(pp,o))
            if rng2 is not None and rng2[0] == rng2[1]: stats["ip_pinned_rank_auditor"] += w
            if hf_ra: stats["homog_forced_rank_auditor"] += w
            rng3 = milp_range(n, eqAZ, eqVZ, ubA, ubV, cell(pp,o))
            if rng3 is not None and rng3[0] == rng3[1]: stats["ip_pinned_auditorB"] += w
            if hf_rb: stats["homog_forced_auditorB"] += w
    return stats

if __name__ == "__main__":
    rows = []
    for name, loader in [("Hospital", load_hospital), ("NHANES", load_nhanes), ("Adult", load_adult)]:
        rows.append(run(name, loader()))
    df = pd.DataFrame(rows)
    df.to_csv(result_path(f"ip_attacker_k{K}.csv"), index=False)
    print(df.to_string(index=False))
