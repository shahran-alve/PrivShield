"""One range filter on an ordered attribute (Section VII-B).

The base query language has equality filters only, so every query of a local
grid is a cell, a row, a column or the total. Education in NHANES is ordered
(<9th, 9-11th, HS/GED, SomeCollege, CollegeGrad). This script adds one filter
of the form education <= L or education >= L, for a fixed diabetes outcome and
across all outcomes, allowed only when the count is at least k, and counts the
at-risk targets whose cell is then in the span of the answerable queries.

The rank auditor is then run on the extended query set of every grid that
holds an attackable target, with the range queries first and with them last.
A target is prevented when its cell is in the span of the released queries in
neither order (independent exact rank test).

Output: results/range_filter.csv

Run:  python experiments/range_filter.py
"""
import pandas as pd

from common import in_span, load_nhanes, local_queries, prep, protected_cells, result_path, target_of
from rank_auditor import RankAuditor

K = 10
EDU_ORDER = ["<9th", "9-11th", "HS/GED", "SomeCollege", "CollegeGrad"]


def range_vectors(cell, parts, outs):
    """0/1 vectors of the answerable prefix and suffix queries on education."""
    P, O = len(parts), len(outs)

    def n(i, o):
        return cell.get((parts[i], o), 0)

    def vec(rows, cols):
        v = [0] * (P * O)
        for i in rows:
            for j in cols:
                v[i * O + j] = 1
        return v

    V = []
    for cols in [[j] for j in range(O)] + [list(range(O))]:
        for L in range(P):
            for rows in (list(range(0, L + 1)), list(range(L, P))):
                if 1 < len(rows) < P and sum(n(i, outs[j]) for i in rows for j in cols) >= K:
                    V.append(vec(rows, cols))
    return V


if __name__ == "__main__":
    data = load_nhanes()
    loc, _, outs, pool = prep(data, K)
    parts = EDU_ORDER
    O = len(outs)
    pidx = {p: i for i, p in enumerate(parts)}
    oidx = {o: j for j, o in enumerate(outs)}

    grids = {}
    equality = extended = prevented_eq = prevented_ext = 0
    for _, t in pool.iterrows():
        bk, tp, ty = target_of(t)
        cell = loc.get(bk, {})
        if bk not in grids:
            eq = [q["vec"] for q in local_queries(cell, parts, outs, K)]
            rng = range_vectors(cell, parts, outs)
            released = []
            for order in (rng + eq, eq + rng):
                aud = RankAuditor(protected_cells(cell, parts, outs, K))
                released.append([v for v in order if aud.ask([i for i, x in enumerate(v) if x])])
            grids[bk] = (eq, eq + rng, released)
        eq, both, released = grids[bk]
        tgt = [0] * (len(parts) * O)
        tgt[pidx[tp] * O + oidx[ty]] = 1
        safe = not any(rel and in_span(rel, tgt) for rel in released)
        if eq and in_span(eq, tgt):
            equality += 1
            prevented_eq += int(safe)
        if both and in_span(both, tgt):
            extended += 1
            prevented_ext += int(safe)

    df = pd.DataFrame([
        dict(queries="equality only", attackable_targets=equality,
             rank_auditor_prevented=prevented_eq),
        dict(queries="equality + one range filter", attackable_targets=extended,
             rank_auditor_prevented=prevented_ext),
    ])
    df.to_csv(result_path("range_filter.csv"), index=False)
    print(f"at-risk targets: {len(pool)}")
    print(df.to_string(index=False))
