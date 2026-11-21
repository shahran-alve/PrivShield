"""Local k-clean search and rank auditor at k = 5, 10 and 20 (Table V, "Local").

For each threshold the script counts the at-risk targets, the targets with a
k-clean certificate, and those whose group is a singleton or has one outcome.
It then runs the rank auditor on every local grid that contains an attackable
target. The auditor sees all answerable queries of the grid twice, once with
the marginals first and once with the single cells first, and protects every
cell with a count from 1 to k-1. A target counts as prevented when, in both
orders, its cell is not in the span of the queries the auditor released. That
last test uses an independent exact rank computation, not the auditor's state.

Output: results/ksweep.csv

Run:  python experiments/kclean_ksweep.py
"""
import pandas as pd

from common import (cells_of, certificate, exposed, in_span, load_adult, load_hospital,
                    load_nhanes, local_queries, prep, protected_cells, result_path, target_of)
from rank_auditor import RankAuditor


def released_sets(cell, parts, outs, k):
    """Queries the auditor releases, for the marginal-first and the cell-first order."""
    Q = local_queries(cell, parts, outs, k)
    prot = protected_cells(cell, parts, outs, k)
    out = []
    for order in (Q, list(reversed(Q))):
        aud = RankAuditor(prot)
        out.append([q["vec"] for q in order if aud.ask(cells_of(q))])
        assert not aud.computable()
    return out


def sweep(data, k):
    loc, parts, outs, pool = prep(data, k)
    O = len(outs)
    pidx = {p: i for i, p in enumerate(parts)}
    oidx = {o: j for j, o in enumerate(outs)}
    attack = disclosed = prevented = 0
    has_cert = {}
    audits = {}
    for _, t in pool.iterrows():
        bk, tp, ty = target_of(t)
        key = (bk, tp, ty)
        if key not in has_cert:
            has_cert[key] = certificate(loc.get(bk, {}), parts, outs, tp, ty, k) is not None
        if not has_cert[key]:
            continue
        attack += 1
        disclosed += int(exposed(loc[bk], tp, outs))
        if bk not in audits:
            audits[bk] = released_sets(loc[bk], parts, outs, k)
        tgt = [0] * (len(parts) * O)
        tgt[pidx[tp] * O + oidx[ty]] = 1
        if not any(rel and in_span(rel, tgt) for rel in audits[bk]):
            prevented += 1
    return dict(at_risk=len(pool), bypass=attack, disclosed=disclosed, auditor_prevented=prevented)


if __name__ == "__main__":
    rows = []
    for name, loader in [("NHANES", load_nhanes), ("Hospital", load_hospital), ("Adult", load_adult)]:
        data = loader()
        for k in (5, 10, 20):
            r = dict(dataset=name, k=k, **sweep(data, k))
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(result_path("ksweep.csv"), index=False)
