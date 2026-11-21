"""The naive tracker against each defense (Fig. 2 and the first result of Section VI).

The tracker of Algorithm 1 puts the outcome in the filter and issues ungrouped
counts only. For each outcome y it asks for the count of the target's
(age_band, race, gender) group with outcome y, then for the count of every
other partition value with outcome y, and subtracts. It fails as soon as one
of its queries is refused.

Defenses
  undefended         every query answered; an empty count returns 0
  empty-suppression  only empty counts are refused
  patient-k          counts below k are refused, empty counts included
  sym. containment   patient-k, and a query is refused when its filter set
                     strictly contains or is strictly contained in an earlier one
  PrivShield-Lite    patient-k plus the rank auditor

For every at-risk target the script records whether the target's cell was
reconstructed and whether that discloses the target (singleton or one-outcome
group). `people` counts the distinct individuals in reconstructed cells of
that kind.

Output: results/ladder.csv

Run:  python experiments/tracker_ladder.py
"""
import numpy as np
import pandas as pd

from common import (load_adult, load_hospital, load_nhanes, prep, protected_cells,
                    result_path, target_of)
from rank_auditor import RankAuditor

K = 10
DEFENSES = ["undefended", "empty-suppression", "patient-k", "sym. containment", "PrivShield-Lite"]


def tracker(cell, tp, parts, outs, defense):
    """Run the tracker for one target. Returns (reconstruction or None, queries issued)."""
    O = len(outs)
    pidx = {p: i for i, p in enumerate(parts)}
    oidx = {o: j for j, o in enumerate(outs)}
    prev = []
    aud = RankAuditor(protected_cells(cell, parts, outs, K)) if defense == "PrivShield-Lite" else None
    issued = 0

    def answer(n, filters, cells):
        nonlocal issued
        issued += 1
        fs = frozenset(filters)
        refuse = False
        if defense == "empty-suppression":
            refuse = n == 0
        elif defense != "undefended":
            refuse = n < K
        if not refuse and defense == "sym. containment":
            refuse = any(pf < fs or fs < pf for pf in prev)
        if not refuse and aud is not None:
            refuse = not aud.ask(cells)
        prev.append(fs)
        return None if refuse else n

    recon = {}
    for y in outs:
        column = [pidx[p] * O + oidx[y] for p in parts]
        acc = answer(sum(cell.get((p, y), 0) for p in parts), ("a", "r", "g", "o=" + y), column)
        if acc is None:
            return None, issued
        for d in parts:
            if d == tp:
                continue
            c = answer(cell.get((d, y), 0), ("a", "r", "g", "p=" + d, "o=" + y),
                       [pidx[d] * O + oidx[y]])
            if c is None:
                return None, issued
            acc -= c
        recon[y] = acc
    return recon, issued


def run(name, data):
    loc, parts, outs, pool = prep(data, K)
    rows = []
    for defense in DEFENSES:
        recon_n = disclosed = singleton = homog = 0
        exposed_cells = {}
        issued = []
        for _, t in pool.iterrows():
            bk, tp, _ = target_of(t)
            cell = loc.get(bk, {})
            truth = {o: cell.get((tp, o), 0) for o in outs}
            total = sum(truth.values())
            is_single = total == 1
            is_homog = max(truth.values()) == total
            singleton += int(is_single)
            homog += int(is_homog)
            recon, nq = tracker(cell, tp, parts, outs, defense)
            issued.append(nq)
            if recon is not None and recon == truth:
                recon_n += 1
                if is_single or is_homog:
                    disclosed += 1
                    exposed_cells[(bk, tp)] = total
        rows.append(dict(dataset=name, defense=defense, targets=len(pool), reconstructed=recon_n,
                         singleton=singleton, one_outcome=homog, disclosed=disclosed,
                         people=sum(exposed_cells.values()),
                         mean_queries=round(float(np.mean(issued)), 2)))
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    rows = []
    for name, loader in [("Hospital", load_hospital), ("Adult", load_adult), ("NHANES", load_nhanes)]:
        rows += run(name, loader())
    pd.DataFrame(rows).to_csv(result_path("ladder.csv"), index=False)
