"""Which defenses stop the k-clean certificates (Table I).

Every at-risk target with a k-clean certificate at k = 10 is replayed against
each defense. The attacker chooses the order of the certificate's queries, so a
defense stops a certificate only if it refuses at least one query in every
order tried: all permutations for up to six queries, otherwise marginals
first, cells first and 60 random orders.

Defenses
  patient_k        refuse counts below k (no memory, so it never refuses here)
  overlap_one_dir  also refuse a query whose filter set strictly contains an
                   earlier one
  overlap_sym      also refuse when it is strictly contained in an earlier one
  history_only     no threshold; refuse a query that resembles three earlier
                   ones (Jaccard >= 2/3 on filter sets) and narrows one of them
  rank             rank auditor without the threshold
  lite             patient-k plus the rank auditor (PrivShield-Lite)

Output: results/kclean_matrix.csv

Run:  python experiments/defense_matrix.py
"""
import itertools
import random

import pandas as pd

from common import (cells_of, certificate, exposed, load_adult, load_hospital, load_nhanes,
                    prep, protected_cells, result_path, target_of)
from rank_auditor import RankAuditor

K = 10
DEFENSES = ["patient_k", "overlap_one_dir", "overlap_sym", "history_only", "rank", "lite"]


def orders(used, rng):
    if len(used) <= 6:
        return [list(p) for p in itertools.permutations(used)]
    marg = [q for q in used if q["kind"] == "m"]
    cell = [q for q in used if q["kind"] == "c"]
    cands = [marg + cell, cell + marg]
    for _ in range(60):
        shuffled = used[:]
        rng.shuffle(shuffled)
        cands.append(shuffled)
    return cands


def all_released(seq, defense, protected):
    """True if the defense answers every query of the sequence."""
    prev = []
    aud = RankAuditor(protected) if defense in ("rank", "lite") else None
    for q in seq:
        fs = q["fs"]
        if defense != "rank" and defense != "history_only" and q["cnt"] < K:
            return False
        if defense == "overlap_one_dir" and any(pf < fs for pf in prev):
            return False
        if defense == "overlap_sym" and any(pf < fs or fs < pf for pf in prev):
            return False
        if defense == "history_only":
            similar = [pf for pf in prev if len(pf & fs) / (len(pf | fs) or 1) >= 2 / 3]
            if len(similar) >= 3 and any(pf < fs for pf in similar):
                return False
        if aud is not None and not aud.ask(cells_of(q)):
            return False
        prev.append(fs)
    return True


def evaluate(name, data):
    loc, parts, outs, pool = prep(data, K)
    certs = []
    for _, t in pool.iterrows():
        bk, tp, ty = target_of(t)
        cell = loc.get(bk, {})
        used = certificate(cell, parts, outs, tp, ty, K)
        if used is not None:
            certs.append((used, protected_cells(cell, parts, outs, K), exposed(cell, tp, outs)))
    rows = []
    for d in DEFENSES:
        rng = random.Random(42)
        succeed = disclosed = 0
        for used, protected, disc in certs:
            if any(all_released(seq, d, protected) for seq in orders(used, rng)):
                succeed += 1
                disclosed += int(disc)
        rows.append(dict(dataset=name, defense=d, certificates=len(certs), succeed=succeed,
                         prevented=len(certs) - succeed, disclosures=disclosed))
    return rows


if __name__ == "__main__":
    rows = []
    for name, loader in [("NHANES", load_nhanes), ("Hospital", load_hospital), ("Adult", load_adult)]:
        rows += evaluate(name, loader())
    df = pd.DataFrame(rows)
    df.to_csv(result_path("kclean_matrix.csv"), index=False)
    print(df.to_string(index=False))
    total = df.groupby("defense", sort=False)[["certificates", "prevented"]].sum()
    print("\nAll datasets:")
    print(total.to_string())
