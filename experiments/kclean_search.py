"""Local k-clean search at one threshold (default k = 10).

For every at-risk target the script fixes (age_band, race, gender), lists the
ungrouped COUNT queries of the local partition x outcome grid whose count is at
least k, and tests exactly, over the rationals, whether the indicator of the
target's cell lies in their span. If it does, the cell can be computed from
answers that all pass the threshold; if not, no exact linear attack exists in
this grid.

Output: results/kclean_search.csv with, per dataset, the at-risk targets, the
attackable targets, and those whose group is a singleton or has one outcome.

Run:  python experiments/kclean_search.py [k]
"""
import sys

import pandas as pd

from common import (certificate, exposed, load_adult, load_hospital, load_nhanes,
                    prep, result_path, target_of)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def search(data, k):
    loc, parts, outs, pool = prep(data, k)
    attack = disclosed = 0
    cache = {}
    for _, t in pool.iterrows():
        bk, tp, ty = target_of(t)
        key = (bk, tp, ty)
        if key not in cache:
            cache[key] = certificate(loc.get(bk, {}), parts, outs, tp, ty, k) is not None
        if cache[key]:
            attack += 1
            disclosed += int(exposed(loc[bk], tp, outs))
    return dict(at_risk=len(pool), kclean_attack=attack, disclosed=disclosed)


if __name__ == "__main__":
    rows = []
    for name, loader in [("Hospital", load_hospital), ("Adult", load_adult), ("NHANES", load_nhanes)]:
        rows.append(dict(dataset=name, k=K, **search(loader(), K)))
    df = pd.DataFrame(rows)
    df.to_csv(result_path("kclean_search.csv" if K == 10 else f"kclean_search_k{K}.csv"), index=False)
    print(df.to_string(index=False))
