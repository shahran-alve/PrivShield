"""Why protecting empty cells changes one NHANES result in Table VI (133 to 135).

The rank auditor is online: it refuses the first query that would complete a
protected cell. Adding the empty cells to the protected set changes which
query that is, and therefore which later queries are released. This script
lists the at-risk groups for which the feasibility attacker of ip_attacker.py
reaches a different verdict on forced homogeneity under the two protected sets.

Run:  python experiments/empty_cell_case.py
"""
from collections import defaultdict
from itertools import combinations

import numpy as np

import ip_attacker as ip
from common import load_nhanes, local_queries, prep

K = 10


if __name__ == "__main__":
    data = load_nhanes()
    loc, parts, outs, _ = prep(data, K)
    P, O = len(parts), len(outs)
    n = P * O
    sizes = defaultdict(int)
    for bk, cell in loc.items():
        for (p, _), c in cell.items():
            sizes[(bk, p)] += c

    for (bk, pp), size in sorted(sizes.items()):
        if not 0 < size < K:
            continue
        cell = loc[bk]
        counts = [cell.get((p, o), 0) for p in parts for o in outs]
        released = [(q["vec"], q["cnt"]) for q in local_queries(cell, parts, outs, K)]
        refused = []
        for i in range(P):
            v = [0] * n
            for j in range(O):
                v[i * O + j] = 1
            if sum(counts[i * O + j] for j in range(O)) < K:
                refused.append(v)
        for j in range(O):
            v = [0] * n
            for i in range(P):
                v[i * O + j] = 1
            if sum(counts[i * O + j] for i in range(P)) < K:
                refused.append(v)
        for c in range(n):
            if counts[c] < K:
                v = [0] * n
                v[c] = 1
                refused.append(v)
        if sum(counts) < K:
            refused.append([1] * n)
        bounds = [K - 1] * len(refused)

        def forced(allowed):
            row = parts.index(pp) * O
            for o1, o2 in combinations(range(O), 2):
                e1 = [0] * n
                e1[row + o1] = 1
                e2 = [0] * n
                e2[row + o2] = 1
                if ip.feasible(n, [v for v, _ in allowed], [x for _, x in allowed], refused, bounds,
                               [(e1, 1, np.inf), (e2, 1, np.inf)]):
                    return False
            return True

        nonempty = [c for c in range(n) if 0 < counts[c] < K]
        with_empty = [c for c in range(n) if counts[c] < K]
        a = ip.rank_auditor_allowed(released, nonempty)
        b = ip.rank_auditor_allowed(released, with_empty)
        fa, fb = forced(a), forced(b)
        if fa != fb:
            print(f"group {bk}, partition {pp}: {size} people; forced homogeneous "
                  f"{fa} (cells 1..k-1 protected) vs {fb} (empty cells protected too); "
                  f"released {len(a)} vs {len(b)} of {len(released)} queries")
