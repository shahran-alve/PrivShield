"""Latency of the rank auditor on a local grid as the history grows (Section VI, Cost).

A local grid has d = partition values x outcomes cells (d <= 45 in the three
datasets), so the released span has at most d dimensions however many queries
have been answered. The benchmark uses a 15 x 3 grid with eight protected
cells. Each request is drawn uniformly from the queries of the grid (total,
outcome totals, partition totals, single cells other than the protected ones)
and passed to the auditor, which uses exact rational arithmetic. The script
reports the median time per request for histories of 10^3, 10^4 and 10^5
requests.

Output: results/auditor_latency.csv

Run:  python experiments/auditor_latency.py
"""
import time

import numpy as np
import pandas as pd

from common import result_path
from rank_auditor import RankAuditor

P, O = 15, 3
N_PROTECTED = 8


def bench(history, seed=0):
    rng = np.random.default_rng(seed)
    d = P * O
    protected = sorted(int(x) for x in rng.choice(d, size=N_PROTECTED, replace=False))
    queries = [list(range(d))]
    queries += [[p * O + o for p in range(P)] for o in range(O)]
    queries += [[p * O + o for o in range(O)] for p in range(P)]
    queries += [[c] for c in range(d) if c not in protected]
    auditor = RankAuditor(protected)
    picks = rng.integers(0, len(queries), size=history)
    times = np.empty(history)
    for i, q in enumerate(picks):
        t0 = time.perf_counter()
        auditor.ask(queries[q])
        times[i] = time.perf_counter() - t0
    assert not auditor.computable()
    return dict(history=history, median_us=round(float(np.median(times)) * 1e6, 1),
                p95_us=round(float(np.percentile(times, 95)) * 1e6, 1),
                basis_size=len(auditor.basis), refused=auditor.refused)


if __name__ == "__main__":
    df = pd.DataFrame([bench(n) for n in (1000, 10000, 100000)])
    df.to_csv(result_path("auditor_latency.csv"), index=False)
    print(df.to_string(index=False))
