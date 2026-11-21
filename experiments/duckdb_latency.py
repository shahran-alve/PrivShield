"""Latency of a query with and without the small-cell guard (Section VI, Cost).

Synthetic tables of 1M, 10M and 100M rows are generated inside DuckDB. The
unprotected query is a filtered COUNT. The protected query is one grouped pass
that returns the answer and the smallest cell the guard needs. Each is timed
20 times after a warm-up run, with DuckDB limited to 3 GB of memory and four
threads. Synthetic data are used for timing only.

Output: results/scale.csv (median and tail latency in milliseconds)

Run:  python experiments/duckdb_latency.py     (needs duckdb)
"""
import os
import time

import duckdb
import numpy as np
import pandas as pd

from common import RESULTS, result_path

COHORT = "ageb=30 AND race=1 AND gender=0 AND part=2"


def bench(n):
    db = os.path.join(RESULTS, f"scale_{n}.db")
    if os.path.exists(db):
        os.remove(db)
    con = duckdb.connect(db)
    con.execute("PRAGMA memory_limit='3GB'; PRAGMA threads=4")
    con.execute(f"""
      CREATE TABLE t AS
      SELECT (i%10)*10 AS ageb, i%6 AS race, i%2 AS gender, i%15 AS part, i%3 AS outcome
      FROM range({n}) tbl(i)""")

    def unprotected():
        con.execute(f"SELECT COUNT(*) FROM t WHERE {COHORT} AND outcome=0").fetchone()

    def protected():
        con.execute(f"SELECT outcome, COUNT(*) c FROM t WHERE {COHORT} GROUP BY outcome").fetchall()

    unprotected()
    protected()
    base, prot = [], []
    for _ in range(20):
        s = time.perf_counter()
        unprotected()
        base.append((time.perf_counter() - s) * 1000)
        s = time.perf_counter()
        protected()
        prot.append((time.perf_counter() - s) * 1000)
    con.close()
    os.remove(db)
    base, prot = np.array(base), np.array(prot)
    return dict(rows=n, base_p50=round(float(np.percentile(base, 50)), 2),
                prot_p50=round(float(np.percentile(prot, 50)), 2),
                prot_p95=round(float(np.percentile(prot, 95)), 2),
                prot_p99=round(float(np.percentile(prot, 99)), 2),
                overhead_ms=round(float(np.percentile(prot, 50) - np.percentile(base, 50)), 2),
                throughput_qps=round(float(1000 / np.mean(prot)), 1))


if __name__ == "__main__":
    rows = []
    for n in (1_000_000, 10_000_000, 100_000_000):
        rows.append(bench(n))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(result_path("scale.csv"), index=False)
