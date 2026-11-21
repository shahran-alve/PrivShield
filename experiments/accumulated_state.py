"""Rank auditor with state kept per account (Table III, last two columns).

The per-session numbers of hard_benign.py start every session with an empty
auditor. A deployment would keep the state with the account, so the released
span grows over many sessions. Here one account runs S consecutive sessions of
the same workload generator with the auditor state and the query history
carried over, for S = 1, 10, 50 and 150. The total is 150 sessions per dataset
in every setting, so the rows are comparable.

Output: results/accumulated_state_k{k}.csv with the share of legitimate
queries (count >= k) refused by the rank auditor and by symmetric containment.

Run:  python experiments/accumulated_state.py [k]
"""
import sys

import numpy as np
import pandas as pd

from common import Schema, load_adult, load_hospital, load_nhanes, result_path
from hard_benign import score, session_queries, small_cohort_queries
from rank_auditor import P61, RankAuditor

PLAN = [(1, 150), (10, 15), (50, 3), (150, 1)]   # (sessions per account, accounts)


def run_accounts(name, schema, per_account, accounts, seed=7):
    rng = np.random.default_rng(seed)
    legit = rank_fp = sym_fp = 0
    for _ in range(accounts):
        auditor = RankAuditor(schema.protected, P61)
        prev = []
        for _ in range(per_account):
            Q = session_queries(rng, schema.levels) + small_cohort_queries(rng, schema)
            n, a, b = score(schema, Q, auditor, prev)
            legit += n
            rank_fp += a
            sym_fp += b
    return dict(dataset=name, sessions_per_account=per_account, accounts=accounts,
                legit_queries=legit, rank_fp=rank_fp, rank_fp_rate=round(rank_fp / legit, 4),
                sym_fp_rate=round(sym_fp / legit, 4))


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows = []
    for name, loader in [("Hospital", load_hospital), ("NHANES", load_nhanes), ("Adult", load_adult)]:
        schema = Schema(loader(), K)
        for per_account, accounts in PLAN:
            rows.append(run_accounts(name, schema, per_account, accounts))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(result_path(f"accumulated_state_k{K}.csv"), index=False)
