"""Answer and exact rates on the population workload (Table IV).

The workload is 120 broad grouped COUNT queries by one analyst account: group
by one of age band, race, gender or admission type, with no filter or one
filter on another of these attributes. It runs on the one-row-per-patient
hospital table.

  patient-k        the whole result is refused when its smallest group has
                   fewer than k patients
  PrivShield-Lite  patient-k plus the rank auditor, with state kept across the
                   120 queries. The auditor works on the grid of the four
                   workload attributes and protects every cell with a count
                   from 1 to k-1.
  PrivShield-Full  the complete controller in src/ (policy, threshold, history
                   features, learned risk model, perturbation), using the
                   model in models/

A query is answered when the defense returns a result, and exact when that
result is unperturbed and complete. The output gives counts out of 120.

Output: results/population_workload.csv

Run:  python experiments/population_workload.py
"""
import copy
import os
import warnings
from itertools import product

import pandas as pd

from common import result_path
from rank_auditor import P61, RankAuditor
from src import audit_logger, workload
from src.config import K_THRESHOLD, load_field_registry
from src.contracts import QueryRequest
from src.datasets import PATIENT_CSV

warnings.filterwarnings("ignore")
FIELDS = workload.BROAD_FIELDS
N_QUERIES = 120


def lite(data, registry):
    """Patient-k plus the rank auditor on the request stream."""
    d = data[FIELDS].astype(str)
    levels = [sorted(d[f].unique()) for f in FIELDS]
    keys = list(product(*levels))
    index = {key: i for i, key in enumerate(keys)}
    counts = {tuple(key): int(v) for key, v in d.groupby(FIELDS, observed=True).size().items()}
    protected = [index[key] for key, c in counts.items() if 0 < c < K_THRESHOLD]
    auditor = RankAuditor(protected, P61)
    answered = 0
    for gb, filters in workload.requests(data, N_QUERIES):
        request = QueryRequest(aggregate="COUNT", target_field=None, group_by=gb, filters=filters)
        result, _, _ = workload.defense_k_only(None, request, data, registry)
        if result is None:
            continue
        fixed = {FIELDS.index(f): v for f, v in filters.items()}
        g = FIELDS.index(gb)
        groups = {}
        for key in keys:
            if all(key[i] == v for i, v in fixed.items()):
                groups.setdefault(key[g], []).append(index[key])
        returned = [cells for cells in groups.values() if sum(counts.get(keys[i], 0) for i in cells) > 0]
        # a grouped result is one equation per returned group; release all of them or none
        trial = copy.deepcopy(auditor)
        if all(trial.ask(cells) for cells in returned):
            auditor = trial
            answered += 1
    return answered


if __name__ == "__main__":
    data = pd.read_csv(PATIENT_CSV)
    registry = load_field_registry()
    rows = []
    for name, defense in [("patient-k", workload.defense_k_only),
                          ("PrivShield-Full", workload.defense_full)]:
        if os.path.exists(audit_logger.AUDIT_LOG):
            os.remove(audit_logger.AUDIT_LOG)
        r = workload.run(data, registry, defense, n=N_QUERIES)
        rows.append(dict(defense=name, queries=N_QUERIES, answered=r["answered"], exact=r["exact"]))
    n = lite(data, registry)
    rows.insert(1, dict(defense="PrivShield-Lite", queries=N_QUERIES, answered=n, exact=n))
    df = pd.DataFrame(rows)
    df.to_csv(result_path("population_workload.csv"), index=False)
    print(df.to_string(index=False))
