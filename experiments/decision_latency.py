"""End-to-end decision latency of the complete controller (Section IX).

For samples of 10,000, 25,000 and 50,000 hospital visits, 100 policy-allowed
queries are drawn and sent through the controller in src/ (validation, policy,
aggregation, history features, risk model, action). The time of each decision
is recorded after one warm-up call.

Output: results/decision_latency.csv (milliseconds)

Run:  python experiments/decision_latency.py     (needs config/, models/ and
      the samples written by python -m src.data_preprocessing)
"""
import datetime as dt
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import result_path
from src.config import DATA_DIR, GLOBAL_SEED, get_user, load_field_registry
from src.decision_engine import decide
from src.policy_engine import evaluate_policy
from src.training_data import _random_request
from src.utils import derive_seed

warnings.filterwarnings("ignore")
PER_SIZE = 100


if __name__ == "__main__":
    registry = load_field_registry()
    rng = np.random.default_rng(derive_seed(GLOBAL_SEED, "latency"))
    admin = get_user("admin_01")
    now = dt.datetime(2026, 1, 15, 12, 0, 0, tzinfo=dt.timezone.utc)
    rows = []
    for size in (10000, 25000, 50000):
        data = pd.read_csv(os.path.join(DATA_DIR, "processed", f"diabetes_{size}.csv"))
        cats = {f: sorted({str(x) for x in data[f].dropna().unique()}) for f in registry.filterable()}
        pool = []
        while len(pool) < PER_SIZE + 1:
            request = _random_request(rng, registry, cats, intent="mixed")
            if evaluate_policy(admin, request, registry).allowed:
                pool.append(request)
        decide(admin, pool[0], data, now_utc=now, write_audit=False)
        lat = []
        for request in pool[1:]:
            t0 = time.perf_counter()
            decide(admin, request, data, now_utc=now, write_audit=False)
            lat.append((time.perf_counter() - t0) * 1000)
        lat = np.array(lat)
        rows.append(dict(dataset_size=size, n_queries=len(lat), mean_ms=round(float(lat.mean()), 3),
                         median_ms=round(float(np.median(lat)), 3),
                         p95_ms=round(float(np.percentile(lat, 95)), 3),
                         max_ms=round(float(lat.max()), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(result_path("decision_latency.csv"), index=False)
    print(df.to_string(index=False))
