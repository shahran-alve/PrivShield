"""Write every local k-clean certificate to a JSON file.

One entry per distinct protected cell that has a certificate. Each entry gives
the group, the protected cell and its true count, the queries used with the
count each returns and its rational coefficient, and a `verified` flag: the
weighted sum of the returned counts equals the true count and every query
used returns at least k. Arithmetic is exact (Python fractions).

`true_count` is the number of people in the cell, so the entries of a dataset
add up to its number of attackable at-risk targets.

Output: results/certificates/local_certificates_k{k}.json

Run:  python experiments/export_certificates.py [k]
"""
import json
import os
import sys

from common import (RESULTS, certificate, load_adult, load_hospital, load_nhanes, prep,
                    target_of)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def certificates(name, data):
    loc, parts, outs, pool = prep(data, K)
    seen = set()
    out = []
    for _, t in pool.iterrows():
        bk, tp, ty = target_of(t)
        if (bk, tp, ty) in seen:
            continue
        seen.add((bk, tp, ty))
        cell = loc.get(bk, {})
        used = certificate(cell, parts, outs, tp, ty, K)
        if used is None:
            continue
        true = cell.get((tp, ty), 0)
        recon = sum(q["coef"] * q["cnt"] for q in used)
        all_ge_k = all(q["cnt"] >= K for q in used)
        out.append({
            "dataset": name, "k": K,
            "group": {"age_band": bk[0], "race": bk[1], "gender": bk[2]},
            "protected_cell": {"partition": tp, "outcome": ty},
            "true_count": int(true),
            "reconstructed_count": int(recon) if recon == int(recon) else str(recon),
            "all_queries_at_least_k": bool(all_ge_k),
            "verified": bool(recon == true and all_ge_k),
            "terms": [{"query": q["meta"], "returns": q["cnt"], "coefficient": str(q["coef"])}
                      for q in used],
        })
    return out


if __name__ == "__main__":
    everything = []
    for name, loader in [("NHANES", load_nhanes), ("Hospital", load_hospital), ("Adult", load_adult)]:
        certs = certificates(name, loader())
        print(f"{name}: {len(certs)} cells, {sum(c['true_count'] for c in certs)} people, "
              f"{sum(c['verified'] for c in certs)} verified")
        everything += certs
    folder = os.path.join(RESULTS, "certificates")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"local_certificates_k{K}.json")
    with open(path, "w") as fh:
        json.dump(everything, fh, indent=1)
    print(f"wrote {path}")
