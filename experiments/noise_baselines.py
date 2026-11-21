"""Rounding and Laplace noise against the k-clean certificates (Section VI).

An attacker who holds a certificate combines the released answers with the
certificate's coefficients and rounds. A recovery is counted when the rounded
value equals the true count of the protected cell.

  rounding          every count is rounded to the nearest multiple of R
  Laplace, memoized one draw per distinct query (scale 1/eps, sensitivity 1),
                    so repeating a query returns the same value
  Laplace, fresh    a new draw per request; the attacker repeats each query
                    and averages

Utility is measured on a benign workload made of every (age_band, race,
gender, partition) cell with at least k people.

Output
  results/baseline_dp.csv          memoized Laplace, recovery rate per eps
  results/baseline_rounding.csv    certificates still recovered per base R
  results/baseline_utility.csv     error on the benign cells
  results/laplace_averaging.csv    fresh noise, recovery per number of repeats

Run:  python experiments/noise_baselines.py
"""
from fractions import Fraction

import numpy as np
import pandas as pd

from common import QUASI, certificate, load_hospital, load_nhanes, prep, result_path, target_of

K = 10
DRAWS_MEMOIZED = 3000
DRAWS_FRESH = 1500


def coefficients(used, n_cells, target_index):
    """Solve for the weights of the certificate's queries (float), by Gauss-Jordan."""
    m = len(used)
    aug = [[Fraction(used[j]["vec"][i]) for j in range(m)] + [Fraction(int(i == target_index))]
           for i in range(n_cells)]
    pr = 0
    piv = []
    for c in range(m):
        p_ = next((r for r in range(pr, n_cells) if aug[r][c] != 0), None)
        if p_ is None:
            continue
        aug[pr], aug[p_] = aug[p_], aug[pr]
        inv = aug[pr][c]
        aug[pr] = [x / inv for x in aug[pr]]
        for r in range(n_cells):
            if r != pr and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[pr])]
        piv.append(c)
        pr += 1
        if pr == n_cells:
            break
    coef = [0.0] * m
    for i, c in enumerate(piv):
        coef[c] = float(aug[i][m])
    return coef


def load_certificates(datasets):
    """One (counts, coefficients, true count) triple per attackable at-risk target."""
    certs = []
    for data in datasets:
        loc, parts, outs, pool = prep(data, K)
        O = len(outs)
        pidx = {p: i for i, p in enumerate(parts)}
        oidx = {o: j for j, o in enumerate(outs)}
        for _, t in pool.iterrows():
            bk, tp, ty = target_of(t)
            cell = loc.get(bk, {})
            used = certificate(cell, parts, outs, tp, ty, K)
            if used is None:
                continue
            coef = coefficients(used, len(parts) * O, pidx[tp] * O + oidx[ty])
            certs.append((np.array([q["cnt"] for q in used], float), np.array(coef),
                          cell.get((tp, ty), 0)))
    return certs


def memoized_recovery(certs, scale, rng):
    hits = total = 0
    for counts, coef, true in certs:
        noise = rng.laplace(0, scale, size=(DRAWS_MEMOIZED, len(counts)))
        est = (coef * (counts + noise)).sum(axis=1)
        hits += int(np.sum(np.round(est) == true))
        total += DRAWS_MEMOIZED
    return hits / total


def rounding_recovered(certs, base):
    return sum(int(round(float((coef * np.array([base * round(x / base) for x in counts])).sum())) == true)
               for counts, coef, true in certs)


def averaging_recovery(certs, scale, repeats, rng):
    hits = total = 0
    for counts, coef, true in certs:
        for _ in range(DRAWS_FRESH):
            avg = counts + rng.laplace(0, scale, size=(len(counts), repeats)).mean(axis=1)
            hits += int(round((coef * avg).sum()) == true)
            total += 1
    return hits / total


def benign_error(named, rng, base=5, scales=(1.0, 2.0, 4.0)):
    rows = []
    for name, data in named:
        g = data.groupby(QUASI, observed=True).size()
        counts = np.array([v for v in g.values if v >= K], float)
        row = {"dataset": name, "benign_cells": len(counts)}
        for b in scales:
            noise = rng.laplace(0, b, size=(len(counts),))
            row[f"laplace_scale{b:g}_within1"] = round(float(np.mean(np.abs(noise) < 1)), 3)
            row[f"laplace_scale{b:g}_mae"] = round(float(np.mean(np.abs(noise))), 2)
        err = np.array([abs(x - base * round(x / base)) for x in counts])
        row[f"round{base}_exact"] = round(float(np.mean(err == 0)), 3)
        row[f"round{base}_mae"] = round(float(np.mean(err)), 2)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    nhanes, hospital = load_nhanes(), load_hospital()
    certs = load_certificates([nhanes, hospital])
    print(f"certificates: {len(certs)}")

    rng = np.random.default_rng(0)
    dp = pd.DataFrame([dict(eps=eps, scale=1 / eps, draws_per_certificate=DRAWS_MEMOIZED,
                            recovery_rate=round(memoized_recovery(certs, 1 / eps, rng), 4))
                       for eps in (1.0, 0.5, 0.25)])
    dp.to_csv(result_path("baseline_dp.csv"), index=False)
    print(dp.to_string(index=False))

    rounding = pd.DataFrame([dict(base=R, certificates=len(certs), recovered=rounding_recovered(certs, R))
                             for R in (5, 10)])
    rounding["prevented"] = rounding.certificates - rounding.recovered
    rounding.to_csv(result_path("baseline_rounding.csv"), index=False)
    print(rounding.to_string(index=False))

    utility = benign_error([("NHANES", nhanes), ("Hospital", hospital)], rng)
    utility.to_csv(result_path("baseline_utility.csv"), index=False)
    print(utility.to_string(index=False))

    rng = np.random.default_rng(0)
    avg = pd.DataFrame([dict(eps=1.0, repeats=r, draws_per_certificate=DRAWS_FRESH,
                             recovery_rate=round(averaging_recovery(certs, 1.0, r, rng), 4))
                        for r in (1, 10, 100, 1000)])
    avg.to_csv(result_path("laplace_averaging.csv"), index=False)
    print(avg.to_string(index=False))
