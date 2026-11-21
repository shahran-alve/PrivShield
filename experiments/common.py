"""Paths, dataset loaders and local-grid helpers shared by the experiment scripts.

A local grid is the partition x outcome table of one (age_band, race, gender)
group. Cell (p, o) has flat index pidx[p] * len(outs) + oidx[o].
"""
import os
import sys
from collections import defaultdict
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets import load_adult, load_hospital, load_nhanes  # noqa: E402

RESULTS = os.environ.get("PRIVSHIELD_RESULTS", "results")
os.makedirs(RESULTS, exist_ok=True)

QUASI = ["age_band", "race", "gender", "partition"]
OUT = "outcome"


def result_path(name):
    return os.path.join(RESULTS, name)


def prep(data, k):
    """Local grids, the partition and outcome levels, and the at-risk pool.

    loc[(age, race, gender)][(partition, outcome)] is a count. The pool holds
    every person whose (age, race, gender, partition) group has fewer than k
    members.
    """
    g = data.groupby(QUASI + [OUT], observed=True).size()
    loc = defaultdict(dict)
    for (a, r, gg, p, o), c in g.items():
        loc[(str(a), str(r), str(gg))][(str(p), str(o))] = int(c)
    parts = sorted(data["partition"].astype(str).unique().tolist())
    outs = sorted(data[OUT].astype(str).unique().tolist())
    size = data.groupby(QUASI, observed=True)[OUT].transform("size")
    return loc, parts, outs, data[size < k].copy()


def target_of(row):
    """(base key, partition, outcome) of one row of the at-risk pool."""
    return ((str(row["age_band"]), str(row["race"]), str(row["gender"])),
            str(row["partition"]), str(row[OUT]))


def rank(rows):
    """Rank of a list of Fraction rows by Gauss-Jordan elimination."""
    rows = [r[:] for r in rows]
    if not rows:
        return 0
    pr = 0
    for c in range(len(rows[0])):
        piv = next((r for r in range(pr, len(rows)) if rows[r][c] != 0), None)
        if piv is None:
            continue
        rows[pr], rows[piv] = rows[piv], rows[pr]
        inv = rows[pr][c]
        rows[pr] = [x / inv for x in rows[pr]]
        for r in range(len(rows)):
            if r != pr and rows[r][c] != 0:
                f = rows[r][c]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[pr])]
        pr += 1
        if pr == len(rows):
            break
    return pr


def in_span(vecs, target):
    """Exact test: is `target` in the rational row space of `vecs`?"""
    V = [[Fraction(int(x)) for x in v] for v in vecs]
    return rank(V) == rank(V + [[Fraction(int(x)) for x in target]])


def local_queries(cell, parts, outs, k):
    """Answerable ungrouped COUNT queries of one local grid.

    In order: the group total, one query per outcome, one per partition value,
    then the single cells. A query is kept only if its count is at least k.
    Each query is a dict with its 0/1 vector, count, filter set and kind
    ("m" for a marginal, "c" for a single cell).
    """
    P, O = len(parts), len(outs)
    pidx = {p: i for i, p in enumerate(parts)}
    oidx = {o: j for j, o in enumerate(outs)}

    def flat(cells):
        v = [0] * (P * O)
        for p, o in cells:
            v[pidx[p] * O + oidx[o]] = 1
        return v

    def n(p, o):
        return cell.get((p, o), 0)

    Q = []
    tot = sum(cell.values())
    if tot >= k:
        Q.append(dict(vec=flat({(p, o) for p in parts for o in outs}), cnt=tot,
                      fs=frozenset("arg"), kind="m", meta={"type": "total"}))
    for o in outs:
        s = sum(n(p, o) for p in parts)
        if s >= k:
            Q.append(dict(vec=flat({(p, o) for p in parts}), cnt=s,
                          fs=frozenset(["a", "r", "g", "o=" + o]), kind="m",
                          meta={"type": "outcome_total", "outcome": o}))
    for p in parts:
        s = sum(n(p, o) for o in outs)
        if s >= k:
            Q.append(dict(vec=flat({(p, o) for o in outs}), cnt=s,
                          fs=frozenset(["a", "r", "g", "p=" + p]), kind="m",
                          meta={"type": "partition_total", "partition": p}))
    for p in parts:
        for o in outs:
            if n(p, o) >= k:
                Q.append(dict(vec=flat({(p, o)}), cnt=n(p, o),
                              fs=frozenset(["a", "r", "g", "p=" + p, "o=" + o]), kind="c",
                              meta={"type": "cell", "partition": p, "outcome": o}))
    return Q


def cells_of(query):
    return [i for i, x in enumerate(query["vec"]) if x]


def protected_cells(cell, parts, outs, k):
    """Flat indices of the cells with a count from 1 to k-1."""
    O = len(outs)
    return [i * O + j for i, p in enumerate(parts) for j, o in enumerate(outs)
            if 0 < cell.get((p, o), 0) < k]


def certificate(cell, parts, outs, tp, ty, k):
    """A k-clean certificate for cell (tp, ty), or None if there is none.

    Returns the queries with a nonzero coefficient, each with key "coef" added.
    The weighted sum of their counts equals the count of the target cell.
    """
    P, O = len(parts), len(outs)
    pidx = {p: i for i, p in enumerate(parts)}
    oidx = {o: j for j, o in enumerate(outs)}
    Q = local_queries(cell, parts, outs, k)
    tgt = [0] * (P * O)
    tgt[pidx[tp] * O + oidx[ty]] = 1
    if not Q or not in_span([q["vec"] for q in Q], tgt):
        return None
    aug = [[Fraction(q["vec"][i]) for q in Q] + [Fraction(tgt[i])] for i in range(P * O)]
    pr = 0
    piv = []
    for c in range(len(Q)):
        p_ = next((r for r in range(pr, P * O) if aug[r][c] != 0), None)
        if p_ is None:
            continue
        aug[pr], aug[p_] = aug[p_], aug[pr]
        inv = aug[pr][c]
        aug[pr] = [x / inv for x in aug[pr]]
        for r in range(P * O):
            if r != pr and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[pr])]
        piv.append(c)
        pr += 1
        if pr == P * O:
            break
    coef = [Fraction(0)] * len(Q)
    for i, c in enumerate(piv):
        coef[c] = aug[i][len(Q)]
    assert sum(coef[i] * Q[i]["cnt"] for i in range(len(Q))) == cell.get((tp, ty), 0)
    return [dict(q, coef=coef[i]) for i, q in enumerate(Q) if coef[i] != 0]


def exposed(cell, tp, outs):
    """True if the target's group is a singleton or has a single outcome."""
    tc = [cell.get((tp, o), 0) for o in outs if cell.get((tp, o), 0) > 0]
    return bool(tc) and (sum(tc) == 1 or max(tc) == sum(tc))


# Schema grid: the full product of the observed levels of the five attributes.

DIMS = QUASI + [OUT]


class Schema:
    """Counts over the full attribute grid, and the cells an auditor protects."""

    def __init__(self, data, k):
        from itertools import product
        d = data[DIMS].astype(str)
        self.k = k
        self.counts = {tuple(key): int(v) for key, v in d.groupby(DIMS, observed=True).size().items()}
        self.levels = [sorted(d[c].unique().tolist()) for c in DIMS]
        self.index = {key: i for i, key in enumerate(product(*self.levels))}
        self.protected = sorted(self.index[key] for key, c in self.counts.items() if 0 < c < k)

    def keys(self, filters):
        """Grid cells matching a filter dict; attributes left out range over all levels."""
        from itertools import product
        opts = [[filters[d]] if d in filters else self.levels[i] for i, d in enumerate(DIMS)]
        return list(product(*opts))

    def cells(self, filters):
        return [self.index[key] for key in self.keys(filters)]

    def count(self, filters):
        return sum(self.counts.get(key, 0) for key in self.keys(filters))
