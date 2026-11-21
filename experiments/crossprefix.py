"""Queries that drop a quasi-identifier: attack and rank auditor (Section VII-C, Table V).

The local search fixes age band, race and gender, so it only sees one
partition x outcome grid. The query language also allows COUNT queries with
fewer filters, and such a query adds up cells from several grids. This script
rebuilds the answerable set over the full attribute grid: every count that
fixes any subset of the five attributes and is at least k.

Attack. An at-risk target is attackable when the indicator of its cell lies in
the span of the answerable queries. The grid includes empty cells, because the
attacker does not know which cells are empty. Rank is computed modulo two
primes (2^61-1 and 2^31-1) and a target is counted only if both agree. Mode
"local" restricts the queries to those that fix all three quasi-identifiers
and reproduces the local search.

Auditor. The rank auditor sees the whole cross-prefix query set, once from the
broadest query to the narrowest and once in reverse, and protects every cell
with a count from 1 to k-1. After each run the script recomputes, from the
released queries alone, how many protected cells are in their span. The audit
is repeated modulo the second prime and must make the same decisions.

Output
  results/crossprefix_k{k}.csv        attackable targets, local and cross-prefix
  results/crossprefix_audit_k{k}.csv  auditor run: refusals, cells left
                                      computable, time per query

Run:  python experiments/crossprefix.py [k]
"""
import sys
import time
from itertools import combinations

import pandas as pd

from common import Schema, load_adult, load_hospital, load_nhanes, result_path
from rank_auditor import P31, P61, RankAuditor, in_span, span_basis


def answerable(schema, mode):
    """Cell lists of every answerable query, broadest first."""
    k = schema.k
    grid = list(schema.index)
    queries = []
    for size in range(0, 6):
        for fixed in combinations(range(5), size):
            if mode == "local" and not {0, 1, 2} <= set(fixed):
                continue
            groups = {}
            for key in grid:
                groups.setdefault(tuple(key[i] for i in fixed), []).append(key)
            for members in groups.values():
                if sum(schema.counts.get(key, 0) for key in members) >= k:
                    queries.append([schema.index[key] for key in members])
    return queries


def at_risk_cells(schema):
    """Nonempty cells of groups (age, race, gender, partition) with fewer than k members."""
    size = {}
    for key, c in schema.counts.items():
        size[key[:4]] = size.get(key[:4], 0) + c
    return [key for key, c in schema.counts.items() if 0 < size[key[:4]] < schema.k]


def search(name, schema, mode):
    queries = answerable(schema, mode)
    bases = [(p, span_basis(queries, p)) for p in (P61, P31)]
    group = {}
    for key in at_risk_cells(schema):
        group.setdefault(key[:4], []).append(key)
    at_risk = attack = disclosed = 0
    for cells in group.values():
        one_outcome = len(cells) == 1
        for key in cells:
            w = schema.counts[key]
            at_risk += w
            if all(in_span([schema.index[key]], b, p) for p, b in bases):
                attack += w
                disclosed += w if one_outcome else 0
    return dict(dataset=name, mode=mode, grid=len(schema.index), n_query_vecs=len(queries),
                at_risk=at_risk, bypasses=attack, disclosed=disclosed)


def audit(name, schema):
    queries = answerable(schema, "cross")
    targets = [schema.index[key] for key in at_risk_cells(schema)]
    full = span_basis(queries)
    row = dict(dataset=name, k=schema.k, grid=len(schema.index), queries=len(queries),
               protected_cells=len(schema.protected), target_cells=len(targets),
               targets_computable_no_auditor=sum(in_span([t], full) for t in targets))
    for label, seq in (("fwd", queries), ("rev", list(reversed(queries)))):
        aud = RankAuditor(schema.protected, P61)
        t0 = time.perf_counter()
        decisions = [aud.ask(q) for q in seq]
        elapsed = time.perf_counter() - t0
        released = [q for q, ok in zip(seq, decisions) if ok]
        left = 0
        for p in (P61, P31):
            basis = span_basis(released, p)
            left = max(left, sum(in_span([c], basis, p) for c in schema.protected))
        second = RankAuditor(schema.protected, P31)
        row[f"refused_{label}"] = aud.refused
        row[f"protected_computable_{label}"] = left
        row[f"primes_agree_{label}"] = decisions == [second.ask(q) for q in seq]
        row[f"ms_per_query_{label}"] = round(1000 * elapsed / len(seq), 3)
    return row


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    found, audits = [], []
    for name, loader in [("Hospital", load_hospital), ("NHANES", load_nhanes), ("Adult", load_adult)]:
        schema = Schema(loader(), K)
        for mode in ("local", "cross"):
            found.append(search(name, schema, mode))
            print(found[-1], flush=True)
        audits.append(audit(name, schema))
        print(audits[-1], flush=True)
    pd.DataFrame(found).to_csv(result_path(f"crossprefix_k{K}.csv"), index=False)
    pd.DataFrame(audits).to_csv(result_path(f"crossprefix_audit_k{K}.csv"), index=False)
