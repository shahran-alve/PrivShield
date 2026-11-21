"""Check the rank auditor against a brute-force rank computation.

On random small grids, every decision of RankAuditor (rationals and both
primes) must equal the answer of a direct test: would the query put a
protected cell in the span of the released queries? After each run the
released span is also checked with the independent echelon routine.

Run:  python tests/test_rank_auditor.py
"""
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from common import rank  # noqa: E402
from rank_auditor import P31, P61, RankAuditor, in_span, span_basis  # noqa: E402


def vector(cells, n):
    return [Fraction(int(i in cells)) for i in range(n)]


def exposes(released, query, protected, n):
    rows = [vector(q, n) for q in released] + [vector(query, n)]
    r = rank(rows)
    return any(rank(rows + [vector({t}, n)]) == r for t in protected)


def main(trials=4000, seed=3):
    rng = random.Random(seed)
    decisions = refusals = 0
    for _ in range(trials):
        n = rng.randint(3, 9)
        protected = rng.sample(range(n), rng.randint(1, min(3, n)))
        queries = [set(rng.sample(range(n), rng.randint(1, n))) for _ in range(rng.randint(2, 9))]
        for prime in (None, P61, P31):
            auditor = RankAuditor(protected, prime)
            released = []
            for q in queries:
                expected = not exposes(released, q, protected, n)
                assert auditor.ask(q) == expected
                decisions += 1
                if expected:
                    released.append(q)
                else:
                    refusals += 1
            assert not auditor.computable()
            basis = span_basis([sorted(q) for q in released])
            assert not any(in_span([t], basis) for t in protected)

    # a single cell answered before the total that contains it
    for order in ([[1, 3], [3]], [[3], [1, 3]]):
        auditor = RankAuditor([1])
        assert [auditor.ask(q) for q in order] == [True, False]

    print(f"{decisions} decisions checked, {refusals} refusals, all equal to the brute-force test")


if __name__ == "__main__":
    main()
