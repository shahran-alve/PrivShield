"""Online rank auditor.

The auditor keeps the span of the query vectors it has released. A query is a
set of grid cells whose counts it adds up. Before releasing a query the auditor
checks whether the release would put the indicator of any protected cell in
that span, and refuses if it would.

State. `basis` is a reduced row-echelon basis of the released span: every row
has coefficient 1 at its pivot and 0 at every other pivot. `resid[t]` is the
indicator of protected cell t reduced against that basis. Reduction against a
reduced echelon basis is a linear map whose kernel is the span, so cell t lies
in span(basis + v) exactly when resid[t] is a scalar multiple of reduce(v).

Arithmetic is exact: rationals when `prime` is None, otherwise the field of
integers modulo `prime`.
"""
from collections import defaultdict
from fractions import Fraction

P61 = (1 << 61) - 1
P31 = (1 << 31) - 1


class RankAuditor:
    def __init__(self, protected, prime=None):
        self.p = prime
        self.one = 1 if prime else Fraction(1)
        self.basis = {}
        self.basis_cols = defaultdict(set)     # column -> pivots whose row uses it
        self.resid = {t: {t: self.one} for t in protected}
        self.resid_cols = defaultdict(set)     # column -> protected cells whose residual uses it
        for t in self.resid:
            self.resid_cols[t].add(t)
        self.released = 0
        self.refused = 0

    # field helpers
    def _inv(self, x):
        return pow(x, self.p - 2, self.p) if self.p else 1 / x

    def _fix(self, x):
        return x % self.p if self.p else x

    def _axpy(self, row, f, other, index=None, key=None):
        """row <- row - f * other, keeping the column index up to date."""
        for c, v in other.items():
            nv = self._fix(row.get(c, 0) - f * v)
            if nv:
                if index is not None and c not in row:
                    index[c].add(key)
                row[c] = nv
            elif c in row:
                del row[c]
                if index is not None:
                    index[c].discard(key)

    def reduce(self, cells):
        """Vector of a query (sum of the given cells) reduced against the basis."""
        row = {}
        for c in cells:
            row[c] = self._fix(row.get(c, 0) + self.one)
        row = {c: v for c, v in row.items() if v}
        for c in [c for c in row if c in self.basis]:
            f = row.get(c)
            if f:
                self._axpy(row, f, self.basis[c])
        return row

    def _normalise(self, row):
        c0 = min(row)
        inv = self._inv(row[c0])
        return c0, {c: self._fix(v * inv) for c, v in row.items()}

    def exposed_by(self, cells):
        """Protected cells that would become computable if the query were released."""
        row = self.reduce(cells)
        if not row:
            return []
        c0, n = self._normalise(row)
        hit = []
        for t in self.resid_cols.get(c0, ()):
            r = self.resid[t]
            if len(r) != len(n):
                continue
            f = r[c0]
            if all(c in r and r[c] == self._fix(f * v) for c, v in n.items()):
                hit.append(t)
        return hit

    def release(self, cells):
        """Add a query to the released span (no check)."""
        self.released += 1
        row = self.reduce(cells)
        if not row:
            return
        cp, n = self._normalise(row)
        for piv in list(self.basis_cols.get(cp, ())):
            b = self.basis[piv]
            self._axpy(b, b[cp], n, self.basis_cols, piv)
        self.basis[cp] = n
        for c in n:
            self.basis_cols[c].add(cp)
        for t in list(self.resid_cols.get(cp, ())):
            r = self.resid[t]
            self._axpy(r, r[cp], n, self.resid_cols, t)

    def ask(self, cells):
        """Release the query unless it exposes a protected cell. Returns True if released."""
        if self.exposed_by(cells):
            self.refused += 1
            return False
        self.release(cells)
        return True

    def computable(self):
        """Protected cells that lie in the released span (should stay empty)."""
        return [t for t, r in self.resid.items() if not r]


def span_basis(vectors, prime=P61):
    """Echelon basis of a list of cell lists, for independent membership checks."""
    basis = {}
    for v in vectors:
        row = _lead_reduce({c: 1 for c in v}, basis, prime)
        if row:
            c = min(row)
            inv = pow(row[c], prime - 2, prime)
            basis[c] = {c2: (x * inv) % prime for c2, x in row.items()}
    return basis


def _lead_reduce(row, basis, prime):
    row = {c: v % prime for c, v in row.items() if v % prime}
    while row:
        c = min(row)
        if c not in basis:
            break
        f = row[c]
        for c2, x in basis[c].items():
            nv = (row.get(c2, 0) - f * x) % prime
            if nv:
                row[c2] = nv
            else:
                row.pop(c2, None)
    return row


def in_span(cells, basis, prime=P61):
    return not _lead_reduce({c: 1 for c in cells}, basis, prime)
