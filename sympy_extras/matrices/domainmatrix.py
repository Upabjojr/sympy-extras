r"""Exact linear algebra over a typed domain: fraction-free elimination.

A :class:`DomainMatrix` is a dense matrix whose entries are elements of a
:class:`~sympy_extras.polys.domains.domain.Domain` (integers, rationals,
a finite field, polynomials in parameters). Gaussian elimination over a
field of fractions makes the entries grow at every step (numerators and
denominators of rational functions, with a gcd for each operation);
the algorithms here stay in the ring instead:

* **Bareiss' one-step fraction-free elimination** [1]_: with $p_{-1} = 1$
  and $p_k$ the pivot of step $k$, the update

  .. math:: a_{ij} \leftarrow \frac{p_k\, a_{ij} - a_{ik}\, a_{kj}}{p_{k-1}}

  is an exact division in the ring, since by Sylvester's identity every
  entry produced is a minor of the original matrix. The last pivot is the
  determinant (up to the sign of the row exchanges), and the entries stay
  as small as minors can be.
* The same update applied to the rows *above* the pivot as well gives the
  fraction-free Gauss–Jordan form [2]_: a matrix $R$ and a denominator $d$
  with $R/d$ the reduced row echelon form. The rank, a basis of the null
  space, the inverse ($[A \mid I]$) and the solutions of linear systems
  ($[A \mid B]$) are read off it, with a single denominator.
* **Berkowitz' algorithm** [3]_ for the characteristic polynomial, which
  uses no division at all: the characteristic polynomial of the leading
  principal submatrix $A_{k+1} = \begin{pmatrix} A_k & C \\ R & a
  \end{pmatrix}$ is the product of the lower triangular Toeplitz matrix
  with first column $(1, -a, -RC, -RA_kC, -RA_k^2C, \dots)$ and the
  coefficient vector of that of $A_k$. The adjugate of a singular matrix
  follows from it by the Cayley–Hamilton theorem.

Examples
========

>>> from sympy_extras.polys.domains import ZZ, QQ
>>> from sympy_extras.matrices.domainmatrix import DomainMatrix
>>> A = DomainMatrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]], ZZ)
>>> A.det(), A.rank(), A.charpoly()
(-1, 3, [1, -5, 4, 1])
>>> N, d = A.inv_den()
>>> N, d, A * N == DomainMatrix.eye(3, ZZ).scale(d)
(DomainMatrix([[0, 0, 1], [-2, 1, 3], [3, -1, -5]], ZZ), 1, True)
>>> B = DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ZZ)
>>> B.rref_den()
(DomainMatrix([[3, 0, -3], [0, 3, 6], [0, 0, 0]], ZZ), 3, [0, 1])
>>> B.nullspace(), B.det()
([[3, -6, 3]], 0)

References
==========

.. [1] E. H. Bareiss, *Sylvester's identity and multistep
       integer-preserving Gaussian elimination*, Mathematics of
       Computation 22 (1968), 565-578.
.. [2] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Section 9.3 (fraction-free elimination).
.. [3] S. J. Berkowitz, *On computing the determinant in small parallel
       time using a small number of processors*, Information Processing
       Letters 18 (1984), 147-150.
"""
from __future__ import annotations

from typing import Generic, Optional, Sequence, TypeVar

from sympy_extras.polys.domains.domain import Domain

__all__ = ['DomainMatrix']

T = TypeVar('T')


class DomainMatrix(Generic[T]):
    """A dense ``nrows`` by ``ncols`` matrix over ``domain`` (immutable:
    every operation returns a new matrix). ``ncols`` is only needed for a
    matrix without rows.

    The fraction-free methods need an integral domain with exact division
    (``exquo``); they work in a field too.
    """

    __slots__ = ('rows', 'nrows', 'ncols', 'domain')

    def __init__(self, rows: Sequence[Sequence[T]], domain: Domain[T], ncols: Optional[int] = None) -> None:
        width = len(rows[0]) if rows else (0 if ncols is None else ncols)
        if ncols is not None and ncols != width:
            raise ValueError('rows of length %d for %d columns' % (width, ncols))
        if any(len(r) != width for r in rows):
            raise ValueError('rows of different lengths')
        self.rows: list[list[T]] = [list(r) for r in rows]
        self.nrows: int = len(rows)
        self.ncols: int = width
        self.domain: Domain[T] = domain

    # -- construction

    @classmethod
    def zeros(cls, nrows: int, ncols: int, domain: Domain[T]) -> DomainMatrix[T]:
        return cls([[domain.zero] * ncols for _ in range(nrows)], domain, ncols)

    @classmethod
    def eye(cls, n: int, domain: Domain[T]) -> DomainMatrix[T]:
        return cls([[domain.one if i == j else domain.zero for j in range(n)] for i in range(n)], domain, n)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, DomainMatrix) and self.domain == other.domain and self.shape == other.shape
                and self.rows == other.rows)

    def __hash__(self) -> int:
        return hash((self.nrows, self.ncols, self.domain))

    def __repr__(self) -> str:
        K = self.domain
        body = ', '.join('[' + ', '.join(K.to_str(a) for a in row) + ']' for row in self.rows)
        return 'DomainMatrix([%s], %s)' % (body, K.name)

    def _square(self) -> int:
        if self.nrows != self.ncols:
            raise ValueError('a %d x %d matrix is not square' % self.shape)
        return self.nrows

    # -- arithmetic

    def __add__(self, other: DomainMatrix[T]) -> DomainMatrix[T]:
        if self.shape != other.shape:
            raise ValueError('shapes %s and %s' % (self.shape, other.shape))
        K = self.domain
        return DomainMatrix([[K.add(a, b) for a, b in zip(r, s)] for r, s in zip(self.rows, other.rows)], K, self.ncols)

    def __sub__(self, other: DomainMatrix[T]) -> DomainMatrix[T]:
        if self.shape != other.shape:
            raise ValueError('shapes %s and %s' % (self.shape, other.shape))
        K = self.domain
        return DomainMatrix([[K.sub(a, b) for a, b in zip(r, s)] for r, s in zip(self.rows, other.rows)], K, self.ncols)

    def __neg__(self) -> DomainMatrix[T]:
        K = self.domain
        return DomainMatrix([[K.neg(a) for a in r] for r in self.rows], K, self.ncols)

    def __mul__(self, other: DomainMatrix[T]) -> DomainMatrix[T]:
        if self.ncols != other.nrows:
            raise ValueError('shapes %s and %s' % (self.shape, other.shape))
        K = self.domain
        columns = other.transpose().rows
        product: list[list[T]] = []
        for r in self.rows:
            line: list[T] = []
            for column in columns:
                total = K.zero
                for a, b in zip(r, column):
                    if not K.is_zero(a) and not K.is_zero(b):
                        total = K.add(total, K.mul(a, b))
                line.append(total)
            product.append(line)
        return DomainMatrix(product, K, other.ncols)

    def __pow__(self, n: int) -> DomainMatrix[T]:
        size = self._square()
        if n < 0:
            raise ValueError('negative power of a matrix over a ring: use inv_den')
        result = DomainMatrix.eye(size, self.domain)
        base = self
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def scale(self, c: T) -> DomainMatrix[T]:
        """The matrix times the element ``c``."""
        K = self.domain
        return DomainMatrix([[K.mul(a, c) for a in r] for r in self.rows], K, self.ncols)

    def transpose(self) -> DomainMatrix[T]:
        return DomainMatrix([[self.rows[i][j] for i in range(self.nrows)] for j in range(self.ncols)], self.domain, self.nrows)

    def hstack(self, other: DomainMatrix[T]) -> DomainMatrix[T]:
        """The columns of ``other`` appended."""
        if self.nrows != other.nrows:
            raise ValueError('shapes %s and %s' % (self.shape, other.shape))
        return DomainMatrix([r + s for r, s in zip(self.rows, other.rows)], self.domain, self.ncols + other.ncols)

    # -- fraction-free elimination

    def _pivot_row(self, rows: list[list[T]], start: int, column: int) -> Optional[int]:
        """The row from ``start`` on with the smallest nonzero entry in
        the column (small pivots keep the intermediate entries small)."""
        K = self.domain
        best: Optional[int] = None
        best_size = 0
        for i in range(start, len(rows)):
            a = rows[i][column]
            if K.is_zero(a):
                continue
            size = K.size(a)
            if best is None or size < best_size:
                best, best_size = i, size
        return best

    def det(self) -> T:
        """The determinant, by Bareiss' fraction-free elimination.

        >>> from sympy_extras.polys.domains import ZZ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]], ZZ).det()
        -3
        """
        n = self._square()
        K = self.domain
        rows = [list(r) for r in self.rows]
        previous = K.one
        negative = False
        for k in range(n):
            chosen = self._pivot_row(rows, k, k)
            if chosen is None:
                return K.zero
            if chosen != k:
                rows[k], rows[chosen] = rows[chosen], rows[k]
                negative = not negative
            pivot = rows[k][k]
            pivot_row = rows[k]
            for i in range(k + 1, n):
                row = rows[i]
                factor = row[k]
                if K.is_zero(factor):
                    for j in range(k + 1, n):
                        row[j] = K.exquo(K.mul(pivot, row[j]), previous)
                else:
                    for j in range(k + 1, n):
                        row[j] = K.exquo(K.sub(K.mul(pivot, row[j]), K.mul(factor, pivot_row[j])), previous)
                row[k] = K.zero
            previous = pivot
        return K.neg(previous) if negative else previous

    def rref_den(self) -> tuple[DomainMatrix[T], T, list[int]]:
        """``(R, d, pivots)``: ``R/d`` is the reduced row echelon form,
        ``pivots`` the pivot columns; ``R`` has ``d`` at the pivot
        positions (fraction-free Gauss–Jordan elimination). ``d`` is one
        for a matrix of rank zero, and in canonical form (positive over the
        integers).

        >>> from sympy_extras.polys.domains import ZZ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> DomainMatrix([[2, 4, 1], [3, 6, 5]], ZZ).rref_den()
        (DomainMatrix([[7, 14, 0], [0, 0, 7]], ZZ), 7, [0, 2])
        """
        reduced, den, pivots, _ = self.elimination()
        return reduced, den, pivots

    def elimination(self) -> tuple[DomainMatrix[T], T, list[int], list[T]]:
        """The result of :meth:`rref_den` followed by the pivots used, in
        order (minors of the matrix): the elimination is the same over any
        ring the entries are mapped to in which these stay nonzero.
        """
        K = self.domain
        rows = [list(r) for r in self.rows]
        previous = K.one
        pivots: list[int] = []
        values: list[T] = []
        r = 0
        for column in range(self.ncols):
            if r == self.nrows:
                break
            chosen = self._pivot_row(rows, r, column)
            if chosen is None:
                continue
            if chosen != r:
                rows[r], rows[chosen] = rows[chosen], rows[r]
            pivot_row = rows[r]
            pivot = pivot_row[column]
            for i in range(self.nrows):
                if i == r:
                    continue
                row = rows[i]
                factor = row[column]
                if K.is_zero(factor):
                    if not K.is_one(previous) or not K.is_one(pivot):
                        rows[i] = [K.exquo(K.mul(pivot, a), previous) if not K.is_zero(a) else a for a in row]
                else:
                    rows[i] = [K.exquo(K.sub(K.mul(pivot, a), K.mul(factor, b)), previous)
                               for a, b in zip(row, pivot_row)]
            previous = pivot
            pivots.append(column)
            values.append(pivot)
            r += 1
        unit = K.canonical_unit(previous)
        if not K.is_one(unit):
            previous = K.mul(previous, unit)
            rows = [[K.mul(a, unit) for a in row] for row in rows]
        return DomainMatrix(rows, K, self.ncols), previous, pivots, values

    def rref(self) -> tuple[DomainMatrix[T], list[int]]:
        """The reduced row echelon form and the pivot columns (over a
        field).

        >>> from fractions import Fraction
        >>> from sympy_extras.polys.domains import QQ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> M = DomainMatrix([[Fraction(2), Fraction(4), Fraction(1)], [Fraction(3), Fraction(6), Fraction(5)]], QQ)
        >>> M.rref() == (DomainMatrix([[Fraction(1), Fraction(2), Fraction(0)],
        ...     [Fraction(0), Fraction(0), Fraction(1)]], QQ), [0, 2])
        True
        """
        K = self.domain
        if not K.is_field:
            raise ValueError('rref needs a field: use rref_den over %s' % K.name)
        reduced, den, pivots = self.rref_den()
        inverse = K.invert(den)
        return reduced.scale(inverse), pivots

    def rank(self) -> int:
        return len(self.rref_den()[2])

    def nullspace(self) -> list[list[T]]:
        """A basis of the (right) null space, without fractions: one
        vector for each column without pivot, with the denominator of
        :meth:`rref_den` at that coordinate.

        >>> from sympy_extras.polys.domains import ZZ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> DomainMatrix([[1, 2, 3], [2, 4, 7]], ZZ).nullspace()
        [[-2, 1, 0]]
        """
        K = self.domain
        reduced, den, pivots = self.rref_den()
        basis: list[list[T]] = []
        for free in range(self.ncols):
            if free in pivots:
                continue
            vector = [K.zero] * self.ncols
            vector[free] = den
            for i, column in enumerate(pivots):
                vector[column] = K.neg(reduced.rows[i][free])
            basis.append(vector)
        return basis

    def inv_den(self) -> tuple[DomainMatrix[T], T]:
        """``(N, d)`` with ``A N = N A = d I``, so that the inverse is
        ``N/d``; ``ValueError`` for a singular matrix.

        >>> from sympy_extras.polys.domains import ZZ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> DomainMatrix([[1, 2], [3, 4]], ZZ).inv_den()
        (DomainMatrix([[-4, 2], [3, -1]], ZZ), 2)
        """
        n = self._square()
        K = self.domain
        reduced, den, pivots = self.hstack(DomainMatrix.eye(n, K)).rref_den()
        if pivots != list(range(n)):
            raise ValueError('the matrix is singular')
        return DomainMatrix([row[n:] for row in reduced.rows], K, n), den

    def inv(self) -> DomainMatrix[T]:
        """The inverse (over a field)."""
        K = self.domain
        if not K.is_field:
            raise ValueError('inv needs a field: use inv_den over %s' % K.name)
        numerator, den = self.inv_den()
        return numerator.scale(K.invert(den))

    def solve_den(self, rhs: DomainMatrix[T]) -> Optional[tuple[DomainMatrix[T], T]]:
        """``(X, d)`` with ``A X = d B`` for the right-hand sides ``B``
        (one system per column), or ``None`` when some system has no
        solution. For an underdetermined system the solution with the free
        unknowns zero is returned; the others differ from it by the
        vectors of :meth:`nullspace`.

        >>> from sympy_extras.polys.domains import ZZ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> A = DomainMatrix([[2, 1], [1, 3]], ZZ)
        >>> A.solve_den(DomainMatrix([[3], [5]], ZZ))
        (DomainMatrix([[4], [7]], ZZ), 5)
        >>> DomainMatrix([[1, 1], [2, 2]], ZZ).solve_den(DomainMatrix([[1], [3]], ZZ)) is None
        True
        """
        if rhs.nrows != self.nrows:
            raise ValueError('shapes %s and %s' % (self.shape, rhs.shape))
        K = self.domain
        reduced, den, pivots = self.hstack(rhs).rref_den()
        if any(column >= self.ncols for column in pivots):
            return None
        solution = [[K.zero] * rhs.ncols for _ in range(self.ncols)]
        for i, column in enumerate(pivots):
            solution[column] = reduced.rows[i][self.ncols:]
        return DomainMatrix(solution, K, rhs.ncols), den

    # -- division-free

    def charpoly(self) -> list[T]:
        r"""The coefficients of the characteristic polynomial
        $\det(\lambda I - A)$, highest degree first, by Berkowitz'
        division-free algorithm.

        >>> from sympy_extras.polys.domains import ZZ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> DomainMatrix([[1, 2], [3, 4]], ZZ).charpoly()
        [1, -5, -2]
        """
        n = self._square()
        K = self.domain
        polynomial: list[T] = [K.one]
        for k in range(n):
            last = self.rows[k][:k]
            column = [self.rows[i][k] for i in range(k)]
            toeplitz: list[T] = [K.one, K.neg(self.rows[k][k])]
            vector = column
            for step in range(k):
                total = K.zero
                for a, b in zip(last, vector):
                    total = K.add(total, K.mul(a, b))
                toeplitz.append(K.neg(total))
                if step + 1 < k:
                    updated: list[T] = []
                    for i in range(k):
                        entry = K.zero
                        for a, b in zip(self.rows[i][:k], vector):
                            entry = K.add(entry, K.mul(a, b))
                        updated.append(entry)
                    vector = updated
            product: list[T] = []
            for i in range(k + 2):
                total = K.zero
                for j in range(max(0, i - k - 1), min(i, k) + 1):
                    total = K.add(total, K.mul(toeplitz[i - j], polynomial[j]))
                product.append(total)
            polynomial = product
        return polynomial

    def adjugate(self) -> DomainMatrix[T]:
        """The adjugate (the transpose of the matrix of cofactors):
        ``A adj(A) = det(A) I``.

        >>> from sympy_extras.polys.domains import ZZ
        >>> from sympy_extras.matrices.domainmatrix import DomainMatrix
        >>> DomainMatrix([[1, 2], [3, 4]], ZZ).adjugate(), DomainMatrix([[1, 2], [2, 4]], ZZ).adjugate()
        (DomainMatrix([[4, -2], [-3, 1]], ZZ), DomainMatrix([[4, -2], [-2, 1]], ZZ))
        """
        n = self._square()
        K = self.domain
        if n == 0:
            return self
        determinant = self.det()
        if not K.is_zero(determinant):
            numerator, den = self.inv_den()
            # N/d is the inverse and adj = det * inverse; det = +-d
            if determinant == den:
                return numerator
            return DomainMatrix([[K.exquo(K.mul(a, determinant), den) for a in r] for r in numerator.rows], K, n)
        # Cayley-Hamilton: adj(A) = (-1)**(n-1) (A**(n-1) + c1 A**(n-2) + ... + c_(n-1) I)
        coefficients = self.charpoly()
        identity = DomainMatrix.eye(n, K)
        total = identity
        for c in coefficients[1:n]:
            total = self * total + identity.scale(c)
        return total if n % 2 == 1 else -total
