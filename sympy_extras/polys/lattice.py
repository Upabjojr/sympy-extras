r"""Lattice basis reduction over the integers: the LLL algorithm in its
all-integer form, and the Hermite normal form.

A lattice is given by integer row vectors $b_1, \dots, b_n$. With the
Gram–Schmidt vectors $b_i^* = b_i - \sum_{j<i} \mu_{ij} b_j^*$, the basis
is *LLL reduced* with parameter $\delta \in (1/4, 1]$ when $|\mu_{ij}|
\le 1/2$ (size reduction) and $\|b_i^*\|^2 \ge (\delta - \mu_{i,i-1}^2)
\|b_{i-1}^*\|^2$ (Lovász). Then $\|b_1\| \le 2^{(n-1)/2} \lambda_1$ for
$\delta = 3/4$, and, what the applications use most: a lattice vector
which is not in the span of $b_1, \dots, b_t$ is at least as long as
$\min_{j > t} \|b_j^*\|$.

The implementation is de Weger's integral version (Cohen, Algorithm
2.6.7): with $d_i = \det(b_j \cdot b_k)_{j,k \le i}$ (the Gram
determinants, $d_0 = 1$), the numbers $d_i$ and $\lambda_{ij} = d_j
\mu_{ij}$ are integers, $\|b_i^*\|^2 = d_i/d_{i-1}$, and every division
of the algorithm is exact: no fractions, no floating point, hence no
limit on the size of the entries. Linearly dependent generators are
replaced first by a basis (the Hermite normal form, by Euclidean row
operations).

Examples
========

>>> from sympy_extras.polys.lattice import lll, hermite_normal_form, lll_reduction
>>> lll([[1, 0, 3], [0, 1, 5], [0, 0, 7]])
[[-1, -1, -1], [-1, 2, 0], [-2, 0, 1]]
>>> lll([[2, 0], [0, 2], [1, 1]])
[[1, 1], [-1, 1]]
>>> hermite_normal_form([[2, 0], [0, 2], [1, 1]])
[[1, 1], [0, 2]]
>>> reduction = lll_reduction([[1, 0, 1000], [0, 1, 1001]])
>>> reduction.basis
[[-1, 1, 1], [334, -333, 667]]
>>> reduction.norms_squared() == [3, reduction.gram[2]/3]
True

References
==========

.. [1] A. K. Lenstra, H. W. Lenstra, L. Lovász, *Factoring polynomials
       with rational coefficients*, Mathematische Annalen 261 (1982),
       515-534.
.. [2] H. Cohen, *A Course in Computational Algebraic Number Theory*,
       Springer, 1993, Sections 2.4.2 (Hermite normal form) and 2.6
       (Algorithm 2.6.7, the integral LLL algorithm).
.. [3] B. M. M. de Weger, *Algorithms for Diophantine equations*, CWI
       Tract 65, 1989, Section 3.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Sequence

__all__ = ['lll', 'lll_reduction', 'LatticeReduction', 'hermite_normal_form', 'is_lll_reduced']


class LatticeReduction:
    """An LLL reduced basis with its Gram determinants.

    ``basis`` are the reduced row vectors, ``gram`` the determinants
    ``d_0 = 1, d_1, ..., d_n`` of the leading Gram matrices, so that the
    squared length of the ``i``-th Gram–Schmidt vector (from 0) is
    ``gram[i + 1]/gram[i]``.
    """

    def __init__(self, basis: list[list[int]], gram: list[int]) -> None:
        self.basis: Final[list[list[int]]] = basis
        self.gram: Final[list[int]] = gram

    def norms_squared(self) -> list[Fraction]:
        """The squared lengths of the Gram–Schmidt vectors."""
        return [Fraction(self.gram[i + 1], self.gram[i]) for i in range(len(self.basis))]

    def short_part(self, bound_squared: Fraction) -> list[list[int]]:
        """The first vectors ``b_1, ..., b_t`` of the basis, ``t`` the
        smallest index such that every later Gram–Schmidt vector is
        longer than the bound: every lattice vector ``v`` with ``|v|**2
        <= bound_squared`` is an integer combination of them."""
        t = len(self.basis)
        while t > 0 and self.gram[t] > bound_squared*self.gram[t - 1]:
            t -= 1
        return self.basis[:t]


def _dot(u: Sequence[int], v: Sequence[int]) -> int:
    total = 0
    for a, c in zip(u, v):
        total += a*c
    return total


def hermite_normal_form(rows: Sequence[Sequence[int]]) -> list[list[int]]:
    """The Hermite normal form of the lattice spanned by the integer row
    vectors: the basis in row echelon form with positive pivots and, above
    each pivot, entries in ``[0, pivot)``. It depends on the lattice only,
    so that two families span the same lattice exactly when their normal
    forms are equal.

    Examples
    ========

    >>> from sympy_extras.polys.lattice import hermite_normal_form
    >>> hermite_normal_form([[4, 6], [6, 4]])
    [[2, 8], [0, 10]]
    >>> hermite_normal_form([[1, 2, 3], [2, 4, 6]])
    [[1, 2, 3]]
    """
    work = [[int(v) for v in row] for row in rows]
    width = len(work[0]) if work else 0
    for row in work:
        if len(row) != width:
            raise ValueError('the vectors must have the same length')
    result: list[list[int]] = []
    pivots: list[int] = []
    for column in range(width):
        # Euclid's algorithm on the column: one row keeps the gcd
        while True:
            candidates = [row for row in work if row[column]]
            if len(candidates) <= 1:
                break
            smallest = min(candidates, key=lambda row: abs(row[column]))
            for row in candidates:
                if row is smallest:
                    continue
                q = row[column]//smallest[column]
                if q:
                    for j in range(column, width):
                        row[j] -= q*smallest[j]
        pivot_rows = [row for row in work if row[column]]
        if not pivot_rows:
            continue
        pivot = pivot_rows[0]
        work = [row for row in work if row is not pivot]
        if pivot[column] < 0:
            pivot = [-v for v in pivot]
        for previous in result:
            q = previous[column]//pivot[column]
            if q:
                for j in range(column, width):
                    previous[j] -= q*pivot[j]
        result.append(pivot)
        pivots.append(column)
    return result


def _integral_lll(b: list[list[int]], num: int, den: int) -> tuple[bool, list[int]]:
    """de Weger's integral LLL on the rows ``b`` (in place), ``delta =
    num/den``. Returns ``(False, [])`` when the rows are linearly
    dependent, ``(True, gram)`` otherwise."""
    n = len(b)
    # d[i + 1] is the Gram determinant d_i of the text, d[0] = 1
    d: list[int] = [1]*(n + 1)
    lam: list[list[int]] = [[0]*n for _ in range(n)]
    if n == 0:
        return True, d
    d[1] = _dot(b[0], b[0])
    if d[1] == 0:
        return False, []
    k = 1
    kmax = 0
    while k < n:
        if k > kmax:
            kmax = k
            bk = b[k]
            lamk = lam[k]
            for j in range(k + 1):
                u = _dot(bk, b[j])
                lamj = lam[j]
                for i in range(j):
                    u = (d[i + 1]*u - lamk[i]*lamj[i])//d[i]
                if j < k:
                    lamk[j] = u
                else:
                    d[k + 1] = u
            if d[k + 1] == 0:
                return False, []
        _reduce(b, lam, d, k, k - 1)
        l = lam[k][k - 1]
        if den*(d[k + 1]*d[k - 1] + l*l) < num*d[k]*d[k]:
            _swap(b, lam, d, k, kmax)
            k = max(1, k - 1)
        else:
            for j in range(k - 2, -1, -1):
                _reduce(b, lam, d, k, j)
            k += 1
    return True, d


def _reduce(b: list[list[int]], lam: list[list[int]], d: list[int], k: int, l: int) -> None:
    """Size reduction of ``b_k`` against ``b_l``."""
    dl = d[l + 1]
    value = lam[k][l]
    if 2*abs(value) <= dl:
        return
    q = (2*value + dl)//(2*dl)
    bl = b[l]
    b[k] = [a - q*c for a, c in zip(b[k], bl)]
    lamk, laml = lam[k], lam[l]
    lamk[l] = value - q*dl
    for i in range(l):
        lamk[i] -= q*laml[i]


def _swap(b: list[list[int]], lam: list[list[int]], d: list[int], k: int, kmax: int) -> None:
    """Exchange ``b_k`` and ``b_{k-1}`` and update the Gram data."""
    b[k], b[k - 1] = b[k - 1], b[k]
    lamk, lamk1 = lam[k], lam[k - 1]
    for j in range(k - 1):
        lamk[j], lamk1[j] = lamk1[j], lamk[j]
    l = lamk[k - 1]
    dk1 = d[k]
    dk = d[k + 1]
    new = (d[k - 1]*dk + l*l)//dk1
    for i in range(k + 1, kmax + 1):
        t = lam[i][k]
        lam[i][k] = (dk*lam[i][k - 1] - l*t)//dk1
        lam[i][k - 1] = (new*t + l*lam[i][k])//dk
    d[k] = new


def lll_reduction(rows: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4)) -> LatticeReduction:
    """An LLL reduced basis of the lattice spanned by the integer row
    vectors, with its Gram determinants.

    Parameters
    ==========

    rows : the generators; they need not be independent (a basis is
        extracted first), zero vectors are ignored.
    delta : the parameter of the Lovász condition, ``1/4 < delta <= 1``.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.lattice import lll_reduction
    >>> reduction = lll_reduction([[1, 1, 1], [-1, 0, 2], [3, 5, 6]])
    >>> reduction.basis
    [[0, 1, 0], [1, 0, 1], [-1, 0, 2]]
    >>> reduction.gram
    [1, 1, 2, 9]
    >>> reduction.short_part(Fraction(2))
    [[0, 1, 0], [1, 0, 1]]
    """
    if not Fraction(1, 4) < delta <= 1:
        raise ValueError('delta must be in (1/4, 1]')
    b = [[int(v) for v in row] for row in rows if any(row)]
    for row in b:
        if len(row) != len(b[0]):
            raise ValueError('the vectors must have the same length')
    num, den = delta.numerator, delta.denominator
    independent, gram = _integral_lll(b, num, den)
    if not independent:
        b = hermite_normal_form(rows)
        independent, gram = _integral_lll(b, num, den)
        if not independent:
            raise ArithmeticError('the Hermite normal form is not a basis')
    return LatticeReduction(b, gram)


def lll(rows: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4)) -> list[list[int]]:
    """An LLL reduced basis of the lattice spanned by the integer row
    vectors (exact integer arithmetic, entries of any size).

    Examples
    ========

    >>> from sympy_extras.polys.lattice import lll
    >>> lll([[1, 1, 1], [-1, 0, 2], [3, 5, 6]])
    [[0, 1, 0], [1, 0, 1], [-1, 0, 2]]

    An integer relation between ``1``, ``sqrt(2)`` and ``sqrt(8)``
    (scaled by ``10**6`` and rounded) is the short vector:

    >>> lll([[1, 0, 0, 1000000], [0, 1, 0, 1414214], [0, 0, 1, 2828427]])[0]
    [0, -2, 1, -1]
    """
    return lll_reduction(rows, delta).basis


def is_lll_reduced(rows: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4)) -> bool:
    """Whether the rows are independent and LLL reduced with the
    parameter ``delta`` (checked from the definition, with rational
    Gram–Schmidt coefficients).

    Examples
    ========

    >>> from sympy_extras.polys.lattice import is_lll_reduced
    >>> is_lll_reduced([[1, 0], [0, 1]])
    True
    >>> is_lll_reduced([[1, 0], [5, 1]])
    False
    """
    star: list[list[Fraction]] = []
    norms: list[Fraction] = []
    previous_mu = Fraction(0)
    for i, row in enumerate(rows):
        v = [Fraction(c) for c in row]
        current = list(v)
        for j in range(i):
            mu = sum((a*c for a, c in zip(v, star[j])), Fraction(0))/norms[j]
            if abs(mu) > Fraction(1, 2):
                return False
            current = [a - mu*c for a, c in zip(current, star[j])]
            if j == i - 1:
                previous_mu = mu
        norm = sum((a*a for a in current), Fraction(0))
        if norm == 0:
            return False
        if i and norm < (delta - previous_mu*previous_mu)*norms[i - 1]:
            return False
        star.append(current)
        norms.append(norm)
    return True
