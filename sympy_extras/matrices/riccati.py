r"""The continuous algebraic Riccati equation, solved exactly, and the
linear-quadratic regulator.

For the system $x' = A x + B u$ and the cost $\int_0^\infty (x^T Q x +
u^T R u)\, dt$ with $Q$ symmetric positive semidefinite and $R$ symmetric
positive definite, the optimal control is the feedback $u = -K x$ with
$K = R^{-1} B^T X$, where $X$ is the *stabilising* solution of

.. math:: A^T X + X A - X B R^{-1} B^T X + Q = 0,

the symmetric solution for which the closed loop matrix $A - B K$ has
all its eigenvalues in the open left half-plane. It exists and is unique
when $(A, B)$ is stabilisable and $(A, Q)$ has no unobservable mode on
the imaginary axis, and it is then positive semidefinite [1]_ [2]_.

The numerical methods (the Schur vectors of the Hamiltonian matrix)
give floating point matrices. Here the data are rational numbers and the
equation is solved as what it is, a system of $n (n + 1)/2$ quadratic
equations in the entries of $X$:

* all its complex solutions are found exactly
  (:func:`sympy_extras.polys.zerodim.solve_zero_dimensional`: a Gröbner
  basis, the change of order and the shape lemma), as algebraic numbers;
* the real ones are the real symmetric solutions
  (:func:`riccati_solutions`);
* for each the criterion of Routh and Hurwitz is applied to the
  characteristic polynomial of the closed loop, first in interval
  arithmetic on rational enclosures of the solution, refined until every
  sign is certain, and in exact arithmetic on algebraic numbers
  (:class:`~sympy_extras.polys.numberfields.qqbar.Algebraic`) if a sign
  stays undecided (a quantity which is exactly zero); the stabilising
  solution is selected with certainty (:func:`care_solve`, :func:`lqr`).

The discrete equation $A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A
+ Q = 0$ of $x_{k+1} = A x_k + B u_k$ is rational in $X$; with the gain
$K = (R + B^T X B)^{-1} B^T X A$ as a second unknown it is the polynomial
system $(R + B^T X B) K = B^T X A$, $X = A^T X (A - B K) + Q$, solved in
the same way (:func:`dare_solve`, :func:`dlqr`); the stabilising solution
has the eigenvalues of $A - B K$ inside the unit circle, which the
Möbius map $z = (s + 1)/(s - 1)$ reduces to the criterion of Routh and
Hurwitz.

The entries of the results are rational numbers, radicals for quadratic
irrationalities, and :class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`
objects (a polynomial and the index of one of its real roots) otherwise.
The cost is that of the polynomial system: state dimensions up to three
or so.

Examples
========

>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.riccati import care_solve, lqr, riccati_solutions
>>> A, B = ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1])        # the double integrator
>>> X = care_solve(A, B, ExprMatrix.eye(2), ExprMatrix.eye(1))
>>> X
ExprMatrix([[sqrt(3), 1], [1, sqrt(3)]])
>>> lqr(A, B, ExprMatrix.eye(2), ExprMatrix.eye(1))[0]
ExprMatrix([[1, sqrt(3)]])
>>> riccati_solutions(ExprMatrix.from_rows([[1]]), ExprMatrix.from_rows([[1]]), ExprMatrix.from_rows([[3]]), ExprMatrix.eye(1))
[ExprMatrix([[-1]]), ExprMatrix([[3]])]

References
==========

.. [1] P. Lancaster, L. Rodman, *Algebraic Riccati Equations*, Oxford
       University Press, 1995, Chapters 7 and 8 (existence and uniqueness
       of the stabilising solution, the set of all solutions).
.. [2] B. D. O. Anderson, J. B. Moore, *Optimal Control: Linear Quadratic
       Methods*, Prentice-Hall, 1990, Chapter 3.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.power import sqrt
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.rings import PolyElement, PolyRing, lex

from .dense import ExprMatrix
from .domainmatrix import DomainMatrix
from .linalg import _square

__all__ = ['riccati_solutions', 'care_solve', 'lqr', 'closed_loop_is_stable', 'discrete_riccati_solutions', 'dare_solve',
           'dlqr']

_Rows = list[list[Fraction]]


def _rational_rows(M: ExprMatrix, name: str) -> _Rows:
    rows: _Rows = []
    for i in range(M.rows):
        line: list[Fraction] = []
        for e in M.row(i):
            if not isinstance(e, Rational):
                raise ValueError('the entries of %s must be rational numbers: %s' % (name, e))
            line.append(Fraction(e.p, e.q))
        rows.append(line)
    return rows


def _product(a: _Rows, b: _Rows) -> _Rows:
    columns = list(zip(*b)) if b else []
    return [[sum((p*q for p, q in zip(row, column)), Fraction(0)) for column in columns] for row in a]


def _transpose(a: _Rows) -> _Rows:
    return [list(column) for column in zip(*a)]


class _Problem:
    """The rational data: ``A``, ``Q`` and ``S = B R**-1 B^T``, with the
    gain map ``G = R**-1 B^T``."""

    def __init__(self, A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> None:
        n = _square(A)
        m = _square(R)
        if B.shape != (n, m) or Q.shape != (n, n):
            raise ValueError('shapes: A %s, B %s, Q %s, R %s' % (A.shape, B.shape, Q.shape, R.shape))
        self.n: int = n
        self.A: _Rows = _rational_rows(A, 'A')
        self.Q: _Rows = _rational_rows(Q, 'Q')
        b, r = _rational_rows(B, 'B'), _rational_rows(R, 'R')
        if self.Q != _transpose(self.Q) or r != _transpose(r):
            raise ValueError('Q and R must be symmetric')
        try:
            inverse = DomainMatrix(r, QQ, m).inv().rows
        except ValueError:
            raise ValueError('R must be invertible') from None
        self.G: _Rows = _product(inverse, _transpose(b))
        self.S: _Rows = _product(b, self.G)

    def solutions(self) -> list[list[list[Algebraic]]]:
        """The real symmetric solutions, as matrices of algebraic
        numbers."""
        n = self.n
        if n == 0:
            return [[]]
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
        ring = PolyRing(QQ, ['x_%d_%d' % pair for pair in pairs], lex)
        unknown = dict(zip(pairs, ring.gens))
        X = [[unknown[min(i, j), max(i, j)] for j in range(n)] for i in range(n)]
        zero = ring.zero

        def left(c: _Rows, M: list[list[PolyElement[Fraction]]]) -> list[list[PolyElement[Fraction]]]:
            # the rational matrix c times the polynomial matrix M
            result: list[list[PolyElement[Fraction]]] = []
            for row in c:
                line: list[PolyElement[Fraction]] = []
                for j in range(n):
                    total = zero
                    for k, coefficient in enumerate(row):
                        if coefficient:
                            total = total + M[k][j].mul_ground(coefficient)
                    line.append(total)
                result.append(line)
            return result

        AtX = left(_transpose(self.A), X)
        SX = left(self.S, X)
        equations: list[PolyElement[Fraction]] = []
        for i, j in pairs:
            # (A^T X)_ij + (A^T X)_ji - (X S X)_ij + Q_ij
            total = AtX[i][j] + AtX[j][i] + ring.ground(self.Q[i][j])
            for k in range(n):
                total = total - X[i][k]*SX[k][j]
            equations.append(total)
        from sympy_extras.polys.zerodim import solve_zero_dimensional
        found: list[list[list[Algebraic]]] = []
        for point in solve_zero_dimensional(equations, ring):
            if all(value.is_real for value in point):
                values = dict(zip(pairs, point))
                found.append([[values[min(i, j), max(i, j)] for j in range(n)] for i in range(n)])
        return found

    def closed_loop(self, X: Sequence[Sequence[Algebraic]]) -> list[list[Algebraic]]:
        """``A - S X``."""
        n = self.n
        result: list[list[Algebraic]] = []
        for i in range(n):
            line: list[Algebraic] = []
            for j in range(n):
                total = Algebraic.from_rational(self.A[i][j])
                for k in range(n):
                    if self.S[i][k]:
                        total = total - X[k][j]*self.S[i][k]
                line.append(total)
            result.append(line)
        return result


def _characteristic(M: Sequence[Sequence[Algebraic]]) -> list[Algebraic]:
    """The coefficients of ``det(x I - M)``, highest degree first
    (Faddeev and LeVerrier)."""
    n = len(M)
    one, zero = Algebraic.from_rational(1), Algebraic.from_rational(0)
    coefficients = [one]
    current = [[one if i == j else zero for j in range(n)] for i in range(n)]
    for k in range(1, n + 1):
        product = [[sum((M[i][l]*current[l][j] for l in range(n)), zero) for j in range(n)] for i in range(n)]
        c = -sum((product[i][i] for i in range(n)), zero)/k
        coefficients.append(c)
        current = [[product[i][j] + (c if i == j else zero) for j in range(n)] for i in range(n)]
    return coefficients


def _hurwitz(coefficients: Sequence[Algebraic]) -> bool:
    """Whether every root of the monic polynomial has a negative real
    part: the first column of Routh's array is positive."""
    zero = Algebraic.from_rational(0)
    degree = len(coefficients) - 1
    first = list(coefficients[0::2])
    second = list(coefficients[1::2])
    for _ in range(degree):
        if not second or second[0].sign() <= 0:
            return False
        ratio = first[0]/second[0]
        following: list[Algebraic] = []
        for k in range(1, len(first)):
            below = second[k] if k < len(second) else zero
            following.append(first[k] - ratio*below)
        first, second = second, following
    return True


_Interval = tuple[Fraction, Fraction]


def _i_add(a: _Interval, b: _Interval) -> _Interval:
    return a[0] + b[0], a[1] + b[1]


def _i_sub(a: _Interval, b: _Interval) -> _Interval:
    return a[0] - b[1], a[1] - b[0]


def _i_mul(a: _Interval, b: _Interval) -> _Interval:
    products = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return min(products), max(products)


def _i_scale(a: _Interval, c: Fraction) -> _Interval:
    return (a[0]*c, a[1]*c) if c >= 0 else (a[1]*c, a[0]*c)


def _interval_characteristic(closed: Sequence[Sequence[_Interval]]) -> list[_Interval]:
    """Enclosures of the coefficients of the characteristic polynomial
    (Faddeev and LeVerrier)."""
    n = len(closed)
    zero: _Interval = (Fraction(0), Fraction(0))
    one: _Interval = (Fraction(1), Fraction(1))
    coefficients: list[_Interval] = [one]
    current = [[one if i == j else zero for j in range(n)] for i in range(n)]
    for k in range(1, n + 1):
        product: list[list[_Interval]] = []
        for i in range(n):
            row: list[_Interval] = []
            for j in range(n):
                total = zero
                for l in range(n):
                    total = _i_add(total, _i_mul(closed[i][l], current[l][j]))
                row.append(total)
            product.append(row)
        trace = zero
        for i in range(n):
            trace = _i_add(trace, product[i][i])
        c = _i_scale(trace, Fraction(-1, k))
        coefficients.append(c)
        current = [[_i_add(product[i][j], c) if i == j else product[i][j] for j in range(n)] for i in range(n)]
    return coefficients


def _interval_hurwitz(coefficients: Sequence[_Interval]) -> Optional[bool]:
    """Routh's test on enclosures of the coefficients (any sign of the
    leading one): ``None`` when a sign is not certain."""
    zero: _Interval = (Fraction(0), Fraction(0))
    leading = coefficients[0]
    if leading == zero:
        return False
    if leading[0] <= 0 <= leading[1]:
        return None
    if leading[1] < 0:
        coefficients = [(-c[1], -c[0]) for c in coefficients]
    first, second = list(coefficients[0::2]), list(coefficients[1::2])
    for _ in range(len(coefficients) - 1):
        if not second:
            return False
        pivot = second[0]
        if pivot[1] < 0 or pivot == zero:
            return False
        if pivot[0] <= 0:
            return None
        inverse: _Interval = (1/pivot[1], 1/pivot[0])
        ratio = _i_mul(first[0], inverse)
        following: list[_Interval] = []
        for k in range(1, len(first)):
            below = second[k] if k < len(second) else zero
            following.append(_i_sub(first[k], _i_mul(ratio, below)))
        first, second = second, following
    return True


def _moebius_weights(n: int) -> list[list[int]]:
    """Row ``k``: the coefficients (highest degree first) of ``(s +
    1)**(n - k) (s - 1)**k``."""
    rows: list[list[int]] = []
    for k in range(n + 1):
        polynomial = [1]
        for factor in [1]*(n - k) + [-1]*k:
            polynomial = [a + factor*b for a, b in zip(polynomial + [0], [0] + polynomial)]
        rows.append(polynomial)
    return rows


def _interval_moebius(coefficients: Sequence[_Interval]) -> list[_Interval]:
    n = len(coefficients) - 1
    weights = _moebius_weights(n)
    result: list[_Interval] = []
    for j in range(n + 1):
        total: _Interval = (Fraction(0), Fraction(0))
        for k in range(n + 1):
            total = _i_add(total, _i_scale(coefficients[k], Fraction(weights[k][j])))
        result.append(total)
    return result


def _stable_by_intervals(problem: _Problem, X: Sequence[Sequence[Algebraic]], eps: Fraction) -> Optional[bool]:
    """The criterion of Routh and Hurwitz on rational enclosures of
    width ``eps`` of the solution: ``True`` or ``False`` when every sign
    met is certain, ``None`` otherwise."""
    n = problem.n
    boxes = [[X[i][j].enclosure(eps)[:2] for j in range(n)] for i in range(n)]
    closed: list[list[_Interval]] = []
    for i in range(n):
        line: list[_Interval] = []
        for j in range(n):
            total: _Interval = (problem.A[i][j], problem.A[i][j])
            for k in range(n):
                if problem.S[i][k]:
                    total = _i_sub(total, _i_scale(boxes[k][j], problem.S[i][k]))
            line.append(total)
        closed.append(line)
    return _interval_hurwitz(_interval_characteristic(closed))


def _stabilising(problem: _Problem, X: Sequence[Sequence[Algebraic]]) -> bool:
    """Whether the closed loop of the solution is stable: by interval
    arithmetic on finer and finer enclosures, and by exact arithmetic on
    algebraic numbers when a sign stays undecided (a quantity which is
    zero)."""
    for bits in (32, 96, 256):
        decided = _stable_by_intervals(problem, X, Fraction(1, 2**bits))
        if decided is not None:
            return decided
    return _hurwitz(_characteristic(problem.closed_loop(X)))


def _expression(a: Algebraic) -> Expr:
    """A rational number, a quadratic irrationality in radicals, or an
    :class:`IndexedRoot`."""
    if a.is_rational:
        return number_from_fraction(a.rational_value())
    minpoly = a.minpoly
    if len(minpoly) == 3:
        p, q, r = minpoly
        root = sqrt(Integer(q*q - 4*p*r))
        center = Fraction(-q, 2*p)
        larger = a._compare(center) > 0
        return (-q + root)/(2*p) if larger == (p > 0) else (-q - root)/(2*p)
    from sympy_extras.polys.cad.indexedroot import real_root
    return real_root(a, 'x')


def _matrix(X: Sequence[Sequence[Algebraic]]) -> ExprMatrix:
    n = len(X)
    return ExprMatrix(n, n, [_expression(a) for row in X for a in row])


class _DiscreteProblem:
    """``A^T X A - X - A^T X B (R + B^T X B)**-1 B^T X A + Q = 0`` with
    the gain ``K = (R + B^T X B)**-1 B^T X A`` as an unknown: the
    polynomial system ``(R + B^T X B) K = B^T X A``, ``X = A^T X (A - B
    K) + Q``."""

    def __init__(self, A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> None:
        n = _square(A)
        m = _square(R)
        if B.shape != (n, m) or Q.shape != (n, n):
            raise ValueError('shapes: A %s, B %s, Q %s, R %s' % (A.shape, B.shape, Q.shape, R.shape))
        self.n: int = n
        self.m: int = m
        self.A: _Rows = _rational_rows(A, 'A')
        self.B: _Rows = _rational_rows(B, 'B')
        self.Q: _Rows = _rational_rows(Q, 'Q')
        self.R: _Rows = _rational_rows(R, 'R')
        if self.Q != _transpose(self.Q) or self.R != _transpose(self.R):
            raise ValueError('Q and R must be symmetric')

    def solutions(self) -> list[tuple[list[list[Algebraic]], list[list[Algebraic]]]]:
        """The real solutions ``(X, K)``."""
        n, m = self.n, self.m
        if n == 0:
            return [([], [[] for _ in range(m)])]
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
        gains = [(i, j) for i in range(m) for j in range(n)]
        ring = PolyRing(QQ, ['x_%d_%d' % pair for pair in pairs] + ['k_%d_%d' % pair for pair in gains], lex)
        unknown = dict(zip(pairs, ring.gens[:len(pairs)]))
        X = [[unknown[min(i, j), max(i, j)] for j in range(n)] for i in range(n)]
        K = [[ring.gens[len(pairs) + i*n + j] for j in range(n)] for i in range(m)]
        zero = ring.zero

        def constant(c: _Rows) -> list[list[PolyElement[Fraction]]]:
            return [[ring.ground(v) for v in row] for row in c]

        def times(a: list[list[PolyElement[Fraction]]], b: list[list[PolyElement[Fraction]]]
                  ) -> list[list[PolyElement[Fraction]]]:
            columns = len(b[0]) if b else 0
            result: list[list[PolyElement[Fraction]]] = []
            for row in a:
                line: list[PolyElement[Fraction]] = []
                for j in range(columns):
                    total = zero
                    for l, entry in enumerate(row):
                        if entry.terms and b[l][j].terms:
                            total = total + entry*b[l][j]
                    line.append(total)
                result.append(line)
            return result

        A, B = constant(self.A), constant(self.B)
        At, Bt = constant(_transpose(self.A)), constant(_transpose(self.B))
        BtX = times(Bt, X)
        AtX = times(At, X)
        gain_left = times(BtX, B)                 # B^T X B
        gain_right = times(BtX, A)                # B^T X A
        gain_product = times(gain_left, K)
        RK = times(constant(self.R), K)
        equations: list[PolyElement[Fraction]] = []
        for i in range(m):
            for j in range(n):
                equations.append(RK[i][j] + gain_product[i][j] - gain_right[i][j])
        BK = times(B, K)
        closed = [[A[i][j] - BK[i][j] for j in range(n)] for i in range(n)]
        update = times(AtX, closed)
        for i in range(n):
            for j in range(n):
                equations.append(update[i][j] + ring.ground(self.Q[i][j]) - X[i][j])
        from sympy_extras.polys.zerodim import solve_zero_dimensional
        found: list[tuple[list[list[Algebraic]], list[list[Algebraic]]]] = []
        for point in solve_zero_dimensional(equations, ring):
            if all(value.is_real for value in point):
                values = dict(zip(pairs, point[:len(pairs)]))
                solution = [[values[min(i, j), max(i, j)] for j in range(n)] for i in range(n)]
                gain = [[point[len(pairs) + i*n + j] for j in range(n)] for i in range(m)]
                found.append((solution, gain))
        return found

    def stabilising(self, K: Sequence[Sequence[Algebraic]]) -> bool:
        """Whether ``A - B K`` has its eigenvalues inside the unit
        circle."""
        n, m = self.n, self.m
        for bits in (32, 96, 256):
            eps = Fraction(1, 2**bits)
            boxes = [[K[i][j].enclosure(eps)[:2] for j in range(n)] for i in range(m)]
            closed: list[list[_Interval]] = []
            for i in range(n):
                line: list[_Interval] = []
                for j in range(n):
                    total: _Interval = (self.A[i][j], self.A[i][j])
                    for l in range(m):
                        if self.B[i][l]:
                            total = _i_sub(total, _i_scale(boxes[l][j], self.B[i][l]))
                    line.append(total)
                closed.append(line)
            decided = _interval_hurwitz(_interval_moebius(_interval_characteristic(closed)))
            if decided is not None:
                return decided
        # exactly: the Moebius transform of the characteristic polynomial in algebraic numbers
        exact = [[Algebraic.from_rational(self.A[i][j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                for l in range(m):
                    if self.B[i][l]:
                        exact[i][j] = exact[i][j] - K[l][j]*self.B[i][l]
        coefficients = _characteristic(exact)
        weights = _moebius_weights(n)
        transformed: list[Algebraic] = []
        for j in range(n + 1):
            total_exact = Algebraic.from_rational(0)
            for k in range(n + 1):
                if weights[k][j]:
                    total_exact = total_exact + coefficients[k]*weights[k][j]
            transformed.append(total_exact)
        sign = transformed[0].sign()
        if sign == 0:
            return False
        return _hurwitz([c/transformed[0] for c in transformed])


def riccati_solutions(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> list[ExprMatrix]:
    """All the real symmetric solutions ``X`` of ``A^T X + X A - X B
    R**-1 B^T X + Q = 0`` for matrices of rational numbers; ``ValueError``
    when there are infinitely many.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.riccati import riccati_solutions
    >>> A, B = ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1])
    >>> for X in riccati_solutions(A, B, ExprMatrix.eye(2), ExprMatrix.eye(1)):
    ...     print(X)
    ExprMatrix([[-sqrt(3), 1], [1, -sqrt(3)]])
    ExprMatrix([[sqrt(3), 1], [1, sqrt(3)]])
    """
    return [_matrix(X) for X in _Problem(A, B, Q, R).solutions()]


def care_solve(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> Optional[ExprMatrix]:
    """The stabilising solution of the continuous algebraic Riccati
    equation (every eigenvalue of ``A - B R**-1 B^T X`` has a negative
    real part), or ``None`` when there is none.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.riccati import care_solve
    >>> one = ExprMatrix.eye(1)
    >>> care_solve(ExprMatrix.from_rows([[1]]), one, ExprMatrix.from_rows([[3]]), one)
    ExprMatrix([[3]])
    >>> care_solve(ExprMatrix.from_rows([[1]]), ExprMatrix.zeros(1, 1), one, one) is None      # not stabilisable
    True
    """
    problem = _Problem(A, B, Q, R)
    for X in problem.solutions():
        if _stabilising(problem, X):
            return _matrix(X)
    return None


def lqr(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
    """``(K, X)``: the gain of the linear-quadratic regulator ``u = -K
    x`` and the stabilising solution of the Riccati equation, with ``K =
    R**-1 B^T X``; ``ValueError`` when no stabilising solution exists.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.riccati import lqr
    >>> K, X = lqr(ExprMatrix.from_rows([[0]]), ExprMatrix.eye(1), ExprMatrix.from_rows([[4]]), ExprMatrix.eye(1))
    >>> K, X
    (ExprMatrix([[2]]), ExprMatrix([[2]]))
    """
    problem = _Problem(A, B, Q, R)
    for X in problem.solutions():
        if _stabilising(problem, X):
            n = problem.n
            gain: list[Expr] = []
            for row in problem.G:
                for j in range(n):
                    total = Algebraic.from_rational(0)
                    for k in range(n):
                        if row[k]:
                            total = total + X[k][j]*row[k]
                    gain.append(_expression(total))
            return ExprMatrix(len(problem.G), n, gain), _matrix(X)
    raise ValueError('the Riccati equation has no stabilising solution')


def closed_loop_is_stable(A: ExprMatrix, B: ExprMatrix, K: ExprMatrix) -> bool:
    """Whether ``A - B K`` (rational matrices) has all its eigenvalues in
    the open left half-plane, by the criterion of Routh and Hurwitz.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.riccati import closed_loop_is_stable
    >>> A, B = ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1])
    >>> closed_loop_is_stable(A, B, ExprMatrix.from_rows([[1, 2]])), closed_loop_is_stable(A, B, ExprMatrix.from_rows([[1, 0]]))
    (True, False)
    """
    a, b, k = _rational_rows(A, 'A'), _rational_rows(B, 'B'), _rational_rows(K, 'K')
    bk = _product(b, k)
    closed = [[Algebraic.from_rational(p - q) for p, q in zip(row, other)] for row, other in zip(a, bk)]
    return _hurwitz(_characteristic(closed))


def discrete_riccati_solutions(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> list[ExprMatrix]:
    """All the real symmetric solutions ``X`` of the discrete algebraic
    Riccati equation ``A^T X A - X - A^T X B (R + B^T X B)**-1 B^T X A + Q
    = 0`` for matrices of rational numbers.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.riccati import discrete_riccati_solutions
    >>> one = ExprMatrix.eye(1)
    >>> discrete_riccati_solutions(one, one, one, one)
    [ExprMatrix([[1/2 - sqrt(5)/2]]), ExprMatrix([[1/2 + sqrt(5)/2]])]
    """
    return [_matrix(X) for X, _ in _DiscreteProblem(A, B, Q, R).solutions()]


def dare_solve(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> Optional[ExprMatrix]:
    """The stabilising solution of the discrete algebraic Riccati
    equation (every eigenvalue of ``A - B K``, ``K = (R + B^T X B)**-1 B^T
    X A``, is inside the unit circle), or ``None`` when there is none.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.riccati import dare_solve
    >>> one = ExprMatrix.eye(1)
    >>> dare_solve(one, one, one, one)                 # the golden ratio
    ExprMatrix([[1/2 + sqrt(5)/2]])
    """
    problem = _DiscreteProblem(A, B, Q, R)
    for X, K in problem.solutions():
        if problem.stabilising(K):
            return _matrix(X)
    return None


def dlqr(A: ExprMatrix, B: ExprMatrix, Q: ExprMatrix, R: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
    """``(K, X)``: the gain of the discrete linear-quadratic regulator
    ``u_k = -K x_k`` for ``x_(k+1) = A x_k + B u_k`` and the cost ``sum
    x^T Q x + u^T R u``, with the stabilising solution of the Riccati
    equation; ``ValueError`` when there is none.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.riccati import dlqr
    >>> one = ExprMatrix.eye(1)
    >>> dlqr(one, one, one, one)[0]
    ExprMatrix([[-1/2 + sqrt(5)/2]])
    """
    problem = _DiscreteProblem(A, B, Q, R)
    for X, K in problem.solutions():
        if problem.stabilising(K):
            gain = ExprMatrix(problem.m, problem.n, [_expression(a) for row in K for a in row])
            return gain, _matrix(X)
    raise ValueError('the discrete Riccati equation has no stabilising solution')
