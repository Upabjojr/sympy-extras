r"""Linear algebra on matrices of expressions: determinant, inverse, rank,
row reduction, null space, linear systems, characteristic polynomial.

The entries are read as rational functions with integer coefficients in
their *kernels*: the symbols and every other subexpression which is not a
sum, a product or an integer power (``sin(t)``, ``sqrt(2)``, ``I``). The
matrix is scaled to one over the polynomial ring $\mathbb{Z}[k_1, \dots,
k_m]$ (over $\mathbb{Z}$ itself for a matrix of numbers), and the
fraction-free algorithms of :mod:`sympy_extras.matrices.domainmatrix`
run there: no rational function is ever added to another one, which is
what makes elimination on expressions slow, and the results come out as
quotients of expanded polynomials in lowest terms.

The determinant, the adjugate and the characteristic polynomial are
polynomial identities in the entries, so they hold whatever relations
there are between the kernels (``sin(t)**2 + cos(t)**2 = 1``). The rank,
the row echelon form, the null space, the inverse and the solutions of
systems depend on which quantities are zero. With symbols only a nonzero
polynomial is a nonzero quantity, and the answer is the generic one (for
the values of the symbols at which no pivot vanishes; see
:func:`sympy_extras.assumptions.resolve` for case distinctions). With
other kernels each pivot of the elimination is submitted to the zero test
of :mod:`sympy_extras.simplify.structure`, and ``NotImplementedError`` is
raised unless all are proved nonzero.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sin, cos
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.linalg import det, inverse, rank, nullspace, charpoly, linear_solve
>>> a, b, c, d, x, t = (Symbol(s) for s in 'abcdxt')
>>> M = ExprMatrix.from_rows([[a, b], [c, d]])
>>> det(M), inverse(M)
(a*d - b*c, ExprMatrix([[d/(a*d - b*c), -b/(a*d - b*c)], [-c/(a*d - b*c), a/(a*d - b*c)]]))
>>> charpoly(M, x)
a*d - b*c + x**2 + x*(-a - d)
>>> V = ExprMatrix.from_rows([[1, a, a**2], [1, b, b**2], [1, c, c**2]])
>>> det(V)
-a**2*b + a**2*c + a*b**2 - a*c**2 - b**2*c + b*c**2
>>> S = ExprMatrix.from_rows([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> rank(S), nullspace(S)
(2, [ExprMatrix([[1], [-2], [1]])])
>>> R = ExprMatrix.from_rows([[cos(t), -sin(t)], [sin(t), cos(t)]])
>>> det(R)
sin(t)**2 + cos(t)**2
>>> linear_solve(ExprMatrix.from_rows([[a, 1], [1, a]]), ExprMatrix.column([1, 1])).particular
ExprMatrix([[1/(a + 1)], [1/(a + 1)]])

References
==========

.. [1] E. H. Bareiss, *Sylvester's identity and multistep
       integer-preserving Gaussian elimination*, Mathematics of
       Computation 22 (1968), 565-578.
.. [2] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Chapter 9 (solving systems of equations).
"""
from __future__ import annotations

from math import gcd
from typing import Final, Generic, Optional, Sequence, TypeVar

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing
from sympy_extras.polys.rings import PolyElement

from .dense import ExprMatrix
from .domainmatrix import DomainMatrix

__all__ = ['normal', 'det', 'adjugate', 'inverse', 'rank', 'rref', 'nullspace', 'linear_solve', 'LinearSolution', 'charpoly',
           'charpoly_coefficients']

T = TypeVar('T')


# ---------------------------------------------------------------------
# from expressions to a ring and back

def _is_kernel(e: Expr) -> bool:
    if isinstance(e, (Rational, Add, Mul)):
        return False
    return not (isinstance(e, Pow) and isinstance(e.exp, Integer))


def _collect_kernels(e: Expr, found: dict[Expr, int]) -> None:
    if isinstance(e, Rational):
        return
    if isinstance(e, (Add, Mul)):
        for a in e.args:
            if isinstance(a, Expr):
                _collect_kernels(a, found)
        return
    if isinstance(e, Pow) and isinstance(e.exp, Integer):
        _collect_kernels(e.base, found)
        return
    if e not in found:
        found[e] = len(found)


class _Lift(Generic[T]):
    """The entries as fractions over a ring ``domain``; ``exact`` when a
    nonzero element of the ring is a nonzero quantity."""

    def __init__(self, domain: Domain[T], exact: bool) -> None:
        self.domain: Final[Domain[T]] = domain
        self.exact: Final[bool] = exact

    def atom(self, e: Expr) -> T:
        raise NotImplementedError

    def expr(self, a: T) -> Expr:
        raise NotImplementedError

    def cancel(self, n: T, d: T) -> tuple[T, T]:
        raise NotImplementedError

    def fraction(self, e: Expr) -> tuple[T, T]:
        """``(n, d)`` with ``e = n/d`` (not in lowest terms)."""
        K = self.domain
        if isinstance(e, Rational):
            return K.from_int(e.p), K.from_int(e.q)
        if isinstance(e, Add):
            n, d = K.zero, K.one
            for a in e.args:
                if not isinstance(a, Expr):
                    raise TypeError('%s is not an expression' % (a,))
                p, q = self.fraction(a)
                if q == d:
                    n = K.add(n, p)
                else:
                    n, d = K.add(K.mul(n, q), K.mul(p, d)), K.mul(d, q)
            return n, d
        if isinstance(e, Mul):
            n, d = K.one, K.one
            for a in e.args:
                if not isinstance(a, Expr):
                    raise TypeError('%s is not an expression' % (a,))
                p, q = self.fraction(a)
                n, d = K.mul(n, p), K.mul(d, q)
            return n, d
        if isinstance(e, Pow):
            exponent = e.exp
            if isinstance(exponent, Integer):
                p, q = self.fraction(e.base)
                k = exponent.p
                if k < 0:
                    if K.is_zero(p):
                        raise ZeroDivisionError('division by zero in %s' % (e,))
                    p, q, k = q, p, -k
                return K.pow(p, k), K.pow(q, k)
        return self.atom(e), K.one

    def ratio(self, n: T, d: T) -> Expr:
        """The expression ``n/d``, in lowest terms."""
        p, q = self.cancel(n, d)
        if self.domain.is_one(q):
            return self.expr(p)
        return self.expr(p) / self.expr(q)


class _IntegerLift(_Lift[int]):
    def __init__(self) -> None:
        super().__init__(ZZ, True)

    def atom(self, e: Expr) -> int:
        raise ValueError('%s is not a rational number' % (e,))

    def expr(self, a: int) -> Expr:
        return Integer(a)

    def cancel(self, n: int, d: int) -> tuple[int, int]:
        g = gcd(n, d)
        if d < 0:
            g = -g
        return n//g, d//g


class _PolynomialLift(_Lift[PolyElement[int]]):
    def __init__(self, kernels: Sequence[Expr]) -> None:
        ring = IntegerPolynomialRing(['_k%d' % i for i in range(len(kernels))])
        super().__init__(ring, all(isinstance(k, Symbol) for k in kernels))
        self.ring: Final[IntegerPolynomialRing] = ring
        self.kernels: Final[tuple[Expr, ...]] = tuple(kernels)
        self.index: Final[dict[Expr, PolyElement[int]]] = dict(zip(kernels, ring.ring.gens))
        # the same denominator is converted back for every entry of an inverse
        self._expressions: Final[dict[PolyElement[int], Expr]] = {}

    def atom(self, e: Expr) -> PolyElement[int]:
        return self.index[e]

    def expr(self, a: PolyElement[int]) -> Expr:
        known = self._expressions.get(a)
        if known is not None:
            return known
        parts: list[Expr] = []
        for monomial, coefficient in a.terms.items():
            factors: list[Expr] = [Integer(coefficient)]
            for kernel, exponent in zip(self.kernels, monomial):
                if exponent == 1:
                    factors.append(kernel)
                elif exponent:
                    factors.append(pow(kernel, Integer(exponent)))
            parts.append(mul(*factors))
        result = add(*parts) if parts else Integer(0)
        self._expressions[a] = result
        return result

    def cancel(self, n: PolyElement[int], d: PolyElement[int]) -> tuple[PolyElement[int], PolyElement[int]]:
        return self.ring.cancel(n, d)


def _kernels(matrices: Sequence[ExprMatrix]) -> list[Expr]:
    found: dict[Expr, int] = {}
    for M in matrices:
        for e in M.entries:
            _collect_kernels(e, found)
    return list(found)


def _scaled(lift: _Lift[T], M: ExprMatrix, common: bool) -> tuple[DomainMatrix[T], list[T]]:
    """The matrix over the ring with row ``i`` multiplied by the returned
    ``dens[i]`` (the same element for every row with ``common``)."""
    K = lift.domain
    fractions = [[lift.cancel(*lift.fraction(e)) for e in M.row(i)] for i in range(M.rows)]
    dens: list[T] = []
    for line in fractions:
        d = K.one
        for _, q in line:
            if not K.is_one(q):
                d = K.lcm(d, q)
        dens.append(d)
    if common:
        total = K.one
        for d in dens:
            if not K.is_one(d):
                total = K.lcm(total, d)
        dens = [total] * M.rows
    rows = [[p if q == d else K.mul(p, K.exquo(d, q)) for p, q in line] for line, d in zip(fractions, dens)]
    return DomainMatrix(rows, K, M.cols), dens


def _certify(lift: _Lift[T], pivots: Sequence[T]) -> None:
    """The pivots of an elimination are nonzero quantities, or
    ``NotImplementedError``."""
    if lift.exact:
        return
    from sympy_extras.interop import to_sympy
    from sympy_extras.simplify.structure import is_zero
    for value in pivots:
        e = lift.expr(value)
        if isinstance(e, Rational) and e.p != 0:
            continue
        if is_zero(to_sympy(e)) is not False:
            raise NotImplementedError('the pivot %s is not proved nonzero: simplify the entries of the matrix first' % (e,))


def _square(M: ExprMatrix) -> int:
    if M.rows != M.cols:
        raise ValueError('a %d x %d matrix is not square' % (M.rows, M.cols))
    return M.rows


# ---------------------------------------------------------------------
# the operations, over a lift

def _det(lift: _Lift[T], M: ExprMatrix) -> Expr:
    K = lift.domain
    scaled, dens = _scaled(lift, M, False)
    denominator = K.one
    for d in dens:
        denominator = K.mul(denominator, d)
    return lift.ratio(scaled.det(), denominator)


def _charpoly(lift: _Lift[T], M: ExprMatrix) -> list[Expr]:
    K = lift.domain
    scaled, dens = _scaled(lift, M, True)
    d = dens[0] if dens else K.one
    coefficients: list[Expr] = []
    power = K.one
    for c in scaled.charpoly():
        coefficients.append(lift.ratio(c, power))
        power = K.mul(power, d)
    return coefficients


def _adjugate(lift: _Lift[T], M: ExprMatrix) -> ExprMatrix:
    K = lift.domain
    n = M.rows
    scaled, dens = _scaled(lift, M, True)
    d = K.pow(dens[0], n - 1) if n else K.one
    return ExprMatrix(n, n, [lift.ratio(a, d) for row in scaled.adjugate().rows for a in row])


def _inverse(lift: _Lift[T], M: ExprMatrix) -> ExprMatrix:
    K = lift.domain
    n = M.rows
    scaled, dens = _scaled(lift, M, False)
    reduced, den, pivots, values = scaled.hstack(DomainMatrix.eye(n, K)).elimination()
    if pivots != list(range(n)):
        raise ValueError('the matrix is singular')
    _certify(lift, values)
    # A = D**-1 A' with D the diagonal of the row denominators: A**-1 = A'**-1 D
    return ExprMatrix(n, n, [lift.ratio(K.mul(a, dens[j]), den) for row in reduced.rows for j, a in enumerate(row[n:])])


def _rref(lift: _Lift[T], M: ExprMatrix) -> tuple[ExprMatrix, tuple[int, ...]]:
    scaled, _ = _scaled(lift, M, False)
    reduced, den, pivots, values = scaled.elimination()
    _certify(lift, values)
    return ExprMatrix(M.rows, M.cols, [lift.ratio(a, den) for row in reduced.rows for a in row]), tuple(pivots)


def _rank(lift: _Lift[T], M: ExprMatrix) -> int:
    scaled, _ = _scaled(lift, M, False)
    _, _, pivots, values = scaled.elimination()
    _certify(lift, values)
    return len(pivots)


def _nullspace_vectors(lift: _Lift[T], reduced: DomainMatrix[T], den: T, pivots: Sequence[int], ncols: int) -> list[ExprMatrix]:
    basis: list[ExprMatrix] = []
    for free in range(ncols):
        if free in pivots:
            continue
        vector: list[Expr] = [Integer(0)] * ncols
        vector[free] = Integer(1)
        for i, column in enumerate(pivots):
            vector[column] = lift.ratio(lift.domain.neg(reduced.rows[i][free]), den)
        basis.append(ExprMatrix.column(vector))
    return basis


def _nullspace(lift: _Lift[T], M: ExprMatrix) -> list[ExprMatrix]:
    scaled, _ = _scaled(lift, M, False)
    reduced, den, pivots, values = scaled.elimination()
    _certify(lift, values)
    return _nullspace_vectors(lift, reduced, den, pivots, M.cols)


class LinearSolution:
    """The solutions of ``A X = B``: ``particular`` (the free unknowns
    zero) plus any combination of the vectors of ``nullspace``."""

    __slots__ = ('particular', 'nullspace')

    def __init__(self, particular: ExprMatrix, nullspace: Sequence[ExprMatrix]) -> None:
        self.particular: Final[ExprMatrix] = particular
        self.nullspace: Final[tuple[ExprMatrix, ...]] = tuple(nullspace)

    @property
    def is_unique(self) -> bool:
        return not self.nullspace

    def __repr__(self) -> str:
        return 'LinearSolution(%r, %r)' % (self.particular, list(self.nullspace))


def _linear_solve(lift: _Lift[T], A: ExprMatrix, B: ExprMatrix) -> Optional[LinearSolution]:
    augmented = A
    for j in range(B.cols):
        augmented = augmented.col_insert(augmented.cols, ExprMatrix.column([B[i, j] for i in range(B.rows)]))
    scaled, _ = _scaled(lift, augmented, False)
    reduced, den, pivots, values = scaled.elimination()
    _certify(lift, values)
    if any(column >= A.cols for column in pivots):
        return None
    zero: Expr = Integer(0)
    solution = [[zero] * B.cols for _ in range(A.cols)]
    for i, column in enumerate(pivots):
        solution[column] = [lift.ratio(a, den) for a in reduced.rows[i][A.cols:]]
    particular = ExprMatrix(A.cols, B.cols, [e for line in solution for e in line])
    return LinearSolution(particular, _nullspace_vectors(lift, reduced, den, pivots, A.cols))


# ---------------------------------------------------------------------
# the public functions

def normal(e: Expr) -> Expr:
    """The expression as a quotient of two expanded polynomials in its
    kernels, in lowest terms (the form in which the functions of this
    module return their results). Relations between kernels which are
    not symbols are not used.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import cos
    >>> from sympy_extras.matrices.linalg import normal
    >>> x, t = Symbol('x'), Symbol('t')
    >>> normal(1/(x - 1) - 1/(x + 1)), normal((cos(t)**2 - 1)/(cos(t) + 1) + x/2)
    (2/(x**2 - 1), x/2 + cos(t) - 1)
    """
    found: dict[Expr, int] = {}
    _collect_kernels(e, found)
    if not found:
        integers = _IntegerLift()
        return integers.ratio(*integers.fraction(e))
    lift = _PolynomialLift(list(found))
    return lift.ratio(*lift.fraction(e))


def det(M: ExprMatrix) -> Expr:
    """The determinant, expanded and in lowest terms (Bareiss'
    fraction-free elimination).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import det
    >>> x, y = Symbol('x'), Symbol('y')
    >>> det(ExprMatrix.from_rows([[x, 1, 0], [1, x, 1], [0, 1, x]])), det(ExprMatrix.from_rows([[1/x, y], [1, x]]))
    (x**3 - 2*x, 1 - y)
    """
    _square(M)
    kernels = _kernels([M])
    return _det(_PolynomialLift(kernels), M) if kernels else _det(_IntegerLift(), M)


def charpoly_coefficients(M: ExprMatrix) -> list[Expr]:
    r"""The coefficients of $\det(\lambda I - M)$, highest degree first
    (Berkowitz' division-free algorithm)."""
    _square(M)
    kernels = _kernels([M])
    return _charpoly(_PolynomialLift(kernels), M) if kernels else _charpoly(_IntegerLift(), M)


def charpoly(M: ExprMatrix, x: Expr) -> Expr:
    r"""The characteristic polynomial $\det(x I - M)$.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import charpoly
    >>> x = Symbol('x')
    >>> charpoly(ExprMatrix.from_rows([[2, 1], [1, 2]]), x)
    x**2 - 4*x + 3
    """
    coefficients = charpoly_coefficients(M)
    n = len(coefficients) - 1
    return add(*[mul(c, pow(x, Integer(n - k))) for k, c in enumerate(coefficients)])


def adjugate(M: ExprMatrix) -> ExprMatrix:
    """The adjugate: ``M * adjugate(M) = det(M) * I`` (singular matrices
    included).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import adjugate
    >>> adjugate(ExprMatrix.from_rows([[1, 2], [2, 4]]))
    ExprMatrix([[4, -2], [-2, 1]])
    """
    _square(M)
    kernels = _kernels([M])
    return _adjugate(_PolynomialLift(kernels), M) if kernels else _adjugate(_IntegerLift(), M)


def inverse(M: ExprMatrix) -> ExprMatrix:
    """The inverse; ``ValueError`` for a singular matrix.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import inverse
    >>> x = Symbol('x')
    >>> inverse(ExprMatrix.from_rows([[1, x], [0, 1]])), inverse(ExprMatrix.from_rows([[2, 1], [1, 1]]))
    (ExprMatrix([[1, -x], [0, 1]]), ExprMatrix([[1, -1], [-1, 2]]))
    """
    _square(M)
    kernels = _kernels([M])
    return _inverse(_PolynomialLift(kernels), M) if kernels else _inverse(_IntegerLift(), M)


def rank(M: ExprMatrix) -> int:
    """The rank (for generic values of the symbols).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import rank
    >>> x = Symbol('x')
    >>> rank(ExprMatrix.from_rows([[x, x**2], [1, x]])), rank(ExprMatrix.from_rows([[x, 1], [1, x]]))
    (1, 2)
    """
    kernels = _kernels([M])
    return _rank(_PolynomialLift(kernels), M) if kernels else _rank(_IntegerLift(), M)


def rref(M: ExprMatrix) -> tuple[ExprMatrix, tuple[int, ...]]:
    """The reduced row echelon form and the pivot columns.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import rref
    >>> x = Symbol('x')
    >>> rref(ExprMatrix.from_rows([[1, x, 2], [x, 1, 3]]))
    (ExprMatrix([[1, 0, (3*x - 2)/(x**2 - 1)], [0, 1, (2*x - 3)/(x**2 - 1)]]), (0, 1))
    """
    kernels = _kernels([M])
    return _rref(_PolynomialLift(kernels), M) if kernels else _rref(_IntegerLift(), M)


def nullspace(M: ExprMatrix) -> list[ExprMatrix]:
    """A basis of the null space: one column vector for each column
    without pivot, with one at that coordinate.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import nullspace
    >>> x = Symbol('x')
    >>> nullspace(ExprMatrix.from_rows([[1, x, 0], [0, 1, x]]))
    [ExprMatrix([[x**2], [-x], [1]])]
    """
    kernels = _kernels([M])
    return _nullspace(_PolynomialLift(kernels), M) if kernels else _nullspace(_IntegerLift(), M)


def linear_solve(A: ExprMatrix, B: ExprMatrix) -> Optional[LinearSolution]:
    """The solutions of ``A X = B`` (any shape, one system per column of
    ``B``), or ``None`` when there is none.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.linalg import linear_solve
    >>> a = Symbol('a')
    >>> linear_solve(ExprMatrix.from_rows([[2, 1], [1, 3]]), ExprMatrix.column([3, 5]))
    LinearSolution(ExprMatrix([[4/5], [7/5]]), [])
    >>> linear_solve(ExprMatrix.from_rows([[1, 1, 1], [1, 2, a]]), ExprMatrix.column([1, 2]))
    LinearSolution(ExprMatrix([[0], [1], [0]]), [ExprMatrix([[a - 2], [1 - a], [1]])])
    >>> linear_solve(ExprMatrix.from_rows([[1, 1], [1, 1]]), ExprMatrix.column([1, 2])) is None
    True
    """
    if A.rows != B.rows:
        raise ValueError('%d equations and %d right-hand sides' % (A.rows, B.rows))
    kernels = _kernels([A, B])
    return _linear_solve(_PolynomialLift(kernels), A, B) if kernels else _linear_solve(_IntegerLift(), A, B)
