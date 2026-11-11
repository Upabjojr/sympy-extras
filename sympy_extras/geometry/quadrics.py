r"""Conics and quadric surfaces from their equations: the affine type,
the centre, the axes and the canonical equation.

A polynomial of degree two in $x_1, \dots, x_n$ ($n = 2$ or $3$) is
$\tilde x^T M \tilde x$ with $\tilde x = (x_1, \dots, x_n, 1)$ and a
symmetric matrix $M = \begin{pmatrix} A & b \\ b^T & c \end{pmatrix}$,
$A$ the matrix of the quadratic part. The ranks and the signatures of
$A$ and $M$ do not change under affine changes of coordinates and
determine the affine type of the zero set [1]_ [2]_: an ellipse has
$\operatorname{rank} M = 3$ and $A$ definite with $M$ indefinite, a
hyperbola $A$ indefinite, a parabola $\operatorname{rank} A = 1$; the
degenerate cases (pairs of lines, a point, the empty set) have
$\operatorname{rank} M < 3$ or $M$ definite. The table for surfaces
(ellipsoids, the two hyperboloids, the two paraboloids, cones, cylinders,
pairs of planes) is read in the same way.

The signatures are computed without eigenvalues, by Sylvester's law of
inertia (:func:`sympy_extras.matrices.decompositions.inertia`), so that
the type is decided exactly for rational coefficients and for symbols
whose signs are declared.

For a central conic ($\det A \ne 0$) the centre solves $A x_0 = -b$ and
in the coordinates of the principal axes the equation is $\lambda_1 X^2 +
\lambda_2 Y^2 + \det M / \det A = 0$ with $\lambda_i$ the eigenvalues of
$A$, from which the semi-axes and the eccentricity follow; a parabola
has the parameter $p = \sqrt{-\det M / (\operatorname{tr} A)^3}$
(canonical equation $Y^2 = 2 p X$) [2]_.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.geometry.quadrics import classify_conic, classify_quadric
>>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
>>> c = classify_conic(5*x**2 + 4*x*y + 8*y**2 - 36, x, y)
>>> c.kind, c.center, c.semi_axes, c.eccentricity
('ellipse', (0, 0), (3, 2), sqrt(5)/3)
>>> classify_conic(x*y - 1, x, y).kind, classify_conic(x**2 - 2*x*y + y**2 - x, x, y).kind
('hyperbola', 'parabola')
>>> classify_conic(x**2 - y**2, x, y).kind, classify_conic(x**2 + y**2 + 1, x, y).kind
('two intersecting lines', 'empty set')
>>> classify_quadric(x**2 + y**2 - z**2 - 1, x, y, z).kind, classify_quadric(x*y - z, x, y, z).kind
('hyperboloid of one sheet', 'hyperbolic paraboloid')

References
==========

.. [1] M. Berger, *Geometry II*, Springer, 1987, Chapters 14 and 15
       (affine classification of quadrics by rank and signature).
.. [2] G. A. Korn, T. M. Korn, *Mathematical Handbook for Scientists and
       Engineers*, 2nd ed., McGraw-Hill, 1968, Sections 2.4 (conics:
       invariants, canonical forms) and 3.5 (quadric surfaces).
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.matrices.decompositions import inertia
from sympy_extras.matrices.linalg import det, linear_solve, normal

__all__ = ['Conic', 'Quadric', 'quadric_matrices', 'classify_conic', 'classify_quadric']


def quadric_matrices(equation: ExprLike, variables: Sequence[Symbol]) -> tuple[ExprMatrix, ExprMatrix]:
    """``(A, M)``: the symmetric matrices of the quadratic part and of
    the whole polynomial of degree at most two in ``variables``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.geometry.quadrics import quadric_matrices
    >>> x, y = Symbol('x'), Symbol('y')
    >>> quadric_matrices(x**2 + 4*x*y + 6*y - 1, [x, y])
    (ExprMatrix([[1, 2], [2, 0]]), ExprMatrix([[1, 2, 0], [2, 0, 3], [0, 3, -1]]))
    """
    polynomial = expand(to_expr(equation))
    n = len(variables)
    half: Expr = Rational(1, 2)
    zero_point = {v: Integer(0) for v in variables}

    def at_origin(e: Expr) -> Expr:
        for v, value in zero_point.items():
            e = e.subs(v, value)
        return e

    first = [polynomial.diff(v) for v in variables]
    A = [[normal(first[i].diff(variables[j])*half) for j in range(n)] for i in range(n)]
    b = [normal(at_origin(first[i])*half) for i in range(n)]
    c = normal(at_origin(polynomial))
    if any(e.free_symbols & set(variables) for row in A for e in row):
        raise ValueError('%s is not a polynomial of degree at most two in %s' % (equation, list(variables)))
    rebuilt: Expr = c
    for i in range(n):
        rebuilt = rebuilt + 2*b[i]*variables[i]
        for j in range(n):
            rebuilt = rebuilt + A[i][j]*variables[i]*variables[j]
    if normal(rebuilt - polynomial) != 0:
        raise ValueError('%s is not a polynomial of degree at most two in %s' % (equation, list(variables)))
    rows = [A[i] + [b[i]] for i in range(n)] + [b + [c]]
    return ExprMatrix.from_rows(A), ExprMatrix.from_rows(rows)


def _signature(M: ExprMatrix) -> tuple[int, int]:
    """``(rank, |positive - negative|)``."""
    positive, negative, _ = inertia(M)
    return positive + negative, abs(positive - negative)


class Conic:
    """A classified conic: ``kind``, the ``center`` of a central conic
    (``None`` otherwise), the ``semi_axes`` of an ellipse (major first)
    or a hyperbola (transverse first), the ``eccentricity``, the
    ``parameter`` of a parabola, and the coefficients ``canonical`` of
    the equation in principal axes ``l1*X**2 + l2*Y**2 + k = 0`` of a
    central conic."""

    __slots__ = ('kind', 'center', 'semi_axes', 'eccentricity', 'parameter', 'canonical')

    def __init__(self, kind: str, center: Optional[tuple[Expr, ...]], semi_axes: Optional[tuple[Expr, Expr]],
                 eccentricity: Optional[Expr], parameter: Optional[Expr], canonical: Optional[tuple[Expr, Expr, Expr]]) -> None:
        self.kind: Final[str] = kind
        self.center: Final[Optional[tuple[Expr, ...]]] = center
        self.semi_axes: Final[Optional[tuple[Expr, Expr]]] = semi_axes
        self.eccentricity: Final[Optional[Expr]] = eccentricity
        self.parameter: Final[Optional[Expr]] = parameter
        self.canonical: Final[Optional[tuple[Expr, Expr, Expr]]] = canonical

    def __repr__(self) -> str:
        return 'Conic(%s)' % self.kind


class Quadric:
    """A classified quadric surface: ``kind`` and the ``center`` of a
    central one (``None`` otherwise)."""

    __slots__ = ('kind', 'center')

    def __init__(self, kind: str, center: Optional[tuple[Expr, ...]]) -> None:
        self.kind: Final[str] = kind
        self.center: Final[Optional[tuple[Expr, ...]]] = center

    def __repr__(self) -> str:
        return 'Quadric(%s)' % self.kind


def _center(A: ExprMatrix, M: ExprMatrix) -> Optional[tuple[Expr, ...]]:
    n = A.rows
    if det(A) == 0:
        return None
    solution = linear_solve(A, ExprMatrix.column([-M[i, n] for i in range(n)]))
    if solution is None:
        return None
    return tuple(solution.particular[i, 0] for i in range(n))


def _conic_kind(A: ExprMatrix, M: ExprMatrix) -> str:
    rank_a, sign_a = _signature(A)
    rank_m, sign_m = _signature(M)
    if rank_m == 3:
        if rank_a == 2:
            if sign_a == 0:
                return 'hyperbola'
            if sign_m == 3:
                return 'empty set'
            return 'circle' if A[0, 1] == 0 and normal(A[0, 0] - A[1, 1]) == 0 else 'ellipse'
        return 'parabola'
    if rank_m == 2:
        if rank_a == 2:
            return 'point' if sign_a == 2 else 'two intersecting lines'
        if rank_a == 1:
            return 'two parallel lines' if sign_m == 0 else 'empty set'
        return 'line'
    if rank_m == 1:
        return 'double line' if rank_a == 1 else 'empty set'
    return 'plane'


def classify_conic(equation: ExprLike, x: Symbol, y: Symbol) -> Conic:
    """The conic ``equation = 0`` (a polynomial of degree at most two in
    ``x`` and ``y``): see :class:`Conic`. The kinds are ``'ellipse'``,
    ``'circle'``, ``'hyperbola'``, ``'parabola'``, ``'two intersecting
    lines'``, ``'two parallel lines'``, ``'double line'``, ``'line'``,
    ``'point'``, ``'empty set'`` and ``'plane'``. ``ValueError`` when the
    signs needed are not known (parameters without assumptions).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.geometry.quadrics import classify_conic
    >>> x, y = Symbol('x'), Symbol('y')
    >>> h = classify_conic(x**2 - 4*y**2 - 2*x - 3, x, y)
    >>> h.kind, h.center, h.semi_axes, h.eccentricity
    ('hyperbola', (1, 0), (2, 1), sqrt(5)/2)
    >>> p = classify_conic(y**2 - 4*x, x, y)
    >>> p.kind, p.parameter
    ('parabola', 2)
    >>> classify_conic((x - 1)**2 + (y + 2)**2 - 9, x, y).kind
    circle
    """
    A, M = quadric_matrices(equation, [x, y])
    kind = _conic_kind(A, M)
    center = _center(A, M)
    semi_axes: Optional[tuple[Expr, Expr]] = None
    eccentricity: Optional[Expr] = None
    parameter: Optional[Expr] = None
    canonical: Optional[tuple[Expr, Expr, Expr]] = None
    if kind in ('ellipse', 'circle', 'hyperbola'):
        delta, big = det(A), det(M)
        trace = normal(A[0, 0] + A[1, 1])
        # the eigenvalues of A: (trace -+ root)/2
        root = sqrt(normal(trace**2 - 4*delta))
        first, second = normal((trace - root)/2), normal((trace + root)/2)
        constant = normal(big/delta)
        canonical = (first, second, constant)
        squares = [normal(-constant/first), normal(-constant/second)]
        if kind == 'hyperbola':
            transverse = squares[0] if squares[0].is_positive else squares[1]
            conjugate = squares[1] if squares[0].is_positive else squares[0]
            semi_axes = (sqrt(transverse), sqrt(normal(-conjugate)))
            eccentricity = sqrt(normal(1 - conjugate/transverse))
        else:
            larger, smaller = squares[0], squares[1]
            if (smaller - larger).is_positive:
                larger, smaller = smaller, larger
            semi_axes = (sqrt(larger), sqrt(smaller))
            eccentricity = sqrt(normal(1 - smaller/larger))
    elif kind == 'parabola':
        trace = normal(A[0, 0] + A[1, 1])
        parameter = sqrt(normal(-det(M)/trace**3))
    return Conic(kind, center, semi_axes, eccentricity, parameter, canonical)


def _quadric_kind(A: ExprMatrix, M: ExprMatrix) -> str:
    rank_a, sign_a = _signature(A)
    rank_m, sign_m = _signature(M)
    if rank_m == 4:
        if rank_a == 3:
            if sign_a == 3:
                return 'ellipsoid' if sign_m == 2 else 'empty set'
            return 'hyperboloid of one sheet' if sign_m == 0 else 'hyperboloid of two sheets'
        return 'elliptic paraboloid' if sign_a == 2 else 'hyperbolic paraboloid'
    if rank_m == 3:
        if rank_a == 3:
            return 'point' if sign_a == 3 else 'cone'
        if rank_a == 2:
            if sign_a == 0:
                return 'hyperbolic cylinder'
            return 'elliptic cylinder' if sign_m == 1 else 'empty set'
        return 'parabolic cylinder'
    if rank_m == 2:
        if rank_a == 2:
            return 'line' if sign_a == 2 else 'two intersecting planes'
        if rank_a == 1:
            return 'two parallel planes' if sign_m == 0 else 'empty set'
        return 'plane'
    if rank_m == 1:
        return 'double plane' if rank_a == 1 else 'empty set'
    return 'space'


def classify_quadric(equation: ExprLike, x: Symbol, y: Symbol, z: Symbol) -> Quadric:
    """The quadric surface ``equation = 0`` (a polynomial of degree at
    most two in ``x``, ``y``, ``z``): its affine type and its centre.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.geometry.quadrics import classify_quadric
    >>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    >>> q = classify_quadric(x**2 + 2*y**2 + 3*z**2 - 2*x - 5, x, y, z)
    >>> q.kind, q.center
    ('ellipsoid', (1, 0, 0))
    >>> [classify_quadric(e, x, y, z).kind for e in (x**2 + y**2 - z**2, x**2 + y**2 - z, x**2 - y**2 - 1, x**2 - 1)]
    ['cone', 'elliptic paraboloid', 'hyperbolic cylinder', 'two parallel planes']
    """
    A, M = quadric_matrices(equation, [x, y, z])
    return Quadric(_quadric_kind(A, M), _center(A, M))
