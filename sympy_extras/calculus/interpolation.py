r"""Exact interpolation and fitting: Newton and Hermite interpolating
polynomials, cubic splines, and linear least squares.

* :func:`interpolating_polynomial`: the polynomial of degree below $n$
  through $n$ points, by Newton's divided differences [1]_; the nodes
  and the values may be symbolic.
* :func:`hermite_interpolation`: the polynomial with prescribed values
  *and derivatives* at the nodes, by divided differences with repeated
  nodes ($f[x_i, \dots, x_i] = f^{(k)}(x_i)/k!$) [1]_.
* :func:`cubic_spline`: the piecewise cubic of class $C^2$ through the
  points. With $h_i$ the lengths of the intervals and $M_i$ the second
  derivatives at the nodes, continuity of the first derivative is the
  tridiagonal system

  .. math:: h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1}
            = 6 \Bigl(\frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}\Bigr),

  closed by the boundary conditions: *natural* ($M_0 = M_n = 0$),
  *clamped* (given end slopes), *not-a-knot* (the third derivative is
  continuous at the second and the last but one node) or *periodic*
  [2]_. The system is solved exactly and the result is a
  ``Piecewise``.
* :func:`least_squares`: the combination of given functions which best
  fits the points in the sense of least squares, from the normal
  equations $A^T A c = A^T y$ solved exactly [3]_; a line, a polynomial
  (:func:`polynomial_fit`) or any basis.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.calculus.interpolation import (interpolating_polynomial, hermite_interpolation, cubic_spline,
...     polynomial_fit)
>>> x, a = Symbol('x'), Symbol('a')
>>> interpolating_polynomial([(0, 1), (1, 3), (2, 11)], x)
3*x**2 - x + 1
>>> interpolating_polynomial([(0, 0), (a, 1)], x)
x/a
>>> hermite_interpolation([(0, [0, 1]), (1, [1, 0])], x)            # values and slopes at 0 and 1
-x**3 + x**2 + x
>>> cubic_spline([(0, 0), (1, 1), (2, 0)], x)
Piecewise((-x**3/2 + 3*x/2, x <= 1), (x**3/2 - 3*x**2 + 9*x/2 - 1, True))
>>> polynomial_fit([(0, 1), (1, 2), (2, 2), (3, 4)], x, 1)
9*x/10 + 9/10

References
==========

.. [1] J. Stoer, R. Bulirsch, *Introduction to Numerical Analysis*, 3rd
       ed., Springer, 2002, Sections 2.1 (Newton's formula, divided
       differences, Hermite interpolation) and 2.4 (spline functions).
.. [2] C. de Boor, *A Practical Guide to Splines*, revised ed., Springer,
       2001, Chapter IV (cubic spline interpolation and its end
       conditions).
.. [3] Å. Björck, *Numerical Methods for Least Squares Problems*, SIAM,
       1996, Section 1.1 (the normal equations).
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.numbers import Integer
from sympy_extras.core.relational import Le
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import piecewise
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.matrices.linalg import linear_solve, normal

__all__ = ['divided_differences', 'interpolating_polynomial', 'hermite_interpolation', 'cubic_spline', 'least_squares',
           'polynomial_fit']


def _newton(nodes: Sequence[Expr], coefficients: Sequence[Expr], x: Expr) -> Expr:
    """Horner's rule on Newton's form."""
    result: Expr = Integer(0)
    for node, c in zip(reversed(nodes), reversed(coefficients)):
        result = result*(x - node) + c
    return normal(expand(result))


def divided_differences(points: Sequence[tuple[ExprLike, ExprLike]]) -> list[Expr]:
    """The divided differences ``f[x_0], f[x_0, x_1], ..., f[x_0, ...,
    x_n]`` of the points ``(x_i, f(x_i))`` (distinct nodes): the
    coefficients of Newton's form of the interpolating polynomial.

    >>> from sympy_extras.calculus.interpolation import divided_differences
    >>> divided_differences([(0, 1), (1, 3), (2, 11)])
    [1, 2, 3]
    """
    nodes = [to_expr(p) for p, _ in points]
    table = [to_expr(v) for _, v in points]
    n = len(nodes)
    for i in range(n):
        for j in range(i):
            if normal(nodes[i] - nodes[j]) == 0:
                raise ValueError('the nodes of interpolation are not distinct')
    result: list[Expr] = []
    for level in range(n):
        result.append(table[0])
        table = [normal((table[i + 1] - table[i])/(nodes[i + level + 1] - nodes[i])) for i in range(n - level - 1)]
    return result


def interpolating_polynomial(points: Sequence[tuple[ExprLike, ExprLike]], x: Expr) -> Expr:
    """The polynomial in ``x`` of degree below the number of points
    through the points ``(x_i, y_i)``, expanded.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.calculus.interpolation import interpolating_polynomial
    >>> x = Symbol('x')
    >>> interpolating_polynomial([(1, 1), (2, 8), (3, 27), (4, 64)], x)
    x**3
    """
    nodes = [to_expr(p) for p, _ in points]
    return _newton(nodes, divided_differences(points), x)


def hermite_interpolation(data: Sequence[tuple[ExprLike, Sequence[ExprLike]]], x: Expr) -> Expr:
    """The polynomial of least degree with, at each node ``x_i`` of
    ``data = [(x_i, [f(x_i), f'(x_i), ...]), ...]``, the prescribed value
    and derivatives (as many as given).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.calculus.interpolation import hermite_interpolation
    >>> x = Symbol('x')
    >>> hermite_interpolation([(0, [1, 1, 1])], x)                      # a Taylor polynomial
    x**2/2 + x + 1
    >>> hermite_interpolation([(-1, [0, 0]), (1, [0, 0]), (0, [1])], x)
    x**4 - 2*x**2 + 1
    """
    nodes: list[Expr] = []
    owner: list[int] = []
    derivatives: list[list[Expr]] = []
    for index, (node, values) in enumerate(data):
        if not values:
            raise ValueError('a node without a value')
        derivatives.append([to_expr(v) for v in values])
        nodes.extend([to_expr(node)]*len(values))
        owner.extend([index]*len(values))
    for i, (first, _) in enumerate(data):
        for second, _ in data[:i]:
            if normal(to_expr(first) - to_expr(second)) == 0:
                raise ValueError('the nodes of interpolation are not distinct')
    n = len(nodes)
    # the table of divided differences with repeated nodes: f[x_i, ..., x_i] (k + 1 times) = f^(k)(x_i)/k!
    table = [derivatives[owner[i]][0] for i in range(n)]
    coefficients: list[Expr] = []
    factorial = 1
    for level in range(n):
        coefficients.append(table[0])
        if level + 1 == n:
            break
        factorial *= level + 1
        following: list[Expr] = []
        for i in range(n - level - 1):
            if owner[i] == owner[i + level + 1]:
                following.append(normal(derivatives[owner[i]][level + 1]/factorial))
            else:
                following.append(normal((table[i + 1] - table[i])/(nodes[i + level + 1] - nodes[i])))
        table = following
    return _newton(nodes, coefficients, x)


def cubic_spline(points: Sequence[tuple[ExprLike, ExprLike]], x: Symbol, boundary: str = 'natural',
                 slopes: Optional[tuple[ExprLike, ExprLike]] = None) -> Expr:
    """The cubic spline through the points (nodes in increasing order):
    a ``Piecewise`` of cubics, twice continuously differentiable, each
    valid up to its right node (the first and the last extend beyond the
    ends). ``boundary`` is ``'natural'``, ``'clamped'`` (with the end
    ``slopes``), ``'not-a-knot'`` or ``'periodic'`` (equal end values).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.calculus.interpolation import cubic_spline
    >>> x = Symbol('x')
    >>> cubic_spline([(0, 0), (1, 1), (2, 8), (3, 27)], x, 'not-a-knot')          # reproduces a cubic
    x**3
    >>> cubic_spline([(0, 0), (1, 1)], x, 'clamped', (0, 0))
    -2*x**3 + 3*x**2
    """
    nodes = [to_expr(p) for p, _ in points]
    values = [to_expr(v) for _, v in points]
    n = len(nodes) - 1
    if n < 1:
        raise ValueError('a spline needs two points at least')
    h = [normal(nodes[i + 1] - nodes[i]) for i in range(n)]
    if any(step.is_positive is not True for step in h):
        raise ValueError('the nodes must be increasing (with known signs of their differences)')
    slope = [normal((values[i + 1] - values[i])/h[i]) for i in range(n)]
    zero: Expr = Integer(0)
    rows: list[list[Expr]] = []
    rhs: list[Expr] = []

    def row(entries: Sequence[tuple[int, Expr]], value: Expr) -> None:
        # a position may occur twice (the end conditions with few intervals): the entries add up
        line = [zero]*(n + 1)
        for position, entry in entries:
            line[position] = line[position] + entry
        rows.append(line)
        rhs.append(value)

    for i in range(1, n):
        row([(i - 1, h[i - 1]), (i, 2*(h[i - 1] + h[i])), (i + 1, h[i])], 6*(slope[i] - slope[i - 1]))
    if boundary == 'natural':
        row([(0, Integer(1))], zero)
        row([(n, Integer(1))], zero)
    elif boundary == 'clamped':
        if slopes is None:
            raise ValueError('a clamped spline needs the slopes at the two ends')
        first, last = to_expr(slopes[0]), to_expr(slopes[1])
        row([(0, 2*h[0]), (1, h[0])], 6*(slope[0] - first))
        row([(n - 1, h[n - 1]), (n, 2*h[n - 1])], 6*(last - slope[n - 1]))
    elif boundary == 'not-a-knot':
        if n < 3:
            # too few intervals for the condition: the interpolating polynomial is the spline
            return interpolating_polynomial(points, x)
        row([(0, h[1]), (1, -(h[0] + h[1])), (2, h[0])], zero)
        row([(n - 2, h[n - 1]), (n - 1, -(h[n - 2] + h[n - 1])), (n, h[n - 2])], zero)
    elif boundary == 'periodic':
        if normal(values[0] - values[n]) != 0:
            raise ValueError('a periodic spline needs equal values at the two ends')
        row([(0, Integer(1)), (n, Integer(-1))], zero)
        row([(0, 2*h[0]), (1, h[0]), (n - 1, h[n - 1]), (n, 2*h[n - 1])], 6*(slope[0] - slope[n - 1]))
    else:
        raise ValueError('unknown boundary condition %r' % (boundary,))
    solution = linear_solve(ExprMatrix.from_rows(rows), ExprMatrix.column(rhs))
    if solution is None or not solution.is_unique:
        raise ArithmeticError('the spline is not determined')
    M = [solution.particular[i, 0] for i in range(n + 1)]
    pieces: list[Expr] = []
    for i in range(n):
        t = x - nodes[i]
        cubic = (values[i] + (slope[i] - h[i]*(2*M[i] + M[i + 1])/6)*t + M[i]*t**2/2 + (M[i + 1] - M[i])*t**3/(6*h[i]))
        pieces.append(normal(expand(cubic)))
    if all(normal(piece - pieces[0]) == 0 for piece in pieces):
        return pieces[0]
    branches: list[tuple[Expr, Boolean]] = [(piece, Le(x, nodes[i + 1])) for i, piece in enumerate(pieces[:-1])]
    branches.append((pieces[-1], true))
    return piecewise(*branches)


def least_squares(points: Sequence[tuple[ExprLike, ExprLike]], x: Symbol, basis: Sequence[ExprLike]) -> Expr:
    """The combination of the functions ``basis`` of ``x`` which
    minimises the sum of the squares of the residuals at the points
    ``(x_i, y_i)``, with exact coefficients; ``ValueError`` when the fit
    is not unique (dependent functions on these abscissas).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.calculus.interpolation import least_squares
    >>> x = Symbol('x')
    >>> least_squares([(0, 0), (1, 1), (2, 1), (3, 3)], x, [1, x])
    9*x/10 - 1/10
    >>> least_squares([(1, 2), (2, 4), (3, 7)], x, [x])                    # a line through the origin
    31*x/14
    """
    functions = [to_expr(f) for f in basis]
    if not functions:
        raise ValueError('an empty basis')
    abscissas = [to_expr(p) for p, _ in points]
    ordinates = [to_expr(v) for _, v in points]
    design = [[normal(f.subs(x, p)) for f in functions] for p in abscissas]
    k = len(functions)
    gram = [[normal(add(*[row[i]*row[j] for row in design])) for j in range(k)] for i in range(k)]
    moments = [normal(add(*[row[i]*v for row, v in zip(design, ordinates)])) for i in range(k)]
    solution = linear_solve(ExprMatrix.from_rows(gram), ExprMatrix.column(moments))
    if solution is None or not solution.is_unique:
        raise ValueError('the least squares fit is not unique: the functions are dependent on these abscissas')
    return add(*[solution.particular[i, 0]*functions[i] for i in range(k)])


def polynomial_fit(points: Sequence[tuple[ExprLike, ExprLike]], x: Symbol, degree: int) -> Expr:
    """The polynomial of degree at most ``degree`` which fits the points
    best in the sense of least squares (the regression line for ``degree
    = 1``).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.calculus.interpolation import polynomial_fit
    >>> x = Symbol('x')
    >>> polynomial_fit([(-1, 1), (0, 0), (1, 1), (2, 5)], x, 2)
    5*x**2/4 + x/20 - 3/20
    """
    if degree < 0:
        raise ValueError('a negative degree')
    return least_squares(points, x, [x**k for k in range(degree + 1)])
