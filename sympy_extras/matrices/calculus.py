r"""Matrices of derivatives: the Jacobian, the Hessian (bordered for
constraints), the Wronskian, and the second derivative test for critical
points.

* :func:`jacobian`: the matrix $\partial f_i / \partial x_j$ of a vector
  of expressions; :func:`gradient`: the column of the partial
  derivatives of a scalar.
* :func:`hessian`: the symmetric matrix $\partial^2 f / \partial x_i
  \partial x_j$; with constraints $g_1 = \dots = g_m = 0$ the *bordered*
  Hessian $\begin{pmatrix} 0 & J_g \\ J_g^T & H \end{pmatrix}$ of the
  test for constrained extrema [1]_.
* :func:`wronskian`: the determinant of the derivatives of orders $0,
  \dots, n - 1$ of $n$ functions; it is not identically zero exactly
  when analytic functions are linearly independent [2]_.
* :func:`classify_critical_point`: at a point where the gradient
  vanishes, the signs of the eigenvalues of the Hessian are found
  without computing them, by Sylvester's law of inertia
  (:func:`sympy_extras.matrices.decompositions.inertia`): a positive
  definite Hessian means a strict local minimum, a negative definite one
  a strict local maximum, eigenvalues of both signs a saddle, and
  otherwise (a semidefinite Hessian) the test does not decide [1]_.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import cos, exp, sin
>>> from sympy_extras.matrices.calculus import jacobian, hessian, wronskian, classify_critical_point
>>> x, y, r, t = Symbol('x'), Symbol('y'), Symbol('r'), Symbol('t')
>>> J = jacobian([r*cos(t), r*sin(t)], [r, t])
>>> J, J.det()
(ExprMatrix([[cos(t), -r*sin(t)], [sin(t), r*cos(t)]]), r*sin(t)**2 + r*cos(t)**2)
>>> hessian(x**3 + x*y**2, [x, y])
ExprMatrix([[6*x, 2*y], [2*y, 2*x]])
>>> wronskian([exp(x), exp(2*x)], x)
exp(3*x)
>>> f = x**3 - 3*x + y**2
>>> classify_critical_point(f, [x, y], [1, 0]), classify_critical_point(f, [x, y], [-1, 0])
('minimum', 'saddle')

References
==========

.. [1] J. E. Marsden, A. Tromba, *Vector Calculus*, 6th ed., Freeman,
       2012, Sections 3.3 (the second derivative test) and 3.4
       (constrained extrema, the bordered Hessian).
.. [2] M. Bôcher, *Certain cases in which the vanishing of the Wronskian
       is a sufficient condition for linear dependence*, Transactions of
       the American Mathematical Society 2 (1901), 139-149.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Symbol

from .dense import ExprMatrix

__all__ = ['jacobian', 'gradient', 'hessian', 'wronskian', 'classify_critical_point']


def _tidy(e: Expr) -> Expr:
    from sympy_extras.simplify.normalform import normal_form
    return normal_form(e)


def jacobian(functions: Sequence[ExprLike], variables: Sequence[Symbol]) -> ExprMatrix:
    """The Jacobian matrix: row ``i`` holds the partial derivatives of
    ``functions[i]`` with respect to the variables.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.calculus import jacobian
    >>> x, y = Symbol('x'), Symbol('y')
    >>> jacobian([x**2*y, x + y], [x, y])
    ExprMatrix([[2*x*y, x**2], [1, 1]])
    """
    expressions = [to_expr(f) for f in functions]
    return ExprMatrix(len(expressions), len(variables), [f.diff(v) for f in expressions for v in variables])


def gradient(f: ExprLike, variables: Sequence[Symbol]) -> ExprMatrix:
    """The column of the partial derivatives of a scalar.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.calculus import gradient
    >>> x, y = Symbol('x'), Symbol('y')
    >>> gradient(x**2*y, [x, y])
    ExprMatrix([[2*x*y], [x**2]])
    """
    expression = to_expr(f)
    return ExprMatrix.column([expression.diff(v) for v in variables])


def hessian(f: ExprLike, variables: Sequence[Symbol], constraints: Sequence[ExprLike] = ()) -> ExprMatrix:
    """The Hessian matrix of the second partial derivatives of ``f``;
    with ``constraints`` (expressions which are zero), the bordered
    Hessian: the Jacobian of the constraints above and on the left of the
    Hessian, a zero block in the corner.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.calculus import hessian
    >>> x, y = Symbol('x'), Symbol('y')
    >>> hessian(x*y, [x, y], [x + y - 1])
    ExprMatrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    """
    expression = to_expr(f)
    n = len(variables)
    first = [expression.diff(v) for v in variables]
    H = [[first[i].diff(variables[j]) for j in range(n)] for i in range(n)]
    if not constraints:
        return ExprMatrix(n, n, [e for row in H for e in row])
    border = [[to_expr(g).diff(v) for v in variables] for g in constraints]
    m = len(border)
    zero: Expr = Integer(0)
    rows: list[list[Expr]] = [[zero]*m + border[i] for i in range(m)]
    rows += [[border[k][i] for k in range(m)] + H[i] for i in range(n)]
    return ExprMatrix(m + n, m + n, [e for row in rows for e in row])


def wronskian(functions: Sequence[ExprLike], x: Symbol) -> Expr:
    """The Wronskian determinant of the functions of ``x``, reduced
    modulo the relations of sines, cosines and radicals
    (:func:`~sympy_extras.simplify.normalform.normal_form`).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import cos, sin
    >>> from sympy_extras.matrices.calculus import wronskian
    >>> x = Symbol('x')
    >>> wronskian([cos(x), sin(x)], x), wronskian([x, x**2, x**3], x), wronskian([x, 2*x], x)
    (1, 2*x**3, 0)
    """
    n = len(functions)
    rows: list[list[Expr]] = []
    current = [to_expr(f) for f in functions]
    for _ in range(n):
        rows.append(current)
        current = [f.diff(x) for f in current]
    if n == 0:
        return Integer(1)
    from .linalg import det
    return _tidy(det(ExprMatrix(n, n, [e for row in rows for e in row])))


def classify_critical_point(f: ExprLike, variables: Sequence[Symbol], point: Sequence[ExprLike]) -> Optional[str]:
    """The nature of the critical point ``point`` of ``f`` by the second
    derivative test: ``'minimum'``, ``'maximum'`` (strict local ones),
    ``'saddle'``, or ``None`` when the Hessian is semidefinite and the
    test does not decide. ``ValueError`` when the gradient does not
    vanish at the point, or when the signs of the pivots of the Hessian
    there are not known (parameters without assumptions).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.calculus import classify_critical_point
    >>> x, y = Symbol('x'), Symbol('y')
    >>> classify_critical_point(x**2 - y**2, [x, y], [0, 0]), classify_critical_point(-x**2 - y**4, [x, y], [0, 0]) is None
    ('saddle', True)
    """
    if len(point) != len(variables):
        raise ValueError('%d coordinates for %d variables' % (len(point), len(variables)))
    expression = to_expr(f)
    rule: dict[Basic, Basic] = {v: to_expr(p) for v, p in zip(variables, point)}

    def at_point(e: Expr) -> Expr:
        return _tidy(e.xreplace(rule).doit())

    if any(at_point(expression.diff(v)) != 0 for v in variables):
        raise ValueError('%s is not a critical point: the gradient does not vanish there' % (list(point),))
    H = hessian(expression, variables).applyfunc(at_point)
    from .decompositions import inertia
    positive, negative, zero = inertia(H)
    if positive and negative:
        return 'saddle'
    if zero:
        return None
    return 'minimum' if positive else 'maximum'
