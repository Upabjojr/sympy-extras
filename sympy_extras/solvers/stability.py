r"""Stability of polynomials and matrices with parameters: the criterion
of Routh and Hurwitz and its discrete counterpart, as explicit
conditions on the parameters.

A real polynomial $p(s) = a_0 s^n + a_1 s^{n-1} + \dots + a_n$ is
*Hurwitz stable* when all its roots have negative real parts: the
equilibrium of $x' = A x$ is asymptotically stable exactly when the
characteristic polynomial of $A$ is. With the Hurwitz matrix

.. math:: H = \begin{pmatrix} a_1 & a_3 & a_5 & \cdots \\ a_0 & a_2 & a_4 & \cdots \\
          0 & a_1 & a_3 & \cdots \\ 0 & a_0 & a_2 & \cdots \\ \vdots & & & \ddots \end{pmatrix}

of order $n$ and its leading principal minors $\Delta_1, \dots,
\Delta_n$, the polynomial with $a_0 > 0$ is stable exactly when all the
$\Delta_i$ are positive [1]_ [2]_. For any sign of $a_0$ this reads
$a_0^i \Delta_i > 0$, the $\Delta_i$ being homogeneous of degree $i$ in
the coefficients.

A polynomial is *Schur stable* when its roots are inside the unit
circle (the stability of $x_{k+1} = A x_k$). The Möbius map $z = (s +
1)/(s - 1)$ sends the left half-plane onto the unit disc, so that $p$
of degree $n$ is Schur stable exactly when $q(s) = (s - 1)^n p\bigl((s
+ 1)/(s - 1)\bigr)$ is Hurwitz stable of degree $n$ (its leading
coefficient is $p(1)$) [3]_.

With parameters in the coefficients these are polynomial inequalities
in the parameters. :func:`stability_conditions` returns them simplified
by quantifier-free simplification over the real numbers
(:func:`sympy_extras.assumptions.resolve.resolve`: a cylindrical
decomposition of the parameter space), which removes the redundant ones
and solves for a single parameter: the gain range of a feedback loop.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.solvers.stability import stability_conditions, hurwitz_determinants, routh_array
>>> s, z, k, a, b = (Symbol(n) for n in 'szkab')
>>> stability_conditions((s + 1)**3 + k, s)                   # the loop of 1/(s + 1)**3 with the gain k
(k > -1) & (k < 8)
>>> hurwitz_determinants(s**3 + a*s**2 + b*s + 1, s)
[a, a*b - 1, a*b - 1]
>>> stability_conditions(z**2 + a*z + b, z, discrete=True)     # the stability triangle
(b - 1 < 0) & (a + b + 1 > 0) & (a - b - 1 < 0)
>>> stability_conditions(ExprMatrix.from_rows([[0, 1], [-k, -1]]))
k > 0
>>> routh_array(s**3 + 2*s**2 + 3*s + k, s)
[[1, 3], [2, k], [3 - k/2, 0], [k, 0]]

References
==========

.. [1] A. Hurwitz, *Über die Bedingungen, unter welchen eine Gleichung
       nur Wurzeln mit negativen reellen Theilen besitzt*, Mathematische
       Annalen 46 (1895), 273-284.
.. [2] F. R. Gantmacher, *The Theory of Matrices*, vol. 2, Chelsea, 1959,
       Chapter XV (the problem of Routh and Hurwitz).
.. [3] E. I. Jury, *Theory and Application of the z-Transform Method*,
       Wiley, 1964, Chapter 3 (stability tests; the bilinear
       transformation).
"""
from __future__ import annotations

from typing import Optional, Union

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import And, Boolean, false, true
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.core.relational import Gt, Ne
from sympy_extras.core.symbol import Symbol
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.matrices.eigen import _coefficients, _fresh_symbol
from sympy_extras.matrices.linalg import charpoly, det, normal
from sympy_extras.simplify.radsimp import fraction

__all__ = ['hurwitz_matrix', 'hurwitz_determinants', 'routh_array', 'stability_conditions', 'is_stable']


def _polynomial_coefficients(p: ExprLike, x: Symbol) -> tuple[list[Expr], Expr]:
    """The coefficients (highest degree first) of the numerator of ``p``
    as a polynomial in ``x``, and the denominator (free of ``x``)."""
    numerator, denominator = fraction(normal(to_expr(p)))
    if denominator.has(x):
        raise ValueError('%s is not a polynomial in %s' % (p, x))
    expanded = expand(numerator)
    if expanded == 0:
        raise ValueError('the zero polynomial')
    return _coefficients(expanded, x), denominator


def hurwitz_matrix(p: ExprLike, x: Symbol) -> ExprMatrix:
    """The Hurwitz matrix of a polynomial in ``x`` of degree ``n``: of
    order ``n``, with the entry ``a_(2j - i + 1)`` in row ``i`` and column
    ``j`` (indices from zero, ``a_k`` the coefficient of ``x**(n - k)``).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.stability import hurwitz_matrix
    >>> s, a, b, c = (Symbol(n) for n in 'sabc')
    >>> hurwitz_matrix(s**3 + a*s**2 + b*s + c, s)
    ExprMatrix([[a, c, 0], [1, b, 0], [0, a, c]])
    """
    coefficients, _ = _polynomial_coefficients(p, x)
    n = len(coefficients) - 1
    zero: Expr = Integer(0)
    entries: list[Expr] = []
    for i in range(n):
        for j in range(n):
            k = 2*j - i + 1
            entries.append(coefficients[k] if 0 <= k <= n else zero)
    return ExprMatrix(n, n, entries)


def hurwitz_determinants(p: ExprLike, x: Symbol) -> list[Expr]:
    """The leading principal minors of the Hurwitz matrix, of orders
    ``1, ..., n``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.stability import hurwitz_determinants
    >>> s = Symbol('s')
    >>> hurwitz_determinants(s**3 + 6*s**2 + 11*s + 6, s), hurwitz_determinants(s**2 - s + 1, s)
    ([6, 60, 360], [-1, -1])
    """
    H = hurwitz_matrix(p, x)
    return [det(ExprMatrix(k, k, [H[i, j] for i in range(k) for j in range(k)])) for k in range(1, H.rows + 1)]


def routh_array(p: ExprLike, x: Symbol) -> list[list[Expr]]:
    """Routh's array of a polynomial in ``x``: ``n + 1`` rows, the first
    two holding the coefficients of even and odd rank; the polynomial with
    a positive leading coefficient is stable exactly when the first
    column is positive. ``ValueError`` when an entry of the first column
    vanishes identically (the singular cases of the array:
    :func:`stability_conditions` has none).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.stability import routh_array
    >>> s = Symbol('s')
    >>> routh_array(s**4 + 2*s**3 + 3*s**2 + 4*s + 5, s)             # two sign changes: two unstable roots
    [[1, 3, 5], [2, 4, 0], [1, 5, 0], [-6, 0, 0], [5, 0, 0]]
    """
    coefficients, _ = _polynomial_coefficients(p, x)
    n = len(coefficients) - 1
    width = n//2 + 1
    zero: Expr = Integer(0)

    def padded(row: list[Expr]) -> list[Expr]:
        return row + [zero]*(width - len(row))

    rows = [padded(coefficients[0::2])]
    if n >= 1:
        rows.append(padded(coefficients[1::2]))
    for _ in range(2, n + 1):
        above, last = rows[-2], rows[-1]
        pivot = last[0]
        if pivot == 0:
            raise ValueError('a zero in the first column of the Routh array: use stability_conditions')
        rows.append(padded([normal((pivot*above[k + 1] - above[0]*last[k + 1])/pivot) for k in range(width - 1)]))
    return rows


def _moebius(coefficients: list[Expr], x: Symbol) -> list[Expr]:
    """The coefficients of ``(x - 1)**n p((x + 1)/(x - 1))``."""
    n = len(coefficients) - 1
    total = add(*[mul(c, pow(x + 1, Integer(n - k)), pow(x - 1, Integer(k))) for k, c in enumerate(coefficients)])
    result = _coefficients(expand(total), x) if expand(total) != 0 else [Integer(0)]
    zero: Expr = Integer(0)
    return [zero]*(n + 1 - len(result)) + result


def stability_conditions(p: Union[ExprLike, ExprMatrix], x: Optional[Symbol] = None, discrete: bool = False,
                         simplify: bool = True) -> Boolean:
    """The conditions on the parameters under which all the roots in
    ``x`` of the polynomial ``p`` have negative real parts (inside the
    unit circle with ``discrete``); for a square matrix, under which all
    its eigenvalues do. A Boolean: ``true`` or ``false`` without
    parameters. With ``simplify`` (the default) the conjunction of the
    Hurwitz inequalities is simplified by a decomposition of the space
    of the parameters; without, it is returned as it is.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.stability import stability_conditions
    >>> s, K = Symbol('s'), Symbol('K')
    >>> stability_conditions(s*(s + 1)*(s + 2) + K, s)
    (K > 0) & (K < 6)
    >>> stability_conditions(s**2 + 1, s), stability_conditions(s**2 + 3*s + 2, s)
    (False, True)
    >>> stability_conditions(s**2 + K*s + 1, s, simplify=False)
    (K > 0) & (K > 0)
    """
    variable: Symbol
    polynomial: Expr
    if isinstance(p, ExprMatrix):
        variable = _fresh_symbol(p) if x is None else x
        polynomial = charpoly(p, variable)
    else:
        if x is None:
            raise ValueError('the variable of the polynomial is needed')
        variable, polynomial = x, to_expr(p)
    coefficients, denominator = _polynomial_coefficients(polynomial, variable)
    conditions: list[Boolean] = []
    if denominator.free_symbols:
        conditions.append(Ne(denominator, 0))
    if discrete:
        coefficients = _moebius(coefficients, variable)
    n = len(coefficients) - 1
    leading = coefficients[0]
    if leading == 0:
        return false
    if leading.free_symbols:
        conditions.append(Ne(leading, 0))
    if n > 0:
        rebuilt = add(*[mul(c, pow(variable, Integer(n - k))) for k, c in enumerate(coefficients)])
        for order, minor in enumerate(hurwitz_determinants(rebuilt, variable), start=1):
            quantity = normal(minor*leading**order)
            if not quantity.free_symbols:
                if quantity.is_positive is not True:
                    return false
                continue
            conditions.append(Gt(quantity, 0))
    if not conditions:
        return true
    formula: Boolean = conditions[0] if len(conditions) == 1 else And(*conditions)
    if not simplify:
        return formula
    from sympy_extras.assumptions.resolve import resolve
    try:
        return resolve(formula)
    except (ValueError, NotImplementedError):
        return formula


def is_stable(p: Union[ExprLike, ExprMatrix], x: Optional[Symbol] = None, discrete: bool = False) -> bool:
    """Whether a polynomial or a matrix without parameters is stable.

    >>> from sympy_extras.core import Rational, Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.solvers.stability import is_stable
    >>> z = Symbol('z')
    >>> is_stable(ExprMatrix.from_rows([[-1, 5], [0, -2]])), is_stable(z**2 - z + Rational(1, 2), z, discrete=True)
    (True, True)
    """
    result = stability_conditions(p, x, discrete)
    if result is true:
        return True
    if result is false:
        return False
    raise ValueError('the stability depends on parameters: %s' % (result,))
