"""The SymPy face of the Dirichlet, Frullani and Borwein integrals
(:mod:`sympy_extras.integrals.dirichlet`), for the integrators which are
still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back: the values of
:mod:`.conditions_sympy`, the trigonometric sums with SymPy's ``sin`` and
``cos`` as the kinds of the terms. Nothing of the module stays on SymPy.

Examples
========

>>> from sympy import symbols, sin, cos, oo
>>> from sympy_extras.integrals.dirichlet_sympy import dirichlet_integral
>>> x = symbols('x')
>>> dirichlet_integral(sin(x)**2/x**2, x, 0, oo)
ConditionalValue(pi/2)
>>> dirichlet_integral((cos(x) - cos(2*x))/x, x, 0, oo)
ConditionalValue(log(2))
>>> dirichlet_integral(sin(x)*sin(3*x)*sin(5*x)*sin(7*x)/(105*x**4), x, -oo, oo)
ConditionalValue(44*pi/315)
>>> dirichlet_integral(sin(x)**2/x, x, 0, oo) is None
True
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.functions.elementary.trigonometric import Sin
from sympy_extras.interop import from_sympy

from . import dirichlet as native
from .conditions_sympy import ConditionalValue

__all__ = ['dirichlet_integral', 'trigonometric_sum', 'TrigonometricSum']

#: a trigonometric sum: the constant term and ``{(w, kind): coefficient}``
#: for the terms ``coefficient * kind(w * x)``, ``kind`` being ``sin`` or ``cos``
TrigonometricSum = tuple[Expr, dict[tuple[Expr, type], Expr]]


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def trigonometric_sum(g: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[TrigonometricSum]:
    """:func:`sympy_extras.integrals.dirichlet.trigonometric_sum` on SymPy
    objects: ``g`` as a constant plus terms ``c*sin(w*x)`` and
    ``c*cos(w*x)`` with ``w > 0``; ``None`` when ``g`` is not such a sum or
    the sign of a frequency is unknown.

    >>> from sympy import symbols, sin, cos, pi
    >>> from sympy_extras.integrals.dirichlet_sympy import trigonometric_sum
    >>> x = symbols('x')
    >>> constant, terms = trigonometric_sum(sin(x)**2, x)
    >>> constant, terms
    (1/2, {(2, cos): -1/2})
    >>> trigonometric_sum(sin(x + pi/4), x)[1]
    {(1, sin): sqrt(2)/2, (1, cos): sqrt(2)/2}
    """
    found = native.trigonometric_sum(core_expr(g), _core_symbol(x), core_assumptions(assumptions))
    if found is None:
        return None
    constant, terms = found
    return sympy_expr(constant), {(sympy_expr(w), sin if kind is Sin else cos): sympy_expr(c)
                                  for (w, kind), c in terms.items()}


def dirichlet_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.dirichlet.dirichlet_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` for ``f`` a trigonometric sum (or
    product) over ``x**n`` and the range ``(0, oo)``, ``(-oo, 0)`` or
    ``(-oo, oo)``; ``None`` when ``f`` is not of this shape or the
    integral diverges.

    Examples
    ========

    >>> from sympy import symbols, sin, cos, oo
    >>> from sympy_extras.integrals.dirichlet_sympy import dirichlet_integral
    >>> x = symbols('x')
    >>> dirichlet_integral(sin(x)**3/x**3, x, 0, oo)
    ConditionalValue(3*pi/8)
    >>> dirichlet_integral((1 - cos(x))/x**2, x, -oo, oo)
    ConditionalValue(pi)
    >>> dirichlet_integral(sin(x)/x**2, x, 0, oo) is None
    True
    """
    found = native.dirichlet_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                      core_expr(sympify(b)), core_assumptions(assumptions))
    return None if found is None else ConditionalValue.from_core(found)
