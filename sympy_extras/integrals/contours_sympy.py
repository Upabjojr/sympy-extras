"""The SymPy face of the three further contours of the residue theorem
(:mod:`sympy_extras.integrals.contours`): the rectangle, the sector and
the indented real line, for the integrators which are still written on
SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). Nothing of the module stays on SymPy.

Examples
========

>>> from sympy import symbols, exp, sinh, sin, oo
>>> from sympy_extras.integrals.contours_sympy import contour_integral
>>> x = symbols('x')
>>> a, b, n = symbols('a b n', positive=True)
>>> contour_integral(x/sinh(x), x, -oo, oo)
ConditionalValue(pi**2/2)
>>> contour_integral(exp(a*x)/(1 + exp(x)), x, -oo, oo, a < 1)
ConditionalValue(pi/sin(pi*a), a < 1)
>>> contour_integral(x**a/(1 + x**n), x, 0, oo, a < n - 1)
ConditionalValue(pi/(n*sin(pi*(a + 1)/n)), a < n - 1)
>>> contour_integral(sin(a*x)/(x*(x**2 + b**2)), x, 0, oo)
ConditionalValue(pi*(exp(a*b) - 1)*exp(-a*b)/(2*b**2))
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.symbol import Symbol

from sympy_extras._numeric_sympy import core_expr
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy

from . import contours as native
from .conditions import ConditionalValue as CoreConditionalValue
from .conditions_sympy import ConditionalValue

__all__ = ['rectangular_integral', 'sector_integral', 'indented_integral', 'contour_integral']


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _conditional(v: Optional[CoreConditionalValue]) -> Optional[ConditionalValue]:
    return None if v is None else ConditionalValue.from_core(v)


def rectangular_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.contours.rectangular_integral` on SymPy
    objects: ``Integral(x**n * exp(k*x) * R(exp(c*x)), (x, -oo, oo))`` by
    the rectangular contour.

    >>> from sympy import symbols, cosh
    >>> from sympy_extras.integrals.contours_sympy import rectangular_integral
    >>> x = symbols('x')
    >>> rectangular_integral(1/cosh(x), x)
    ConditionalValue(pi)
    >>> rectangular_integral(x**2/cosh(x), x)
    ConditionalValue(pi**3/4)
    >>> rectangular_integral(1/(2*cosh(x) + 1), x)
    ConditionalValue(2*sqrt(3)*pi/9)
    """
    return _conditional(native.rectangular_integral(core_expr(as_expr(f)), _core_symbol(x),
                                                    core_assumptions(assumptions)))


def sector_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.contours.sector_integral` on SymPy
    objects: ``Integral(x**a/(b + c*x**n), (x, 0, oo))`` by the sector of
    angle ``2 pi/n``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.contours_sympy import sector_integral
    >>> x = symbols('x')
    >>> n = symbols('n', positive=True)
    >>> sector_integral(1/(1 + x**n), x, n > 1)
    ConditionalValue(pi/(n*sin(pi/n)), 0 < n - 1)
    >>> sector_integral(x**2/(3 + 2*x**6), x)
    ConditionalValue(sqrt(6)*pi/36)
    """
    return _conditional(native.sector_integral(core_expr(as_expr(f)), _core_symbol(x),
                                               core_assumptions(assumptions)))


def indented_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.contours.indented_integral` on SymPy
    objects: ``Integral(R(x)*sin(k*x), (x, -oo, oo))`` (or ``cos``) by the
    indented contour; over ``(0, oo)`` for an even integrand.

    >>> from sympy import symbols, sin, oo
    >>> from sympy_extras.integrals.contours_sympy import indented_integral
    >>> x = symbols('x')
    >>> indented_integral(sin(x)/x, x, -oo, oo)
    ConditionalValue(pi)
    >>> indented_integral(sin(x)/(x*(x**2 + 1)), x, 0, oo)
    ConditionalValue(-pi*exp(-1)/2 + pi/2)
    """
    return _conditional(native.indented_integral(core_expr(as_expr(f)), _core_symbol(x), core_expr(as_expr(a)),
                                                 core_expr(as_expr(b)), core_assumptions(assumptions)))


def contour_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.contours.contour_integral` on SymPy
    objects: the first of the three contours which applies to
    ``Integral(f, (x, a, b))``, or ``None``.

    >>> from sympy import symbols, cosh, oo
    >>> from sympy_extras.integrals.contours_sympy import contour_integral
    >>> x = symbols('x')
    >>> contour_integral(1/cosh(x), x, -oo, oo)
    ConditionalValue(pi)
    """
    return _conditional(native.contour_integral(core_expr(as_expr(f)), _core_symbol(x), core_expr(as_expr(a)),
                                                core_expr(as_expr(b)), core_assumptions(assumptions)))
