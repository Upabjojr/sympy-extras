"""The SymPy face of the mean value (Laurent coefficient) integrals of
periodic integrands (:mod:`sympy_extras.integrals.periodic`), for the
integrators which are still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). Nothing of the module stays on SymPy: only the
formal power series of a function other than ``exp``, ``sin``, ``cos``,
``sinh`` and ``cosh`` reaches SymPy's ``fps``, through the native
:mod:`.brackets`.

Examples
========

>>> from sympy import symbols, exp, cos, sin, pi
>>> from sympy_extras.integrals.periodic_sympy import mean_value_integral, laurent_coefficient
>>> x, z = symbols('x z')
>>> mean_value_integral(exp(cos(x))*cos(sin(x)), x, 0, 2*pi)
ConditionalValue(2*pi)
>>> mean_value_integral(exp(cos(x)), x, 0, 2*pi)
ConditionalValue(2*pi*besseli(0, 1))
>>> laurent_coefficient(exp(z)*exp(2/z), z, 1)
sqrt(2)*besseli(1, 2*sqrt(2))/2
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy

from . import periodic as native
from .conditions_sympy import ConditionalValue

__all__ = ['to_laurent', 'laurent_coefficient', 'mean_value_integral']


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def to_laurent(f: Expr, x: Symbol, z: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.periodic.to_laurent` on SymPy objects:
    ``f`` with the trigonometric functions of integer multiples of ``x``
    (and ``exp(I*k*x)``) written in ``z = exp(I*x)``; ``None`` when ``x``
    occurs otherwise.

    >>> from sympy import symbols, cos, sin
    >>> from sympy_extras.integrals.periodic_sympy import to_laurent
    >>> x, z = symbols('x z')
    >>> to_laurent(cos(2*x) + sin(x), x, z)
    z**2/2 - I*(z - 1/z)/2 + 1/(2*z**2)
    """
    found = native.to_laurent(core_expr(f), _core_symbol(x), _core_symbol(z))
    return None if found is None else sympy_expr(found)


def laurent_coefficient(F: ExprLike, z: Symbol, m: ExprLike = 0) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.periodic.laurent_coefficient` on SymPy
    objects: the coefficient of ``z**m`` in the Laurent expansion of ``F``
    about 0, for ``F`` a sum of terms ``constant * z**n * g(c z**p) * h(d
    z**q)``; ``None`` when ``F`` is not of this form or a sum does not
    close.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.periodic_sympy import laurent_coefficient
    >>> z = symbols('z')
    >>> laurent_coefficient(exp(z/2)*exp(1/(2*z)), z)
    besseli(0, 1)
    >>> laurent_coefficient(exp(z), z, 3)
    1/6
    >>> laurent_coefficient((z + 1/z)**4, z)
    6
    """
    found = native.laurent_coefficient(core_expr(sympify(F)), _core_symbol(z), core_expr(sympify(m)))
    return None if found is None else sympy_expr(found)


def mean_value_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                        assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.periodic.mean_value_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` over ``k`` full periods (``b - a =
    2 pi k``) as ``2 pi k`` times the constant Laurent coefficient of the
    integrand in ``z = exp(I x)``; ``None`` when the method does not apply.

    Examples
    ========

    >>> from sympy import symbols, exp, cos, sin, pi
    >>> from sympy_extras.integrals.periodic_sympy import mean_value_integral
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> mean_value_integral(exp(a*cos(x))*cos(x), x, 0, 2*pi)
    ConditionalValue(2*pi*besseli(1, a))
    >>> mean_value_integral(exp(cos(x))*sin(sin(x)), x, -pi, pi)
    ConditionalValue(0)
    >>> mean_value_integral(cos(x)**4, x, 0, 4*pi)
    ConditionalValue(3*pi/2)
    """
    found = native.mean_value_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                       core_expr(sympify(b)), core_assumptions(assumptions))
    return None if found is None else ConditionalValue.from_core(found)
