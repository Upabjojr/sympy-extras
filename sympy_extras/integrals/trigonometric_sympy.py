"""The SymPy face of the antiderivatives of trigonometric and hyperbolic
integrands (:mod:`sympy_extras.integrals.trigonometric`).

:func:`trigonometric_antiderivative` takes and returns SymPy objects,
converted by :mod:`sympy_extras.interop` around the native function; the
integrators still written on SymPy (:mod:`.indefinite`) use it.

Examples
========

>>> from sympy import symbols, sin, cos, exp, tan, asin
>>> from sympy_extras.integrals.trigonometric_sympy import trigonometric_antiderivative
>>> x = symbols('x')
>>> trigonometric_antiderivative(sin(x)**2, x)
(x - sin(x)*cos(x))/2
>>> trigonometric_antiderivative(tan(x)**3, x)
log(cos(x)) + tan(x)**2/2
>>> trigonometric_antiderivative(x*exp(x)*sin(x), x)
(2*x*(sin(x) - cos(x)) + 2*cos(x))*exp(x)/4
>>> trigonometric_antiderivative(asin(x), x)
x*asin(x) + sqrt(1 - x**2)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol

from . import trigonometric as native

__all__ = ['trigonometric_antiderivative']


def trigonometric_antiderivative(f: Expr, x: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.trigonometric.trigonometric_antiderivative`
    on SymPy objects: an antiderivative of ``f`` in ``x`` by the reduction
    formulas and substitutions of the trigonometric and hyperbolic
    families, checked by differentiation; ``None`` when ``f`` is outside
    them or the check fails.

    >>> from sympy import symbols, sin, cos, sinh, sec
    >>> from sympy_extras.integrals.trigonometric_sympy import trigonometric_antiderivative
    >>> x = symbols('x')
    >>> trigonometric_antiderivative(sin(x)**3*cos(x)**2, x)
    cos(x)**5/5 - cos(x)**3/3
    >>> trigonometric_antiderivative(sec(x), x)
    log(tan(x) + sec(x))
    >>> trigonometric_antiderivative(sinh(x)**2, x)
    (-x + sinh(x)*cosh(x))/2
    >>> trigonometric_antiderivative(1/(2 + cos(x)), x)
    2*sqrt(3)*atan(sqrt(3)*tan(x/2)/3)/3
    >>> trigonometric_antiderivative(x**2, x) is None
    True
    """
    found = native.trigonometric_antiderivative(core_expr(as_expr(f)), _core_symbol(x))
    return None if found is None else sympy_expr(found)


def _sin_cos(n: int, m: int, u: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.trigonometric._sin_cos` on SymPy
    objects: ``Integral(sin(u)**n * cos(u)**m, u)``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.trigonometric_sympy import _sin_cos
    >>> _sin_cos(0, -2, symbols('u'))
    sin(u)/cos(u)
    """
    found = native._sin_cos(n, m, _core_symbol(u))
    return None if found is None else sympy_expr(found)


def _sinh_cosh(n: int, m: int, u: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.trigonometric._sinh_cosh` on SymPy
    objects: ``Integral(sinh(u)**n * cosh(u)**m, u)``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.trigonometric_sympy import _sinh_cosh
    >>> _sinh_cosh(1, 0, symbols('u'))
    cosh(u)
    """
    found = native._sinh_cosh(n, m, _core_symbol(u))
    return None if found is None else sympy_expr(found)


def _verified(F: Expr, f: Expr, x: Symbol) -> bool:
    """:func:`sympy_extras.integrals.trigonometric._verified` on SymPy
    objects: whether ``F`` is an antiderivative of ``f``.

    >>> from sympy import symbols, sin, cos
    >>> from sympy_extras.integrals.trigonometric_sympy import _verified
    >>> x = symbols('x')
    >>> _verified(-cos(x), sin(x), x), _verified(cos(x), sin(x), x)
    (True, False)
    """
    return native._verified(core_expr(as_expr(F)), core_expr(as_expr(f)), _core_symbol(x))
