"""The SymPy face of the reliable numerical values
(:mod:`sympy_extras._numeric`).

:func:`reliable_form` and :func:`reliable_value` take and return SymPy
expressions, converted by :mod:`sympy_extras.interop` around the native
functions: the hidden zeros are proved, and the values computed, on the
core (mpmath at raised precisions, the core's ``is_zero`` and the native
minimal polynomials) rather than by SymPy's ``evalf`` and ``is_zero``.

Examples
========

>>> from sympy import log, sqrt, sin, cos
>>> from sympy.abc import x
>>> from sympy_extras._numeric_sympy import reliable_form, reliable_value
>>> z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
>>> reliable_form(log(-2*sqrt(2) - 2*sqrt(z), evaluate=False))
log(2*sqrt(2)) + I*pi
>>> reliable_value(log(-x - sqrt(z), evaluate=False), 15, {x: 2})
0.693147180559945 + 3.14159265358979*I
"""
from __future__ import annotations

from typing import Mapping, Optional, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._typing import as_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import _numeric as native

__all__ = ['reliable_form', 'reliable_value', 'core_expr', 'sympy_expr', 'core_values']


def core_expr(e: Union[Basic, int]) -> CoreExpr:
    """A SymPy expression (or a Python integer) as an expression of the
    core."""
    converted = from_sympy(sympify(e))
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def sympy_expr(e: CoreBasic) -> Expr:
    """An expression of the core as a SymPy expression."""
    return as_expr(to_sympy(e))


def core_values(values: Mapping[Symbol, Union[Expr, int]]) -> dict[CoreSymbol, CoreExpr]:
    """Values of SymPy symbols as values of the symbols of the core."""
    converted: dict[CoreSymbol, CoreExpr] = {}
    for s, v in values.items():
        symbol = from_sympy(s)
        if not isinstance(symbol, CoreSymbol):
            raise TypeError("%s is not a symbol" % (s,))
        converted[symbol] = core_expr(v)
    return converted


def reliable_form(e: Expr, digits: int = 20) -> Optional[Expr]:
    """:func:`sympy_extras._numeric.reliable_form` on SymPy objects: ``e``
    with its constant sums which are exactly zero written 0, or ``None``
    when the numerical value of a constant in it depends on a rounding
    error.

    Examples
    ========

    >>> from sympy import log, sqrt, sin, cos
    >>> from sympy_extras._numeric_sympy import reliable_form
    >>> z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
    >>> reliable_form(log(-2*sqrt(2) - 2*sqrt(z), evaluate=False))
    log(2*sqrt(2)) + I*pi
    >>> reliable_form(sqrt(sin(1)**2 + cos(1)**2 - 1) + 1) is None
    True
    >>> reliable_form(sin(1)**2 + cos(1)**2 - 1) is None
    False
    """
    try:
        converted = core_expr(e)
    except (TypeError, ValueError, RecursionError):
        return None
    form = native.reliable_form(converted, digits)
    if form is None:
        return None
    # nothing resolved: the expression itself, not its conversion back
    return e if form is converted else sympy_expr(form)


def reliable_value(e: Expr, digits: int, values: Optional[Mapping[Symbol, Union[Expr, int]]] = None) -> Optional[Expr]:
    """:func:`sympy_extras._numeric.reliable_value` on SymPy objects: the
    numerical value with ``digits`` digits of ``e`` at the ``values`` of
    its symbols, or ``None`` when it depends on a rounding error or is not
    a finite number.

    Examples
    ========

    >>> from sympy import log, sqrt, sin, cos
    >>> from sympy.abc import x
    >>> from sympy_extras._numeric_sympy import reliable_value
    >>> z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
    >>> reliable_value(log(-2*sqrt(2) - 2*sqrt(z), evaluate=False), 15)
    1.03972077083992 + 3.14159265358979*I
    >>> reliable_value(log(-x - sqrt(z), evaluate=False), 15, {x: 2})
    0.693147180559945 + 3.14159265358979*I
    >>> reliable_value(log(-1 - sqrt(sin(1)**2 + cos(1)**2 - 1)), 15) is None
    True
    """
    try:
        converted = core_expr(e)
        point = core_values(values) if values else None
    except (TypeError, ValueError, RecursionError):
        return None
    value = native.reliable_value(converted, digits, point)
    return None if value is None else sympy_expr(value)
