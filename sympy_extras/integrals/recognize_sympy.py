"""The SymPy face of the symbolic-numeric recognition of constants
(:mod:`sympy_extras.integrals.recognize`), for the integrators which are
still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the closed forms back; the integrands are
evaluated by the numerical rules of the core's nodes. Nothing of the
module stays on SymPy.

Examples
========

>>> from sympy import Float, symbols
>>> from sympy_extras.integrals.recognize_sympy import recognize_constant, recognize_integral
>>> recognize_constant(Float('1.6449340668482264364724151666460251892', 38))
pi**2/6
>>> x = symbols('x')
>>> recognize_integral(1/(x**3 + 1), (x, 0, 1))
log(2)/3 + sqrt(3)*pi/9
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import mpmath

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy

from . import recognize as native

__all__ = ['DEFAULT_CONSTANTS', 'recognize_constant', 'recognize_integral', 'quadrature']

#: the constants a closed form is searched in (those of the native module)
DEFAULT_CONSTANTS: tuple[Expr, ...] = tuple(sympy_expr(c) for c in native.DEFAULT_CONSTANTS)


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _core_constants(constants: Sequence[Expr]) -> list[CoreExpr]:
    return [core_expr(c) for c in constants]


def recognize_constant(value: Union[Expr, float, str], constants: Sequence[Expr] = DEFAULT_CONSTANTS,
                       digits: int = 30) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.recognize.recognize_constant` on SymPy
    objects: a closed form of ``value`` as a rational combination of 1 and
    at most two of the ``constants``, found by PSLQ; ``None`` when none is
    found or numerical checks are off.

    >>> from sympy import Float, pi, zeta
    >>> from sympy_extras.integrals.recognize_sympy import recognize_constant
    >>> recognize_constant(Float('0.91596559417721901505460351493238411077', 38))
    Catalan
    >>> recognize_constant(pi**2/6 + zeta(3)/7)
    zeta(3)/7 + pi**2/6
    """
    converted: Union[CoreExpr, float, str] = value if isinstance(value, (float, str)) else core_expr(value)
    found = native.recognize_constant(converted, _core_constants(constants), digits)
    return None if found is None else sympy_expr(found)


def quadrature(f: Expr, x: Symbol, a: Expr, b: Expr, digits: int = 30) -> Optional[mpmath.mpf]:
    """:func:`sympy_extras.integrals.recognize.quadrature` on SymPy
    objects: ``Integral(f, (x, a, b))`` by two quadrature rules which must
    agree to ``digits - 5`` digits; ``None`` otherwise.

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.recognize_sympy import quadrature
    >>> x = symbols('x')
    >>> print(quadrature(exp(-x), x, 0, oo, 20))
    1.0
    """
    return native.quadrature(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)), core_expr(sympify(b)),
                             digits)


def recognize_integral(f: ExprLike, limits: tuple[Symbol, ExprLike, ExprLike],
                       assumptions: Assumptions = None, constants: Sequence[Expr] = DEFAULT_CONSTANTS,
                       digits: int = 30) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.recognize.recognize_integral` on SymPy
    objects: a conjectured closed form of ``Integral(f, (x, a, b))`` from
    its numerical value; ``None`` when nothing is recognised.

    >>> from sympy import symbols, atan
    >>> from sympy_extras.integrals.recognize_sympy import recognize_integral
    >>> x = symbols('x')
    >>> recognize_integral(atan(x)/x, (x, 0, 1))
    Catalan
    """
    x, a, b = limits
    found = native.recognize_integral(core_expr(sympify(f)), (_core_symbol(x), core_expr(sympify(a)),
                                                            core_expr(sympify(b))),
                                      core_assumptions(assumptions), _core_constants(constants), digits)
    return None if found is None else sympy_expr(found)
