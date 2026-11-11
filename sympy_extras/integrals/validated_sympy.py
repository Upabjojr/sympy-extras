"""The SymPy face of the validated numerical integration
(:mod:`sympy_extras.integrals.validated`), for the integrators which are
still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the floats back; the enclosures are mpmath
intervals in both. Nothing of the module stays on SymPy.

Examples
========

>>> from sympy import symbols, exp, sqrt, pi, erf, oo
>>> from sympy_extras.integrals.validated_sympy import validated_integral
>>> x = symbols('x')
>>> value, error = validated_integral(exp(-x**2), x, 0, 1)
>>> round(value, 12), error < 1e-15
(0.746824132812, True)
>>> abs(value - sqrt(pi)*erf(1)/2) < error
True
>>> value, error = validated_integral(exp(-x**2), x, 0, oo)
>>> abs(value - sqrt(pi)/2) < error, error < 1e-15
(True, True)
"""
from __future__ import annotations

from typing import Optional

import mpmath
from mpmath.ctx_iv import ivmpf

from sympy.core.expr import Expr
from sympy.core.numbers import Float, Rational
from sympy.core.symbol import Symbol

from sympy_extras._numeric_sympy import core_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.numbers import Float as CoreFloat
from sympy_extras.core.numbers import Rational as CoreRational
from sympy_extras.core.numbers import fraction_to_mpf
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy

from . import validated as native

__all__ = ['validated_integral', 'enclosure']


def _core_symbol(s: Symbol) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _core_rational(r: Rational) -> CoreRational:
    converted = core_expr(r)
    if not isinstance(converted, CoreRational):
        raise TypeError("%s is not a rational number" % (r,))
    return converted


def _sympy_float(f: CoreFloat) -> Float:
    """A float of the core as a SymPy ``Float`` of the same precision
    (zero stays a ``Float``: SymPy makes ``Float(mpf(0))`` the integer
    ``0``)."""
    if f.as_fraction() == 0:
        return Float(0, precision=f.precision)
    with mpmath.workprec(f.precision):
        return Float(fraction_to_mpf(f.as_fraction()), precision=f.precision)


def validated_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, digits: int = 15,
                       assumptions: Assumptions = None) -> Optional[tuple[Expr, Expr]]:
    """:func:`sympy_extras.integrals.validated.validated_integral` on SymPy
    objects: ``(value, error)`` with ``|Integral(f, (x, a, b)) - value| <=
    error`` proved by interval arithmetic, the error aimed at
    ``10**(-digits)``; ``None`` when the integrand has parameters, uses
    functions the interval evaluator does not know, or cannot be bounded.

    Examples
    ========

    >>> from sympy import symbols, sqrt, pi, oo, Rational, log, sin
    >>> from sympy_extras.integrals.validated_sympy import validated_integral
    >>> x = symbols('x')
    >>> value, error = validated_integral(sqrt(x), x, 0, 1)
    >>> abs(value - Rational(2, 3)) < error, error < 1e-15
    (True, True)
    >>> value, error = validated_integral(1/(1 + x**2), x, 0, oo, 20)
    >>> abs(value - pi/2) < error, error < 1e-20
    (True, True)
    >>> value, error = validated_integral(log(x), x, 0, 1)
    >>> abs(value + 1) < error
    True
    >>> value, error = validated_integral(sin(x)/x, x, 0, oo, 8)
    >>> abs(value - pi/2) < error, error < 1e-8
    (True, True)
    >>> validated_integral(1/x, x, 0, 1) is None
    True
    """
    found = native.validated_integral(core_expr(f), _core_symbol(x), core_expr(a), core_expr(b), digits,
                                      core_assumptions(assumptions))
    if found is None:
        return None
    return (_sympy_float(found[0]), _sympy_float(found[1]))


def enclosure(f: Expr, x: Symbol, a: Rational, b: Rational, tolerance: mpmath.mpf,
              max_panels: int = native._MAX_PANELS) -> Optional[ivmpf]:
    """:func:`sympy_extras.integrals.validated.enclosure` on SymPy objects:
    an interval containing ``Integral(f, (x, a, b))`` for rational ``a <
    b``, refined until its width is below ``tolerance``; ``None`` when a
    panel cannot be bounded.

    >>> import mpmath
    >>> from sympy import Rational, symbols, log
    >>> from sympy_extras.integrals.validated_sympy import enclosure
    >>> x = symbols('x')
    >>> r = enclosure(log(1 + x), x, Rational(0), Rational(1), mpmath.mpf('1e-12'))
    >>> r.a < 2*mpmath.log(2) - 1 < r.b
    True
    """
    return native.enclosure(core_expr(f), _core_symbol(x), _core_rational(a), _core_rational(b), tolerance,
                            max_panels)
