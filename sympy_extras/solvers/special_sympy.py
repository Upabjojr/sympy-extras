"""The SymPy face of the second order linear equations solved by special
functions (:mod:`sympy_extras.solvers.special`): the functions take and
return SymPy objects, converted by :mod:`sympy_extras.interop` around the
native functions.

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x, a, n
>>> from sympy_extras.solvers.special_sympy import special_solutions
>>> y = Function('y')(x)
>>> special_solutions(y.diff(x, 2) + x*y, y)
[sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
>>> special_solutions((1 - x**2)*y.diff(x, 2) - 2*x*y.diff(x) + n*(n + 1)*y, y)
[sqrt(1/2 - x/2)*sqrt(x/2 + 1/2)*hyper((-n, n + 1), (1,), 1/2 - x/2)/sqrt(x**2 - 1)]
>>> special_solutions(x*y.diff(x, 2) + y.diff(x) + a*y, y)
[besselj(0, 2*sqrt(a)*sqrt(x)), bessely(0, 2*sqrt(a)*sqrt(x))]
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Symbol

from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.expr import Expr as CoreExpr

from . import special as native
from .linear_ode_sympy import _core_basic, _core_function

__all__ = ['special_solutions', 'bessel_solutions', 'whittaker_solutions', 'hypergeometric_solutions',
           'whittaker_m']


def _optional(found: Optional[list[CoreExpr]]) -> Optional[list[Expr]]:
    return None if found is None else [sympy_expr(e) for e in found]


def whittaker_m(kappa: Expr, mu: Expr, t: Expr) -> Expr:
    """Whittaker's function ``M_{kappa, mu}(t) = exp(-t/2) t**(mu + 1/2)
    1F1(mu - kappa + 1/2; 1 + 2 mu; t)``.

    >>> from sympy import S
    >>> from sympy.abc import t
    >>> from sympy_extras.solvers.special_sympy import whittaker_m
    >>> whittaker_m(S(0), S(1)/2, t)
    t*exp(-t/2)*hyper((1,), (2,), t)
    """
    return sympy_expr(native.whittaker_m(core_expr(as_expr(kappa)), core_expr(as_expr(mu)), core_expr(as_expr(t))))


def bessel_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` in Bessel functions, or ``None``.

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.special_sympy import bessel_solutions
    >>> bessel_solutions(-x, x)   # Airy
    [sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
    """
    return _optional(native.bessel_solutions(core_expr(as_expr(r)), _core_symbol(x)))


def whittaker_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` in Whittaker functions (through
    ``1F1``), or ``None``.

    >>> from sympy.abc import x, a
    >>> from sympy_extras.solvers.special_sympy import whittaker_solutions
    >>> whittaker_solutions(x**2 + a, x)[0]
    (x**2)**(3/4)*exp(-x**2/2)*hyper((a/4 + 3/4,), (3/2,), x**2)/sqrt(x)
    """
    return _optional(native.whittaker_solutions(core_expr(as_expr(r)), _core_symbol(x)))


def hypergeometric_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` through Gauss's hypergeometric function,
    or ``None``.

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.special_sympy import hypergeometric_solutions
    >>> hypergeometric_solutions(-1/(4*x**2) - 1/(4*(x - 1)**2) + 1/(x*(x - 1)), x)[0]
    sqrt(x)*sqrt(1 - x)*hyper((1/2 - sqrt(3)/2, 1/2 + sqrt(3)/2), (1,), 1 - x)
    """
    return _optional(native.hypergeometric_solutions(core_expr(as_expr(r)), _core_symbol(x)))


def special_solutions(equation: Basic, f: AppliedUndef) -> Optional[list[Expr]]:
    """Solutions of a second order linear equation with rational
    coefficients in Bessel, Whittaker or hypergeometric functions, or
    ``None`` (:func:`sympy_extras.solvers.special.special_solutions` on
    SymPy objects).

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.special_sympy import special_solutions
    >>> y = Function('y')(x)
    >>> special_solutions(y.diff(x, 2) + x*y, y)
    [sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
    """
    return _optional(native.special_solutions(_core_basic(equation), _core_function(f)))
