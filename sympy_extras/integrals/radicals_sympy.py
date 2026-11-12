"""The SymPy face of the real antiderivatives of `x^n Q(x)^{m/2}`
(:mod:`sympy_extras.integrals.radicals`).

:func:`quadratic_radical_antiderivative` and
:func:`euler_substitution_antiderivative` take and return SymPy objects,
converted by :mod:`sympy_extras.interop` around the native functions;
the integrators still written on SymPy (:mod:`.definite`) use them.

Examples
========

>>> from sympy import symbols, sqrt
>>> from sympy_extras.integrals.radicals_sympy import quadratic_radical_antiderivative
>>> x = symbols('x')
>>> quadratic_radical_antiderivative(sqrt(1 - x**2), x)
x*sqrt(1 - x**2)/2 + asin(x)/2
>>> quadratic_radical_antiderivative(x**2/sqrt(x**2 + 1), x)
x*sqrt(x**2 + 1)/2 - asinh(x)/2
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, core_assumptions

from . import radicals as native

__all__ = ['quadratic_radical_antiderivative', 'euler_substitution_antiderivative']


def quadratic_radical_antiderivative(f: Expr, x: Symbol, assumptions: Assumptions = None,
                                     euler: bool = True) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.radicals.quadratic_radical_antiderivative`
    on SymPy objects: a real antiderivative of a sum of terms ``c * x**n *
    Q**(m/2)``, ``Q`` quadratic or linear, ``m`` odd, from the reduction
    formulas of Gradshteyn and Ryzhik 2.26 (a rational function of ``x``
    and ``sqrt(Q)`` of another form by an Euler substitution, not with
    ``euler=False``), checked by differentiation; ``None`` otherwise.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.radicals_sympy import quadratic_radical_antiderivative
    >>> x = symbols('x')
    >>> quadratic_radical_antiderivative(1/sqrt(2*x - x**2), x)
    asin(x - 1)
    >>> quadratic_radical_antiderivative(x**3*sqrt(x**2 + 1) + x, x)
    x**2*(x**2 + 1)**(3/2)/5 + x**2/2 - 2*(x**2 + 1)**(3/2)/15
    >>> quadratic_radical_antiderivative(sqrt(x**2 + 8) - 2*sqrt(x**2 + 4*x + 2), x)
    x*sqrt(x**2 + 8)/2 - (2*x + 4)*sqrt(x**2 + 4*x + 2)/2 + 2*log(2*x + 2*sqrt(x**2 + 4*x + 2) + 4) + 4*asinh(sqrt(2)*x/4)
    >>> quadratic_radical_antiderivative(sqrt(x**3 + 1), x) is None
    True
    """
    found = native.quadratic_radical_antiderivative(core_expr(as_expr(f)), _core_symbol(x),
                                                    core_assumptions(assumptions), euler)
    return None if found is None else sympy_expr(found)


def euler_substitution_antiderivative(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.radicals.euler_substitution_antiderivative`
    on SymPy objects: an antiderivative of a rational function of ``x``
    and ``sqrt(Q)``, ``Q`` of degree one or two, by the substitution
    ``t = sqrt(Q)`` or an Euler substitution, checked by differentiation;
    ``None`` otherwise.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.radicals_sympy import euler_substitution_antiderivative
    >>> x = symbols('x')
    >>> euler_substitution_antiderivative(1/((x + 3)*sqrt(x**2 - 1)), x)
    sqrt(2)*log(x + sqrt(x**2 - 1) - 2*sqrt(2) + 3)/4 - sqrt(2)*log(x + sqrt(x**2 - 1) + 2*sqrt(2) + 3)/4
    >>> euler_substitution_antiderivative(sqrt(x + 1)/x, x)
    2*sqrt(x + 1) + log(sqrt(x + 1) - 1) - log(sqrt(x + 1) + 1)
    """
    found = native.euler_substitution_antiderivative(core_expr(as_expr(f)), _core_symbol(x),
                                                     core_assumptions(assumptions))
    return None if found is None else sympy_expr(found)


def _odd_power(base: Expr, x: Symbol) -> tuple[Expr, int]:
    """:func:`sympy_extras.integrals.radicals._odd_power` on SymPy objects:
    ``(R, k)`` with ``base == R**k`` for an odd ``k``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.radicals_sympy import _odd_power
    >>> x = symbols('x')
    >>> _odd_power((x**2 + 1)**3, x)
    (x**2 + 1, 3)
    """
    R, k = native._odd_power(core_expr(as_expr(base)), _core_symbol(x))
    return sympy_expr(R), k
