"""The SymPy face of the real antiderivatives of `x^n Q(x)^{m/2}`
(:mod:`sympy_extras.integrals.radicals`), and the step which stays on
SymPy.

:func:`quadratic_radical_antiderivative` and
:func:`euler_substitution_antiderivative` take and return SymPy objects,
converted by :mod:`sympy_extras.interop` around the native functions;
the integrators still written on SymPy (:mod:`.antiderivative`,
:mod:`.definite`, :mod:`.indefinite`) use them.

The step of the native module written on SymPy is called from it at run
time:

- :func:`ratint_of_core`: the rational integrator of
  :mod:`sympy_extras.integrals.risch.rationaltools` (SymPy's ``ratint``
  on SymPy objects), for the rational function of `t` which an Euler
  substitution leaves, and for the rational functions of
  :mod:`sympy_extras.integrals.trigonometric`.

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
from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol

from . import radicals as native

__all__ = ['quadratic_radical_antiderivative', 'euler_substitution_antiderivative', 'ratint_of_core']


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


# ---------------------------------------------------------------------------
# the step of the native module which stays on SymPy

def ratint_of_core(f: CoreExpr, x: CoreSymbol, seconds: Optional[float]) -> Optional[CoreExpr]:
    """The antiderivative of the rational function ``f`` of ``x`` by the
    ``ratint`` of :mod:`sympy_extras.integrals.risch.rationaltools`
    (written on SymPy) within ``seconds``: ``None`` when the time is up or
    it gives up as :func:`sympy_extras._timeout.attempt` says; the
    ``IndexError`` its internals raise on some inputs goes on to the
    caller.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.radicals_sympy import ratint_of_core
    >>> t = symbols('t')
    >>> ratint_of_core(1/(t**2 - 1), t, None)
    log(t - 1)/2 - log(t + 1)/2
    """
    from .risch.rationaltools import ratint
    integrand, variable = sympy_expr(f), _sympy_symbol(x)
    found = attempt(lambda: as_expr(ratint(integrand, variable)), seconds)
    return None if found is None else core_expr(found)
