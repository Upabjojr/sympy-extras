"""The SymPy face of the antiderivatives of exponentials of powers, of
quadratics and of rational functions of an exponential
(:mod:`sympy_extras.integrals.exponential`), for the integrators which are
still written on SymPy (:mod:`.indefinite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the antiderivatives back. The step of the
native module which stays on SymPy is here too: SymPy's rational
integrator ``ratint`` (:func:`ratint_of_core`, the port of
:mod:`.risch.rationaltools` which works on SymPy objects), which the
substitution ``u = exp(c*x)`` needs.

Examples
========

>>> from sympy import symbols, exp
>>> from sympy_extras.integrals.exponential_sympy import exponential_antiderivative
>>> x = symbols('x')
>>> v = symbols('v', positive=True)
>>> a = symbols('a', negative=True)
>>> exponential_antiderivative(x**(v - 1)*exp(a*x**2), x)
-uppergamma(v/2, -a*x**2)/(2*(-a)**(v/2))
>>> exponential_antiderivative(x**2*exp(x**2), x)
x*exp(x**2)/2 - sqrt(pi)*erfi(x)/4
>>> exponential_antiderivative(exp(x)/(exp(2*x) + 1), x)
atan(exp(x))
"""
from __future__ import annotations

from typing import Callable, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions as CoreAssumptions
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import exponential as native

__all__ = ['exponential_antiderivative', 'power_exponential', 'exponential_rational', 'nested_power_exponential',
           'rational_exponential', 'composite_gaussian', 'reciprocal_square_exponential', 'ratint_of_core']

#: a function of the native module
_Native = Callable[[CoreExpr, CoreSymbol, CoreAssumptions], Optional[CoreExpr]]


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    converted = to_sympy(s)
    if not isinstance(converted, Symbol):
        raise TypeError("%s is not a symbol" % (converted,))
    return converted


def ratint_of_core(f: CoreExpr, x: CoreSymbol, seconds: Optional[float]) -> Optional[CoreExpr]:
    """SymPy's ``ratint(f, x)`` (the port in :mod:`.risch.rationaltools`)
    within ``seconds``, on core expressions; ``None`` when it does not
    finish or its antiderivative cannot be converted."""
    from .risch.rationaltools import ratint
    integrand, variable = sympy_expr(f), _sympy_symbol(x)
    found = attempt(lambda: as_expr(ratint(integrand, variable)), seconds)
    if found is None:
        return None
    try:
        return core_expr(found)
    except (TypeError, ValueError, NotImplementedError):
        return None


def _run(function: _Native, f: ExprLike, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    found = function(core_expr(sympify(f)), _core_symbol(x), core_assumptions(assumptions))
    return None if found is None else sympy_expr(found)


def exponential_antiderivative(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.exponential.exponential_antiderivative`
    on SymPy objects: an antiderivative of ``f`` by the routes of the
    module, each result checked by differentiation; ``None`` when none
    applies.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential_sympy import exponential_antiderivative
    >>> x = symbols('x')
    >>> exponential_antiderivative(x*exp(x**2 + 2*x), x)
    exp(-1)*exp((x + 1)**2)/2 - sqrt(pi)*exp(-1)*erfi(x + 1)/2
    """
    f_ = sympify(f)
    if not f_.has(exp):
        # the native module's first test, without converting the integrand
        return None
    return _run(native.exponential_antiderivative, f_, x, assumptions)


def power_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.exponential.power_exponential` on SymPy
    objects: ``Integral(f, x)`` for a sum of terms ``c * x**(v - 1) *
    exp(a*x**n + b)``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential_sympy import power_exponential
    >>> x = symbols('x')
    >>> power_exponential(exp(-x**2), x)
    sqrt(pi)*erf(x)/2
    >>> power_exponential(x**3*exp(x**2), x)
    x**2*exp(x**2)/2 - exp(x**2)/2
    >>> power_exponential(exp(2*x)/x, x)
    Ei(2*x)
    """
    return _run(native.power_exponential, f, x, assumptions)


def exponential_rational(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.exponential.exponential_rational` on
    SymPy objects: ``Integral(f, x)`` for a rational function of
    ``exp(c*x)``, or one times ``(a + b*exp(c*x))**(m/q)``.

    >>> from sympy import symbols, exp, sqrt
    >>> from sympy_extras.integrals.exponential_sympy import exponential_rational
    >>> x = symbols('x')
    >>> exponential_rational(1/(exp(x) + 1), x)
    x - log(exp(x) + 1)
    >>> exponential_rational(exp(x)/sqrt(exp(x) + 1), x)
    2*sqrt(exp(x) + 1)
    """
    return _run(native.exponential_rational, f, x, assumptions)


def nested_power_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.exponential.nested_power_exponential`
    on SymPy objects: ``Integral(f, x)`` for terms ``c * V * exp(a*W + b)``
    with nested powers ``W = (x**r)**p``.

    >>> from sympy import symbols, exp, sqrt
    >>> from sympy_extras.integrals.exponential_sympy import nested_power_exponential
    >>> x, a, r, p = symbols('x a r p')
    >>> nested_power_exponential(exp(a*(x**r)**p), x)
    -x*uppergamma(1/(p*r), -a*(x**r)**p)/(p*r*(-a*(x**r)**p)**(1/(p*r)))
    >>> nested_power_exponential(exp(a*sqrt(x**2)), x)
    x*exp(a*sqrt(x**2))/(a*sqrt(x**2))
    """
    return _run(native.nested_power_exponential, f, x, assumptions)


def rational_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.exponential.rational_exponential` on
    SymPy objects: ``Integral(f, x)`` for ``R(x)*exp(c*x + d)`` by the
    exponential integral at the roots of the denominator.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential_sympy import rational_exponential
    >>> x, c = symbols('x c')
    >>> rational_exponential(exp(c*x)/(x**2 - 1), x)
    exp(c)*Ei(c*(x - 1))/2 - exp(-c)*Ei(c*(x + 1))/2
    """
    return _run(native.rational_exponential, f, x, assumptions)


def composite_gaussian(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.exponential.composite_gaussian` on
    SymPy objects: ``Integral(f, x)`` for ``k*g'(x)*exp(a*g(x)**2 + c)``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential_sympy import composite_gaussian
    >>> x = symbols('x')
    >>> composite_gaussian((1 - 1/x**2)*exp(-(x + 1/x)**2), x)
    sqrt(pi)*erf((x**2 + 1)/x)/2
    >>> composite_gaussian(exp((x + 1/x)**2)/x**2, x)
    -sqrt(pi)*erfi((x**2 + 1)/x)/4 + sqrt(pi)*exp(4)*erfi(x - 1/x)/4
    """
    return _run(native.composite_gaussian, f, x, assumptions)


def reciprocal_square_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.exponential.reciprocal_square_exponential`
    on SymPy objects: ``Integral(f, x)`` for ``c*exp(alpha*x**2 +
    beta/x**2 + gamma)``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential_sympy import reciprocal_square_exponential
    >>> x = symbols('x')
    >>> reciprocal_square_exponential(exp(-x**2 - 1/x**2), x)
    sqrt(pi)*(exp(-2)*erf(x - 1/x) + exp(2)*erf(x + 1/x))/4
    """
    return _run(native.reciprocal_square_exponential, f, x, assumptions)
