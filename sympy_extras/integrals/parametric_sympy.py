"""The SymPy face of differentiation under the integral sign
(:mod:`sympy_extras.integrals.parametric`), for the integrators which are
still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). The native module asks its definite integrals
and its numerical checks of a :class:`~sympy_extras.integrals.parametric.ParametricIntegrator`,
the native :class:`~sympy_extras.integrals.definite.DefiniteIntegrator`;
its antiderivative in the parameter is SymPy's ``integrate``
(:func:`antiderivative_of_core`).

Examples
========

>>> from sympy import symbols, exp, sin, log, oo
>>> from sympy_extras.integrals.parametric_sympy import parametric_integral
>>> x = symbols('x')
>>> p, t = symbols('p t', positive=True)
>>> parametric_integral(exp(-p*x)*sin(x)/x, x, 0, oo, p)
ConditionalValue(-atan(p) + pi/2)
>>> parametric_integral(log(1 + t**2*x**2)/(1 + x**2), x, 0, oo, t)
ConditionalValue(pi*log(t + 1))
>>> parametric_integral((x**p - 1)/log(x), x, 0, 1, p)
ConditionalValue(log(p + 1))
"""
from __future__ import annotations

from typing import Optional

from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.integrals.integrals import Integral, integrate

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr, as_symbol
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import to_sympy

from . import parametric as native
from .conditions_sympy import ConditionalValue
from .definite import DefiniteIntegrator

__all__ = ['parametric_integral', 'candidate_parameters', 'antiderivative_of_core']


def antiderivative_of_core(f: CoreExpr, x: CoreSymbol, seconds: Optional[float]) -> Optional[CoreExpr]:
    """SymPy's ``integrate(f, x)`` within ``seconds``, on core expressions
    (the step of :mod:`sympy_extras.integrals.parametric` which stays on
    SymPy); ``None`` when it does not finish, leaves an ``Integral`` or
    its antiderivative cannot be converted."""
    integrand, variable = sympy_expr(f), _sympy_symbol(x)
    found = attempt(lambda: as_expr(integrate(integrand, variable)), seconds)
    if found is None or found.has(Integral):
        return None
    try:
        return core_expr(found)
    except (TypeError, ValueError, NotImplementedError):
        return None


def candidate_parameters(f: ExprLike, x: Symbol) -> list[Symbol]:
    """:func:`sympy_extras.integrals.parametric.candidate_parameters` on
    SymPy objects: the parameters of ``f`` worth differentiating with
    respect to, sorted by name.

    >>> from sympy import symbols, exp, sin, log
    >>> from sympy_extras.integrals.parametric_sympy import candidate_parameters
    >>> x = symbols('x')
    >>> p, t = symbols('p t', positive=True)
    >>> candidate_parameters(exp(-p*x)*sin(x)/x, x), candidate_parameters(log(1 + t**2*x**2)/(1 + x**2), x)
    ([p], [t])
    """
    return [as_symbol(to_sympy(s)) for s in native.candidate_parameters(core_expr(sympify(f)), _core_symbol(x))]


def parametric_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, parameter: Symbol,
                        assumptions: Assumptions = None, depth: int = 0) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.parametric.parametric_integral` on
    SymPy objects: ``Integral(f, (x, a, b))`` by differentiation under the
    integral sign with respect to ``parameter``, the integrals by
    the native :class:`~sympy_extras.integrals.definite.DefiniteIntegrator`;
    ``None`` when a step fails or the
    numerical check rejects the result.

    >>> from sympy import symbols, atan, oo
    >>> from sympy_extras.integrals.parametric_sympy import parametric_integral
    >>> x = symbols('x')
    >>> p = symbols('p', positive=True)
    >>> parametric_integral(atan(p*x)/(x*(1 + x**2)), x, 0, oo, p)
    ConditionalValue(pi*log(2*p + 2)/2 - pi*log(2)/2)
    """
    found = native.parametric_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                       core_expr(sympify(b)), _core_symbol(parameter),
                                       core_assumptions(assumptions), depth,
                                       DefiniteIntegrator(core_assumptions(assumptions)))
    return None if found is None else ConditionalValue.from_core(found)
