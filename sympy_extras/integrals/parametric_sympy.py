"""The SymPy face of differentiation under the integral sign
(:mod:`sympy_extras.integrals.parametric`), for the integrators which are
still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). The native module asks its definite integrals
and its numerical checks of a :class:`~sympy_extras.integrals.parametric.ParametricIntegrator`:
here :class:`DefiniteIntegrator`, on the integrator of
:mod:`.definite` (``_Integrator`` without the method, and
``verify_numerically``); its antiderivative in the parameter is SymPy's
``integrate`` (:func:`antiderivative_of_core`).

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

from typing import Final, Optional

from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.integrals.integrals import Integral, integrate

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr, as_symbol
from sympy_extras.assumptions import ask as native_ask
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import to_sympy

from . import conditions as native_conditions
from . import parametric as native
from .conditions_sympy import ConditionalValue
from .definite import _Integrator, verify_numerically
from .laplace_sympy import _sympy_assumptions

__all__ = ['parametric_integral', 'candidate_parameters', 'DefiniteIntegrator', 'definite_integrator',
           'antiderivative_of_core']


class DefiniteIntegrator(native.ParametricIntegrator):
    """The integrator of :mod:`.definite` (without differentiation under
    the integral sign) and its ``verify_numerically``, under assumptions
    given with SymPy objects, on core expressions."""

    __slots__ = ('assumptions', '_integrator')

    def __init__(self, assumptions: Assumptions) -> None:
        self.assumptions: Final[Assumptions] = assumptions
        self._integrator: Final[_Integrator] = _Integrator(assumptions, parametric=False)

    def integrate(self, f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                  depth: int) -> Optional[native_conditions.ConditionalValue]:
        found = self._integrator.integrate(sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b),
                                           depth, True)
        if found is None:
            return None
        try:
            return found.to_core()
        except (TypeError, ValueError, NotImplementedError):
            return None

    def verify(self, value: CoreExpr, f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr) -> Optional[bool]:
        return verify_numerically(sympy_expr(value), sympy_expr(f), _sympy_symbol(x), sympy_expr(a),
                                  sympy_expr(b), self.assumptions)


def definite_integrator(assumptions: native_ask.Assumptions) -> DefiniteIntegrator:
    """The :class:`DefiniteIntegrator` under assumptions of the core (the
    default integrator of :func:`sympy_extras.integrals.parametric.parametric_integral`)."""
    return DefiniteIntegrator(_sympy_assumptions(assumptions))


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
    :class:`DefiniteIntegrator`; ``None`` when a step fails or the
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
                                       core_assumptions(assumptions), depth, DefiniteIntegrator(assumptions))
    return None if found is None else ConditionalValue.from_core(found)
