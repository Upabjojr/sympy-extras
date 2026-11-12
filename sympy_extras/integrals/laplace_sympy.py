"""The SymPy face of the operational rules of the Laplace transform
(:mod:`sympy_extras.integrals.laplace`), for the integrators which are
still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). The steps of the native integrators of this
package which stay on SymPy are here too, on core expressions (the
functions ending in ``_of_core``, imported by the native modules at run
time): the definite integrals of the pieces by
:func:`~sympy_extras.integrals.definite.definite_integral` and
:func:`~sympy_extras.integrals.definite.conditional_integral`, not ported
yet (:func:`definite_integral_of_core`, :func:`conditional_integral_of_core`,
also used by :mod:`.summability`, :mod:`.transformations` and
:mod:`.parametric`).

Examples
========

>>> from sympy import symbols, cos, sin, Abs, oo, exp
>>> from sympy_extras.integrals.laplace_sympy import laplace_rules, laplace_integral
>>> t = symbols('t')
>>> s, a, b = symbols('s a b', positive=True)
>>> laplace_rules((cos(a*t) - cos(b*t))/t, t, s)
ConditionalValue((-log(a**2 + s**2) + log(b**2 + s**2))/2)
>>> laplace_rules(Abs(sin(a*t)), t, s)
ConditionalValue(a/((a**2 + s**2)*tanh(pi*s/(2*a))))
>>> laplace_integral(t**2*exp(a*t)*exp(-s*t), t, 0, oo, s > a)
ConditionalValue(2/(-a + s)**3, -a + s > 0)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions import ask as native_ask
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions, sympy_boolean
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean, to_boolean
from sympy_extras.core.symbol import Symbol as CoreSymbol

from . import conditions as native_conditions
from . import laplace as native
from .conditions_sympy import ConditionalValue

__all__ = ['laplace_rules', 'laplace_integral', 'laplace_of_convolution', 'transform', 'real_arguments',
           'definite_integral_of_core', 'conditional_integral_of_core']


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


def transform(f: ExprLike, t: Symbol, u: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.laplace.transform` on SymPy objects:
    ``Integral(f*exp(-u*t), (t, 0, oo))`` by
    :func:`~sympy_extras.integrals.definite.definite_integral`, ``None``
    when it is not found or comes with an undecided condition.

    >>> from sympy import symbols, sin
    >>> from sympy_extras.integrals.laplace_sympy import transform
    >>> t, v = symbols('t'), symbols('v', positive=True)
    >>> transform(sin(t), t, v)
    1/(v**2 + 1)
    """
    found = native.transform(core_expr(sympify(f)), _core_symbol(t), _core_symbol(u), core_assumptions(assumptions))
    return None if found is None else sympy_expr(found)


def laplace_of_convolution(f: ExprLike, t: Symbol, s: ExprLike,
                           assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.laplace.laplace_of_convolution` on
    SymPy objects: ``L[Integral(p(u) q(t - u), (u, 0, t))](s) = P(s)
    Q(s)``, with the integral rule ``L[Integral(p, (u, 0, t))] = P(s)/s``
    as the case ``q = 1``; ``None`` when ``f`` is not such an integral.

    Examples
    ========

    >>> from sympy import symbols, Integral, sin
    >>> from sympy_extras.integrals.laplace_sympy import laplace_of_convolution
    >>> t, u = symbols('t u')
    >>> s = symbols('s', positive=True)
    >>> laplace_of_convolution(Integral(sin(t - u)*u, (u, 0, t)), t, s)
    ConditionalValue(1/(s**2*(s**2 + 1)))
    """
    return _conditional(native.laplace_of_convolution(core_expr(sympify(f)), _core_symbol(t),
                                                      core_expr(sympify(s)), core_assumptions(assumptions)))


def real_arguments(value: ExprLike, assumptions: Assumptions = None) -> Expr:
    """:func:`sympy_extras.integrals.laplace.real_arguments` on SymPy
    objects: ``arg(x + I*y)`` with the signs of the real numbers ``x`` and
    ``y`` decided under the assumptions written as ``atan(y/x)`` (``x >
    0``) or ``atan(y/x) +- pi`` (``x < 0``); then simplified.

    >>> from sympy import symbols, arg, I, pi
    >>> from sympy_extras.integrals.laplace_sympy import real_arguments
    >>> s, a = symbols('s a', positive=True)
    >>> real_arguments(arg(-s - I*a)/2 - arg(-s + I*a)/2 + pi)
    atan(a/s)
    """
    return sympy_expr(native.real_arguments(core_expr(sympify(value)), core_assumptions(assumptions)))


def laplace_rules(f: ExprLike, t: Symbol, s: ExprLike, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.laplace.laplace_rules` on SymPy
    objects: ``Integral(f*exp(-s*t), (t, 0, oo))`` by the first applicable
    rule (the shift by a step function, the exponential shift, a period,
    a convolution, the division by ``t``, the multiplication by ``t``),
    or ``None``.

    Examples
    ========

    >>> from sympy import symbols, sin, cos, Heaviside, exp
    >>> from sympy_extras.integrals.laplace_sympy import laplace_rules
    >>> t = symbols('t')
    >>> s, a = symbols('s a', positive=True)
    >>> laplace_rules(sin(a*t)/t, t, s)
    ConditionalValue(atan(a/s))
    >>> laplace_rules(Heaviside(t - 1)*exp(-t), t, s)
    ConditionalValue(exp(-s - 1)/(s + 1))
    """
    return _conditional(native.laplace_rules(core_expr(sympify(f)), _core_symbol(t), core_expr(sympify(s)),
                                             core_assumptions(assumptions)))


def laplace_integral(f: ExprLike, t: Symbol, a: ExprLike, b: ExprLike,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.laplace.laplace_integral` on SymPy
    objects: ``Integral(f, (t, a, b))`` when it is a Laplace transform
    ``Integral(g(t)*exp(-s*t), (t, 0, oo))`` with ``s`` a symbol or a
    positive number, by :func:`laplace_rules` on ``g``; ``None``
    otherwise.

    Examples
    ========

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.laplace_sympy import laplace_integral
    >>> t = symbols('t')
    >>> s, a = symbols('s a', positive=True)
    >>> laplace_integral((exp(-a*t) - exp(-2*a*t))*exp(-s*t)/t, t, 0, oo)
    ConditionalValue(log((2*a + s)/(a + s)))
    """
    return _conditional(native.laplace_integral(core_expr(sympify(f)), _core_symbol(t), core_expr(sympify(a)),
                                                core_expr(sympify(b)), core_assumptions(assumptions)))


# ---------------------------------------------------------------------------
# the steps of the native integrators which stay on SymPy

def _sympy_assumptions(assumptions: native_ask.Assumptions) -> Assumptions:
    if assumptions is None:
        return None
    if isinstance(assumptions, bool):
        return assumptions
    if isinstance(assumptions, CoreBoolean):
        return sympy_boolean(assumptions)
    items: list[Boolean] = [sympy_boolean(to_boolean(a)) for a in assumptions]
    return items


def definite_integral_of_core(f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                              assumptions: native_ask.Assumptions, special_values: bool,
                              seconds: Optional[float]) -> Optional[CoreExpr]:
    """:func:`~sympy_extras.integrals.definite.definite_integral` of
    ``Integral(f, (x, a, b))`` within ``seconds``, on core expressions
    (not ported yet); ``None`` when it does not finish or its value
    cannot be converted."""
    from .definite import definite_integral
    integrand, variable, lower, upper = sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b)
    facts = _sympy_assumptions(assumptions)
    found = attempt(lambda: definite_integral(integrand, (variable, lower, upper), facts,
                                              special_values=special_values), seconds)
    if found is None:
        return None
    try:
        return core_expr(found)
    except (TypeError, ValueError, NotImplementedError):
        return None


def conditional_integral_of_core(f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                                 assumptions: native_ask.Assumptions,
                                 seconds: Optional[float]) -> Optional[native_conditions.ConditionalValue]:
    """:func:`~sympy_extras.integrals.definite.conditional_integral` of
    ``Integral(f, (x, a, b))`` within ``seconds`` (no limit for
    ``None``), on core expressions (not ported yet); ``None`` when it is
    not found, does not finish or cannot be converted."""
    from .definite import conditional_integral
    integrand, variable, lower, upper = sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b)
    facts = _sympy_assumptions(assumptions)
    found = attempt(lambda: conditional_integral(integrand, variable, lower, upper, facts), seconds)
    if found is None:
        return None
    try:
        return found.to_core()
    except (TypeError, ValueError, NotImplementedError):
        return None
