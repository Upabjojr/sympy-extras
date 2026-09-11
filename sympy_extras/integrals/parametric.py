"""Differentiation under the integral sign (Feynman's trick).

For an integral depending on a parameter, `I(p) = \\int_a^b f(x, p)\\, dx`,
differentiating under the integral sign gives

.. math::

    I'(p) = \\int_a^b \\frac{\\partial f}{\\partial p}\\, dx,

which is often a simpler integral (a factor `1/x` disappears against
`\\exp(-p x)` or `\\sin(p x)`, a logarithm against `x^p`). Integrating
`I'(p)` back in `p` and fixing the constant at a value `p_0` where `I`
is known (`p_0 = 0`, `1`, or a limit) gives `I(p)`. This is the way
Mathematica's ``Integrate`` handles many parametric integrals
[Wolfram]_, and the method Feynman made famous [Feynman]_; the
interchange of the derivative and the integral is justified when the
derivative converges uniformly near `p`, which is checked here
numerically when ``settings.numerical_checks`` is on (the value is
compared with quadrature at random values of the parameters satisfying
the assumptions) and otherwise assumed.

Examples
========

>>> from sympy import symbols, exp, sin, log, oo
>>> from sympy_extras.integrals.parametric import parametric_integral
>>> x = symbols('x')
>>> p, t = symbols('p t', positive=True)
>>> parametric_integral(exp(-p*x)*sin(x)/x, x, 0, oo, p)
ConditionalValue(-atan(p) + pi/2)
>>> parametric_integral(log(1 + t**2*x**2)/(1 + x**2), x, 0, oo, t)
ConditionalValue(pi*log(t + 1))
>>> parametric_integral((x**p - 1)/log(x), x, 0, 1, p)
ConditionalValue(log(p + 1))

References
==========

.. [Feynman] R. P. Feynman, *Surely You're Joking, Mr. Feynman!*, 1985
   ("A different box of tools"); the method is due to Leibniz, see
   F. S. Woods, *Advanced Calculus*, Ginn, 1926, section 61.
.. [Wolfram] Wolfram Research, *Some notes on internal implementation*,
   Integrate: "differentiation and integration with respect to
   parameters".
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.power import Pow
from sympy.core.numbers import nan, oo, zoo
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.integrals.integrals import Integral, integrate
from sympy.series.limits import Limit

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.limits import limit
from sympy_extras.settings import settings
from .conditions import ConditionalValue

__all__ = ['parametric_integral', 'candidate_parameters']

#: the values of the parameter at which the integral is evaluated to fix
#: the constant of integration
_ANCHORS: tuple[Expr, ...] = (S.Zero, S.One)


def candidate_parameters(f: Expr, x: Symbol) -> list[Symbol]:
    """The parameters of ``f`` worth differentiating with respect to: the
    free symbols other than ``x`` which appear inside a function of
    ``x`` (``exp(-p*x)``, ``x**p``, ``log(1 + p*x)``), sorted by name."""
    found: set[Symbol] = set()
    for node in f.atoms(Function):
        if node.has(x):
            found |= free_symbols(node) - {x}
    for node in f.atoms(Pow):
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if base.has(x) and exponent.has(x) is False:
            found |= free_symbols(exponent) - {x}
        if exponent.has(x) and not base.has(x):
            found |= free_symbols(base) - {x}
    return sorted_symbols(found)


def parametric_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike, parameter: Symbol,
                        assumptions: Assumptions = None, depth: int = 0) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` by differentiation under the integral
    sign with respect to ``parameter``: the derivative is integrated by
    :func:`~sympy_extras.integrals.conditional_integral` (which does not
    use this method again), integrated back in the parameter, and the
    constant is fixed at ``parameter = 0`` or ``1`` where the integral is
    computed directly, or by the limit of ``f`` at infinity when it
    vanishes there. ``None`` when a step fails or the numerical check
    rejects the result."""
    from .definite import _Integrator, verify_numerically
    p = parameter
    a, b = as_expr(a), as_expr(b)
    if not f.has(p) or a.has(p) or b.has(p):
        return None
    derivative = as_expr(f.diff(p))
    if derivative == 0:
        return None
    integrator = _Integrator(assumptions, parametric=False)
    inner = integrator.integrate(derivative, x, a, b, depth + 1, True)
    if inner is None or inner.value.has(Integral):
        return None
    q = Dummy('q', real=True)
    primitive = attempt(lambda: as_expr(integrate(inner.value.subs(p, q), q)), settings.timeout)
    if primitive is None or primitive.has(Integral):
        return None
    for anchor in _ANCHORS + (oo,):
        value = _anchor_value(f, x, a, b, p, anchor, integrator, depth)
        if value is None:
            continue
        shift = _difference(primitive, q, p, anchor, assumptions)
        if shift is None:
            continue
        result = as_expr(value.value + shift)
        if result.has(nan, zoo, oo, -oo, Limit):
            continue
        if settings.numerical_checks and verify_numerically(result, f, x, a, b, assumptions) is False:
            return None
        return ConditionalValue(result, as_boolean(inner.condition & value.condition))
    return None


def _anchor_value(f: Expr, x: Symbol, a: Expr, b: Expr, p: Symbol, anchor: Expr,
                  integrator: object, depth: int) -> Optional[ConditionalValue]:
    """``I(anchor)``: the integral at ``p = anchor`` computed directly, or
    ``0`` at ``anchor = oo`` when the integrand tends to zero there."""
    from .definite import _Integrator
    assert isinstance(integrator, _Integrator)
    if anchor == oo:
        vanishes = attempt(lambda: limit(f, p, oo, assumptions=integrator.assumptions), settings.timeout)
        if vanishes is not None and vanishes == 0 and _decays(f, x, p):
            return ConditionalValue(S.Zero)
        return None
    specialised = as_expr(f.subs(p, anchor))
    if specialised.has(nan, zoo):
        return None
    if integrator.ask(as_boolean(anchor > 0)) is False and integrator.assumptions is not None:
        # the anchor may violate the assumptions on the parameter (p > 0):
        # still usable when the integral there is the limit of I(p)
        pass
    return integrator.integrate(specialised, x, a, b, depth + 1, True)


def _decays(f: Expr, x: Symbol, p: Symbol) -> bool:
    """Whether ``f`` carries a factor ``exp(-p*g(x))`` with ``g > 0`` on
    the range, so that the integral tends to zero with ``p -> oo`` (the
    only case in which the anchor at infinity is used)."""
    from sympy.functions.elementary.exponential import exp
    for node in f.atoms(exp):
        argument = as_expr(node.args[0])
        coefficient, rest = argument.as_independent(p, as_Add=False)
        if as_expr(rest) == p and as_expr(coefficient).is_negative:
            return True
        if as_expr(rest) == p and as_expr(-coefficient).is_positive:
            return True
        if rest == p * x or rest == -p * x:
            return as_expr(coefficient * (rest / p / x)).is_negative is True
    return False


def _difference(primitive: Expr, q: Symbol, p: Symbol, anchor: Expr,
                assumptions: Assumptions) -> Optional[Expr]:
    """``primitive(p) - primitive(anchor)``, the latter as a limit."""
    at_p = as_expr(primitive.subs(q, p))
    at_anchor = attempt(lambda: limit(primitive, q, anchor, assumptions=assumptions), settings.timeout)
    if at_anchor is None or at_anchor.has(oo, -oo, zoo, nan, Limit):
        return None
    return as_expr(at_p - at_anchor)
