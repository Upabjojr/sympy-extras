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

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native derivatives and limits under assumptions. The
definite integrals of the derivative and at the anchors, and the
numerical check against a quadrature, are those of
:func:`~sympy_extras.integrals.definite.conditional_integral`: they are
asked of a :class:`ParametricIntegrator`, by default the native one of
:mod:`.definite` (:func:`~sympy_extras.integrals.definite.definite_integrator`),
and the antiderivative in the parameter is that of the native
:func:`~sympy_extras.integrals.indefinite.closed_antiderivative` (the
generic antiderivative, without the cases of the special values of the
other parameters: the value found is checked numerically).
The functions on SymPy objects are in :mod:`.parametric_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, oo
>>> from sympy_extras.functions import exp, sin, log
>>> from sympy_extras.integrals.parametric import parametric_integral
>>> x = Symbol('x')
>>> p, t = Symbol('p', positive=True), Symbol('t', positive=True)
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

import itertools
from abc import ABC, abstractmethod
from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.limits import limit
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Application
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import and_
from sympy_extras.core.numbers import nan, oo, zoo
from sympy_extras.core.power import Pow
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.settings import settings

from .conditions import ConditionalValue
from .laplace import has_integral

__all__ = ['parametric_integral', 'candidate_parameters', 'ParametricIntegrator']

#: the values of the parameter at which the integral is evaluated to fix
#: the constant of integration
_ANCHORS: Final[tuple[Expr, ...]] = (S.Zero, S.One)

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so a dummy ``q`` beside a parameter ``q`` would be
    taken for it."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


class ParametricIntegrator(ABC):
    """The definite integrator which differentiation under the integral
    sign calls, for the derivative and for the integrals at the anchors
    (it must not use the method again), and the numerical check of a
    result: the boundary with
    :func:`~sympy_extras.integrals.definite.conditional_integral`
    (:class:`sympy_extras.integrals.definite.DefiniteIntegrator`).
    """

    @abstractmethod
    def integrate(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """``Integral(f, (x, a, b))`` at the recursion depth ``depth``, or
        ``None``."""

    @abstractmethod
    def verify(self, value: Expr, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[bool]:
        """Whether ``value`` agrees with a quadrature of ``Integral(f, (x,
        a, b))`` at values of the parameters satisfying the assumptions:
        ``False`` when it disagrees, ``None`` when nothing was checked."""


def candidate_parameters(f: Expr, x: Symbol) -> list[Symbol]:
    """The parameters of ``f`` worth differentiating with respect to: the
    free symbols other than ``x`` which appear inside a function of
    ``x`` (``exp(-p*x)``, ``x**p``, ``log(1 + p*x)``), sorted by name.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.parametric import candidate_parameters
    >>> x, p, t = Symbol('x'), Symbol('p', positive=True), Symbol('t', positive=True)
    >>> candidate_parameters(exp(-p*x)*sin(x)/x, x), candidate_parameters(t*exp(-x), x)
    ([p], [])
    """
    found: set[Symbol] = set()
    for node in f.atoms(Application):
        if node.has(x):
            found |= node.free_symbols - {x}
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if base.has(x) and not exponent.has(x):
            found |= exponent.free_symbols - {x}
        if exponent.has(x) and not base.has(x):
            found |= base.free_symbols - {x}
    return sorted(found, key=lambda s: s.name)


def parametric_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike, parameter: Symbol,
                        assumptions: Assumptions = None, depth: int = 0,
                        integrator: Optional[ParametricIntegrator] = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` by differentiation under the integral
    sign with respect to ``parameter``: the derivative is integrated by
    the ``integrator`` (by default
    :func:`~sympy_extras.integrals.definite.conditional_integral` under the
    assumptions, which does not use this method again:
    :func:`sympy_extras.integrals.definite.definite_integrator`),
    integrated back in the parameter, and the constant is fixed at
    ``parameter = 0`` or ``1`` where the integral is computed directly,
    or by the limit of ``f`` at infinity when it vanishes there. ``None``
    when a step fails or the numerical check rejects the result."""
    p = parameter
    a_, b_ = to_expr(a), to_expr(b)
    if not f.has(p) or a_.has(p) or b_.has(p):
        return None
    derivative = f.diff(p)
    if derivative == 0:
        return None
    if integrator is None:
        from .definite import definite_integrator
        integrator = definite_integrator(assumptions)
    inner = integrator.integrate(derivative, x, a_, b_, depth + 1)
    if inner is None or has_integral(inner.value):
        return None
    from .indefinite import closed_antiderivative
    q = _fresh('q', real=True)
    primitive = closed_antiderivative(inner.value.subs(p, q), q, seconds=settings.timeout)
    if primitive is None:
        return None
    for anchor in _ANCHORS + (oo,):
        value = _anchor_value(f, x, a_, b_, p, anchor, integrator, assumptions, depth)
        if value is None:
            continue
        shift = _difference(primitive, q, p, anchor, assumptions)
        if shift is None:
            continue
        result = value.value + shift
        if result.has(nan, zoo, oo, -oo, Limit):
            continue
        if settings.numerical_checks and integrator.verify(result, f, x, a_, b_) is False:
            return None
        return ConditionalValue(result, and_(inner.condition, value.condition))
    return None


def _anchor_value(f: Expr, x: Symbol, a: Expr, b: Expr, p: Symbol, anchor: Expr,
                  integrator: ParametricIntegrator, assumptions: Assumptions,
                  depth: int) -> Optional[ConditionalValue]:
    """``I(anchor)``: the integral at ``p = anchor`` computed directly, or
    ``0`` at ``anchor = oo`` when the integrand tends to zero there. (The
    anchor may violate the assumptions on the parameter, ``p > 0``: it is
    still usable when the integral there is the limit of ``I(p)``.)"""
    if anchor == oo:
        vanishes = attempt(lambda: limit(f, p, oo, assumptions=assumptions), settings.timeout, native=True)
        if vanishes is not None and vanishes == 0 and _decays(f, x, p):
            return ConditionalValue(S.Zero)
        return None
    specialised = f.subs(p, anchor)
    if specialised.has(nan, zoo):
        return None
    return integrator.integrate(specialised, x, a, b, depth + 1)


def _decays(f: Expr, x: Symbol, p: Symbol) -> bool:
    """Whether ``f`` carries a factor ``exp(-p*g(x))`` with ``g > 0`` on
    the range, so that the integral tends to zero with ``p -> oo`` (the
    only case in which the anchor at infinity is used)."""
    for node in f.atoms(Exp):
        argument = node.args[0]
        if not isinstance(argument, Expr):
            continue
        coefficient, rest = argument.as_independent(p, as_Add=False)
        if rest == p and coefficient.is_negative:
            return True
        if rest == p and (-coefficient).is_positive:
            return True
        if rest == p * x or rest == -p * x:
            return (coefficient * (rest / p / x)).is_negative is True
    return False


def _difference(primitive: Expr, q: Symbol, p: Symbol, anchor: Expr,
                assumptions: Assumptions) -> Optional[Expr]:
    """``primitive(p) - primitive(anchor)``, the latter as a limit."""
    at_p = primitive.subs(q, p)
    at_anchor = attempt(lambda: limit(primitive, q, anchor, assumptions=assumptions), settings.timeout, native=True)
    if at_anchor is None or at_anchor.has(oo, -oo, zoo, nan, Limit):
        return None
    return at_p - at_anchor
