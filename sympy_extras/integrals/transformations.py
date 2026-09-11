"""Integral identities which change the integrand rather than compute
it: Frullani's theorem, the Cauchy–Schlömilch transformation and
Glasser's master theorem.

**Frullani's theorem** [Frullani]_: for a function with finite limits at
`0` and `\\infty`,

.. math::

    \\int_0^\\infty \\frac{f(ax) - f(bx)}{x}\\, dx = \\bigl(f(0) - f(\\infty)\\bigr) \\log\\frac{b}{a},
    \\qquad a, b > 0.

**Glasser's master theorem** [Glasser]_: for `a_i > 0`,

.. math::

    \\int_{-\\infty}^\\infty F\\left(x - \\sum_i \\frac{a_i}{x - b_i}\\right) dx
    = \\int_{-\\infty}^\\infty F(x)\\, dx,

whenever the right side converges; the **Cauchy–Schlömilch
transformation** is the case of one term and an even `F`,
`\\int_0^\\infty F\\bigl((x - a/x)^2\\bigr) dx = \\int_0^\\infty F(x^2)\\, dx`
[Amdeberhan]_. The theorem holds because the map `x \\mapsto x - \\sum a_i/(x - b_i)`
is a bijection of each interval between consecutive poles onto the real
line with Jacobians summing to one (Boros and Moll, chapter 13).

The functions here recognise the shapes and hand the transformed integral
to :func:`sympy_extras.integrals.definite_integral`.

Examples
========

>>> from sympy import symbols, exp, cos, atan, oo
>>> from sympy_extras.integrals.transformations import frullani, glasser
>>> x = symbols('x')
>>> a, b = symbols('a b', positive=True)
>>> frullani((exp(-a*x) - exp(-b*x))/x, x, a > 0)
ConditionalValue(log(b/a))
>>> glasser(exp(-(x - 1/x)**2), x, 0, oo)
ConditionalValue(sqrt(pi)/2)

References
==========

.. [Frullani] G. Frullani, *Sopra gli integrali definiti*, Memorie della
   Società Italiana delle Scienze 20 (1828); the modern statement in
   G. Boros, V. Moll, *Irresistible Integrals*, Cambridge University Press,
   2004, section 5.2.
.. [Glasser] M. L. Glasser, *A remarkable property of definite integrals*,
   Mathematics of Computation 40 (1983) 561–563.
.. [Amdeberhan] T. Amdeberhan, M. L. Glasser, M. C. Jones, V. Moll,
   R. Posey, D. Varela, *The Cauchy–Schlömilch transformation*, Integral
   Transforms and Special Functions 30 (2019); Boros and Moll, chapter 13.
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import nan, oo, zoo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log
from sympy.polys.partfrac import apart
from sympy.series.limits import Limit
from sympy.calculus.accumulationbounds import AccumBounds

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.limits import limit
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .mellin import monomial

__all__ = ['frullani', 'glasser', 'transformation_integral']


def _limit(f: Expr, x: Symbol, point: Expr, assumptions: Assumptions) -> Optional[Expr]:
    value = attempt(lambda: limit(f, x, point, assumptions=assumptions), settings.timeout)
    if value is None or value.has(oo, -oo, zoo, nan, Limit, AccumBounds) or value.free_symbols - f.free_symbols:
        return None
    return value


def frullani(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, 0, oo))`` by Frullani's theorem for
    ``f = c*(g(a*x) - g(b*x))/x``; ``None`` when ``f`` is not of this shape
    or the limits of ``g`` at ``0`` and ``oo`` are not finite. The
    condition ``a > 0`` and ``b > 0`` is reported."""
    terms = [as_expr(t) for t in Add.make_args(as_expr(f.expand()))]
    if len(terms) != 2:
        return None
    parts: list[tuple[Expr, Expr, Expr]] = []
    for term in terms:
        coefficient, rest = term.as_independent(x, as_Add=False)
        rest_ = as_expr(rest)
        if not isinstance(rest_, Mul):
            return None
        factors = [as_expr(u) for u in rest_.args]
        if len(factors) != 2 or Pow(x, -1) not in factors:
            return None
        function = factors[0] if factors[1] == Pow(x, -1) else factors[1]
        if not isinstance(function, Function) or len(function.args) != 1:
            return None
        found = monomial(as_expr(function.args[0]), x)
        if found is None or found[1] != 1:
            return None
        parts.append((as_expr(coefficient), found[0], as_expr(function.func(x))))
    (c1, a, g1), (c2, b, g2) = parts
    if g1 != g2 or as_expr(c1 + c2) != 0:
        return None
    if a.is_negative and b.is_negative:
        # g(a x) = h(|a| x) with h(x) = g(-x)
        a, b, g1 = -a, -b, as_expr(g1.subs(x, -x))
    if ask(as_boolean(a > 0), assumptions) is False or ask(as_boolean(b > 0), assumptions) is False:
        return None
    at_zero = _limit(g1, x, S.Zero, assumptions)
    at_infinity = _limit(g1, x, oo, assumptions)
    if at_zero is None or at_infinity is None:
        return None
    condition = as_boolean((a > 0) & (b > 0))
    return ConditionalValue(c1 * (at_zero - at_infinity) * log(b / a), condition)


def _glasser_map(u: Expr, x: Symbol, assumptions: Assumptions) -> Optional[list[tuple[Expr, Expr]]]:
    """``[(a_i, b_i)]`` when ``u == x - sum(a_i/(x - b_i))`` with ``a_i > 0``."""
    decomposed = attempt(lambda: as_expr(apart(u, x)), settings.timeout)
    if decomposed is None:
        return None
    poles: list[tuple[Expr, Expr]] = []
    linear: Expr = S.Zero
    for term in Add.make_args(decomposed):
        term_ = as_expr(term)
        if term_ == x or not term_.has(x):
            linear = linear + term_
            continue
        coefficient, rest = term_.as_independent(x, as_Add=False)
        rest_ = as_expr(rest)
        if not (isinstance(rest_, Pow) and rest_.exp == -1):
            return None
        base = as_expr(rest_.base)
        found = base.as_poly(x)
        if found is None or found.degree() != 1:
            return None
        slope, intercept = as_expr(found.coeff_monomial(x)), as_expr(found.coeff_monomial(1))
        a_i = as_expr(-coefficient / slope)
        b_i = as_expr(-intercept / slope)
        if ask(as_boolean(a_i > 0), assumptions) is not True:
            return None
        poles.append((a_i, b_i))
    if as_expr(linear - x) != 0 or not poles:
        return None
    return poles


def _inner_arguments(f: Expr, x: Symbol) -> list[Expr]:
    """The subexpressions of ``f`` depending on ``x`` which are arguments
    of functions, or bases of powers, largest first."""
    found: set[Expr] = set()
    for node in f.atoms(Function):
        for argument in node.args:
            argument_ = as_expr(argument)
            if argument_.has(x) and argument_ != x:
                found.add(argument_)
    for node in f.atoms(Pow):
        base = as_expr(node.base)
        if base.has(x) and base != x:
            found.add(base)
    return sorted(found, key=lambda e: -e.count_ops())


def glasser(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
            assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` by Glasser's master theorem, for ``f =
    F(u(x))`` with ``u = x - sum(a_i/(x - b_i))``, ``a_i > 0``, over the
    real line, or over ``(0, oo)`` when ``F`` is even (the
    Cauchy–Schlömilch transformation, ``f = G(u**2)``); the transformed
    integral ``Integral(F, (x, -oo, oo))`` is computed by
    :func:`~sympy_extras.integrals.definite_integral`. ``None`` when the
    shape is not recognised."""
    from .definite import conditional_integral
    a_, b_ = as_expr(a), as_expr(b)
    whole_line = (a_, b_) == (-oo, oo)
    half_line = (a_, b_) == (S.Zero, oo)
    if not whole_line and not half_line:
        return None
    y = Dummy('y', real=True)
    for v in _inner_arguments(f, x):
        poles = _glasser_map(v, x, assumptions)
        if poles is None:
            continue
        replaced = as_expr(f.xreplace({v: y}))
        if replaced.has(x):
            continue
        if half_line and as_expr(replaced.xreplace({y: -y})) != replaced:
            # over the half line F must be even
            continue
        lower = -oo if whole_line else S.Zero
        value = conditional_integral(replaced, y, lower, oo, assumptions)
        if value is not None:
            return value
    return None


def transformation_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                            assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """The first of the identities of this module which applies."""
    a_, b_ = as_expr(a), as_expr(b)
    if (a_, b_) == (S.Zero, oo):
        found = frullani(f, x, assumptions)
        if found is not None:
            return found
    return glasser(f, x, a_, b_, assumptions)
