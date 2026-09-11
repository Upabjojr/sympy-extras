"""The operational rules of the Laplace transform, for definite integrals
``Integral(f(t)*exp(-s*t), (t, 0, oo))`` which the Mellin table does not
reach directly.

The Laplace transform `F(s) = \\int_0^\\infty f(t) e^{-st}\\, dt` obeys rules
which turn the transform of a function built from `f` into an operation
on `F` [Doetsch]_, [AS]_ (table 29.2):

* division by `t`: `\\mathcal{L}[f(t)/t](s) = \\int_s^\\infty F(u)\\, du`, and
  again for `1/t^n`;
* multiplication by `t^n`: `\\mathcal{L}[t^n f](s) = (-1)^n F^{(n)}(s)`;
* the convolution `\\mathcal{L}[\\int_0^t f(u) g(t - u)\\, du] = F G`, with the
  integral rule `\\mathcal{L}[\\int_0^t f] = F/s` as the case `g = 1`;
* the shifts `\\mathcal{L}[f(t - a) \\theta(t - a)] = e^{-as} F(s)` (`a > 0`)
  and `\\mathcal{L}[e^{at} f] = F(s - a)` (`\\operatorname{Re} s > a`);
* a periodic `f` of period `T`:
  `\\mathcal{L}[f](s) = \\int_0^T f e^{-st} dt / (1 - e^{-sT})`.

Each rule reduces the transform of the given integrand to transforms of
simpler ones, which :func:`~sympy_extras.integrals.definite_integral`
computes (the Mellin table, the antiderivative for the outer integral of
the division rule, whose lower endpoint is the symbol `s`). The pairs of
[AS]_ table 29.3 and of [Erdelyi]_ are the test material; Maxima's
``specint`` tests in the benchmarks come from the same tables.

Examples
========

>>> from sympy import symbols, cos, sin, Abs, oo, exp
>>> from sympy_extras.integrals.laplace import laplace_rules, laplace_integral
>>> t = symbols('t')
>>> s, a, b = symbols('s a b', positive=True)
>>> laplace_rules((cos(a*t) - cos(b*t))/t, t, s)
ConditionalValue((-log(a**2 + s**2) + log(b**2 + s**2))/2)
>>> laplace_rules(Abs(sin(a*t)), t, s)
ConditionalValue(a/((a**2 + s**2)*tanh(pi*s/(2*a))))
>>> laplace_integral(t**2*exp(a*t)*exp(-s*t), t, 0, oo, s > a)
ConditionalValue(2/(-a + s)**3, -a + s > 0)

References
==========

.. [Doetsch] G. Doetsch, *Introduction to the Theory and Application of
   the Laplace Transformation*, Springer, 1974, chapters 7–11.
.. [AS] M. Abramowitz, I. A. Stegun, *Handbook of Mathematical
   Functions*, Dover, 1965, section 29.2 (the rules) and 29.3 (the pairs).
.. [Erdelyi] A. Erdélyi, W. Magnus, F. Oberhettinger, F. G. Tricomi,
   *Tables of integral transforms*, vol. I, McGraw-Hill, 1954, chapters
   IV and V.
"""
from __future__ import annotations

from typing import Optional

from sympy.calculus.util import periodicity
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.complexes import arg, re, im
from sympy.functions.elementary.trigonometric import atan
from sympy.core.numbers import pi
from sympy.simplify.simplify import simplify
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.elementary.piecewise import Piecewise
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Boolean, true

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .mellin import monomial
from .marichev import tidy

__all__ = ['laplace_rules', 'laplace_integral', 'laplace_of_convolution', 'transform', 'real_arguments']

#: the powers of ``t`` in the division and multiplication rules are
#: bounded, so that the rules terminate
_MAX_POWER = 3


def _statements(assumptions: Assumptions) -> list[Boolean]:
    if isinstance(assumptions, (Basic, bool)):
        return [as_boolean(assumptions)]
    if assumptions is None:
        return []
    return [as_boolean(a) for a in assumptions]


def _renamed(assumptions: Assumptions, s: Expr, u: Symbol) -> list[Boolean]:
    """The assumptions with the transform variable ``s`` renamed ``u``."""
    return [as_boolean(item.xreplace({s: u})) for item in _statements(assumptions)] + [as_boolean(u > 0)]


def transform(f: Expr, t: Symbol, u: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f*exp(-u*t), (t, 0, oo))`` by :func:`~sympy_extras.integrals.definite_integral`
    (which does not use the rules of this module again), ``None`` when
    it is not found or comes with an undecided condition."""
    from .definite import definite_integral
    value = attempt(lambda: definite_integral(f * exp(-u * t), (t, S.Zero, oo), assumptions), settings.timeout)
    if value is None or value.has(Integral, Piecewise):
        return None
    return value


def _factors(f: Expr) -> list[Expr]:
    return [as_expr(g) for g in (f.args if isinstance(f, Mul) else (f,))]


def _power_of(f: Expr, t: Symbol) -> tuple[int, Expr]:
    """``(n, h)`` with ``f == t**n * h`` and ``h`` without a power of ``t``
    as a factor (``n`` may be negative)."""
    n = 0
    rest: list[Expr] = []
    for g in _factors(f):
        if g == t:
            n += 1
        elif isinstance(g, Pow) and g.base == t and isinstance(g.exp, Integer):
            n += int(g.exp)
        else:
            rest.append(g)
    return n, as_expr(Mul(*rest))


def _division_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[h/t**n](s) = Integral(...Integral(H(u), (u, s, oo))...)``, ``n``
    times."""
    from .definite import conditional_integral
    n, h = _power_of(f, t)
    if n >= 0 or -n > _MAX_POWER:
        return None
    u = Dummy('u', positive=True)
    value = transform(h, t, u, _renamed(assumptions, s, u))
    if value is None:
        return None
    condition: Boolean = true
    for _ in range(-n):
        w = Dummy('w', positive=True)
        outer = conditional_integral(value, u, w, oo, _renamed(assumptions, s, w))
        if outer is None or outer.value.has(Integral):
            return None
        value = real_arguments(as_expr(outer.value.subs(w, u)), _renamed(assumptions, s, u))
        condition = as_boolean(And(condition, as_boolean(outer.condition.subs(w, s))))
    return ConditionalValue(as_expr(value.subs(u, s)), condition)


def _multiplication_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[t**n h](s) = (-1)**n H^(n)(s)``."""
    n, h = _power_of(f, t)
    if n <= 0 or n > _MAX_POWER:
        return None
    u = Dummy('u', positive=True)
    value = transform(h, t, u, _renamed(assumptions, s, u))
    if value is None:
        return None
    derivative = as_expr(value.diff(u, n) * S.NegativeOne**n)
    return ConditionalValue(as_expr(derivative.subs(u, s)))


def _heaviside_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[h(t) theta(t - a)](s) = exp(-a s) L[h(t + a)](s)`` for ``a > 0``."""
    for g in _factors(f):
        if isinstance(g, Heaviside) and g.has(t):
            argument = as_expr(g.args[0])
            found = monomial(argument, t)
            if found is None:
                shift, rest = argument.as_independent(t, as_Add=True)
                shift_, rest_ = as_expr(shift), as_expr(rest)
                if rest_ != t:
                    return None
                a = -shift_
            else:
                return None
            if ask(as_boolean(a > 0), assumptions) is not True:
                return None
            h = as_expr((f / g).subs(t, t + a))
            u = Dummy('u', positive=True)
            value = transform(h, t, u, _renamed(assumptions, s, u))
            if value is None:
                return None
            return ConditionalValue(as_expr(exp(-a * s) * value.subs(u, s)))
    return None


def _exponential_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[exp(c t) h](s) = H(s - c)`` for ``Re s > c``."""
    for g in _factors(f):
        if isinstance(g, exp) and g.has(t):
            found = monomial(as_expr(g.args[0]), t)
            if found is None or found[1] != 1:
                return None
            c = found[0]
            h = as_expr(f / g)
            u = Dummy('u', positive=True)
            value = transform(h, t, u, _renamed(assumptions, s, u))
            if value is None:
                return None
            condition = as_boolean(s - c > 0)
            return ConditionalValue(as_expr(value.subs(u, s - c)), condition)
    return None


def _periodic_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[f](s) = Integral(f exp(-s t), (t, 0, T)) / (1 - exp(-s T))`` for
    a period ``T``."""
    from .definite import conditional_integral
    if f.has(exp, Heaviside, Integral):
        return None
    period = attempt(lambda: periodicity(f, t), settings.timeout)
    if period is None or not isinstance(period, Expr) or period == 0 or period.has(t):
        return None
    u = Dummy('u', positive=True)
    piece = conditional_integral(f * exp(-u * t), t, S.Zero, as_expr(period), _renamed(assumptions, s, u))
    if piece is None or piece.value.has(Integral):
        return None
    value = as_expr(piece.value / (1 - exp(-u * period)))
    return ConditionalValue(as_expr(value.subs(u, s)), as_boolean(piece.condition.subs(u, s)))


def laplace_of_convolution(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``L[Integral(p(u) q(t - u), (u, 0, t))](s) = P(s) Q(s)``, with the
    integral rule ``L[Integral(p, (u, 0, t))] = P(s)/s`` as the case
    ``q = 1``; ``None`` when ``f`` is not such an integral.

    Examples
    ========

    >>> from sympy import symbols, Integral, sin
    >>> from sympy_extras.integrals.laplace import laplace_of_convolution
    >>> t, u = symbols('t u')
    >>> s = symbols('s', positive=True)
    >>> laplace_of_convolution(Integral(sin(t - u)*u, (u, 0, t)), t, s)
    ConditionalValue(1/(s**2*(s**2 + 1)))
    """
    if not isinstance(f, Integral) or len(f.limits) != 1:
        return None
    variable, lower, upper = f.limits[0]
    if not isinstance(variable, Symbol) or lower != 0 or upper != t:
        return None
    integrand = as_expr(f.function)
    p: Expr = S.One
    q: Expr = S.One
    w = Dummy('w')
    for g in _factors(integrand):
        if not g.has(t):
            p = p * g
            continue
        shifted = as_expr(g.subs(variable, t - w))
        if shifted.has(t):
            return None
        q = q * shifted.subs(w, variable)
    v = Dummy('v', positive=True)
    first = transform(p, variable, v, _renamed(assumptions, s, v))
    second = transform(q, variable, v, _renamed(assumptions, s, v))
    if first is None or second is None:
        return None
    return ConditionalValue(as_expr((first * second).subs(v, s)))


def real_arguments(value: Expr, assumptions: Assumptions = None) -> Expr:
    """``arg(x + I*y)`` with the signs of the real numbers ``x`` and ``y``
    decided under the assumptions written as ``atan(y/x)`` (``x > 0``) or
    ``atan(y/x) +- pi`` (``x < 0``), the way the limits of an
    antiderivative come out of SymPy (``arg(-s - I*a)`` for the transform
    of ``sin(a t)/t``); then simplified.

    >>> from sympy import symbols, arg, I
    >>> from sympy_extras.integrals.laplace import real_arguments
    >>> s, a = symbols('s a', positive=True)
    >>> real_arguments(arg(-s - I*a)/2 - arg(-s + I*a)/2 + pi)
    atan(a/s)
    """
    replacement: dict[Expr, Expr] = {}
    for node in value.atoms(arg):
        z = as_expr(node.args[0])
        x, y = as_expr(re(z)), as_expr(im(z))
        if x.has(re, im) or y.has(re, im):
            continue
        sign_x = ask(as_boolean(x > 0), assumptions)
        if sign_x is True:
            replacement[as_expr(node)] = atan(y / x)
        elif ask(as_boolean(x < 0), assumptions) is True:
            sign_y = ask(as_boolean(y >= 0), assumptions)
            if sign_y is True:
                replacement[as_expr(node)] = atan(y / x) + pi
            elif ask(as_boolean(y < 0), assumptions) is True:
                replacement[as_expr(node)] = atan(y / x) - pi
    if not replacement:
        return value
    rewritten = as_expr(value.xreplace(replacement))
    simpler = attempt(lambda: as_expr(simplify(rewritten)), settings.timeout)
    return rewritten if simpler is None else simpler


def laplace_rules(f: ExprLike, t: Symbol, s: ExprLike, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f*exp(-s*t), (t, 0, oo))`` by the first applicable rule
    (the shift by a step function, the exponential shift, a period, a
    convolution, the division by ``t``, the multiplication by ``t``),
    or ``None``.

    Examples
    ========

    >>> from sympy import symbols, sin, cos, Heaviside, exp
    >>> from sympy_extras.integrals.laplace import laplace_rules
    >>> t = symbols('t')
    >>> s, a = symbols('s a', positive=True)
    >>> laplace_rules(sin(a*t)/t, t, s)
    ConditionalValue(atan(a/s))
    >>> laplace_rules(Heaviside(t - 1)*exp(-t), t, s)
    ConditionalValue(exp(-s - 1)/(s + 1))
    """
    f_, s_ = as_expr(f), as_expr(s)
    if isinstance(f_, Add):
        total = ConditionalValue(S.Zero)
        for term in f_.args:
            found = laplace_rules(as_expr(term), t, s_, assumptions)
            if found is None:
                return None
            total = total.add(found)
        return total
    for rule in (laplace_of_convolution, _heaviside_rule, _exponential_rule, _periodic_rule,
                 _division_rule, _multiplication_rule):
        found = rule(f_, t, s_, assumptions)
        if found is not None:
            value = tidy(real_arguments(found.value, assumptions), assumptions, found.condition)
            return ConditionalValue(value, found.condition)
    return None


def laplace_integral(f: ExprLike, t: Symbol, a: ExprLike, b: ExprLike,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (t, a, b))`` when it is a Laplace transform
    ``Integral(g(t)*exp(-s*t), (t, 0, oo))`` with ``s`` a symbol or a
    positive number, by :func:`laplace_rules` on ``g``; ``None``
    otherwise.

    Examples
    ========

    >>> from sympy import symbols, cos, exp, oo
    >>> from sympy_extras.integrals.laplace import laplace_integral
    >>> t = symbols('t')
    >>> s, a = symbols('s a', positive=True)
    >>> laplace_integral((exp(-a*t) - exp(-2*a*t))*exp(-s*t)/t, t, 0, oo)
    ConditionalValue(log((2*a + s)/(a + s)))
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    if a_ != 0 or b_ != oo:
        return None
    for g in _factors(f_):
        if isinstance(g, exp) and g.has(t):
            found = monomial(as_expr(g.args[0]), t)
            if found is None or found[1] != 1:
                continue
            s = as_expr(-found[0])
            if not (isinstance(s, Symbol) or s.is_positive):
                continue
            if isinstance(s, Symbol) and not s.is_positive and ask(as_boolean(s > 0), assumptions) is not True:
                continue
            rest = as_expr(f_ / g)
            if rest == 1 or not rest.has(t):
                continue
            return laplace_rules(rest, t, s, assumptions)
    return None
