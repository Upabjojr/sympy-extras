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
simpler ones, which :func:`~sympy_extras.integrals.definite.definite_integral`
computes (the Mellin table, the antiderivative for the outer integral of
the division rule, whose lower endpoint is the symbol `s`). The pairs of
[AS]_ table 29.3 and of [Erdelyi]_ are the test material; Maxima's
``specint`` tests in the benchmarks come from the same tables.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native ask, conditions, ``simplify`` and ``tidy``. The
transforms of the pieces are computed by
:func:`~sympy_extras.integrals.definite.definite_integral`, which is still
written on SymPy, through the functions of
:mod:`sympy_extras.integrals.laplace_sympy` (``definite_integral_of_core``,
``conditional_integral_of_core``): SymPy is loaded on those paths only.
The periods of the periodic rule are native
(:func:`sympy_extras.calculus.util.periodicity`). The
``Integral`` of the convolution rule is SymPy's, wrapped in a node of the
core (the core has no class of its own). The functions on SymPy objects
are in :mod:`.laplace_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, oo
>>> from sympy_extras.functions import cos, sin, abs_, exp
>>> from sympy_extras.integrals.laplace import laplace_rules, laplace_integral
>>> t = Symbol('t')
>>> s, a, b = [Symbol(n, positive=True) for n in 'sab']
>>> laplace_rules((cos(a*t) - cos(b*t))/t, t, s)
ConditionalValue((-log(a**2 + s**2) + log(b**2 + s**2))/2)
>>> laplace_rules(abs_(sin(a*t)), t, s)
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

import itertools
from typing import Callable, Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.logic import Boolean, and_, to_boolean, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, oo, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.core.traversal import preorder_traversal
from sympy_extras.functions.elementary.complexes import Abs, Arg, Im, Re, im, re
from sympy_extras.functions.elementary.exponential import Exp, exp
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import atan
from sympy_extras.functions.special.delta_functions import Heaviside
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify

from .conditions import ConditionalValue
from .marichev import tidy
from .mellin import monomial

__all__ = ['laplace_rules', 'laplace_integral', 'laplace_of_convolution', 'transform', 'real_arguments',
           'has_integral']

#: the powers of ``t`` in the division and multiplication rules are
#: bounded, so that the rules terminate
_MAX_POWER: Final = 3

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so the variable ``u`` of a rule and the ``u`` of a
    rule applied inside it (or a parameter ``u``) would be taken for one
    symbol."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


def _is_integral(e: Basic) -> bool:
    """Whether ``e`` is an ``Integral`` (SymPy's, wrapped in a node of the
    core: the core has no class of its own)."""
    return e._is_foreign_node() and e._compare_name() == 'Integral'


def has_integral(e: Basic) -> bool:
    """Whether ``e`` has an unevaluated ``Integral`` (SymPy's, wrapped in a
    node of the core).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.integrals.laplace import has_integral
    >>> has_integral(Symbol('t')**2)
    False
    """
    return any(_is_integral(node) for node in preorder_traversal(e))


def _expr(e: Basic) -> Expr:
    if not isinstance(e, Expr):
        raise TypeError("expected an expression, got %s" % (e,))
    return e


def _boolean(b: Basic) -> Boolean:
    if not isinstance(b, Boolean):
        raise TypeError("expected a Boolean, got %s" % (b,))
    return b


def _statements(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Basic, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(a) for a in assumptions]


def _renamed(assumptions: Assumptions, s: Expr, u: Symbol) -> list[Boolean]:
    """The assumptions with the transform variable ``s`` renamed ``u``."""
    rule: dict[Basic, Basic] = {s: u}
    return [_boolean(item._xreplace_evaluated(rule)) for item in _statements(assumptions)] + [u > 0]


def transform(f: Expr, t: Symbol, u: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f*exp(-u*t), (t, 0, oo))`` by
    :func:`~sympy_extras.integrals.definite.definite_integral` (which does
    not use the rules of this module again; SymPy's side, through
    :func:`sympy_extras.integrals.laplace_sympy.definite_integral_of_core`),
    ``None`` when it is not found or comes with an undecided condition."""
    from .laplace_sympy import definite_integral_of_core
    value = definite_integral_of_core(f * exp(-u * t), t, S.Zero, oo, assumptions, False, settings.timeout)
    if value is None or value.has(Piecewise) or has_integral(value):
        return None
    return value


def _conditional_integral(f: Expr, x: Symbol, a: Expr, b: Expr,
                          assumptions: Assumptions) -> Optional[ConditionalValue]:
    """:func:`~sympy_extras.integrals.definite.conditional_integral`
    (SymPy's side, through the face); ``None`` for a value with an
    unevaluated ``Integral``."""
    from .laplace_sympy import conditional_integral_of_core
    found = conditional_integral_of_core(f, x, a, b, assumptions, None)
    if found is None or has_integral(found.value):
        return None
    return found


def _factors(f: Expr) -> tuple[Expr, ...]:
    return Mul.make_args(f)


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
    return n, mul(*rest)


def _division_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[h/t**n](s) = Integral(...Integral(H(u), (u, s, oo))...)``, ``n``
    times."""
    n, h = _power_of(f, t)
    if n >= 0 or -n > _MAX_POWER:
        return None
    u = _fresh('u', positive=True)
    value = transform(h, t, u, _renamed(assumptions, s, u))
    if value is None:
        return None
    condition: Boolean = true
    for _ in range(-n):
        w = _fresh('w', positive=True)
        outer = _conditional_integral(value, u, w, oo, _renamed(assumptions, s, w))
        if outer is None:
            return None
        value = real_arguments(outer.value.subs(w, u), _renamed(assumptions, s, u))
        condition = and_(condition, _boolean(outer.condition.subs(w, s)))
    return ConditionalValue(value.subs(u, s), condition)


def _multiplication_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[t**n h](s) = (-1)**n H^(n)(s)``."""
    n, h = _power_of(f, t)
    if n <= 0 or n > _MAX_POWER:
        return None
    u = _fresh('u', positive=True)
    value = transform(h, t, u, _renamed(assumptions, s, u))
    if value is None:
        return None
    derivative = value.diff((u, n)) * S.NegativeOne**n
    return ConditionalValue(derivative.subs(u, s))


def _heaviside_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[h(t) theta(t - a)](s) = exp(-a s) L[h(t + a)](s)`` for ``a > 0``."""
    for g in _factors(f):
        if isinstance(g, Heaviside) and g.has(t):
            argument = _expr(g.args[0])
            if monomial(argument, t) is not None:
                return None
            shift, rest = argument.as_independent(t, as_Add=True)
            if rest != t:
                return None
            a = -shift
            if ask(a > 0, assumptions) is not True:
                return None
            h = (f / g).subs(t, t + a)
            u = _fresh('u', positive=True)
            value = transform(h, t, u, _renamed(assumptions, s, u))
            if value is None:
                return None
            return ConditionalValue(exp(-a * s) * value.subs(u, s))
    return None


def _exponential_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[exp(c t) h](s) = H(s - c)`` for ``Re s > c``."""
    for g in _factors(f):
        if isinstance(g, Exp) and g.has(t):
            found = monomial(_expr(g.args[0]), t)
            if found is None or found[1] != 1:
                return None
            c = found[0]
            u = _fresh('u', positive=True)
            value = transform(f / g, t, u, _renamed(assumptions, s, u))
            if value is None:
                return None
            return ConditionalValue(value.subs(u, s - c), s - c > 0)
    return None


def _periodic_rule(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``L[f](s) = Integral(f exp(-s t), (t, 0, T)) / (1 - exp(-s T))`` for
    a period ``T`` (:func:`sympy_extras.calculus.util.periodicity`)."""
    if f.has(Exp, Heaviside) or has_integral(f):
        return None
    from sympy_extras.calculus.util import periodicity
    found = attempt(lambda: periodicity(f, t), settings.timeout, native=True)
    if found is None or found == 0 or found.has(t):
        return None
    period: Expr = found
    # 2*pi/Abs(k) for sin(k*t): the sign of k is in the assumptions
    for node in period.atoms(Abs):
        argument = _expr(node.args[0])
        if ask(argument > 0, assumptions) is True:
            period = _expr(period._xreplace_evaluated({node: argument}))
        elif ask(argument < 0, assumptions) is True:
            period = _expr(period._xreplace_evaluated({node: -argument}))
    if period.has(Abs):
        return None
    u = _fresh('u', positive=True)
    piece = _conditional_integral(f * exp(-u * t), t, S.Zero, period, _renamed(assumptions, s, u))
    if piece is None:
        return None
    value = piece.value / (1 - exp(-u * period))
    return ConditionalValue(value.subs(u, s), _boolean(piece.condition.subs(u, s)))


def laplace_of_convolution(f: Expr, t: Symbol, s: Expr, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``L[Integral(p(u) q(t - u), (u, 0, t))](s) = P(s) Q(s)``, with the
    integral rule ``L[Integral(p, (u, 0, t))] = P(s)/s`` as the case
    ``q = 1``; ``None`` when ``f`` is not such an integral (an
    ``Integral`` of SymPy, wrapped in a node of the core: the core has no
    class of its own).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.integrals.laplace import laplace_of_convolution
    >>> t, s = Symbol('t'), Symbol('s', positive=True)
    >>> laplace_of_convolution(sin(t), t, s) is None
    True
    """
    if not _is_integral(f) or len(f.args) != 2:
        return None
    integrand, limits = f.args
    if not isinstance(integrand, Expr) or not isinstance(limits, Tuple) or len(limits.args) != 3:
        return None
    variable, lower, upper = limits.args
    if not isinstance(variable, Symbol) or lower != 0 or upper != t:
        return None
    p: Expr = S.One
    q: Expr = S.One
    w = _fresh('w')
    for g in _factors(integrand):
        if not g.has(t):
            p = p * g
            continue
        shifted = g.subs(variable, t - w)
        if shifted.has(t):
            return None
        q = q * shifted.subs(w, variable)
    v = _fresh('v', positive=True)
    first = transform(p, variable, v, _renamed(assumptions, s, v))
    second = transform(q, variable, v, _renamed(assumptions, s, v))
    if first is None or second is None:
        return None
    return ConditionalValue((first * second).subs(v, s))


def real_arguments(value: Expr, assumptions: Assumptions = None) -> Expr:
    """``arg(x + I*y)`` with the signs of the real numbers ``x`` and ``y``
    decided under the assumptions written as ``atan(y/x)`` (``x > 0``) or
    ``atan(y/x) +- pi`` (``x < 0``), the way the limits of an
    antiderivative come out of SymPy (``arg(-s - I*a)`` for the transform
    of ``sin(a t)/t``); then simplified (the native ``simplify``).

    >>> from sympy_extras.core import I, Symbol, pi
    >>> from sympy_extras.functions import Arg
    >>> from sympy_extras.integrals.laplace import real_arguments
    >>> s, a = Symbol('s', positive=True), Symbol('a', positive=True)
    >>> real_arguments(Arg(-s - I*a)/2 - Arg(-s + I*a)/2 + pi)
    atan(a/s)
    """
    replacement: dict[Basic, Basic] = {}
    for node in value.atoms(Arg):
        z = _expr(node.args[0])
        x, y = re(z), im(z)
        if x.has(Re, Im) or y.has(Re, Im):
            continue
        if ask(x > 0, assumptions) is True:
            replacement[node] = atan(y / x)
        elif ask(x < 0, assumptions) is True:
            if ask(y >= 0, assumptions) is True:
                replacement[node] = atan(y / x) + pi
            elif ask(y < 0, assumptions) is True:
                replacement[node] = atan(y / x) - pi
    if not replacement:
        return value
    rewritten = _expr(value._xreplace_evaluated(replacement))
    simpler = attempt(lambda: simplify(rewritten), settings.timeout, native=True)
    return rewritten if simpler is None else simpler


#: a rule of the module: the transform of ``f(t)`` at ``s`` under the
#: assumptions, or ``None``
Rule = Callable[[Expr, Symbol, Expr, Assumptions], Optional[ConditionalValue]]

#: the rules in the order they are tried
_RULES: Final[tuple[Rule, ...]] = (laplace_of_convolution, _heaviside_rule, _exponential_rule, _periodic_rule,
                                    _division_rule, _multiplication_rule)


def laplace_rules(f: ExprLike, t: Symbol, s: ExprLike, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f*exp(-s*t), (t, 0, oo))`` by the first applicable rule
    (the shift by a step function, the exponential shift, a period, a
    convolution, the division by ``t``, the multiplication by ``t``),
    or ``None``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sin, heaviside, exp
    >>> from sympy_extras.integrals.laplace import laplace_rules
    >>> t = Symbol('t')
    >>> s, a = Symbol('s', positive=True), Symbol('a', positive=True)
    >>> laplace_rules(sin(a*t)/t, t, s)
    ConditionalValue(atan(a/s))
    >>> laplace_rules(heaviside(t - 1)*exp(-t), t, s)
    ConditionalValue(exp(-s - 1)/(s + 1))
    """
    f_, s_ = to_expr(f), to_expr(s)
    if isinstance(f_, Add):
        total = ConditionalValue(S.Zero)
        for term in f_.args:
            found = laplace_rules(term, t, s_, assumptions)
            if found is None:
                return None
            total = total.add(found)
        return total
    for rule in _RULES:
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

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.laplace import laplace_integral
    >>> t = Symbol('t')
    >>> s, a = Symbol('s', positive=True), Symbol('a', positive=True)
    >>> laplace_integral((exp(-a*t) - exp(-2*a*t))*exp(-s*t)/t, t, 0, oo)
    ConditionalValue(log((2*a + s)/(a + s)))
    """
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    if a_ != 0 or b_ != oo:
        return None
    for g in _factors(f_):
        if isinstance(g, Exp) and g.has(t):
            found = monomial(_expr(g.args[0]), t)
            if found is None or found[1] != 1:
                continue
            s = -found[0]
            if not (isinstance(s, Symbol) or s.is_positive):
                continue
            if isinstance(s, Symbol) and not s.is_positive and ask(s > 0, assumptions) is not True:
                continue
            rest = f_ / g
            if rest == 1 or not rest.has(t):
                continue
            return laplace_rules(rest, t, s, assumptions)
    return None
