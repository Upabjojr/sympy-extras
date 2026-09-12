r"""Canonical forms and substitutions before integration.

The integrals of the published test suites come in the spellings their
authors chose, `a^{d z} h^{c z^2 + f z + g}`, `e^{(c z^2 + g) v}` written
as a power of an exponential, `d^{a z + b\sqrt z}`, `e^{\operatorname{asech} x}`,
`x/\sinh(x + 2)`; the integrators of this package (the heuristic and the
transcendental Risch algorithms, the radical table, the trigonometric
integrator) take them once they are written with one exponential, one
logarithm, and one radical. This module supplies

- :func:`rewritten_forms`: the integrand in canonical forms, each equal
  to it wherever it is real, powers of positive bases as exponentials,
  powers of exponentials as one exponential, hyperbolic functions as
  exponentials, inverse hyperbolic functions as logarithms, radicals of
  products as radicals of positive factors, `|x|` by its sign;
- :func:`power_substitutions`: the substitutions `x = t^k` for the
  fractional powers of `x` (or of `a x + b`), `u = e^{c x}` for rational
  functions of exponentials and `x = e^t` for rational functions of
  `\log x`, each with its Jacobian, so that the new integrand is free of
  the fractional powers, or rational;
- :func:`substitute_back`: the antiderivative in the new variable
  returned to `x`.

Examples
========

>>> from sympy import symbols, exp, sinh, sqrt
>>> from sympy_extras.integrals.rewriting import rewritten_forms, power_substitutions
>>> x, z = symbols('x z')
>>> a, c = symbols('a c', positive=True)
>>> rewritten_forms(a**(c*z)*exp(z**2), z)
[exp(z**2)*exp(c*z*log(a)), exp(c*z*log(a) + z**2)]
>>> rewritten_forms(x/sinh(x + 2), x)
[x/(-exp(-x - 2)/2 + exp(x + 2)/2)]
>>> s = power_substitutions(sqrt(x)/(1 + x), x)[0]
>>> s.integrand, s.back
(2*_t**2/(_t**2 + 1), sqrt(x))
"""
from __future__ import annotations

from math import gcd, lcm
from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import HyperbolicFunction, InverseHyperbolicFunction
from sympy.logic.boolalg import Boolean
from sympy.polys.polytools import cancel
from sympy.simplify.powsimp import powsimp

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.settings import settings

__all__ = ['Substitution', 'rewritten_forms', 'power_substitutions', 'substitute_back']

#: the most forms and substitutions returned
_LIMIT = 5
_SUBSTITUTIONS = 3


class Substitution:
    """A change of variable ``x -> t``: ``integrand`` is ``g(t)`` with
    ``f(x) dx = g(t) dt``, ``variable`` the new symbol ``t``, ``back`` the
    expression of ``t`` in ``x`` and ``forward`` that of ``x`` in ``t``."""

    def __init__(self, integrand: Expr, variable: Symbol, back: Expr, forward: Expr) -> None:
        self.integrand = integrand
        self.variable = variable
        self.back = back
        self.forward = forward

    def __repr__(self) -> str:
        return "Substitution(%s, %s, back=%s)" % (self.integrand, self.variable, self.back)


def _real_in(e: Expr, x: Symbol, assumptions: Assumptions) -> bool:
    """Whether ``e`` is real for a real ``x``, the other symbols being
    declared real or asked real under the assumptions (a parameter about
    which nothing is known is not taken real)."""
    if not e.has(x) and (e.is_extended_real or ask(element(e, S.Reals), assumptions) is True):
        return True
    real = Dummy(x.name, real=True)
    others: dict[Symbol, Expr] = {}
    for s in free_symbols(e) - {x}:
        if s.is_extended_real:
            continue
        if ask(element(s, S.Reals), assumptions) is not True:
            return False
        others[s] = Dummy(s.name, real=True)
    candidate = as_expr(e.xreplace({x: real}).xreplace(others))
    return candidate.is_extended_real is True


def _positive(e: Expr, assumptions: Assumptions) -> bool:
    if e.is_positive is True:
        return True
    if not e.free_symbols:
        return False
    return ask(as_boolean(e > 0), assumptions) is True


def _bases_to_exponentials(f: Expr, x: Symbol, assumptions: Assumptions) -> Expr:
    """``a**g`` with a positive ``a`` free of ``x`` as ``exp(g*log(a))``,
    and ``exp(X)**v`` as ``exp(v*X)`` for a real ``v`` and a real ``X``."""
    def rewrite(node: Basic) -> Basic:
        if not isinstance(node, Pow):
            return node
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if not exponent.has(x):
            if isinstance(base, exp) and _real_in(exponent, x, assumptions) and _real_in(as_expr(base.args[0]), x, assumptions):
                return exp(exponent * base.args[0])
            return node
        if base.has(x) or base == S.Exp1:
            return node
        if _positive(base, assumptions):
            return exp(exponent * log(base))
        return node

    return as_expr(f.replace(lambda n: isinstance(n, Pow), rewrite))


def _hyperbolic_to_exponentials(f: Expr) -> Expr:
    if not f.has(HyperbolicFunction):
        return f
    return as_expr(f.rewrite(exp))


def _inverse_hyperbolic_to_logarithms(f: Expr) -> Expr:
    if not f.has(InverseHyperbolicFunction):
        return f
    return as_expr(f.rewrite(log))


def _combined_exponentials(f: Expr) -> Expr:
    found = attempt(lambda: as_expr(powsimp(f, combine='exp', deep=True)), settings.timeout)
    return f if found is None else found


def _products_of_exponentials(f: Expr) -> bool:
    """Whether some product of ``f`` has two exponential factors."""
    return any(sum(isinstance(as_expr(a), exp) for a in node.args) > 1 for node in f.atoms(Mul))


def _recombined_radicals(f: Expr, x: Symbol, assumptions: Assumptions) -> Expr:
    """Products of square roots (and other rational powers with one
    denominator) whose radicands are known nonnegative wherever ``x`` is
    real, ``sqrt(u)*sqrt(v)`` with ``u, v >= 0``, combined into one
    radical; the identity fails when both radicands are negative, so a
    radicand of unknown sign is left alone."""
    def rewrite(node: Basic) -> Basic:
        if not isinstance(node, Mul):
            return node
        radicals: list[tuple[Expr, Rational]] = []
        rest: list[Expr] = []
        for factor in node.args:
            factor_ = as_expr(factor)
            if isinstance(factor_, Pow) and isinstance(factor_.exp, Rational) and factor_.exp.q > 1 \
                    and factor_.has(x) and _nonnegative(as_expr(factor_.base), x, assumptions):
                radicals.append((as_expr(factor_.base), factor_.exp))
            else:
                rest.append(factor_)
        if len(radicals) < 2:
            return node
        denominators = {r.q for _, r in radicals}
        if len(denominators) != 1:
            return node
        q = denominators.pop()
        radicand: Expr = S.One
        for base, r in radicals:
            radicand = radicand * base**r.p
        return Mul(*rest) * Pow(as_expr(radicand), Rational(1, q))

    return as_expr(f.replace(lambda n: isinstance(n, Mul), rewrite))


def _nonnegative(e: Expr, x: Symbol, assumptions: Assumptions) -> bool:
    """Whether ``e >= 0`` for every real ``x`` under the assumptions."""
    if e.is_nonnegative is True:
        return True
    real = Dummy(x.name, real=True)
    candidate = as_expr(e.xreplace({x: real}))
    if candidate.is_nonnegative is True:
        return True
    return ask(as_boolean(e >= 0), _with(assumptions, [element(x, S.Reals)])) is True


def _with(assumptions: Assumptions, more: list[Boolean]) -> list[Boolean]:
    items: list[Boolean] = list(more)
    if isinstance(assumptions, (Boolean, bool)):
        items.append(as_boolean(assumptions))
    elif assumptions is not None:
        items.extend(as_boolean(a) for a in assumptions)
    return items


def _absolute_values(f: Expr, x: Symbol, assumptions: Assumptions) -> Expr:
    """``Abs(e)`` replaced by ``e`` or ``-e`` when the assumptions fix the
    sign of ``e``."""
    def rewrite(node: Basic) -> Basic:
        if not isinstance(node, Abs):
            return node
        inner = as_expr(node.args[0])
        if ask(as_boolean(inner >= 0), assumptions) is True:
            return inner
        if ask(as_boolean(inner <= 0), assumptions) is True:
            return -inner
        return node

    return as_expr(f.replace(lambda n: isinstance(n, Abs), rewrite))


def rewritten_forms(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> list[Expr]:
    """Canonical forms of ``f`` for integration in ``x``, each equal to
    ``f`` wherever ``f`` is real (``x`` real, the parameters real, and the
    signs the assumptions give): ``a**g(x)`` as ``exp(g(x)*log(a))`` for a
    number or a parameter known positive, ``exp(X)**v`` as ``exp(v*X)``
    for real ``v`` and ``X``, products of exponentials combined,
    hyperbolic functions as exponentials, inverse hyperbolic functions as
    logarithms (their real branches, which agree with the functions on
    the real domains), radicals of products with radicands known
    nonnegative combined, and ``Abs(e)`` by its sign when the assumptions
    fix it. In order of preference, without duplicates and without ``f``,
    at most five.

    >>> from sympy import symbols, exp, cosh, asinh, sqrt, Abs
    >>> from sympy_extras.integrals.rewriting import rewritten_forms
    >>> x = symbols('x')
    >>> v = symbols('v', real=True)
    >>> rewritten_forms(exp(x**2)**v*exp(3*x), x)
    [exp(3*x)*exp(v*x**2), exp(v*x**2 + 3*x)]
    >>> rewritten_forms(2**x*cosh(x), x)
    [(exp(x)/2 + exp(-x)/2)*exp(x*log(2)), exp(-x + x*log(2))/2 + exp(x*log(2) + x)/2]
    >>> rewritten_forms(x*Abs(x), x, [x > 0])
    [x**2]
    >>> rewritten_forms(asinh(x)/sqrt(x**2 + 1), x)
    [log(x + sqrt(x**2 + 1))/sqrt(x**2 + 1)]
    """
    f_ = as_expr(f)
    forms: list[Expr] = []

    def add(candidate: Optional[Expr]) -> None:
        if candidate is not None and candidate != f_ and candidate not in forms and len(forms) < _LIMIT:
            forms.append(candidate)

    signed = _absolute_values(f_, x, assumptions)
    exponential = _bases_to_exponentials(signed, x, assumptions)
    exponential = _hyperbolic_to_exponentials(exponential)
    add(exponential)
    combined = _combined_exponentials(exponential)
    add(combined)
    # a sum of exponentials times an exponential: distributed, then combined
    distributed = _combined_exponentials(as_expr(expand(exponential, power_base=False, power_exp=False)))
    if distributed != combined and not _products_of_exponentials(distributed):
        add(distributed)
        combined = distributed
    logarithmic = _inverse_hyperbolic_to_logarithms(combined)
    add(logarithmic)
    radical = _recombined_radicals(logarithmic, x, assumptions)
    add(radical)
    expanded = attempt(lambda: as_expr(cancel(expand(radical))), settings.timeout)
    if expanded is not None and expanded.count_ops() <= radical.count_ops():
        add(expanded)
    return forms


def _fractional_powers(f: Expr, x: Symbol) -> dict[Expr, int]:
    """The bases ``x`` or ``a*x + b`` of the fractional powers of ``f``
    with the least common denominator of their exponents."""
    found: dict[Expr, int] = {}
    for node in f.atoms(Pow):
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if not isinstance(exponent, Rational) or exponent.q == 1 or not base.has(x):
            continue
        if base != x and not (base.is_polynomial(x) and base.as_poly(x) is not None
                              and base.as_poly(x).degree() == 1):
            continue
        found[base] = lcm(found.get(base, 1), int(exponent.q))
    return found


def _power_substitution(f: Expr, x: Symbol, base: Expr, k: int) -> Optional[Substitution]:
    """``base = t**k``: ``x = (t**k - b)/a`` for ``base = a*x + b``."""
    t = Dummy('t', positive=True)
    poly = base.as_poly(x)
    if poly is None:
        return None
    a, b = as_expr(poly.coeff_monomial(x)), as_expr(poly.coeff_monomial(1))
    if a == 0:
        return None
    forward = as_expr((t**k - b) / a)
    jacobian = as_expr(k * t**(k - 1) / a)
    g = attempt(lambda: as_expr(cancel(powsimp(as_expr(f.subs(x, forward) * jacobian), force=True))),
                settings.timeout)
    if g is None:
        return None
    if any(isinstance(node.exp, Rational) and node.exp.q > 1 and node.has(t) for node in g.atoms(Pow)):
        return None                                         # a fractional power survived
    return Substitution(g, t, as_expr(base**Rational(1, k)), forward)


def _exponential_generator(f: Expr, x: Symbol) -> Optional[Expr]:
    """``c`` such that every ``exp(c_i*x)`` of ``f`` has ``c_i`` an integer
    multiple of ``c``; ``None`` without exponentials linear in ``x`` or
    with an exponential of another shape."""
    coefficients: list[Rational] = []
    symbolic: list[Expr] = []
    for node in f.atoms(exp):
        argument = as_expr(node.args[0])
        coefficient, rest = argument.as_independent(x, as_Add=False)
        if as_expr(rest) != x:
            return None
        if isinstance(coefficient, Rational):
            coefficients.append(coefficient)
        else:
            symbolic.append(as_expr(coefficient))
    if symbolic and (coefficients or len(set(symbolic)) > 1):
        return None
    if symbolic:
        return symbolic[0]
    if not coefficients:
        return None
    numerator = 0
    denominator = 1
    for c in coefficients:
        numerator = gcd(numerator, abs(int(c.p)))
        denominator = lcm(denominator, int(c.q))
    return Rational(numerator, denominator)


def _exponential_substitution(f: Expr, x: Symbol) -> Optional[Substitution]:
    """``u = exp(c*x)``, ``dx = du/(c*u)``, for ``f`` a rational function of
    the exponentials of multiples of ``c*x``."""
    c = _exponential_generator(f, x)
    if c is None:
        return None
    u = Dummy('u', positive=True)
    g = attempt(lambda: as_expr(cancel(powsimp(as_expr(f.subs(x, log(u) / c) / (c * u)), force=True))),
                settings.timeout)
    if g is None or not g.is_rational_function(u):
        return None
    return Substitution(g, u, as_expr(exp(c * x)), as_expr(log(u) / c))


def _logarithmic_substitution(f: Expr, x: Symbol) -> Optional[Substitution]:
    """``x = exp(t)``, ``dx = exp(t) dt``, for ``f`` a rational function of
    ``log(x)`` times a power of ``x``."""
    if not f.has(log(x)):
        return None
    t = Dummy('t', real=True)
    g = attempt(lambda: as_expr(cancel(powsimp(as_expr(f.subs(x, exp(t)) * exp(t)), force=True))),
                settings.timeout)
    if g is None or g.has(log):
        return None
    L = Dummy('L')
    if not as_expr(f.subs(log(x), L)).is_rational_function(L):
        return None
    return Substitution(g, t, as_expr(log(x)), as_expr(exp(t)))


def power_substitutions(f: ExprLike, x: Symbol) -> list[Substitution]:
    """The substitutions which clear the fractional powers of ``x`` or of a
    linear ``a*x + b`` (``base = t**k``, ``k`` the least common denominator
    of their exponents), then ``u = exp(c*x)`` for a rational function of
    exponentials and ``x = exp(t)`` for a rational function of ``log(x)``
    times a power of ``x``; most promising first, at most three, each new
    integrand simplified with ``cancel`` and ``powsimp``.

    >>> from sympy import symbols, sqrt, exp, log, S
    >>> from sympy_extras.integrals.rewriting import power_substitutions
    >>> x = symbols('x')
    >>> [s.integrand for s in power_substitutions(1/(sqrt(x) + x**(S(1)/3)), x)]
    [6*_t**3/(_t + 1)]
    >>> [s.integrand for s in power_substitutions(exp(x)/(exp(2*x) - 1), x)]
    [1/(_u**2 - 1)]
    >>> [s.integrand for s in power_substitutions(1/(x*log(x)**2), x)]
    [_t**(-2)]
    """
    f_ = as_expr(f)
    found: list[Substitution] = []
    for base, k in _fractional_powers(f_, x).items():
        if len(found) >= _SUBSTITUTIONS:
            break
        s = _power_substitution(f_, x, base, k)
        if s is not None:
            found.append(s)
    for maker in (_exponential_substitution, _logarithmic_substitution):
        if len(found) >= _SUBSTITUTIONS:
            break
        s = maker(f_, x)
        if s is not None:
            found.append(s)
    return found


def substitute_back(F: ExprLike, t: Symbol, back: ExprLike) -> Expr:
    """The antiderivative ``F(t)`` returned to ``x``: ``F(back(x))``.

    >>> from sympy import symbols, sqrt, atan
    >>> from sympy_extras.integrals.rewriting import substitute_back
    >>> x, t = symbols('x t')
    >>> substitute_back(2*t - 2*atan(t), t, sqrt(x))
    2*sqrt(x) - 2*atan(sqrt(x))
    """
    return as_expr(as_expr(F).subs(t, as_expr(back)))
