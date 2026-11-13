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

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native ask, ``powsimp``, ``cancel``, ``factor`` and the
rewriting rules of the core's functions. The functions on SymPy objects
are in :mod:`.rewriting_sympy`. Every new variable is a dummy with a name
of its own (``_t3``): the typed polynomials identify their symbols by
name.

Examples
========

>>> from sympy_extras.core import Symbol, sqrt
>>> from sympy_extras.functions import exp, sinh
>>> from sympy_extras.integrals.rewriting import rewritten_forms, power_substitutions
>>> x, z = Symbol('x'), Symbol('z')
>>> a, c = Symbol('a', positive=True), Symbol('c', positive=True)
>>> rewritten_forms(a**(c*z)*exp(z**2), z)
[exp(z**2)*exp(c*z*log(a)), exp(c*z*log(a) + z**2)]
>>> rewritten_forms(x/sinh(x + 2), x)
[x/(-exp(-x - 2)/2 + exp(x + 2)/2)]
>>> s = power_substitutions(sqrt(x)/(1 + x), x)[0]
>>> t = Symbol('t')
>>> s.integrand.subs(s.variable, t), s.back
(2*t**2/(t**2 + 1), sqrt(x))
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from math import gcd, lcm
from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import Boolean, to_boolean
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import E, I, Integer, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, PolarLift
from sympy_extras.functions.elementary.exponential import Exp, ExpPolar, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import (Acosh, Acoth, Acsch, Asech, Asinh, Atanh, Cosh, Coth, Csch,
                                                          Sech, Sinh, Tanh)
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly, cancel, factor
from sympy_extras.sets.sets import Reals
from sympy_extras.settings import settings
from sympy_extras.simplify.powsimp import powsimp
from sympy_extras.simplify.radsimp import fraction

__all__ = ['Substitution', 'rewritten_forms', 'power_substitutions', 'substitute_back', 'implied_assumptions']

#: the most forms and substitutions returned
_LIMIT = 5
_SUBSTITUTIONS = 3

#: SymPy's HyperbolicFunction and InverseHyperbolicFunction
_HYPERBOLIC: Final = (Sinh, Cosh, Tanh, Coth, Sech, Csch)
_INVERSE_HYPERBOLIC: Final = (Asinh, Acosh, Atanh, Acoth, Asech, Acsch)

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so a dummy ``t`` beside a parameter ``t`` would be
    taken for it."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


def _expr(e: Basic) -> Expr:
    if not isinstance(e, Expr):
        raise TypeError("expected an expression, got %s" % (e,))
    return e


def _operations(e: Expr) -> int:
    count = e.count_ops()
    return int(count) if isinstance(count, Integer) else 0


def _coefficients(e: Expr, x: Symbol) -> Optional[dict[int, Expr]]:
    """The nonzero coefficients of the powers of ``x`` in the polynomial
    ``e``, whatever its coefficients (SymPy's ``Poly`` over ``EX``, which
    the native ``Poly`` does not have); ``None`` when ``e`` is not a
    polynomial in ``x``."""
    if e.is_polynomial(x) is not True:
        return None
    found: dict[int, Expr] = {}
    for term in Add.make_args(expand(e)):
        coefficient, rest = term.as_independent(x, as_Add=False)
        if rest == 1:
            n = 0
        elif rest == x:
            n = 1
        elif isinstance(rest, Pow) and rest.base == x and isinstance(rest.exp, Integer) and rest.exp.p > 0:
            n = rest.exp.p
        else:
            return None
        found[n] = found[n] + coefficient if n in found else coefficient
    return {n: c for n, c in found.items() if c != 0}


def _linear(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(a, b)`` with ``e == a*x + b`` and ``a`` not zero, ``None``
    otherwise."""
    found = _coefficients(e, x)
    if found is None or not found or max(found) != 1:
        return None
    return (found[1], found.get(0, S.Zero))


def _degree(e: Expr, x: Symbol) -> Optional[int]:
    """The degree of the polynomial ``e`` in ``x`` (``-1`` for zero),
    ``None`` when ``e`` is not a polynomial."""
    found = _coefficients(e, x)
    if found is None:
        return None
    return max(found) if found else -1


class Substitution:
    """A change of variable ``x -> t``: ``integrand`` is ``g(t)`` with
    ``f(x) dx = g(t) dt``, ``variable`` the new symbol ``t``, ``back`` the
    expression of ``t`` in ``x`` and ``forward`` that of ``x`` in ``t``;
    ``facts`` are the facts about ``x`` under which the substitution holds
    (the extracted factor of a radicand positive): an antiderivative found
    through it is right on that region."""

    def __init__(self, integrand: Expr, variable: Symbol, back: Expr, forward: Expr,
                 facts: Optional[list[Boolean]] = None) -> None:
        self.integrand: Final[Expr] = integrand
        self.variable: Final[Symbol] = variable
        self.back: Final[Expr] = back
        self.forward: Final[Expr] = forward
        self.facts: Final[list[Boolean]] = list(facts or [])

    def __repr__(self) -> str:
        return "Substitution(%s, %s, back=%s)" % (self.integrand, self.variable, self.back)


def _positive(e: Expr, assumptions: Assumptions) -> bool:
    if e.is_positive is True:
        return True
    if not e.free_symbols:
        return False
    return ask(e > 0, assumptions) is True


def _may_be_real(e: Expr, assumptions: Assumptions) -> bool:
    """Whether nothing says ``e`` is not real: the parameters are real by
    the contract of the module, so a symbolic ``e`` without ``I`` may be."""
    if e.has(I, ExpPolar, PolarLift) or e.is_extended_real is False:
        return False
    return ask(element(e, Reals), assumptions) is not False


def _may_be_positive(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is positive, or a symbolic real which nothing says is
    not positive (the ``a`` of ``a**g(x)``, positive wherever the power is
    real on an interval)."""
    if _positive(e, assumptions):
        return True
    if not e.free_symbols or not _may_be_real(e, assumptions):
        return False
    return e.is_positive is not False and ask(e > 0, assumptions) is not False


def implied_assumptions(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> list[Boolean]:
    """The facts the canonical forms of ``f`` assume beyond the given
    assumptions: ``a > 0`` for each base ``a`` written as an exponential
    on the strength of the contract alone (``sqrt(a + b*c**(d*z))`` is
    real only for ``c > 0``, and a check which samples ``c`` of both
    signs would find the integrand real nowhere).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.integrals.rewriting import implied_assumptions
    >>> c, d, z = Symbol('c'), Symbol('d'), Symbol('z')
    >>> implied_assumptions(c**(d*z), z)
    [c > 0]
    """
    f_ = to_expr(f)
    found: list[Boolean] = []
    for node in f_.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if not exponent.has(x) or base.has(x) or base == E or not base.free_symbols:
            continue
        if not _positive(base, assumptions) and _may_be_positive(base, assumptions):
            fact = base > 0
            if fact not in found:
                found.append(fact)
    return found


def _bases_to_exponentials(f: Expr, x: Symbol, assumptions: Assumptions) -> Expr:
    """``a**g`` with ``a`` free of ``x`` and ``g`` varying with ``x`` as
    ``exp(g*log(a))``, ``exp(X)**v`` (and ``(exp(X)*exp(Y)*k)**v`` with a
    constant ``k``, as SymPy writes ``exp(c*t + g)**v`` after a
    substitution) as ``exp(v*X)`` for a ``v`` free of ``x``, and
    ``(x**r)**p`` as ``x**(r*p)``, each wherever the power is real on an
    interval with the parameters real: a real ``a`` with ``a**g`` real
    for every ``x`` of an interval is positive (``g`` is not
    integer-valued there), a real ``exp(X)`` with a real ``X`` is
    positive, and ``(x**r)**p`` is real on an interval for ``x > 0``. A
    base known non-positive or non-real, or an exponent known non-real,
    is left alone."""
    def rewrite(node: Basic) -> Basic:
        if not isinstance(node, Pow):
            return node
        base, exponent = node.base, node.exp
        if exponent.has(x):
            if base.has(x) or base == E:
                return node
            if _may_be_positive(base, assumptions):
                return exp(exponent * log(base))
            return node
        if not _may_be_real(exponent, assumptions):
            return node
        if isinstance(base, Pow) and base.base == x and not base.exp.has(x) \
                and _may_be_real(base.exp, assumptions) and ask(x > 0, assumptions) is True:
            # (x**r)**p is x**(r*p) for x > 0 only: at x < 0 with an even
            # integer r, (x**r)**(1/r) is -x (the bug: the census's
            # exp(c*(z**r)**(1/r))**v came out as exp(c*v*z)/(c*v))
            return x**(base.exp * exponent)
        arguments: list[Expr] = []
        rest: Expr = S.One
        for part in Mul.make_args(base):
            if isinstance(part, Exp) and _may_be_real(part.args[0], assumptions):
                arguments.append(part.args[0])
            elif not part.has(x) and _may_be_positive(part, assumptions):
                rest = rest * part
            else:
                return node
        if not arguments:
            return node
        return exp(exponent * add(*arguments)) * rest**exponent

    return _expr(f.replace(lambda n: isinstance(n, Pow), rewrite))


def _hyperbolic_to_exponentials(f: Expr) -> Expr:
    if not f.has(*_HYPERBOLIC):
        return f
    return f.rewrite(Exp)


def _inverse_hyperbolic_to_logarithms(f: Expr) -> Expr:
    """The inverse hyperbolic functions as logarithms, in SymPy's forms,
    which are the functions' branches on the whole real line: ``acosh(u)``
    is ``log(u + sqrt(u - 1)*sqrt(u + 1))``, which for ``u < -1`` is
    ``acosh(-u) + I*pi`` as the function is (the bug: the hand form
    ``log(u + sqrt(u**2 - 1))`` is ``acosh(-u)`` there, and the integral of
    ``exp(acosh(z))`` came out with the wrong sign of the radical for
    ``z < -1``)."""
    if not f.has(*_INVERSE_HYPERBOLIC):
        return f
    return f.rewrite(Log)


def _combined_exponentials(f: Expr) -> Expr:
    found = attempt(lambda: powsimp(f, combine='exp', deep=True), settings.timeout, native=True)
    return f if found is None else found


def _products_of_exponentials(f: Expr) -> bool:
    """Whether some product of ``f`` has two exponential factors."""
    return any(sum(isinstance(a, Exp) for a in node.args) > 1 for node in f.atoms(Mul))


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
        for part in node.args:
            factor_ = _expr(part)
            if isinstance(factor_, Pow) and isinstance(factor_.exp, Rational) and factor_.exp.q > 1 \
                    and factor_.has(x) and _nonnegative(factor_.base, x, assumptions):
                radicals.append((factor_.base, factor_.exp))
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
        return mul(*rest) * pow(radicand, Rational(1, q))

    return _expr(f.replace(lambda n: isinstance(n, Mul), rewrite))


def _nonnegative(e: Expr, x: Symbol, assumptions: Assumptions) -> bool:
    """Whether ``e >= 0`` for every real ``x`` under the assumptions."""
    if e.is_nonnegative is True:
        return True
    real = _fresh(x.name, real=True)
    candidate = _expr(e._xreplace_evaluated({x: real}))
    if candidate.is_nonnegative is True:
        return True
    return ask(e >= 0, _with(assumptions, [element(x, Reals)])) is True


def _with(assumptions: Assumptions, more: list[Boolean]) -> list[Boolean]:
    items: list[Boolean] = list(more)
    if isinstance(assumptions, (Boolean, bool)):
        items.append(to_boolean(assumptions))
    elif assumptions is not None:
        items.extend(to_boolean(a) for a in assumptions)
    return items


def _absolute_values(f: Expr, x: Symbol, assumptions: Assumptions) -> Expr:
    """``Abs(e)`` replaced by ``e`` or ``-e`` when the assumptions fix the
    sign of ``e``."""
    def rewrite(node: Basic) -> Basic:
        if not isinstance(node, Abs):
            return node
        inner = _expr(node.args[0])
        if ask(inner >= 0, assumptions) is True:
            return inner
        if ask(inner <= 0, assumptions) is True:
            return -inner
        return node

    return _expr(f.replace(lambda n: isinstance(n, Abs), rewrite))


def rewritten_forms(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> list[Expr]:
    """Canonical forms of ``f`` for integration in ``x``, each equal to
    ``f`` wherever ``f`` is real (``x`` real, the parameters real, and the
    signs the assumptions give): ``a**g(x)`` as ``exp(g(x)*log(a))`` for a
    positive number or a parameter not known non-positive (a real
    parameter is positive wherever ``a**g(x)`` is real on an interval),
    ``exp(X)**v`` as ``exp(v*X)`` for a ``v`` not known non-real, products of exponentials combined,
    hyperbolic functions as exponentials, inverse hyperbolic functions as
    logarithms (their real branches, which agree with the functions on
    the real domains), radicals of products with radicands known
    nonnegative combined, and ``Abs(e)`` by its sign when the assumptions
    fix it. In order of preference, without duplicates and without ``f``,
    at most five.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.functions import exp, cosh, asinh, Abs
    >>> from sympy_extras.integrals.rewriting import rewritten_forms
    >>> x = Symbol('x')
    >>> v = Symbol('v', real=True)
    >>> rewritten_forms(exp(x**2)**v*exp(3*x), x)
    [exp(3*x)*exp(v*x**2), exp(v*x**2 + 3*x)]
    >>> rewritten_forms(x*Abs(x), x, [x > 0])
    [x**2]
    >>> rewritten_forms(asinh(x)/sqrt(x**2 + 1), x)
    [log(x + sqrt(x**2 + 1))/sqrt(x**2 + 1)]
    """
    f_ = to_expr(f)
    forms: list[Expr] = []

    def add_form(candidate: Optional[Expr]) -> None:
        if candidate is not None and candidate != f_ and candidate not in forms and len(forms) < _LIMIT:
            forms.append(candidate)

    signed = _absolute_values(f_, x, assumptions)
    exponential = _bases_to_exponentials(signed, x, assumptions)
    exponential = _hyperbolic_to_exponentials(exponential)
    add_form(exponential)
    combined = _combined_exponentials(exponential)
    add_form(combined)
    # a sum of exponentials times an exponential: distributed, then combined
    distributed = _combined_exponentials(expand(exponential, power_base=False, power_exp=False))
    if distributed != combined and not _products_of_exponentials(distributed):
        add_form(distributed)
        combined = distributed
    logarithmic = _inverse_hyperbolic_to_logarithms(combined)
    add_form(logarithmic)
    radical = _recombined_radicals(logarithmic, x, assumptions)
    add_form(radical)
    expanded = attempt(lambda: cancel(expand(radical)), settings.timeout, native=True)
    if expanded is not None and _operations(expanded) <= _operations(radical):
        add_form(expanded)
    return forms


def _fractional_powers(f: Expr, x: Symbol) -> dict[Expr, int]:
    """The bases ``x`` or ``a*x + b`` of the fractional powers of ``f``
    with the least common denominator of their exponents."""
    found: dict[Expr, int] = {}
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if not isinstance(exponent, Rational) or exponent.q == 1 or not base.has(x):
            continue
        if base != x and _degree(base, x) != 1:
            continue
        found[base] = lcm(found.get(base, 1), exponent.q)
    return found


def _power_substitution(f: Expr, x: Symbol, base: Expr, k: int) -> Optional[Substitution]:
    """``base = t**k``: ``x = (t**k - b)/a`` for ``base = a*x + b``."""
    t = _fresh('t', positive=True)
    linear = _linear(base, x)
    if linear is None:
        return None
    a, b = linear
    forward = (t**k - b) / a
    jacobian = k * t**(k - 1) / a
    g = attempt(lambda: cancel(powsimp(f.subs(x, forward) * jacobian, force=True)), settings.timeout, native=True)
    if g is None:
        return None
    if any(isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q > 1 and node.has(t)
           for node in g.atoms(Pow)):
        return None                                         # a fractional power survived
    return Substitution(g, t, base**Rational(1, k), forward)


def _exponential_generator(f: Expr, x: Symbol) -> Optional[Expr]:
    """``c`` such that every ``exp(c_i*x)`` of ``f`` has ``c_i`` an integer
    multiple of ``c``; ``None`` without exponentials linear in ``x`` or
    with an exponential of another shape."""
    coefficients: list[Rational] = []
    symbolic: list[Expr] = []
    for node in f.atoms(Exp):
        if not isinstance(node, Exp):
            continue
        _, argument = _expr(node.args[0]).as_independent(x, as_Add=True)   # exp(x + 14): the constant aside
        coefficient, rest = argument.as_independent(x, as_Add=False)
        if rest != x:
            return None
        if isinstance(coefficient, Rational):
            coefficients.append(coefficient)
        else:
            symbolic.append(coefficient)
    if symbolic and coefficients:
        return None
    if symbolic:
        # c*x and 2*c*x: the multiples of one symbolic coefficient
        first = symbolic[0]
        for other in symbolic[1:]:
            ratio = cancel(other / first)
            if not isinstance(ratio, Rational):
                return None
            coefficients.append(ratio)
        coefficients.append(Integer(1))
        return first * _generator(coefficients)
    if not coefficients:
        return None
    return _generator(coefficients)


def _generator(coefficients: list[Rational]) -> Expr:
    """The positive rational of which every coefficient is an integer multiple."""
    numerator = 0
    denominator = 1
    for c in coefficients:
        numerator = gcd(numerator, abs(c.p))
        denominator = lcm(denominator, c.q)
    return number_from_fraction(Fraction(numerator, denominator))


def _exponential_substitution(f: Expr, x: Symbol) -> Optional[Substitution]:
    """``u = exp(c*x)``, ``dx = du/(c*u)``, for ``f`` a rational function of
    the exponentials of multiples of ``c*x``."""
    c = _exponential_generator(f, x)
    if c is None:
        return None
    u = _fresh('u', positive=True)
    found = attempt(lambda: cancel(powsimp(f.subs(x, log(u) / c) / (c * u), force=True)), settings.timeout,
                    native=True)
    if found is None or not _algebraic_in(found, u):
        return None
    g = found
    if g.is_rational_function(u) is not True:
        # the radical table reads Q**(-5/2), not the expanded denominator cancel leaves
        factored = attempt(lambda: factor(found), settings.timeout, native=True)
        if factored is not None:
            g = factored
    return Substitution(g, u, exp(c * x), log(u) / c)


def _fractional(node: Basic, u: Symbol) -> bool:
    return isinstance(node, Pow) and node.has(u) and isinstance(node.exp, Rational) \
        and not isinstance(node.exp, Integer)


def _algebraic_in(g: Expr, u: Symbol) -> bool:
    """Whether ``g`` is a rational function of ``u`` and of rational powers
    of polynomials in ``u`` (``sqrt(a + b*u)/u``, for the radical table)."""
    if g.is_rational_function(u) is True:
        return True
    for node in g.atoms(Pow):
        if isinstance(node, Pow) and _fractional(node, u) and node.base.is_polynomial(u) is not True:
            return False
    replaced = g
    for node in g.atoms(Pow):
        if _fractional(node, u):
            replaced = _expr(replaced._xreplace_evaluated({node: _fresh('p')}))
    return replaced.is_rational_function(u) is True


def _moebius_substitutions(f: Expr, x: Symbol) -> list[Substitution]:
    """``t**n = (a*x + b)/(c*x + d)`` for radicals which are rational powers
    of one Möbius function (:func:`.algebraic.moebius_root`): the curve
    has genus zero, the integrand becomes rational in ``t``.
    ``((x - 1)**2*(x + 1))**(1/3)`` is ``(x - 1)*((x + 1)/(x - 1))**(1/3)``
    where ``x > 1`` and ``(1 - x)*((x + 1)/(1 - x))**(1/3)`` where ``x < 1``
    (the cube extracted with the sign of the factor): one substitution
    per region, each carrying its fact."""
    from .algebraic import moebius_root
    found_all: list[Substitution] = []
    for g, facts in _extracted_powers(f, x):
        found = moebius_root(g, x)
        if found is None:
            continue
        t = found.t
        rule: dict[Basic, Basic] = {k: v for k, v in found.radicals.items()}
        h = attempt(lambda: cancel(_expr(g._xreplace_evaluated(rule)).subs(x, found.x_of_t) * found.x_of_t.diff(t)),
                    settings.timeout, native=True)
        if h is None or h.is_rational_function(t) is not True or h.has(x):
            continue
        found_all.append(Substitution(h, t, found.t_of_x, found.x_of_t, facts))
    return found_all


def _extracted_powers(f: Expr, x: Symbol) -> list[tuple[Expr, list[Boolean]]]:
    """``f`` with its radical ``(R**k*S)**(p/n)`` written ``R**((k - r)*p/n) *
    (S*R**r)**(p/n)``, ``r`` the remainder of ``k`` by ``n`` which leaves a
    Möbius function under the root: ``((x - 1)**2*(x + 1))**(1/3)`` is
    ``(x - 1)*((x + 1)/(x - 1))**(1/3)``, an identity where the extracted
    factor is positive, and ``(1 - x)*((x + 1)/(1 - x))**(1/3)`` where it
    is negative (``R**k`` is ``(-1)**k*(-R)**k``, the sign into the
    content); one rewriting per region with its facts, none when nothing
    is extracted or ``f`` has not exactly one such radical."""
    atoms = [node for node in f.atoms(Pow) if isinstance(node, Pow) and isinstance(node.exp, Rational)
             and node.exp.q > 1 and node.has(x) and node.base.is_polynomial(x) is True]
    if len(atoms) != 1:
        return []
    node = atoms[0]
    base, exponent = node.base, node.exp
    if not isinstance(exponent, Rational):
        return []
    n = exponent.q
    try:
        content, polys = Poly(base, x).factor_list()
    except BasePolynomialError:
        return []
    factors: list[tuple[Expr, int]] = [(p.as_expr(), k) for p, k in polys]
    if len(factors) < 2:
        return []
    # each factor's exponent k written q*n + r with r in [0, n) or in
    # (-n, 0]: the choice which leaves a Möbius function (numerator and
    # denominator of degree at most one) inside the root
    remainders: Optional[list[int]] = None
    for signs in range(2**len(factors)):
        found: list[int] = []
        inside: Expr = content
        for i, (piece, k) in enumerate(factors):
            r = k % n
            if r and (signs >> i) & 1:
                r -= n
            found.append(r)
            inside = inside * piece**r
        numerator, denominator = fraction(cancel(inside))
        degrees = [_degree(part, x) for part in (numerator, denominator)]
        if (any(k != r for (_, k), r in zip(factors, found))
                and all(d is not None and d <= 1 for d in degrees)):
            remainders = found
            break
    if remainders is None:
        return []
    extracted = [i for i, ((_, k), r) in enumerate(zip(factors, remainders)) if k != r]
    results: list[tuple[Expr, list[Boolean]]] = []
    for signs in range(2**len(extracted)):
        outside: Expr = S.One
        inside = content
        facts: list[Boolean] = []
        for i, ((piece, k), r) in enumerate(zip(factors, remainders)):
            factor_ = piece
            if i in extracted and (signs >> extracted.index(i)) & 1:
                factor_ = -factor_
                inside = inside * Integer(-1)**k
            if k != r:
                outside = outside * factor_**((k - r) // n)
                facts.append(factor_ > 0)
            inside = inside * factor_**r
        rewritten = _expr(f._xreplace_evaluated({node: outside**(exponent * n) * inside**exponent}))
        results.append((rewritten, facts))
    return results


def _binomial_substitution(f: Expr, x: Symbol) -> Optional[Substitution]:
    """Chebyshev's second and third cases for ``x**m*(a + b*x**n)**p``:
    ``t**s = a + b*x**n`` when ``(m + 1)/n`` is an integer and ``t**s = a*x**(-n)
    + b`` when ``(m + 1)/n + p`` is (``s`` the denominator of ``p``); the
    first case, ``x = t**k``, is the power substitution above."""
    from .algebraic import binomial_differential
    found = binomial_differential(f, x)
    if found is None:
        return None
    C, m, a, b, n, p = found
    if not (isinstance(m, Rational) and isinstance(n, Rational) and isinstance(p, Rational)) or n == 0 \
            or isinstance(p, Integer):
        return None
    m_, n_, p_ = Fraction(m.p, m.q), Fraction(n.p, n.q), Fraction(p.p, p.q)
    q = (m_ + 1) / n_
    t = _fresh('t', positive=True)
    s = p_.denominator
    g: Expr
    if q.denominator == 1:
        g = C * Integer(s) / (n * b) * t**(s - 1) * t**number_from_fraction(s * p_) * ((t**s - a) / b)**int(q - 1)
        back = (a + b * x**n)**Rational(1, s)
        forward = ((t**s - a) / b)**(1 / n)
    elif (q + p_).denominator == 1:
        exponent = -int(q + p_) - 1
        g = -C * Integer(s) / (n * a) * t**number_from_fraction(s - 1 + s * p_) * ((t**s - b) / a)**exponent
        back = (a * x**(-n) + b)**Rational(1, s)
        forward = ((t**s - b) / a)**(-1 / n)
    else:
        return None
    g = cancel(g)
    if g.is_rational_function(t) is not True:
        return None
    return Substitution(g, t, back, forward)


def _even_substitution(f: Expr, x: Symbol) -> Optional[Substitution]:
    """``s = x**2``, ``ds = 2*x*dx``, for an odd ``f`` (``f(-x) = -f(x)``):
    ``f = x*g(x**2)`` and the integral is that of ``g(s)/2``, an identity
    on both sides of zero; for ``1/(x*sqrt(a + b*x**2 + c*x**4))``, an
    even quartic under the radical, the radical table in ``s``."""
    if not f.has(x) or not any(isinstance(p, Pow) and p.has(x) and isinstance(p.exp, Rational) and p.exp.q == 2
                               for p in f.atoms(Pow)):
        return None
    odd = attempt(lambda: cancel(f.subs(x, -x) + f), settings.timeout, native=True)
    if odd is None or odd != 0:
        return None
    s = _fresh('s', positive=True)
    g = attempt(lambda: cancel(powsimp(f.subs(x, sqrt(s)) / (2 * sqrt(s)), force=True)), settings.timeout,
                native=True)
    if g is None or g.has(sqrt(s)):
        return None
    return Substitution(g, s, x**2, sqrt(s))


def _logarithmic_substitution(f: Expr, x: Symbol) -> Optional[Substitution]:
    """``x = exp(t)``, ``dx = exp(t) dt``, for ``f`` a rational function of
    ``log(x)`` times a power of ``x``."""
    if not f.has(log(x)):
        return None
    t = _fresh('t', real=True)
    g = attempt(lambda: cancel(powsimp(f.subs(x, exp(t)) * exp(t), force=True)), settings.timeout, native=True)
    if g is None or g.has(Log):
        return None
    L = _fresh('L')
    if f.subs(log(x), L).is_rational_function(L) is not True and not _exponential_polynomial(g, t):
        # exp(-log(x)**2 - 1)/x**3 is exp(-t**2 - 2*t - 1): a rational function
        # of t and of exponentials of polynomials in t
        return None
    return Substitution(g, t, log(x), exp(t))


def _exponential_polynomial(g: Expr, t: Symbol) -> bool:
    """Whether ``g`` is a rational function of ``t`` and of exponentials of
    polynomials in ``t``."""
    replacement: dict[Basic, Basic] = {}
    generators: list[Symbol] = []
    for node in g.atoms(Exp):
        if not isinstance(node, Exp) or not node.has(t):
            continue
        if _expr(node.args[0]).is_polynomial(t) is not True:
            return False
        generator = _fresh('e')
        replacement[node] = generator
        generators.append(generator)
    return _expr(g._xreplace_evaluated(replacement)).is_rational_function(t, *generators) is True


def power_substitutions(f: ExprLike, x: Symbol) -> list[Substitution]:
    """The substitutions which clear the fractional powers of ``x`` or of a
    linear ``a*x + b`` (``base = t**k``, ``k`` the least common denominator
    of their exponents), then ``u = exp(c*x)`` for a rational or algebraic
    function of exponentials, ``x = exp(t)`` for a rational function of
    ``log(x)`` times a power of ``x``, and ``s = x**2`` for an odd
    integrand with a radical; most promising first, at most three, each new
    integrand simplified with ``cancel`` and ``powsimp``.

    >>> from sympy_extras.core import Symbol, sqrt, Rational
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.integrals.rewriting import power_substitutions
    >>> x, t = Symbol('x'), Symbol('t')
    >>> [s.integrand.subs(s.variable, t) for s in power_substitutions(1/(sqrt(x) + x**Rational(1, 3)), x)]
    [6*t**3/(t + 1)]
    >>> [s.integrand.subs(s.variable, t) for s in power_substitutions(exp(x)/(exp(2*x) - 1), x)]
    [1/(t**2 - 1)]
    >>> [s.integrand.subs(s.variable, t) for s in power_substitutions(1/(x*log(x)**2), x)]
    [t**(-2)]
    """
    f_ = to_expr(f)
    found: list[Substitution] = []
    for base, k in _fractional_powers(f_, x).items():
        if len(found) >= _SUBSTITUTIONS:
            break
        s = _power_substitution(f_, x, base, k)
        if s is not None:
            found.append(s)
    for maker in (_exponential_substitution, _logarithmic_substitution, _even_substitution,
                  _binomial_substitution):
        if len(found) >= _SUBSTITUTIONS:
            break
        s = maker(f_, x)
        # t**2 = 2*x + 1 for x*sqrt(2*x + 1) is Chebyshev's first case and
        # the power substitution above: once
        if s is not None and all(s.back != other.back for other in found):
            found.append(s)
    if len(found) < _SUBSTITUTIONS:
        # one substitution per region, all of them or none
        regions = _moebius_substitutions(f_, x)
        if all(region.back != other.back for region in regions for other in found):
            found.extend(regions)
    if len(found) < _SUBSTITUTIONS:
        for s in _nested_substitutions(f_, x):
            if all(s.back != other.back for other in found):
                found.append(s)
    return found


def _nested_substitutions(f: Expr, x: Symbol) -> list[Substitution]:
    """The substitutions of a nested radical: the innermost radical of a
    linear polynomial substituted first (``x = t**2`` for ``sqrt(1 -
    sqrt(x))``), the substitutions of the result (``u**2 = 1 - t``)
    composed with it."""
    radicals = [node for node in f.atoms(Pow) if isinstance(node, Pow) and isinstance(node.exp, Rational)
                and node.exp.q > 1 and node.has(x)]
    if not any(any(other != node and node.base.has(other) for other in radicals) for node in radicals):
        return []
    found: list[Substitution] = []
    for node in radicals:
        base = node.base
        exponent = node.exp
        if any(other != node and base.has(other) for other in radicals) or not isinstance(exponent, Rational):
            continue
        linear = _linear(base, x)
        if linear is None:
            continue
        k = exponent.q
        t = _fresh('t', positive=True)
        a, b = linear
        forward = (t**k - b) / a
        g = attempt(lambda: cancel(powsimp(f.subs(x, forward) * k * t**(k - 1) / a, force=True)), settings.timeout,
                    native=True)
        if g is None:
            continue
        for s in power_substitutions(g, t):
            found.append(Substitution(s.integrand, s.variable, s.back.subs(t, base**Rational(1, k)),
                                      forward.subs(t, s.forward), s.facts))
    return found


def substitute_back(F: ExprLike, t: Symbol, back: ExprLike) -> Expr:
    """The antiderivative ``F(t)`` returned to ``x``: ``F(back(x))``.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.functions import atan
    >>> from sympy_extras.integrals.rewriting import substitute_back
    >>> x, t = Symbol('x'), Symbol('t')
    >>> substitute_back(2*t - 2*atan(t), t, sqrt(x))
    2*sqrt(x) - 2*atan(sqrt(x))
    """
    return to_expr(F).subs(t, to_expr(back))
