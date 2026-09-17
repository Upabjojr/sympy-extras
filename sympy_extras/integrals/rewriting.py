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
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.core.add import Add
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.numbers import I, Integer, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.complexes import polar_lift
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.hyperbolic import HyperbolicFunction, InverseHyperbolicFunction
from sympy.logic.boolalg import Boolean
from sympy.polys.polyerrors import PolynomialError
from sympy.simplify.radsimp import fraction
from sympy.polys.polytools import Poly, factor, cancel
from sympy.simplify.powsimp import powsimp

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.settings import settings

__all__ = ['Substitution', 'rewritten_forms', 'power_substitutions', 'substitute_back', 'implied_assumptions']

#: the most forms and substitutions returned
_LIMIT = 5
_SUBSTITUTIONS = 3


class Substitution:
    """A change of variable ``x -> t``: ``integrand`` is ``g(t)`` with
    ``f(x) dx = g(t) dt``, ``variable`` the new symbol ``t``, ``back`` the
    expression of ``t`` in ``x`` and ``forward`` that of ``x`` in ``t``."""

    def __init__(self, integrand: Expr, variable: Symbol, back: Expr, forward: Expr,
                 facts: Optional[list[Boolean]] = None) -> None:
        self.integrand = integrand
        self.variable = variable
        self.back = back
        self.forward = forward
        #: the facts about ``x`` under which the substitution holds (the
        #: extracted factor of a radicand positive): an antiderivative
        #: found through it is right on that region
        self.facts = list(facts or [])

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


def _may_be_real(e: Expr, assumptions: Assumptions) -> bool:
    """Whether nothing says ``e`` is not real: the parameters are real by
    the contract of the module, so a symbolic ``e`` without ``I`` may be."""
    if e.has(I, exp_polar, polar_lift) or e.is_extended_real is False:
        return False
    return ask(element(e, S.Reals), assumptions) is not False


def _may_be_positive(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is positive, or a symbolic real which nothing says is
    not positive (the ``a`` of ``a**g(x)``, positive wherever the power is
    real on an interval)."""
    if _positive(e, assumptions):
        return True
    if not e.free_symbols or not _may_be_real(e, assumptions):
        return False
    return e.is_positive is not False and ask(as_boolean(e > 0), assumptions) is not False


def implied_assumptions(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> list[Boolean]:
    """The facts the canonical forms of ``f`` assume beyond the given
    assumptions: ``a > 0`` for each base ``a`` written as an exponential
    on the strength of the contract alone (``sqrt(a + b*c**(d*z))`` is
    real only for ``c > 0``, and a check which samples ``c`` of both
    signs would find the integrand real nowhere).

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.rewriting import implied_assumptions
    >>> c, d, z = symbols('c d z')
    >>> implied_assumptions(c**(d*z), z)
    [c > 0]
    """
    f_ = as_expr(f)
    found: list[Boolean] = []
    for node in f_.atoms(Pow):
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if not exponent.has(x) or base.has(x) or base == S.Exp1 or not base.free_symbols:
            continue
        if not _positive(base, assumptions) and _may_be_positive(base, assumptions):
            fact = as_boolean(base > 0)
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
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if exponent.has(x):
            if base.has(x) or base == S.Exp1:
                return node
            if _may_be_positive(base, assumptions):
                return exp(exponent * log(base))
            return node
        if not _may_be_real(exponent, assumptions):
            return node
        if isinstance(base, Pow) and base.base == x and not as_expr(base.exp).has(x) \
                and _may_be_real(as_expr(base.exp), assumptions) \
                and ask(as_boolean(x > 0), assumptions) is True:
            # (x**r)**p is x**(r*p) for x > 0 only: at x < 0 with an even
            # integer r, (x**r)**(1/r) is -x (the bug: the census's
            # exp(c*(z**r)**(1/r))**v came out as exp(c*v*z)/(c*v))
            return x**(as_expr(base.exp) * exponent)
        arguments: list[Expr] = []
        rest: Expr = S.One
        for part in Mul.make_args(base):
            f_ = as_expr(part)
            if isinstance(f_, exp) and _may_be_real(as_expr(f_.args[0]), assumptions):
                arguments.append(as_expr(f_.args[0]))
            elif not f_.has(x) and _may_be_positive(f_, assumptions):
                rest = rest * f_
            else:
                return node
        if not arguments:
            return node
        return exp(exponent * Add(*arguments)) * rest**exponent

    return as_expr(f.replace(lambda n: isinstance(n, Pow), rewrite))


def _hyperbolic_to_exponentials(f: Expr) -> Expr:
    if not f.has(HyperbolicFunction):
        return f
    return as_expr(f.rewrite(exp))


def _inverse_hyperbolic_to_logarithms(f: Expr) -> Expr:
    """The inverse hyperbolic functions as logarithms, in SymPy's forms,
    which are the functions' branches on the whole real line: ``acosh(u)``
    is ``log(u + sqrt(u - 1)*sqrt(u + 1))``, which for ``u < -1`` is
    ``acosh(-u) + I*pi`` as the function is (the bug: the hand form
    ``log(u + sqrt(u**2 - 1))`` is ``acosh(-u)`` there, and the integral of
    ``exp(acosh(z))`` came out with the wrong sign of the radical for
    ``z < -1``)."""
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
        for part in node.args:
            factor_ = as_expr(part)
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
    positive number or a parameter not known non-positive (a real
    parameter is positive wherever ``a**g(x)`` is real on an interval),
    ``exp(X)**v`` as ``exp(v*X)`` for a ``v`` not known non-real, products of exponentials combined,
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
        _, argument = as_expr(node.args[0]).as_independent(x, as_Add=True)   # exp(x + 14): the constant aside
        coefficient, rest = as_expr(argument).as_independent(x, as_Add=False)
        if as_expr(rest) != x:
            return None
        if isinstance(coefficient, Rational):
            coefficients.append(coefficient)
        else:
            symbolic.append(as_expr(coefficient))
    if symbolic and coefficients:
        return None
    if symbolic:
        # c*x and 2*c*x: the multiples of one symbolic coefficient
        first = symbolic[0]
        for other in symbolic[1:]:
            ratio = as_expr(cancel(other / first))
            if not isinstance(ratio, Rational):
                return None
            coefficients.append(ratio)
        coefficients.append(Rational(1))
        return as_expr(first * _generator(coefficients))
    if not coefficients:
        return None
    return _generator(coefficients)


def _generator(coefficients: list[Rational]) -> Rational:
    """The positive rational of which every coefficient is an integer multiple."""
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
    if g is None or not _algebraic_in(g, u):
        return None
    if not g.is_rational_function(u):
        # the radical table reads Q**(-5/2), not the expanded denominator cancel leaves
        factored = attempt(lambda: as_expr(factor(g)), settings.timeout)
        if factored is not None:
            g = factored
    return Substitution(g, u, as_expr(exp(c * x)), as_expr(log(u) / c))


def _algebraic_in(g: Expr, u: Symbol) -> bool:
    """Whether ``g`` is a rational function of ``u`` and of rational powers
    of polynomials in ``u`` (``sqrt(a + b*u)/u``, for the radical table)."""
    if g.is_rational_function(u):
        return True
    for node in g.atoms(Pow):
        if node.has(u) and isinstance(node.exp, Rational) and not node.exp.is_integer:
            if not as_expr(node.base).is_polynomial(u):
                return False
    replaced = g
    for node in g.atoms(Pow):
        if node.has(u) and isinstance(node.exp, Rational) and not node.exp.is_integer:
            replaced = as_expr(replaced.xreplace({node: Dummy('p')}))
    return bool(replaced.is_rational_function(u))


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
        h = attempt(lambda: as_expr(cancel(as_expr(g.xreplace(found.radicals)).subs(x, found.x_of_t)
                                           * found.x_of_t.diff(t))), settings.timeout)
        if h is None or not h.is_rational_function(t) or h.has(x):
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
    atoms = [node for node in f.atoms(Pow) if isinstance(node.exp, Rational) and node.exp.q > 1
             and node.has(x) and as_expr(node.base).is_polynomial(x)]
    if len(atoms) != 1:
        return []
    node = atoms[0]
    base, exponent = as_expr(node.base), node.exp
    if not isinstance(exponent, Rational):
        return []
    n = exponent.q
    try:
        content, factors = Poly(base, x).factor_list()
    except PolynomialError:
        return []
    if len(factors) < 2:
        return []
    # each factor's exponent k written q*n + r with r in [0, n) or in
    # (-n, 0]: the choice which leaves a Möbius function (numerator and
    # denominator of degree at most one) inside the root
    remainders: Optional[list[int]] = None
    for signs in range(2**len(factors)):
        found: list[int] = []
        inside: Expr = as_expr(content)
        for i, (piece, k) in enumerate(factors):
            r = k % n
            if r and (signs >> i) & 1:
                r -= n
            found.append(r)
            inside = inside * piece.as_expr()**r
        numerator, denominator = fraction(cancel(inside))
        if (any(k != r for (_, k), r in zip(factors, found))
                and all(Poly(part, x).degree() <= 1 for part in (numerator, denominator))):
            remainders = found
            break
    if remainders is None:
        return []
    extracted = [i for i, ((_, k), r) in enumerate(zip(factors, remainders)) if k != r]
    results: list[tuple[Expr, list[Boolean]]] = []
    for signs in range(2**len(extracted)):
        outside: Expr = S.One
        inside = as_expr(content)
        facts: list[Boolean] = []
        for i, ((piece, k), r) in enumerate(zip(factors, remainders)):
            factor_ = piece.as_expr()
            if i in extracted and (signs >> extracted.index(i)) & 1:
                factor_ = -factor_
                inside = inside * (-1)**k
            if k != r:
                outside = outside * factor_**((k - r) // n)
                facts.append(as_boolean(factor_ > 0))
            inside = inside * factor_**r
        rewritten = as_expr(f.xreplace({node: outside**(exponent * n) * inside**exponent}))
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
    if not all(isinstance(e, Rational) for e in (m, n, p)) or n == 0 or isinstance(p, Integer):
        return None
    m_, n_, p_ = Rational(m), Rational(n), Rational(p)
    q = (m_ + 1) / n_
    t = Dummy('t', positive=True)
    s = p_.q
    if q.is_integer:
        g = as_expr(C * Rational(s, 1) / (n_ * b) * t**(s - 1) * t**(s * p_) * ((t**s - a) / b)**int(q - 1))
        back = as_expr((a + b * x**n_)**Rational(1, s))
        forward = as_expr(((t**s - a) / b)**(1 / n_))
    elif (q + p_).is_integer:
        exponent = -int(q + p_) - 1
        g = as_expr(-C * Rational(s, 1) / (n_ * a) * t**(s - 1 + s * p_) * ((t**s - b) / a)**exponent)
        back = as_expr((a * x**(-n_) + b)**Rational(1, s))
        forward = as_expr(((t**s - b) / a)**(-1 / n_))
    else:
        return None
    g = as_expr(cancel(g))
    if not g.is_rational_function(t):
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
    odd = attempt(lambda: as_expr(cancel(as_expr(f.subs(x, -x) + f))), settings.timeout)
    if odd is None or odd != 0:
        return None
    s = Dummy('s', positive=True)
    g = attempt(lambda: as_expr(cancel(powsimp(as_expr(f.subs(x, sqrt(s)) / (2 * sqrt(s))), force=True))),
                settings.timeout)
    if g is None or g.has(sqrt(s)):
        return None
    return Substitution(g, s, as_expr(x**2), as_expr(sqrt(s)))


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
    if not as_expr(f.subs(log(x), L)).is_rational_function(L) and not _exponential_polynomial(g, t):
        # exp(-log(x)**2 - 1)/x**3 is exp(-t**2 - 2*t - 1): a rational function
        # of t and of exponentials of polynomials in t
        return None
    return Substitution(g, t, as_expr(log(x)), as_expr(exp(t)))


def _exponential_polynomial(g: Expr, t: Symbol) -> bool:
    """Whether ``g`` is a rational function of ``t`` and of exponentials of
    polynomials in ``t``."""
    replacement: dict[Expr, Expr] = {}
    for node in g.atoms(exp):
        if not node.has(t):
            continue
        if not as_expr(node.args[0]).is_polynomial(t):
            return False
        replacement[as_expr(node)] = Dummy()
    return bool(as_expr(g.xreplace(replacement)).is_rational_function(t, *replacement.values()))


def power_substitutions(f: ExprLike, x: Symbol) -> list[Substitution]:
    """The substitutions which clear the fractional powers of ``x`` or of a
    linear ``a*x + b`` (``base = t**k``, ``k`` the least common denominator
    of their exponents), then ``u = exp(c*x)`` for a rational or algebraic
    function of exponentials, ``x = exp(t)`` for a rational function of
    ``log(x)`` times a power of ``x``, and ``s = x**2`` for an odd
    integrand with a radical; most promising first, at most three, each new
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
    radicals = [node for node in f.atoms(Pow) if isinstance(node.exp, Rational) and node.exp.q > 1 and node.has(x)]
    if not any(any(other != node and as_expr(node.base).has(other) for other in radicals) for node in radicals):
        return []
    found: list[Substitution] = []
    for node in radicals:
        base = as_expr(node.base)
        if any(other != node and base.has(other) for other in radicals):
            continue
        poly = base.as_poly(x)
        if poly is None or poly.degree() != 1:
            continue
        k = node.exp.q
        t = Dummy('t', positive=True)
        a, b = as_expr(poly.coeff_monomial(x)), as_expr(poly.coeff_monomial(1))
        forward = as_expr((t**k - b) / a)
        g = attempt(lambda: as_expr(cancel(powsimp(as_expr(f.subs(x, forward) * k * t**(k - 1) / a), force=True))),
                    settings.timeout)
        if g is None:
            continue
        for s in power_substitutions(g, t):
            found.append(Substitution(s.integrand, s.variable, as_expr(s.back.subs(t, base**Rational(1, k))),
                                      as_expr(forward.subs(t, s.forward)), s.facts))
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
