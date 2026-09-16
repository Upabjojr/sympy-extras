r"""Antiderivatives of exponentials of powers, of quadratics and of rational
functions of an exponential.

The integrands of Maxima's test suite with symbolic exponents,
`z^{v-1} e^{a z^n + c}`, `z^2 e^{a z^2 + b z}`, `a^{d z} z^{v-1}`,
`(a + b e^{c z})^p/(d + e e^{c z})`, are integrated here in real forms.
SymPy's Meijer route writes the first family as polar incomplete gamma
functions which are right for `x > 0` and wrong for `x < 0`.

1. `c\, x^{v-1} e^{a x^n + b}` with `s = v/n`, `n` a number or a symbol. For `a < 0` (known), the
   incomplete gamma form of DLMF 8.2.4

   .. math::

       \int x^{v-1} e^{a x^n}\,dx = -\frac{(-a)^{-s}}{n}\,\Gamma(s, -a x^n),

   real for `-a x^n > 0`; otherwise the confluent form, valid for every
   `a` and every `x` as an identity of power series,

   .. math::

       \int x^{v-1} e^{a x^n}\,dx = \frac{x^v}{v}\, {}_1F_1(s; s + 1; a x^n),

   which follows from `\frac{d}{dx}\bigl[x^v \sum_k \frac{s}{s+k}
   \frac{(a x^n)^k}{k!}\bigr] = v\,x^{v-1} e^{a x^n}`. The special cases
   are elementary: `s` a positive integer by the reduction
   `\int x^{v-1} e^{a x^n} = \frac{x^{v-n} e^{a x^n}}{a n} - \frac{v-n}{a n}
   \int x^{v-n-1} e^{a x^n}`, `s = 0` the exponential integral
   `\operatorname{Ei}(a x^n)/n`, `s` a negative integer by the reduction
   upwards `\int x^{v-1} e^{a x^n} = \frac{x^v e^{a x^n}}{v} - \frac{a n}{v}
   \int x^{v+n-1} e^{a x^n}` down to `s = 0`, and `n = 2`, `v = 1` the error
   functions `\sqrt{\pi}\,\operatorname{erf}(\sqrt{-a}\,x)/(2\sqrt{-a})` for
   `a < 0` and `\sqrt{\pi}\,\operatorname{erfi}(\sqrt{a}\,x)/(2\sqrt{a})`
   for `a > 0` (a ``Piecewise`` when the sign is unknown).
2. `x^m e^{a x^2 + b x + c}` with `m` a nonnegative integer, by completing
   the square and shifting `x = t - b/(2a)`, each term then of the first
   form.
3. `(a x + b)^w e^{c x + d}` by the shift `t = a x + b`, which makes it
   `t^w e^{c t/a + d - b c/a}/a`, of the first form (`e^{c x}/(a x + b)^k`
   is the exponential integral `E_k`, DLMF 8.19.3), and a polynomial
   times an exponential of a quadratic in `t` of the second.
4. Rational functions of `e^{c x}` by `u = e^{c x}`, `dx = du/(c u)`,
   through the rational integrator, and `(a + b e^{c x})^{p} R(e^{c x})`
   with a rational `p = m/q` by `t^q = a + b e^{c x}`, which makes the
   integrand rational in `t`.

Every antiderivative is checked by differentiation before it is
returned, symbolically or at random points where the integrand is real.

Examples
========

>>> from sympy import symbols, exp
>>> from sympy_extras.integrals.exponential import exponential_antiderivative
>>> x = symbols('x')
>>> v = symbols('v', positive=True)
>>> a = symbols('a', negative=True)
>>> exponential_antiderivative(x**(v - 1)*exp(a*x**2), x)
-uppergamma(v/2, -a*x**2)/(2*(-a)**(v/2))
>>> exponential_antiderivative(x**2*exp(x**2), x)
x*exp(x**2)/2 - sqrt(pi)*erfi(x)/4
>>> exponential_antiderivative(exp(x)/(exp(2*x) + 1), x)
atan(exp(x))

References
==========

.. [DLMF] NIST Digital Library of Mathematical Functions, 8.2.4, 8.4.4
   and 13.4.1 (incomplete gamma functions and the confluent
   hypergeometric function).
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi
from sympy.core.power import Pow
from sympy.core.function import expand
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.error_functions import erf, erfi, Ei
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.hyper import hyper
from sympy.logic.boolalg import Boolean
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel, gcd
from sympy.simplify.powsimp import powsimp
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.settings import settings
from .conditions import numerically_equal

__all__ = ['exponential_antiderivative', 'power_exponential', 'exponential_rational']


def _budget() -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / 4


def _facts(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [as_boolean(assumptions)]
    return [as_boolean(a) for a in assumptions]


def _checks(F: Expr, f: Expr, x: Symbol, facts: list[Boolean]) -> bool:
    """Whether ``F`` is an antiderivative of ``f``: ``F' - f`` cancels or
    simplifies to zero, or vanishes numerically under the facts."""
    difference = as_expr(F.diff(x) - f)
    reduced = attempt(lambda: as_expr(cancel(difference)), _budget())
    if reduced is not None and reduced == 0:
        return True
    simpler = attempt(lambda: as_expr(simplify(reduced if reduced is not None else difference)), _budget())
    if simpler is not None and simpler == 0:
        return True
    return numerically_equal(as_expr(F.diff(x)), f, facts)


# ---------------------------------------------------------------------------
# x**(v - 1) exp(a x**n)

def _power_exponential_raw(v: Expr, a: Expr, n: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(x**(v - 1)*exp(a*x**n), x)`` without the constant factors,
    in the real form the case allows; ``None`` when there is none."""
    s = as_expr(cancel(v / n))
    if v == 0:
        return as_expr(Ei(a * x**n) / n)
    if isinstance(s, Integer) and s > 0:
        # downwards to Integral(x**(n-1) exp(a x**n)) = exp(a x**n)/(a n)
        if s == 1:
            return as_expr(exp(a * x**n) / (a * n))
        lower = _power_exponential_raw(v - n, a, n, x, assumptions)
        if lower is None:
            return None
        return as_expr(x**(v - n) * exp(a * x**n) / (a * n) - (v - n) * lower / (a * n))
    if isinstance(s, Integer) and s < 0:
        # upwards to s = 0, the exponential integral
        higher = _power_exponential_raw(v + n, a, n, x, assumptions)
        if higher is None:
            return None
        return as_expr(x**v * exp(a * x**n) / v - a * n * higher / v)
    if n == 2 and isinstance(v, Integer) and v % 2 == 1 and v != 1:
        # an odd power with the Gaussian: the recurrences reach the error function
        if v > 1:
            lower = _power_exponential_raw(v - 2, a, n, x, assumptions)
            if lower is None:
                return None
            return as_expr(x**(v - 2) * exp(a * x**2) / (2 * a) - (v - 2) * lower / (2 * a))
        higher = _power_exponential_raw(v + 2, a, n, x, assumptions)
        if higher is None:
            return None
        return as_expr(x**v * exp(a * x**2) / v - 2 * a * higher / v)
    negative = ask(as_boolean(a < 0), assumptions)
    positive = ask(as_boolean(a > 0), assumptions)
    # the incomplete gamma form is real where -a*x**n > 0: for every x when n is
    # even, for x > 0 otherwise (x*exp(-x**3) at x < -1 needs the confluent form)
    real_everywhere = (isinstance(n, Integer) and n % 2 == 0) or ask(as_boolean(x > 0), assumptions) is True
    if n == 2 and v == 1:
        below = as_expr(sqrt(pi) * erf(sqrt(-a) * x) / (2 * sqrt(-a)))
        above = as_expr(sqrt(pi) * erfi(sqrt(a) * x) / (2 * sqrt(a)))
        if negative is True:
            return below
        if positive is True:
            return above
        return as_expr(Piecewise((below, a < 0), (above, True)))
    if negative is True and real_everywhere:
        return as_expr(-(-a)**(-s) * uppergamma(s, -a * x**n) / n)
    # the confluent form, an identity of power series for every a
    return as_expr(x**v * hyper((s,), (s + 1,), a * x**n) / v)


def _monomial_exponential(term: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr]]:
    """``(c, v, a, n)`` with ``term == c * x**(v - 1) * exp(a * x**n)``, the
    constant ``exp(b)`` of the exponent inside ``c``; ``None`` otherwise."""
    coefficient: Expr = S.One
    power: Expr = S.Zero
    exponent: Optional[Expr] = None
    for factor in Mul.make_args(term):
        f = as_expr(factor)
        if not f.has(x):
            coefficient = coefficient * f
        elif f == x:
            power = power + 1
        elif isinstance(f, Pow) and f.base == x and not as_expr(f.exp).has(x):
            power = power + as_expr(f.exp)
        elif isinstance(f, exp) and exponent is None:
            exponent = as_expr(f.args[0])
        else:
            return None
    if exponent is None:
        return None
    constant, rest = exponent.as_independent(x, as_Add=True)
    # b*x**2*log(a) + c*x**2*log(h), the combined exponent of a**(b*x**2)*h**(c*x**2), is one monomial
    a, monomial = as_expr(factor_terms(rest)).as_independent(x, as_Add=False)
    a_, monomial_ = as_expr(a), as_expr(monomial)
    if monomial_ == x:
        n: Expr = S.One
    elif isinstance(monomial_, Pow) and monomial_.base == x and not as_expr(monomial_.exp).has(x):
        n = as_expr(monomial_.exp)                          # a number, or a symbolic exponent r
    else:
        return None
    if a_ == 0 or n == 0:
        return None
    return (as_expr(coefficient * exp(constant)), as_expr(power + 1), a_, n)


def power_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for a sum of terms ``c * x**(v - 1) * exp(a*x**n + b)``,
    in the real forms of the module documentation, checked by
    differentiation; ``None`` when ``f`` is not of that form or the check
    fails.

    >>> from sympy import symbols, exp, sqrt
    >>> from sympy_extras.integrals.exponential import power_exponential
    >>> x = symbols('x')
    >>> power_exponential(exp(-x**2), x)
    sqrt(pi)*erf(x)/2
    >>> power_exponential(x**3*exp(x**2), x)
    x**2*exp(x**2)/2 - exp(x**2)/2
    >>> power_exponential(exp(2*x)/x, x)
    Ei(2*x)
    """
    f_ = as_expr(f)
    combined = as_expr(powsimp(expand(f_), combine='exp'))
    total: Expr = S.Zero
    for term in Add.make_args(combined):
        found = _monomial_exponential(as_expr(term), x)
        if found is None:
            return None
        c, v, a, n = found
        piece = _power_exponential_raw(v, a, n, x, assumptions)
        if piece is None:
            return None
        total = total + c * piece
    facts = _facts(assumptions)
    if not _checks(total, f_, x, facts):
        return None
    return as_expr(total)


# ---------------------------------------------------------------------------
# x**m exp(a x**2 + b x + c)

def _quadratic_term(term: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(c * x**m * exp(a*x**2 + b*x + d), x)`` for one term, a
    nonnegative integer ``m``, by completing the square and shifting."""
    found = None
    coefficient: Expr = S.One
    m: Expr = S.Zero
    for factor in Mul.make_args(term):
        g = as_expr(factor)
        if not g.has(x):
            coefficient = coefficient * g
        elif g == x:
            m = m + 1
        elif isinstance(g, Pow) and g.base == x and isinstance(g.exp, Integer) and g.exp > 0:
            m = m + as_expr(g.exp)
        elif isinstance(g, exp) and found is None:
            found = as_expr(g.args[0])
        else:
            return None
    if found is None or not found.is_polynomial(x):
        return None
    try:
        poly = Poly(found, x)
    except PolynomialError:
        return None
    if poly.degree() != 2:
        return None
    a, b, c = (as_expr(poly.coeff_monomial(x**2)), as_expr(poly.coeff_monomial(x)), as_expr(poly.coeff_monomial(1)))
    if b == 0:
        return None                                         # the first form
    h = as_expr(b / (2 * a))
    t = Dummy('t')
    shifted = as_expr(expand((t - h)**m * exp(a * t**2) * exp(c - b**2 / (4 * a))))
    inner = power_exponential(shifted, t, assumptions)
    if inner is None:
        return None
    return as_expr(coefficient * inner.subs(t, x + h))


def _quadratic_exponential(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(p(x) * exp(a*x**2 + b*x + c), x)`` for a polynomial ``p``,
    term by term after expansion, checked by differentiation."""
    combined = as_expr(powsimp(expand(f), combine='exp'))
    total: Expr = S.Zero
    for term in Add.make_args(combined):
        piece = _quadratic_term(as_expr(term), x, assumptions)
        if piece is None:
            return None
        total = total + piece
    if not _checks(total, f, x, _facts(assumptions)):
        return None
    return as_expr(total)


# ---------------------------------------------------------------------------
# rational functions of exp(c x)

def _exponential_generator(f: Expr, x: Symbol) -> Optional[Expr]:
    """``c`` such that every ``exp(k x)`` of ``f`` has ``k`` an integer multiple
    of ``c``; ``None`` without exponentials of linear arguments."""
    rates: list[Expr] = []
    for node in f.atoms(exp):
        argument = as_expr(node.args[0])
        if not argument.has(x):
            continue
        k, rest = argument.as_independent(x, as_Add=False)
        if as_expr(rest) != x:
            return None
        rates.append(as_expr(k))
    if not rates:
        return None
    ratios = [as_expr(cancel(k / rates[0])) for k in rates]
    if not all(isinstance(r, Rational) for r in ratios):
        return None
    c = as_expr(rates[0] * gcd([Rational(r) for r in ratios]))
    # a canonical sign: the atoms come from a set, and u = exp(-x) turns
    # (exp(x) + 1)**(1/3) into (1/u + 1)**(1/3), which nothing takes
    if c.could_extract_minus_sign():
        c = as_expr(-c)
    return c


def exponential_rational(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for a rational function of ``exp(c*x)`` by
    ``u = exp(c*x)``, and for ``(a + b*exp(c*x))**(m/q) * R(exp(c*x))`` by
    ``t**q = a + b*exp(c*x)``, through the rational integrator; ``None``
    otherwise.

    >>> from sympy import symbols, exp, sqrt
    >>> from sympy_extras.integrals.exponential import exponential_rational
    >>> x = symbols('x')
    >>> exponential_rational(1/(exp(x) + 1), x)
    x - log(exp(x) + 1)
    >>> exponential_rational(exp(x)/sqrt(exp(x) + 1), x)
    2*sqrt(exp(x) + 1)
    """
    from .risch.rationaltools import ratint
    f_ = as_expr(powsimp(as_expr(f), combine='exp'))
    c = _exponential_generator(f_, x)
    if c is None:
        return None
    u = Dummy('u', positive=True)
    replacement = {node: u**as_expr(cancel(as_expr(node.args[0]).as_independent(x, as_Add=False)[0] / c))
                   for node in f_.atoms(exp) if as_expr(node.args[0]).has(x)}
    g = as_expr(f_.xreplace(replacement))
    if g.has(x):
        return None
    integrand = as_expr(cancel(g / (c * u)))
    F: Optional[Expr] = None
    if integrand.is_rational_function(u):
        F = attempt(lambda: as_expr(ratint(integrand, u)), _budget())
    else:
        F = _binomial_in(integrand, u)
    if F is None:
        return None
    result = as_expr(F.subs(u, exp(c * x)))
    # log(exp(c x)) is c x for a real x
    result = as_expr(result.replace(lambda e: isinstance(e, log) and isinstance(e.args[0], exp),
                                    lambda e: as_expr(e.args[0].args[0])))
    if not _checks(result, as_expr(f), x, _facts(assumptions)):
        return None
    return result


def _binomial_in(integrand: Expr, u: Symbol) -> Optional[Expr]:
    """``Integral(R(u) * (a + b u)**(m/q), u)`` by ``t**q = a + b u``."""
    from .risch.rationaltools import ratint
    radical: Optional[Pow] = None
    for node in integrand.atoms(Pow):
        if isinstance(node.exp, Rational) and node.exp.q > 1 and node.has(u):
            if radical is not None and as_expr(node.base) != as_expr(radical.base):
                return None
            radical = node
    if radical is None:
        return None
    base = as_expr(radical.base)
    try:
        linear = Poly(base, u)
    except PolynomialError:
        return None
    if linear.degree() != 1:
        return None
    b, a = as_expr(linear.coeff_monomial(u)), as_expr(linear.coeff_monomial(1))
    exponents: dict[Pow, Rational] = {}
    for node in integrand.atoms(Pow):
        exponent = node.exp
        if isinstance(exponent, Rational) and node.has(u) and as_expr(node.base) == base:
            exponents[node] = exponent
    q = max(exponent.q for exponent in exponents.values())
    t = Dummy('t', positive=True)
    replacement = {node: t**(exponent.p * (q // exponent.q)) for node, exponent in exponents.items()}
    in_t = as_expr(integrand.xreplace(replacement).subs(u, (t**q - a) / b) * q * t**(q - 1) / b)
    in_t = as_expr(cancel(in_t))
    if not in_t.is_rational_function(t):
        return None
    G = attempt(lambda: as_expr(ratint(in_t, t)), _budget())
    if G is None:
        return None
    return as_expr(G.subs(t, base**Rational(1, q)))


def _linear_bases(f: Expr, x: Symbol) -> list[Expr]:
    """The bases ``a*x + b`` other than ``x`` of the powers of ``f`` whose
    exponent is free of ``x``, without repetition."""
    found: list[Expr] = []
    for node in f.atoms(Pow):
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if exponent.has(x) or not base.has(x) or base == x:
            continue
        poly = base.as_poly(x)
        if poly is not None and poly.degree() == 1 and base not in found:
            found.append(base)
    # a plain linear factor too: (x + 1)*exp(-x**3 - 3*x**2 - 3*x) is t*exp(-t**3)
    for factor in Mul.make_args(f):
        base = as_expr(factor)
        if base.has(x) and base != x and base.is_polynomial(x) and Poly(base, x).degree() == 1 and base not in found:
            found.append(base)
    return found


def _shifted_exponential(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(f, x)`` through ``t = a*x + b`` for a power of a linear
    ``a*x + b`` in ``f``: ``(a*x + b)**w * exp(c*x + d)`` is
    ``t**w * exp(c*t/a + d - b*c/a) / a``, of the first form
    (``exp(c*x)/(a*x + b)**k`` is the exponential integral ``E_k``, DLMF
    8.19.3), and a polynomial in ``t`` times an exponential of a quadratic
    is of the second."""
    for base in _linear_bases(f, x):
        poly = base.as_poly(x)
        if poly is None:
            continue
        a, b = as_expr(poly.coeff_monomial(x)), as_expr(poly.coeff_monomial(1))
        t = Dummy('t')
        shifted = as_expr(f.subs(x, (t - b) / a) / a)
        for route in (power_exponential, _quadratic_exponential):
            found = attempt(lambda: route(shifted, t, assumptions), _budget())
            if found is None:
                continue
            candidate = as_expr(found.subs(t, base))
            if _checks(candidate, f, x, _facts(assumptions)):
                return candidate
    return None


def exponential_antiderivative(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """An antiderivative of ``f`` by the four routes of the module
    documentation, in that order, each result checked by differentiation;
    ``None`` when none applies.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential import exponential_antiderivative
    >>> x = symbols('x')
    >>> exponential_antiderivative(x*exp(x**2 + 2*x), x)
    exp(-1)*exp((x + 1)**2)/2 - sqrt(pi)*exp(-1)*erfi(x + 1)/2
    """
    f_ = as_expr(f)
    if not f_.has(exp):
        return None
    for route in (power_exponential, _quadratic_exponential, _shifted_exponential, exponential_rational,
                  nested_power_exponential, rational_exponential, reciprocal_square_exponential):
        found = attempt(lambda: route(f_, x, assumptions), _budget())
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# x**m exp(a W) for a nested power W = (x**r)**p

def _nested_power(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(W, d)`` for a chain of powers ``W = ((x**r_1)**r_2)...**r_k`` of
    depth at least two with exponents free of ``x`` and formal degree
    ``d = r_1*...*r_k``; ``None`` otherwise."""
    degree: Expr = S.One
    depth = 0
    node = e
    while isinstance(node, Pow) and not as_expr(node.exp).has(x):
        degree = degree * as_expr(node.exp)
        depth += 1
        node = as_expr(node.base)
    if node != x or depth < 2:
        return None
    return e, degree


def _nested_term(term: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr, Expr, Expr]]:
    """``(c, V, q, a, W, d)`` with ``term == c * V * exp(a*W + b)`` (the
    constant ``exp(b)`` inside ``c``), ``V`` a product of powers of ``x``
    and of nested powers of formal degree ``q`` in all, ``W`` a power or a
    nested power of formal degree ``d``, at least one of them nested;
    ``None`` otherwise."""
    coefficient: Expr = S.One
    prefactor: Expr = S.One
    degree: Expr = S.Zero
    nested = False
    exponent: Optional[Expr] = None
    for factor in Mul.make_args(term):
        f = as_expr(factor)
        if not f.has(x):
            coefficient = coefficient * f
        elif f == x:
            prefactor, degree = prefactor * f, degree + 1
        elif isinstance(f, Pow) and f.base == x and not as_expr(f.exp).has(x):
            prefactor, degree = prefactor * f, degree + as_expr(f.exp)
        elif isinstance(f, Pow) and _nested_power(f, x) is not None:
            _, q = _nested_power(f, x) or (f, S.Zero)
            prefactor, degree, nested = prefactor * f, degree + q, True
        elif isinstance(f, exp) and exponent is None:
            exponent = as_expr(f.args[0])
        else:
            return None
    if exponent is None:
        return None
    constant, rest = exponent.as_independent(x, as_Add=True)
    a, monomial = as_expr(factor_terms(rest)).as_independent(x, as_Add=False)
    monomial_ = as_expr(monomial)
    found = _nested_power(monomial_, x)
    if found is not None:
        W, d = found
        nested = True
    elif monomial_ == x:
        W, d = x, S.One
    elif isinstance(monomial_, Pow) and monomial_.base == x and not as_expr(monomial_.exp).has(x):
        W, d = monomial_, as_expr(monomial_.exp)
    else:
        return None
    if not nested or as_expr(a) == 0 or d == 0:
        return None
    return as_expr(coefficient * exp(constant)), prefactor, degree, as_expr(a), W, d


def nested_power_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for a sum of terms ``c * V * exp(a*W + b)`` with a
    nested power ``W = (x**r)**p`` (Maxima's test suite writes them so, for
    symbolic ``r`` and ``p``), which is not ``x**(r*p)`` off the positive
    axis, or a nested power in the prefactor ``V`` (``exp(a*x)/sqrt(x**3)``):
    with ``u = -a*W``, ``d`` the formal degree of ``W`` and ``q`` that of
    ``V``, ``s = (q + 1)/d``,

        Integral(V*exp(-u), x) = -x*V*u**(-s)*uppergamma(s, u)/d,

    an identity for every ``x`` since it uses only ``(x**r)**(p - 1) =
    (x**r)**p/x**r`` and ``x**(r - 1) = x**r/x`` (the derivatives of ``W``
    and ``V`` are ``d*W/x`` and ``q*V/x``); checked by differentiation.
    ``None`` when ``f`` is not of that form.

    >>> from sympy import symbols, exp, sqrt
    >>> from sympy_extras.integrals.exponential import nested_power_exponential
    >>> x, a, r, p = symbols('x a r p')
    >>> nested_power_exponential(exp(a*(x**r)**p), x)
    -x*uppergamma(1/(p*r), -a*(x**r)**p)/(p*r*(-a*(x**r)**p)**(1/(p*r)))
    >>> nested_power_exponential(exp(a*sqrt(x**2)), x)
    x*exp(a*sqrt(x**2))/(a*sqrt(x**2))
    """
    f_ = as_expr(f)
    combined = as_expr(powsimp(expand(f_), combine='exp'))
    total: Expr = S.Zero
    for term in Add.make_args(combined):
        found = _nested_term(as_expr(term), x)
        if found is None:
            return None
        c, V, q, a, W, d = found
        # (x*V*u**(-s))' vanishes for s = (q + 1)/d, as V' = q*V/x and u' = d*u/x
        u = as_expr(-a * W)
        s_ = as_expr((q + 1) / d)
        total = total + c * (-x * V * u**(-s_) * uppergamma(s_, u) / d)
    facts = _facts(assumptions)
    if not _checks(total, f_, x, facts):
        return None
    return as_expr(total)


# ---------------------------------------------------------------------------
# R(x) exp(c x): the exponential integral

def rational_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for ``R(x)*exp(c*x + d)`` with a rational ``R``
    whose denominator has simple roots ``r_i`` (found by ``roots``, so
    symbolic ones too): the polynomial part by the reductions of the
    module, and each partial fraction ``A_i/(x - r_i)`` by

        Integral(exp(c*x)/(x - r), x) = exp(c*r)*Ei(c*(x - r)),

    checked by differentiation. ``None`` for another shape or a
    repeated root.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential import rational_exponential
    >>> x, a, b, c = symbols('x a b c')
    >>> rational_exponential(exp(c*x)/(x**2 - 1), x)
    exp(c)*Ei(c*(x - 1))/2 - exp(-c)*Ei(c*(x + 1))/2
    """
    f_ = as_expr(f)
    exponentials = [as_expr(node) for node in f_.atoms(exp) if node.has(x)]
    if len(exponentials) != 1:
        return None
    argument = as_expr(exponentials[0].args[0])
    d, rest = argument.as_independent(x, as_Add=True)
    c, variable = as_expr(rest).as_independent(x, as_Add=False)
    if as_expr(variable) != x or as_expr(c).has(x):
        return None
    R = as_expr(cancel(f_ / exponentials[0]))
    if R.has(exp) or not R.is_rational_function(x):
        return None
    numerator, denominator = R.as_numer_denom()
    try:
        Q = Poly(denominator, x)
        P = Poly(numerator, x)
    except PolynomialError:
        return None
    if Q.degree() < 1:
        return None
    found = roots(Q)
    if sum(found.values()) != Q.degree() or any(multiplicity != 1 for multiplicity in found.values()):
        return None
    quotient, remainder = P.div(Q)
    total: Expr = S.Zero
    if not quotient.is_zero:
        polynomial = power_exponential(as_expr(quotient.as_expr() * exponentials[0]), x, assumptions)
        if polynomial is None:
            return None
        total = total + polynomial
    derivative = Q.diff()
    c_ = as_expr(c)
    for root in found:
        residue = as_expr(remainder.as_expr().subs(x, root) / derivative.as_expr().subs(x, root))
        total = total + residue * exp(d) * exp(c_ * root) * Ei(c_ * (x - root))
    facts = _facts(assumptions)
    if not _checks(total, f_, x, facts):
        return None
    return as_expr(total)


# ---------------------------------------------------------------------------
# exp(alpha x**2 + beta/x**2)

def reciprocal_square_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for ``c*exp(alpha*x**2 + beta/x**2 + gamma)``: with
    ``A = sqrt(-alpha)``, ``B = sqrt(-beta)``,

        sqrt(pi)/(4*A)*(exp(2*A*B)*erf(A*x + B/x) + exp(-2*A*B)*erf(A*x - B/x)),

    checked by differentiation (an identity of analytic functions of the
    parameters). ``None`` for another shape.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.exponential import reciprocal_square_exponential
    >>> x = symbols('x')
    >>> reciprocal_square_exponential(exp(-x**2 - 1/x**2), x)
    sqrt(pi)*(exp(-2)*erf(x - 1/x) + exp(2)*erf(x + 1/x))/4
    """
    f_ = as_expr(f)
    c, rest = f_.as_independent(x, as_Add=False)
    rest_ = as_expr(rest)
    if not isinstance(rest_, exp):
        return None
    argument = as_expr(expand(as_expr(rest_.args[0])))
    gamma_, dependent = argument.as_independent(x, as_Add=True)
    alpha = as_expr(dependent).coeff(x, 2)
    beta = as_expr(dependent).coeff(x, -2)
    if alpha == 0 or beta == 0 or as_expr(expand(as_expr(dependent) - alpha * x**2 - beta / x**2)) != 0 \
            or alpha.has(x) or beta.has(x):
        return None
    A, B = sqrt(-alpha), sqrt(-beta)
    total = as_expr(c * exp(gamma_) * sqrt(pi) / (4 * A)
                    * (exp(2 * A * B) * erf(A * x + B / x) + exp(-2 * A * B) * erf(A * x - B / x)))
    facts = _facts(assumptions)
    if not _checks(total, f_, x, facts):
        return None
    return total
