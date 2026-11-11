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

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`) with the native polynomials,
``simplify``, ``ask`` and :mod:`.conditions`, and imports no SymPy. One
step stays on SymPy, reached only by the fourth route: the rational
integrator ``ratint`` (:func:`~sympy_extras.integrals.exponential_sympy.ratint_of_core`).
:mod:`sympy_extras.integrals.exponential_sympy` is the face of the module
on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import exp
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

from typing import Callable, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import expand
from sympy_extras.core.logic import Boolean, to_boolean
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, Rational, pi
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.functions.special.error_functions import Ei, erf, erfi
from sympy_extras.functions.special.gamma_functions import uppergamma
from sympy_extras.functions.special.hyper import hyper
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel, gcd
from sympy_extras.polys.polytools import factor as factored_form
from sympy_extras.settings import settings
from sympy_extras.simplify.powsimp import powsimp
from sympy_extras.simplify.simplify import simplify
from .conditions import numerically_equal

__all__ = ['exponential_antiderivative', 'power_exponential', 'exponential_rational', 'nested_power_exponential',
           'rational_exponential', 'composite_gaussian', 'reciprocal_square_exponential']

#: a route of :func:`exponential_antiderivative`
Route = Callable[[Expr, Symbol, Assumptions], Optional[Expr]]


def _budget() -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / 4


def _facts(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(a) for a in assumptions]


def _argument(node: Basic) -> Expr:
    """The first argument of a function node, an expression."""
    return as_expr(node.args[0])


def _checks(F: Expr, f: Expr, x: Symbol, facts: list[Boolean]) -> bool:
    """Whether ``F`` is an antiderivative of ``f``: ``F' - f`` cancels or
    simplifies to zero, or vanishes numerically under the facts."""
    difference = F.diff(x) - f
    reduced = attempt(lambda: cancel(difference), _budget(), native=True)
    if reduced is not None and reduced == 0:
        return True
    simpler = attempt(lambda: simplify(reduced if reduced is not None else difference), _budget(), native=True)
    if simpler is not None and simpler == 0:
        return True
    return numerically_equal(F.diff(x), f, facts)


def _is_power_of(e: Expr, x: Symbol) -> bool:
    """Whether ``e`` is ``x**k`` with ``k`` free of ``x``."""
    return isinstance(e, Pow) and e.base == x and not e.exp.has(x)


def _integer(e: Expr) -> Optional[int]:
    return int(e) if isinstance(e, Integer) else None


def _coefficients(e: Expr, x: Symbol) -> Optional[dict[int, Expr]]:
    """The coefficients of ``e`` as a polynomial in ``x`` (``{degree:
    coefficient}``, the coefficients expanded as SymPy's ``Poly`` gives
    them, of any kind: ``pi``, ``exp(c)``, ``sqrt(2)``); ``None`` when
    ``e`` is not a polynomial in ``x``."""
    coefficients: dict[int, Expr] = {}
    for term in Add.make_args(expand(e)):
        c, monomial = term.as_independent(x, as_Add=False)
        if monomial == 1:
            k = 0
        elif monomial == x:
            k = 1
        elif isinstance(monomial, Pow) and monomial.base == x and _integer(monomial.exp) is not None \
                and int(monomial.exp) > 0:
            k = int(monomial.exp)
        else:
            return None
        coefficients[k] = coefficients.get(k, S.Zero) + c
    return {k: c for k, c in coefficients.items() if c != 0}


def _degree(coefficients: dict[int, Expr]) -> int:
    return max(coefficients) if coefficients else -1


def _from_coefficients(coefficients: dict[int, Expr], x: Expr) -> Expr:
    total: Expr = S.Zero
    for k, c in coefficients.items():
        total = total + c * x**k
    return total


def _divmod(P: dict[int, Expr], Q: dict[int, Expr]) -> tuple[dict[int, Expr], dict[int, Expr]]:
    """The quotient and the remainder of the polynomials ``P`` and ``Q``
    (``Q`` not zero) over the field of their coefficients."""
    remainder = dict(P)
    quotient: dict[int, Expr] = {}
    n = _degree(Q)
    leading = Q[n]
    while remainder and _degree(remainder) >= n:
        m = _degree(remainder)
        c = cancel(remainder[m] / leading)
        quotient[m - n] = c
        for k, q in Q.items():
            value = cancel(remainder.get(k + m - n, S.Zero) - c * q)
            if value == 0:
                remainder.pop(k + m - n, None)
            else:
                remainder[k + m - n] = value
        remainder.pop(m, None)
    return quotient, remainder


class _Generalizer:
    """Every part of a coefficient which is not a polynomial in symbols
    (``pi``, ``exp(c)``, ``sqrt(2)``, ``I``) replaced by a dummy, so that
    the polynomial layer takes it; the substitution back is a ring
    homomorphism, so roots and quotients computed formally stay so."""

    def __init__(self) -> None:
        self.dummies: dict[Expr, Expr] = {}

    def __call__(self, e: Expr) -> Expr:
        if isinstance(e, (Add, Mul)):
            if isinstance(e, Add):
                total: Expr = S.Zero
                for a in e.args:
                    total = total + self(as_expr(a))
                return total
            product: Expr = S.One
            for a in e.args:
                product = product * self(as_expr(a))
            return product
        if isinstance(e, Pow) and isinstance(e.exp, Integer):
            return self(e.base)**e.exp
        if isinstance(e, (Symbol, Rational)):
            return e
        if e not in self.dummies:
            self.dummies[e] = Dummy('g')
        return self.dummies[e]

    def back(self, e: Expr) -> Expr:
        back: dict[Basic, Basic] = {d: original for original, d in self.dummies.items()}
        return as_expr(e._xreplace_evaluated(back))


def _roots(coefficients: dict[int, Expr], x: Symbol) -> Optional[dict[Expr, int]]:
    """The roots of a polynomial in ``x`` with their multiplicities
    (``roots`` of the native polynomials, symbolic ones too); ``None``
    when they are not all found."""
    general = _Generalizer()
    polynomial = _from_coefficients({k: general(c) for k, c in coefficients.items()}, x)
    try:
        found = roots(Poly(polynomial, x))
    except BasePolynomialError:
        return None
    if not isinstance(found, dict):
        return None
    if sum(found.values()) != _degree(coefficients):
        return None
    restored: dict[Expr, int] = {}
    for root, multiplicity in found.items():
        r = general.back(root)
        restored[r] = restored.get(r, 0) + multiplicity
    return restored


# ---------------------------------------------------------------------------
# x**(v - 1) exp(a x**n)

def _power_exponential_raw(v: Expr, a: Expr, n: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(x**(v - 1)*exp(a*x**n), x)`` without the constant factors,
    in the real form the case allows; ``None`` when there is none."""
    s = cancel(v / n)
    if v == 0:
        return Ei(a * x**n) / n
    if isinstance(s, Integer) and int(s) > 0:
        # downwards to Integral(x**(n-1) exp(a x**n)) = exp(a x**n)/(a n)
        if int(s) == 1:
            return exp(a * x**n) / (a * n)
        lower = _power_exponential_raw(v - n, a, n, x, assumptions)
        if lower is None:
            return None
        return x**(v - n) * exp(a * x**n) / (a * n) - (v - n) * lower / (a * n)
    if isinstance(s, Integer) and int(s) < 0:
        # upwards to s = 0, the exponential integral
        higher = _power_exponential_raw(v + n, a, n, x, assumptions)
        if higher is None:
            return None
        return x**v * exp(a * x**n) / v - a * n * higher / v
    if n == 2 and isinstance(v, Integer) and int(v) % 2 == 1 and int(v) != 1:
        # an odd power with the Gaussian: the recurrences reach the error function
        if int(v) > 1:
            lower = _power_exponential_raw(v - 2, a, n, x, assumptions)
            if lower is None:
                return None
            return x**(v - 2) * exp(a * x**2) / (2 * a) - (v - 2) * lower / (2 * a)
        higher = _power_exponential_raw(v + 2, a, n, x, assumptions)
        if higher is None:
            return None
        return x**v * exp(a * x**2) / v - 2 * a * higher / v
    negative = ask(a < 0, assumptions)
    positive = ask(a > 0, assumptions)
    # the incomplete gamma form is real where -a*x**n > 0: for every x when n is
    # even, for x > 0 otherwise (x*exp(-x**3) at x < -1 needs the confluent form)
    real_everywhere = (isinstance(n, Integer) and int(n) % 2 == 0) or ask(x > 0, assumptions) is True
    if n == 2 and v == 1:
        below = sqrt(pi) * erf(sqrt(-a) * x) / (2 * sqrt(-a))
        above = sqrt(pi) * erfi(sqrt(a) * x) / (2 * sqrt(a))
        if negative is True:
            return below
        if positive is True:
            return above
        return piecewise((below, a < 0), (above, True))
    if negative is True and real_everywhere:
        return -(-a)**(-s) * uppergamma(s, -a * x**n) / n
    # the confluent form, an identity of power series for every a
    return x**v * hyper((s,), (s + 1,), a * x**n) / v


def _monomial_exponential(term: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr]]:
    """``(c, v, a, n)`` with ``term == c * x**(v - 1) * exp(a * x**n)``, the
    constant ``exp(b)`` of the exponent inside ``c``; ``None`` otherwise."""
    coefficient: Expr = S.One
    power: Expr = S.Zero
    exponent: Optional[Expr] = None
    for f in Mul.make_args(term):
        if not f.has(x):
            coefficient = coefficient * f
        elif f == x:
            power = power + 1
        elif isinstance(f, Pow) and _is_power_of(f, x):
            power = power + f.exp
        elif isinstance(f, Exp) and exponent is None:
            exponent = f.exp
        else:
            return None
    if exponent is None:
        return None
    constant, rest = exponent.as_independent(x, as_Add=True)
    # b*x**2*log(a) + c*x**2*log(h), the combined exponent of a**(b*x**2)*h**(c*x**2), is one monomial
    a, monomial = factor_terms(rest).as_independent(x, as_Add=False)
    if monomial == x:
        n: Expr = S.One
    elif isinstance(monomial, Pow) and _is_power_of(monomial, x):
        n = monomial.exp                                    # a number, or a symbolic exponent r
    else:
        return None
    if a == 0 or n == 0:
        return None
    return (coefficient * exp(constant), power + 1, a, n)


def power_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for a sum of terms ``c * x**(v - 1) * exp(a*x**n + b)``,
    in the real forms of the module documentation, checked by
    differentiation; ``None`` when ``f`` is not of that form or the check
    fails.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.exponential import power_exponential
    >>> x = symbols('x')
    >>> power_exponential(exp(-x**2), x)
    sqrt(pi)*erf(x)/2
    >>> power_exponential(x**3*exp(x**2), x)
    x**2*exp(x**2)/2 - exp(x**2)/2
    >>> power_exponential(exp(2*x)/x, x)
    Ei(2*x)
    """
    f_ = to_expr(f)
    combined = powsimp(expand(f_), combine='exp')
    total: Expr = S.Zero
    for term in Add.make_args(combined):
        if not term.has(Exp):
            # a term without an exponential: a monomial ((1 - exp(-x**2))**2/x**3 expanded)
            coefficient, monomial = term.as_independent(x, as_Add=False)
            if monomial == 1:
                total = total + coefficient * x
            elif monomial == x:
                total = total + coefficient * x**2 / 2
            elif isinstance(monomial, Pow) and _is_power_of(monomial, x):
                e = monomial.exp
                total = total + (coefficient * log(x) if e == -1 else coefficient * x**(e + 1) / (e + 1))
            else:
                return None
            continue
        found = _monomial_exponential(term, x)
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
    return total


# ---------------------------------------------------------------------------
# x**m exp(a x**2 + b x + c)

def _quadratic_term(term: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(c * x**m * exp(a*x**2 + b*x + d), x)`` for one term, a
    nonnegative integer ``m``, by completing the square and shifting."""
    found: Optional[Expr] = None
    coefficient: Expr = S.One
    m: Expr = S.Zero
    for g in Mul.make_args(term):
        if not g.has(x):
            coefficient = coefficient * g
        elif g == x:
            m = m + 1
        elif isinstance(g, Pow) and g.base == x and isinstance(g.exp, Integer) and int(g.exp) > 0:
            m = m + g.exp
        elif isinstance(g, Exp) and found is None:
            found = g.exp
        else:
            return None
    if found is None or not found.is_polynomial(x):
        return None
    poly = _coefficients(found, x)
    if poly is None or _degree(poly) != 2:
        return None
    a, b, c = poly[2], poly.get(1, S.Zero), poly.get(0, S.Zero)
    if b == 0:
        return None                                         # the first form
    h = b / (2 * a)
    t = Dummy('t')
    shifted = expand((t - h)**m * exp(a * t**2) * exp(c - b**2 / (4 * a)))
    inner = power_exponential(shifted, t, assumptions)
    if inner is None:
        return None
    return coefficient * inner.subs(t, x + h)


def _quadratic_exponential(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(p(x) * exp(a*x**2 + b*x + c), x)`` for a polynomial ``p``,
    term by term after expansion, checked by differentiation."""
    combined = powsimp(expand(f), combine='exp')
    total: Expr = S.Zero
    for term in Add.make_args(combined):
        piece = _quadratic_term(term, x, assumptions)
        if piece is None:
            return None
        total = total + piece
    if not _checks(total, f, x, _facts(assumptions)):
        return None
    return total


# ---------------------------------------------------------------------------
# rational functions of exp(c x)

def _exponential_generator(f: Expr, x: Symbol) -> Optional[Expr]:
    """``c`` such that every ``exp(k x)`` of ``f`` has ``k`` an integer multiple
    of ``c``; ``None`` without exponentials of linear arguments."""
    rates: list[Expr] = []
    for node in f.atoms(Exp):
        argument = _argument(node)
        if not argument.has(x):
            continue
        k, rest = argument.as_independent(x, as_Add=False)
        if rest != x:
            return None
        rates.append(k)
    if not rates:
        return None
    ratios: list[Rational] = []
    for k in rates:
        r = cancel(k / rates[0])
        if not isinstance(r, Rational):
            return None
        ratios.append(r)
    common: Expr = ratios[0]
    for r in ratios[1:]:
        common = gcd(common, r)
    c = rates[0] * common
    # a canonical sign: the atoms come from a set, and u = exp(-x) turns
    # (exp(x) + 1)**(1/3) into (1/u + 1)**(1/3), which nothing takes
    if c.could_extract_minus_sign():
        c = -c
    return c


def _ratint(f: Expr, x: Symbol) -> Optional[Expr]:
    """SymPy's rational integrator, through the face (the step of the
    module which stays on SymPy)."""
    from .exponential_sympy import ratint_of_core
    return ratint_of_core(f, x, _budget())


def _is_log_of_exp(e: Basic) -> bool:
    return isinstance(e, Log) and isinstance(e.args[0], Exp)


def _log_of_exp(e: Basic) -> Basic:
    return _argument(e.args[0])


def exponential_rational(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for a rational function of ``exp(c*x)`` by
    ``u = exp(c*x)``, and for ``(a + b*exp(c*x))**(m/q) * R(exp(c*x))`` by
    ``t**q = a + b*exp(c*x)``, through the rational integrator; ``None``
    otherwise.

    The rational integrator is SymPy's (see the module documentation).

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.exponential import exponential_rational
    >>> x = symbols('x')
    >>> exponential_rational(1/(exp(x) + 1), x)
    x - log(exp(x) + 1)
    >>> exponential_rational(exp(x)/sqrt(exp(x) + 1), x)
    2*sqrt(exp(x) + 1)
    """
    f0 = to_expr(f)
    f_ = powsimp(f0, combine='exp')
    c = _exponential_generator(f_, x)
    if c is None:
        return None
    u = Dummy('u', positive=True)
    replacement: dict[Basic, Basic] = {}
    for node in f_.atoms(Exp):
        argument = _argument(node)
        if argument.has(x):
            replacement[node] = u**cancel(argument.as_independent(x, as_Add=False)[0] / c)
    g = as_expr(f_._xreplace_evaluated(replacement))
    if g.has(x):
        return None
    integrand = cancel(g / (c * u))
    F: Optional[Expr] = None
    if integrand.is_rational_function(u):
        F = _ratint(integrand, u)
    else:
        F = _binomial_in(integrand, u)
    if F is None:
        return None
    result = F.subs(u, exp(c * x))
    # log(exp(c x)) is c x for a real x
    result = as_expr(result.replace(_is_log_of_exp, _log_of_exp))
    if not _checks(result, f0, x, _facts(assumptions)):
        return None
    return result


def _binomial_in(integrand: Expr, u: Symbol) -> Optional[Expr]:
    """``Integral(R(u) * (a + b u)**(m/q), u)`` by ``t**q = a + b u``."""
    radical: Optional[Pow] = None
    for node in integrand.atoms(Pow):
        if isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q > 1 and node.has(u):
            if radical is not None and node.base != radical.base:
                return None
            radical = node
    if radical is None:
        return None
    base = radical.base
    linear = _coefficients(base, u)
    if linear is None or _degree(linear) != 1:
        return None
    b, a = linear[1], linear.get(0, S.Zero)
    exponents: dict[Pow, Rational] = {}
    for node in integrand.atoms(Pow):
        if isinstance(node, Pow) and isinstance(node.exp, Rational) and node.has(u) and node.base == base:
            exponents[node] = node.exp
    q = max(exponent.q for exponent in exponents.values())
    t = Dummy('t', positive=True)
    replacement: dict[Basic, Basic] = {node: t**(exponent.p * (q // exponent.q))
                                       for node, exponent in exponents.items()}
    in_t = as_expr(integrand._xreplace_evaluated(replacement)).subs(u, (t**q - a) / b) * q * t**(q - 1) / b
    in_t = cancel(in_t)
    if not in_t.is_rational_function(t):
        return None
    G = _ratint(in_t, t)
    if G is None:
        return None
    return G.subs(t, base**Rational(1, q))


def _linear_bases(f: Expr, x: Symbol) -> list[Expr]:
    """The bases ``a*x + b`` other than ``x`` of the powers of ``f`` whose
    exponent is free of ``x``, without repetition."""
    found: list[Expr] = []
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if exponent.has(x) or not base.has(x) or base == x:
            continue
        poly = _coefficients(base, x)
        if poly is not None and _degree(poly) == 1 and base not in found:
            found.append(base)
    # a plain linear factor too: (x + 1)*exp(-x**3 - 3*x**2 - 3*x) is t*exp(-t**3)
    for base in Mul.make_args(f):
        if base.has(x) and base != x and base.is_polynomial(x) and base not in found:
            poly = _coefficients(base, x)
            if poly is not None and _degree(poly) == 1:
                found.append(base)
    return found


def _shifted_exponential(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """``Integral(f, x)`` through ``t = a*x + b`` for a power of a linear
    ``a*x + b`` in ``f``: ``(a*x + b)**w * exp(c*x + d)`` is
    ``t**w * exp(c*t/a + d - b*c/a) / a``, of the first form
    (``exp(c*x)/(a*x + b)**k`` is the exponential integral ``E_k``, DLMF
    8.19.3), and a polynomial in ``t`` times an exponential of a quadratic
    is of the second."""
    routes: tuple[Route, ...] = (power_exponential, _quadratic_exponential)
    for base in _linear_bases(f, x):
        poly = _coefficients(base, x)
        if poly is None:
            continue
        a, b = poly[1], poly.get(0, S.Zero)
        t = Dummy('t')
        shifted = f.subs(x, (t - b) / a) / a
        for route in routes:
            found = attempt(lambda: route(shifted, t, assumptions), _budget(), native=True)
            if found is None:
                continue
            candidate = found.subs(t, base)
            if _checks(candidate, f, x, _facts(assumptions)):
                return candidate
    return None


def exponential_antiderivative(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """An antiderivative of ``f`` by the four routes of the module
    documentation, in that order, each result checked by differentiation;
    ``None`` when none applies.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.exponential import exponential_antiderivative
    >>> x = symbols('x')
    >>> exponential_antiderivative(x*exp(x**2 + 2*x), x)
    exp(-1)*exp((x + 1)**2)/2 - sqrt(pi)*exp(-1)*erfi(x + 1)/2
    """
    f_ = to_expr(f)
    if not f_.has(Exp):
        return None
    routes: tuple[Route, ...] = (power_exponential, _quadratic_exponential, _shifted_exponential, exponential_rational,
                                 nested_power_exponential, rational_exponential, reciprocal_square_exponential,
                                 composite_gaussian)
    for route in routes:
        found = attempt(lambda: route(f_, x, assumptions), _budget(), native=True)
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
    while isinstance(node, Pow) and not node.exp.has(x):
        degree = degree * node.exp
        depth += 1
        node = node.base
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
    for f in Mul.make_args(term):
        inner = _nested_power(f, x)
        if not f.has(x):
            coefficient = coefficient * f
        elif f == x:
            prefactor, degree = prefactor * f, degree + 1
        elif isinstance(f, Pow) and _is_power_of(f, x):
            prefactor, degree = prefactor * f, degree + f.exp
        elif isinstance(f, Pow) and inner is not None:
            prefactor, degree, nested = prefactor * f, degree + inner[1], True
        elif isinstance(f, Exp) and exponent is None:
            exponent = f.exp
        else:
            return None
    if exponent is None:
        return None
    constant, rest = exponent.as_independent(x, as_Add=True)
    a, monomial = factor_terms(rest).as_independent(x, as_Add=False)
    found = _nested_power(monomial, x)
    W: Expr
    d: Expr
    if found is not None:
        W, d = found
        nested = True
    elif monomial == x:
        W, d = x, S.One
    elif isinstance(monomial, Pow) and _is_power_of(monomial, x):
        W, d = monomial, monomial.exp
    else:
        return None
    if not nested or a == 0 or d == 0:
        return None
    return coefficient * exp(constant), prefactor, degree, a, W, d


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

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.exponential import nested_power_exponential
    >>> x, a, r, p = symbols('x a r p')
    >>> nested_power_exponential(exp(a*(x**r)**p), x)
    -x*uppergamma(1/(p*r), -a*(x**r)**p)/(p*r*(-a*(x**r)**p)**(1/(p*r)))
    >>> nested_power_exponential(exp(a*sqrt(x**2)), x)
    x*exp(a*sqrt(x**2))/(a*sqrt(x**2))
    """
    f_ = to_expr(f)
    combined = powsimp(expand(f_), combine='exp')
    total: Expr = S.Zero
    for term in Add.make_args(combined):
        found = _nested_term(term, x)
        if found is None:
            return None
        c, V, q, a, W, d = found
        # (x*V*u**(-s))' vanishes for s = (q + 1)/d, as V' = q*V/x and u' = d*u/x
        u = -a * W
        s_ = (q + 1) / d
        total = total + c * (-x * V * u**(-s_) * uppergamma(s_, u) / d)
    facts = _facts(assumptions)
    if not _checks(total, f_, x, facts):
        return None
    return total


# ---------------------------------------------------------------------------
# R(x) exp(c x): the exponential integral

def rational_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for ``R(x)*exp(c*x + d)`` with a rational ``R``
    whose denominator has roots ``r_i`` found by ``roots`` (symbolic ones
    too): the polynomial part by the reductions of the module, and each
    partial fraction ``A_i/(x - r_i)`` by

        Integral(exp(c*x)/(x - r), x) = exp(c*r)*Ei(c*(x - r)),

    the higher powers by parts, checked by differentiation. ``None`` for
    another shape.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.exponential import rational_exponential
    >>> x, c = symbols('x c')
    >>> rational_exponential(exp(c*x)/(x**2 - 1), x)
    exp(c)*Ei(c*(x - 1))/2 - exp(-c)*Ei(c*(x + 1))/2
    """
    f_ = to_expr(f)
    exponentials = [as_expr(node) for node in f_.atoms(Exp) if node.has(x)]
    if len(exponentials) != 1:
        return None
    argument = _argument(exponentials[0])
    d, rest = argument.as_independent(x, as_Add=True)
    c, variable = rest.as_independent(x, as_Add=False)
    if variable != x or c.has(x):
        return None
    R = cancel(f_ / exponentials[0])
    if R.has(Exp) or not R.is_rational_function(x):
        return None
    numerator, denominator = R.as_numer_denom()
    Q = _coefficients(denominator, x)
    P = _coefficients(numerator, x)
    if Q is None or P is None or _degree(Q) < 1:
        return None
    found = _roots(Q, x)
    if found is None:
        return None
    quotient, remainder = _divmod(P, Q)
    total: Expr = S.Zero
    if quotient:
        polynomial = power_exponential(_from_coefficients(quotient, x) * exponentials[0], x, assumptions)
        if polynomial is None:
            return None
        total = total + polynomial
    leading = Q[_degree(Q)]
    rest_ = _from_coefficients(remainder, x)
    for root, multiplicity in found.items():
        # the Laurent coefficients at the root, A_j = ((x - r)**m R)^(m - j)(r)/(m - j)!,
        # with (x - r)**m R written through the other roots (no cancellation
        # of a symbolic radical is needed then)
        others: Expr = leading
        for other, other_multiplicity in found.items():
            if other != root:
                others = others * (x - other)**other_multiplicity
        stripped = rest_ / others
        for j in range(multiplicity, 0, -1):
            order = multiplicity - j
            derivative = stripped
            for _ in range(order):
                derivative = derivative.diff(x)
            coefficient = derivative.subs(x, root) / factorial(order) if order else stripped.subs(x, root)
            coefficient = cancel(coefficient)
            if coefficient == 0:
                continue
            total = total + coefficient * exp(d) * _exponential_over_power(c, root, j, x)
    facts = _facts(assumptions)
    if not _checks(total, f_, x, facts):
        return None
    return total


def _exponential_over_power(c: Expr, r: Expr, k: int, x: Symbol) -> Expr:
    """``Integral(exp(c*x)/(x - r)**k, x)``: ``exp(c*r)*Ei(c*(x - r))`` for
    ``k = 1``, and by parts, ``-exp(c*x)/((k - 1)*(x - r)**(k - 1)) +
    c/(k - 1)*Integral(exp(c*x)/(x - r)**(k - 1))``, above."""
    if k == 1:
        return exp(c * r) * Ei(c * (x - r))
    return -exp(c * x) / ((k - 1) * (x - r)**(k - 1)) + c / (k - 1) * _exponential_over_power(c, r, k - 1, x)


# ---------------------------------------------------------------------------
# k g'(x) exp(a g(x)**2 + c): the error function of a composite argument

def _linear_pair(equations: list[Expr], p: Symbol, q: Symbol) -> Optional[tuple[Expr, Expr]]:
    """The unique solution ``(p, q)`` of the equations, linear in ``p`` and
    ``q`` with coefficients of any kind; ``None`` when there is none or it
    is not unique."""
    rows: list[tuple[Expr, Expr, Expr]] = []
    for e in equations:
        e_ = expand(e)
        a, b = e_.coeff(p), e_.coeff(q)
        rest = cancel(e_ - a * p - b * q)
        if a.has(p, q) or b.has(p, q) or rest.has(p, q):
            return None
        rows.append((a, b, rest))
    for i, (a1, b1, c1) in enumerate(rows):
        for a2, b2, c2 in rows[i + 1:]:
            determinant = cancel(a1 * b2 - a2 * b1)
            if determinant == 0:
                continue
            pv = cancel((b1 * c2 - b2 * c1) / determinant)
            qv = cancel((a2 * c1 - a1 * c2) / determinant)
            if all(cancel(a * pv + b * qv + c) == 0 for a, b, c in rows):
                return pv, qv
            return None
    return None


def composite_gaussian(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for ``f == k*g'(x)*exp(a*g(x)**2 + c)`` with ``g``
    a polynomial in ``x``, ``1/x`` and ``log(x)`` read off the exponent by
    factoring (``exp((x + 1/x)**2)*(1 - 1/x**2)`` is ``exp((x**4 + 2*x**2 +
    1)/x**2)*(x**2 - 1)/x**2`` in FriCAS's suite): ``k*sqrt(pi)*exp(c)*
    erf(sqrt(-a)*g)/(2*sqrt(-a))``, checked by differentiation. ``None``
    for another shape.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.integrals.exponential import composite_gaussian
    >>> x = symbols('x')
    >>> composite_gaussian((1 - 1/x**2)*exp(-(x + 1/x)**2), x)
    sqrt(pi)*erf((x**2 + 1)/x)/2
    >>> composite_gaussian((x + 2*log(x))*exp(x**2 + 2*x + (2*x + 2)*log(x)**2 + log(x)**4 + 1)/x, x)
    sqrt(pi)*erfi(x + log(x)**2 + 1)/2
    >>> composite_gaussian(exp((x + 1/x)**2)/x**2, x)
    -sqrt(pi)*erfi((x**2 + 1)/x)/4 + sqrt(pi)*exp(4)*erfi(x - 1/x)/4
    """
    f_ = to_expr(f)
    exponentials = [as_expr(node) for node in f_.atoms(Exp) if node.has(x)]
    if len(exponentials) != 1:
        return None
    L = Dummy('L')
    argument = expand(_argument(exponentials[0]).subs(log(x), L))
    numerator, denominator = cancel(argument).as_numer_denom()
    if not denominator.is_polynomial(x) or not numerator.is_polynomial(x, L):
        return None
    constant, dependent = argument.as_independent(x, L, as_Add=True)
    rest = cancel(f_ / exponentials[0])
    facts = _facts(assumptions)
    # the whole exponent a square first ((x + 1/x)**2 has the constant 2
    # in it), then the part depending on x
    candidates: tuple[tuple[Expr, Expr], ...] = ((S.Zero, argument), (constant, dependent))
    for c, body in candidates:
        factored = attempt(lambda: factored_form(body), _budget(), native=True)
        if factored is None:
            continue
        a, square = factored.as_independent(x, L, as_Add=False)
        if a == 0 or a.has(x):
            continue
        g: Expr = S.One
        for piece in Mul.make_args(square):
            if not (isinstance(piece, Pow) and isinstance(piece.exp, Integer) and int(piece.exp) % 2 == 0):
                g = S.Zero
                break
            g = g * piece.base**(piece.exp / 2)
        if g == 0:
            continue
        g = g.subs(L, log(x))
        k = cancel(rest / g.diff(x))
        if not k.has(x):
            total = k * _gaussian_primitive(a, g, c)
            if _checks(total, f_, x, facts):
                return total
        # g = u + v with u*v constant: h = u - v has h**2 = g**2 - 4*u*v, so
        # a rest which is p*u' + q*v' is (p + q)/2*g' + (p - q)/2*h', and
        # exp(g**2)/x**2 (FriCAS's integ 122) is a pair of error functions
        terms = Add.make_args(expand(g))
        if len(terms) == 2:
            u, v = terms[0], terms[1]
            product = cancel(u * v)
            if not product.has(x) and u.has(x) and v.has(x):
                p_, q_ = Dummy('p'), Dummy('q')
                difference = cancel(rest - p_ * u.diff(x) - q_ * v.diff(x))
                numerator_ = difference.as_numer_denom()[0]
                coefficients = _coefficients(numerator_, x)
                solution = None if coefficients is None else _linear_pair(list(coefficients.values()), p_, q_)
                if solution is not None and not any(value.has(x) for value in solution):
                    pv, qv = solution
                    h = u - v
                    total = ((pv + qv) / 2 * _gaussian_primitive(a, g, c)
                             + (pv - qv) / 2 * _gaussian_primitive(a, h, c + 4 * a * product))
                    if _checks(total, f_, x, facts):
                        return total
    return None


def _gaussian_primitive(a: Expr, g: Expr, c: Expr) -> Expr:
    """``Integral(g'*exp(a*g**2 + c))``: the imaginary error function for
    a known positive ``a``, the error function of ``sqrt(-a)*g`` otherwise."""
    if a.is_positive:
        return sqrt(pi) * exp(c) * erfi(sqrt(a) * g) / (2 * sqrt(a))
    A = sqrt(-a)
    return sqrt(pi) * exp(c) * erf(A * g) / (2 * A)


# ---------------------------------------------------------------------------
# exp(alpha x**2 + beta/x**2)

def reciprocal_square_exponential(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(f, x)`` for ``c*exp(alpha*x**2 + beta/x**2 + gamma)``: with
    ``A = sqrt(-alpha)``, ``B = sqrt(-beta)``,

        sqrt(pi)/(4*A)*(exp(2*A*B)*erf(A*x + B/x) + exp(-2*A*B)*erf(A*x - B/x)),

    checked by differentiation (an identity of analytic functions of the
    parameters). ``None`` for another shape.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.exponential import reciprocal_square_exponential
    >>> x = symbols('x')
    >>> reciprocal_square_exponential(exp(-x**2 - 1/x**2), x)
    sqrt(pi)*(exp(-2)*erf(x - 1/x) + exp(2)*erf(x + 1/x))/4
    """
    f_ = to_expr(f)
    c, rest = f_.as_independent(x, as_Add=False)
    if not isinstance(rest, Exp):
        return None
    argument = expand(rest.exp)
    gamma_, dependent = argument.as_independent(x, as_Add=True)
    alpha = dependent.coeff(x, 2)
    beta = dependent.coeff(x, -2)
    if alpha == 0 or beta == 0 or expand(dependent - alpha * x**2 - beta / x**2) != 0 \
            or alpha.has(x) or beta.has(x):
        return None
    A, B = sqrt(-alpha), sqrt(-beta)
    total = (c * exp(gamma_) * sqrt(pi) / (4 * A)
             * (exp(2 * A * B) * erf(A * x + B / x) + exp(-2 * A * B) * erf(A * x - B / x)))
    facts = _facts(assumptions)
    if not _checks(total, f_, x, facts):
        return None
    return total

