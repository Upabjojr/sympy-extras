"""Definite integrals of algebraic functions of genus zero, by the
substitutions which make them rational.

An integrand `R(x, y)` with `R` rational and `y` algebraic over
`\\mathbb{Q}(x)` is elementary-integrable only through the geometry of the
curve `y = y(x)` (Trager's algorithm [Trager]_, not implemented here);
when the curve has genus zero it has a rational parametrisation
`x = X(t)`, `y = Y(t)`, and the integral becomes that of a rational
function of `t`, which the other methods of the package compute. Three
classical families are covered.

**Euler's substitutions** [Piskunov]_ for `y = \\sqrt{a x^2 + b x + c}`:

* `a > 0`: `y = \\sqrt{a}\\, x + u`, so `x = (u^2 - c)/(b - 2\\sqrt{a}\\, u)`;
* `c > 0`: `y = x u + \\sqrt{c}`, so `x = (2\\sqrt{c}\\, u - b)/(a - u^2)`;
* real roots `r_1, r_2`: `y = (x - r_1) u`, so `x = (a r_2 - r_1 u^2)/(a - u^2)`;

and for a linear radicand `y = \\sqrt{b x + c} = u`, `x = (u^2 - c)/b`. Each
map is monotone on an interval where the radicand is positive, so the
range is carried to `(u(a), u(b))` by the limits of `u(x)`.

**Chebyshev's theorem** [Chebyshev]_ for the binomial differential
`x^m (a + b x^n)^p` with rational exponents: the integral is elementary
exactly when `p`, `(m + 1)/n` or `(m + 1)/n + p` is an integer, and the
substitutions `x = t^k` (`k` the common denominator of `m` and `n`),
`t^s = a + b x^n` and `t^s = a x^{-n} + b` (`s` the denominator of `p`)
make the integrand rational in the three cases.

The criterion depends only on the logarithmic derivative
`m/x + p n b x^{n-1}/(a + b x^n)`, which an integrand such as
`\\sqrt{x^7/(1 - 5 x^2)}` shares with `x^{7/2} (1 - 5 x^2)^{-1/2}` although the
two differ by a factor `-1` for `x < -1/\\sqrt 5` (:func:`binomial_exponents`).
In the non-elementary case, `H = x (1 - z)^{-p}\\, {}_2F_1(-p, q; q + 1;
z)/(m + 1)` with `q = (m + 1)/n` and `z = -b x^n/a` solves `H' + (f'/f) H = 1`:
with `(1 - z)^{-p} {}_2F_1(-p, q; q + 1; z) = (1 - z)\\, {}_2F_1(1, q + p + 1;
q + 1; z)` (Euler's transformation [DLMF]_ 15.8.1) the equation becomes,
coefficient by coefficient of the series in `z`, the contiguous relation
`(k + q + 1)\\, c_{k+1} = (k + q + p + 1)\\, c_k` of the terms of the second
series, so that `f H` is an antiderivative of `f` for every branch of
`f` (:func:`binomial_hypergeometric_antiderivative`). On `0 < x` small
with `a > 0` it is `a^p x^{m+1}\\, {}_2F_1(-p, q; q + 1; z)/(m + 1)`, the
binomial series of `(1 - z)^p` integrated term by term (the derivation
here, not a table). The same construction on `x^{m + n p} (b + a
x^{-n})^p`, in `1/z`, is real where `z > 1`; with parameters in `a` and
`b` the two are cases of the condition `z > 1` itself, whatever the
signs of the parameters.

**Roots of a Möbius function**: `\\bigl((a x + b)/(c x + d)\\bigr)^{r/n}` (and
of a linear function, `c = 0`) with `t^n = (a x + b)/(c x + d)`, so that
`x = (d t^n - b)/(a - c t^n)`.

The value of the rational integral is computed by
:func:`~sympy_extras.integrals.definite.conditional_integral` and confirmed
numerically by :func:`~sympy_extras.integrals.definite.verify_numerically`
when ``settings.numerical_checks`` is on.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native ask, limits under assumptions, ``cancel``, roots of
quadratics, ``tidy`` and conditional values. The rational integral and
its numerical check are computed by the native :mod:`.definite`. The
functions on SymPy objects are in :mod:`.algebraic_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, sqrt, Rational
>>> from sympy_extras.integrals.algebraic import binomial_exponents, chebyshev_elementary, moebius_root
>>> x = Symbol('x')
>>> binomial_exponents(sqrt(x**7/(1 - 5*x**2)), x)
BinomialDifferential(x**(7/2)*(1 + -5*x**(2))**(-1/2))
>>> chebyshev_elementary(binomial_exponents(x**3*sqrt(x**2 + 1), x))
True
>>> moebius_root((x + 1)**Rational(1, 3)/x, x).t_of_x
(x + 1)**(1/3)

References
==========

.. [Piskunov] N. Piskunov, *Differential and Integral Calculus*, Mir,
   1969, chapter X, section 11 (Euler's substitutions); also T. M.
   Apostol, *Calculus I*, 2nd ed., Wiley, 1967, section 6.22, and
   Gradshteyn–Ryzhik 2.25.
.. [Chebyshev] P. L. Chebyshev, *Sur l'intégration des différentielles
   irrationnelles*, Journal de mathématiques pures et appliquées 18 (1853)
   87–111.
.. [DLMF] NIST Digital Library of Mathematical Functions, chapter 15
   (hypergeometric function), https://dlmf.nist.gov/15.8, Euler's
   transformation 15.8.1.
.. [Trager] B. M. Trager, *Integration of algebraic functions*, PhD
   thesis, MIT, 1984; M. Bronstein, *Symbolic Integration I*, 2nd ed.,
   Springer, 2005, chapter 2 (the general algebraic case, not implemented).
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from math import lcm
from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.limits import limit
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import Boolean, and_, false, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, Rational, nan, number_from_fraction, zoo
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.elementary.integers import ceiling
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.functions.special.gamma_functions import gamma
from sympy_extras.functions.special.hyper import hyper
from sympy_extras.polys.polyroots import roots_quadratic
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .marichev import tidy

__all__ = ['algebraic_integral', 'euler_substitution', 'euler_substitutions', 'binomial_differential', 'moebius_root',
           'rationalised_integral', 'Rationalisation', 'BinomialDifferential', 'binomial_exponents',
           'chebyshev_elementary', 'binomial_hypergeometric_antiderivative']

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


def _rational(e: Expr) -> Rational:
    """``e`` as a rational number (``TypeError`` when it is none)."""
    if not isinstance(e, Rational):
        raise TypeError("expected a rational number, got %s" % (e,))
    return e


def _fraction(r: Rational) -> Fraction:
    return Fraction(r.p, r.q)


def _number(f: Fraction) -> Rational:
    return _rational(number_from_fraction(f))


def _operations(e: Expr) -> int:
    count = e.count_ops()
    return int(count) if isinstance(count, Integer) else 0


class Rationalisation:
    """A rationalising substitution ``x = X(t)``: the new variable, the
    map, the inverse ``t(x)`` (to carry the bounds) and the replacement
    of every radical atom of the integrand by a rational function of
    ``t``."""

    def __init__(self, t: Symbol, x_of_t: Expr, t_of_x: Expr, radicals: dict[Expr, Expr]) -> None:
        self.t: Final[Symbol] = t
        self.x_of_t: Final[Expr] = x_of_t
        self.t_of_x: Final[Expr] = t_of_x
        self.radicals: Final[dict[Expr, Expr]] = dict(radicals)

    def __repr__(self) -> str:
        return "Rationalisation(x = %s, t = %s)" % (self.x_of_t, self.t_of_x)


def _is_rational_in(g: Expr, t: Symbol) -> bool:
    for node in g.atoms(Pow):
        if isinstance(node, Pow) and node.has(t) and not isinstance(node.exp, Integer):
            return False
    return g.is_rational_function(t) is True


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


def _polynomial(e: Expr, x: Symbol) -> Optional[dict[int, Expr]]:
    """The coefficients of a nonzero polynomial ``e`` in ``x``, ``None``
    otherwise."""
    found = _coefficients(e, x)
    if not found:
        return None
    return found


def _degree(coefficients: dict[int, Expr]) -> int:
    return max(coefficients) if coefficients else -1


def _coefficient(coefficients: dict[int, Expr], n: int) -> Expr:
    return coefficients.get(n, S.Zero)


def _radical_atoms(f: Expr, x: Symbol) -> list[Pow]:
    """The powers of ``f`` with a rational non-integer exponent whose base
    depends on ``x``."""
    found: list[Pow] = []
    for node in f.atoms(Pow):
        if isinstance(node, Pow) and node.has(x) and isinstance(node.exp, Rational) \
                and not isinstance(node.exp, Integer):
            found.append(node)
    return found


def _exponent(atom: Pow) -> Rational:
    return _rational(atom.exp)


# ---------------------------------------------------------------------------
# Euler's substitutions

def _real_roots(a2: Expr, b2: Expr, c2: Expr) -> Optional[tuple[Expr, Expr]]:
    """The two distinct real roots of ``a2*x**2 + b2*x + c2`` with rational
    coefficients, the smaller first (SymPy's ``all_roots``); ``None``
    when they are not real and distinct."""
    A, B, C = _rational(a2), _rational(b2), _rational(c2)
    discriminant = _fraction(B)**2 - 4 * _fraction(A) * _fraction(C)
    if discriminant <= 0:
        return None
    y = _fresh('y')
    found = roots_quadratic(Poly(a2 * y**2 + b2 * y + c2, y))
    if len(found) != 2:
        return None
    r1, r2 = found
    if (r1 - r2).is_extended_positive is True:
        r1, r2 = r2, r1
    return (r1, r2)


def euler_substitutions(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                        assumptions: Assumptions = None) -> list[Rationalisation]:
    """Euler's substitutions applicable to an integrand whose radicals are
    half-integer powers of one polynomial ``Q`` of degree one or two
    (bases proportional to ``Q`` with a positive ratio are allowed), in
    the order ``a > 0``, ``c > 0``, real roots; empty when the shape does
    not allow any.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.algebraic import euler_substitutions
    >>> x = Symbol('x')
    >>> [s.t_of_x for s in euler_substitutions(1/sqrt(x**2 + 1), x, 0, 1)]
    [-x + sqrt(x**2 + 1), (sqrt(x**2 + 1) - 1)/x]
    """
    atoms = _radical_atoms(f, x)
    if not atoms or any(_exponent(atom).q != 2 for atom in atoms):
        return []
    bases: list[dict[int, Expr]] = []
    for atom in atoms:
        p = _polynomial(atom.base, x)
        if p is None or _degree(p) > 2:
            return []
        bases.append(p)
    Q = atoms[0].base
    ratios: list[Expr] = []
    for atom in atoms:
        ratio = cancel(atom.base / Q)
        if ratio.has(x) or ask(ratio > 0, assumptions) is not True:
            return []
        ratios.append(ratio)
    Q = expand(Q)
    u = _fresh('u')
    coefficients = bases[0]
    # (x(u), u(x), y(u)) for each applicable substitution, y = sqrt(Q)
    maps: list[tuple[Expr, Expr, Expr]] = []
    if _degree(coefficients) == 1:
        b1, c1 = _coefficient(coefficients, 1), _coefficient(coefficients, 0)
        maps.append(((u**2 - c1) / b1, sqrt(Q), u))
    else:
        a2, b2, c2 = (_coefficient(coefficients, k) for k in (2, 1, 0))
        if ask(a2 > 0, assumptions) is True:
            x_of_u = (u**2 - c2) / (b2 - 2 * sqrt(a2) * u)
            maps.append((x_of_u, sqrt(Q) - sqrt(a2) * x, sqrt(a2) * x_of_u + u))
        if ask(c2 > 0, assumptions) is True:
            x_of_u = (2 * sqrt(c2) * u - b2) / (a2 - u**2)
            maps.append((x_of_u, (sqrt(Q) - sqrt(c2)) / x, x_of_u * u + sqrt(c2)))
        if all(isinstance(c, Rational) for c in (a2, b2, c2)):
            found = _real_roots(a2, b2, c2)
            if found is not None:
                r1, r2 = found
                x_of_u = (a2 * r2 - r1 * u**2) / (a2 - u**2)
                maps.append((x_of_u, sqrt(Q) / (x - r1), (x_of_u - r1) * u))
    substitutions: list[Rationalisation] = []
    for x_of_u, u_of_x, y in maps:
        y_ = cancel(y)
        radicals: dict[Expr, Expr] = {}
        for atom, ratio in zip(atoms, ratios):
            # (ratio Q)^(k/2) = ratio^(k/2) y^k
            exponent = _exponent(atom)
            radicals[atom] = ratio**exponent * y_**exponent.p
        substitutions.append(Rationalisation(u, cancel(x_of_u), u_of_x, radicals))
    return substitutions


def euler_substitution(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None) -> Optional[Rationalisation]:
    """The first of :func:`euler_substitutions`, or ``None``.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.algebraic import euler_substitution
    >>> x = Symbol('x')
    >>> euler_substitution(1/sqrt(x**2 + 1), x, 0, 1).t_of_x
    -x + sqrt(x**2 + 1)
    """
    found = euler_substitutions(f, x, a, b, assumptions)
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Roots of a Möbius function

def moebius_root(f: Expr, x: Symbol) -> Optional[Rationalisation]:
    """The substitution ``t**n = M(x)`` for an integrand whose radicals
    are rational powers of one Möbius (or linear) function ``M`` of ``x``
    (or of its reciprocal); ``None`` otherwise.

    >>> from sympy_extras.core import Symbol, Rational
    >>> from sympy_extras.integrals.algebraic import moebius_root
    >>> x = Symbol('x')
    >>> s = moebius_root((x + 1)**Rational(1, 3)/x, x)
    >>> s.x_of_t.subs(s.t, Symbol('t')), s.t_of_x
    (t**3 - 1, (x + 1)**(1/3))
    """
    atoms = _radical_atoms(f, x)
    if not atoms:
        return None
    base0 = atoms[0].base
    numerator, denominator = cancel(base0).as_numer_denom()
    pn, pd = _polynomial(numerator, x), _polynomial(denominator, x)
    if pn is None or pd is None or _degree(pn) > 1 or _degree(pd) > 1:
        return None
    n = 1
    powers: list[tuple[Pow, Fraction]] = []
    for atom in atoms:
        base = atom.base
        sign: int
        if cancel(base / base0) == 1:
            sign = 1
        elif cancel(base * base0) == 1:
            sign = -1
        else:
            return None
        exponent = _exponent(atom)
        powers.append((atom, sign * _fraction(exponent)))
        n = lcm(n, exponent.q)
    t = _fresh('t', positive=True)
    a1, b1 = _coefficient(pn, 1), _coefficient(pn, 0)
    c1, d1 = _coefficient(pd, 1), _coefficient(pd, 0)
    if a1 - c1 * t**n == 0:
        return None
    x_of_t = cancel((d1 * t**n - b1) / (a1 - c1 * t**n))
    if not x_of_t.has(t):
        return None
    radicals: dict[Expr, Expr] = {}
    for atom, k in powers:
        radicals[atom] = t**int(k * n)
    return Rationalisation(t, x_of_t, base0**Rational(1, n), radicals)


# ---------------------------------------------------------------------------
# Binomial differentials

def _monomial_exponent(monomial: Expr, x: Symbol) -> Optional[Rational]:
    """``k`` for ``monomial == x**k`` with a rational ``k`` (``x`` itself
    is ``k = 1``), ``None`` otherwise."""
    if monomial == x:
        return Integer(1)
    if isinstance(monomial, Pow) and monomial.base == x and isinstance(monomial.exp, Rational):
        return monomial.exp
    return None


def binomial_differential(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr, Expr, Expr]]:
    """``(C, m, a, b, n, p)`` with ``f == C * x**m * (a + b*x**n)**p``,
    ``m``, ``n``, ``p`` rational and the binomial power not an integer
    power of a polynomial ``f`` could be expanded to; ``None`` otherwise.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.algebraic import binomial_differential
    >>> x = Symbol('x')
    >>> binomial_differential(3*x**3*sqrt(x**2 + 1), x)
    (3, 3, 1, 1, 2, 1/2)
    """
    C, rest = f.as_independent(x, as_Add=False)
    m: Expr = S.Zero
    binomial: Optional[tuple[Expr, Expr, Expr, Expr]] = None
    for factor_ in Mul.make_args(rest):
        if factor_ == x:
            m = m + 1
            continue
        if isinstance(factor_, Pow) and factor_.base == x and isinstance(factor_.exp, Rational):
            m = m + factor_.exp
            continue
        if isinstance(factor_, Pow) and isinstance(factor_.exp, Rational) and isinstance(factor_.base, Add) \
                and binomial is None:
            terms = [_expr(term) for term in factor_.base.args]
            if len(terms) != 2:
                return None
            constant_terms = [term for term in terms if not term.has(x)]
            if len(constant_terms) != 1:
                return None
            a_ = constant_terms[0]
            other = terms[0] if terms[1] is a_ else terms[1]
            coefficient, monomial = other.as_independent(x, as_Add=False)
            power = _monomial_exponent(monomial, x)
            if power is None:
                return None
            binomial = (a_, coefficient, power, factor_.exp)
            continue
        return None
    if binomial is None:
        return None
    a0, b0, n0, p0 = binomial
    if isinstance(p0, Integer) and isinstance(m, Integer) and isinstance(n0, Integer):
        return None
    return (C, m, a0, b0, n0, p0)


def _binomial_integral(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """Chebyshev's three cases."""
    found = binomial_differential(f, x)
    if found is None:
        return None
    C, m, a_, b_, n, p = found
    if not (isinstance(m, Rational) and isinstance(n, Rational) and isinstance(p, Rational)) or n == 0:
        return None
    m_, n_, p_ = _fraction(m), _fraction(n), _fraction(p)
    t = _fresh('t', positive=True)
    q = (m_ + 1) / n_
    g: Expr
    if p_.denominator == 1:
        # x = t**k rationalises the powers of x
        k = lcm(m_.denominator, n_.denominator)
        g = C * t**_number(m_ * k) * (a_ + b_ * t**_number(n_ * k))**p * k * t**(k - 1)
        t_of_x = x**Rational(1, k)
    elif q.denominator == 1:
        # t**s = a + b x**n
        s = p_.denominator
        g = C * Integer(s) / (n * b_) * t**(s - 1) * t**_number(s * p_) * ((t**s - a_) / b_)**int(q - 1)
        t_of_x = (a_ + b_ * x**n)**Rational(1, s)
    elif (q + p_).denominator == 1:
        # t**s = a x**(-n) + b
        s = p_.denominator
        exponent = -int(q + p_) - 1
        g = -C * Integer(s) / (n * a_) * t**_number(s - 1 + s * p_) * ((t**s - b_) / a_)**exponent
        t_of_x = (a_ * x**(-n) + b_)**Rational(1, s)
    else:
        # Chebyshev: not elementary
        return None
    return _transformed(f, x, a, b, Rationalisation(t, S.Zero, t_of_x, {}), assumptions, g)


class BinomialDifferential:
    """The exponents and coefficients of ``x**m*(a + b*x**n)**p``, ``n > 0``,
    of which an integrand is a locally constant multiple (see
    :func:`binomial_exponents`); ``a`` and ``b`` as the integrand writes
    them (``5*x**2 - 1`` gives ``a = -1``, ``b = 5``)."""

    def __init__(self, m: Rational, a: Expr, b: Expr, n: Rational, p: Rational) -> None:
        self.m: Final[Rational] = m
        self.a: Final[Expr] = a
        self.b: Final[Expr] = b
        self.n: Final[Rational] = n
        self.p: Final[Rational] = p

    def __repr__(self) -> str:
        return "BinomialDifferential(x**(%s)*(%s + %s*x**(%s))**(%s))" % (self.m, self.a, self.b, self.n, self.p)


#: a binomial ``a + b*x**n`` raised to ``e``, as ``(a, b, n, e)``
_BinomialPower = tuple[Expr, Expr, Fraction, Fraction]


def _binomial_powers(e: Expr, x: Symbol) -> Optional[tuple[Fraction, list[_BinomialPower]]]:
    """``(m, [(a, b, n, e), ...])`` with the logarithmic derivative of ``e``
    that of ``x**m*prod((a + b*x**n)**e)``, ``n > 0``; ``None`` when ``e``
    is not a product of rational powers of ``x`` and of binomials."""
    if not e.has(x):
        return (Fraction(0), [])
    if e == x:
        return (Fraction(1), [])
    if isinstance(e, Pow):
        if not isinstance(e.exp, Rational):
            return None
        exponent = _fraction(e.exp)
        inner = _binomial_powers(e.base, x)
        if inner is None:
            return None
        # (u**r)'/u**r = r*u'/u for the principal power, wherever u is off its cut
        m, powers = inner
        return (m * exponent, [(a, b, n, k * exponent) for a, b, n, k in powers])
    if isinstance(e, Mul):
        total = Fraction(0)
        collected: list[_BinomialPower] = []
        for factor_ in e.args:
            found = _binomial_powers(_expr(factor_), x)
            if found is None:
                return None
            total = total + found[0]
            collected.extend(found[1])
        return (total, collected)
    if isinstance(e, Add):
        # the terms c*x**k, grouped by k
        terms: dict[Fraction, Expr] = {}
        for term in e.args:
            c, monomial = _expr(term).as_independent(x, as_Add=False)
            if monomial == 1:
                k = Fraction(0)
            else:
                power = _monomial_exponent(monomial, x)
                if power is None:
                    return None
                k = _fraction(power)
            terms[k] = terms[k] + c if k in terms else c
        present = sorted(k for k, c in terms.items() if c != 0)
        if len(present) == 1:
            return (present[0], [])
        if len(present) != 2:
            return None
        # c1*x**k1 + c2*x**k2 = x**k1*(c1 + c2*x**(k2 - k1)), k1 < k2
        k1, k2 = present
        return (k1, [(terms[k1], terms[k2], k2 - k1, Fraction(1))])
    return None


def binomial_exponents(f: ExprLike, x: Symbol) -> Optional[BinomialDifferential]:
    """The binomial differential ``x**m*(a + b*x**n)**p`` (``m``, ``n``, ``p``
    rational, ``a`` and ``b`` nonzero numbers or expressions in parameters
    which are not identically zero) of which ``f`` is a locally constant
    multiple, ``None`` when ``f`` is none. The exponents must be explicit
    rational numbers; the coefficients may vanish at isolated values of
    the parameters, where the integrand is a monomial (the callers'
    cases).

    ``f`` is a product of rational powers of ``x``, of the binomial and of
    products and quotients of them, nested in any way:
    ``sqrt(x**7/(1 - 5*x**2))`` is ``x**(7/2)*(1 - 5*x**2)**(-1/2)`` for
    ``0 < x < 1/sqrt(5)`` and ``-x**(7/2)*(1 - 5*x**2)**(-1/2)`` (the two
    principal powers imaginary) for ``x < -1/sqrt(5)``: the split is not
    an identity, but ``f`` and the split form have the same logarithmic
    derivative ``m/x + p*n*b*x**(n - 1)/(a + b*x**n)`` wherever both are
    analytic, and their quotient is constant on every interval where
    neither crosses a branch cut. Chebyshev's criterion
    (:func:`chebyshev_elementary`) and an antiderivative ``f*H`` with ``H``
    solving ``H' + (f'/f)*H = 1`` depend on nothing else.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, sqrt, cbrt
    >>> from sympy_extras.integrals.algebraic import binomial_exponents
    >>> x = Symbol('x')
    >>> binomial_exponents(sqrt(x**7/(1 - 5*x**2)), x)
    BinomialDifferential(x**(7/2)*(1 + -5*x**(2))**(-1/2))
    >>> binomial_exponents(cbrt(x**2*(x**3 + 1))/x, x)
    BinomialDifferential(x**(-1/3)*(1 + 1*x**(3))**(1/3))
    >>> binomial_exponents(sqrt(1 + 1/x**2), x)
    BinomialDifferential(x**(-1)*(1 + 1*x**(2))**(1/2))
    >>> binomial_exponents(sqrt(1 + x**2)*sqrt(1 - x**2), x) is None
    True
    >>> a, b = Symbol('a'), Symbol('b')
    >>> binomial_exponents(sqrt(x**7/(a - b*x**2)), x)
    BinomialDifferential(x**(7/2)*(a + -b*x**(2))**(-1/2))
    """
    found = _binomial_powers(to_expr(f), x)
    if found is None:
        return None
    m, powers = found
    # the binomials proportional to each other are one: (1 - 5*x**2) and (5*x**2 - 1)
    merged: list[_BinomialPower] = []
    for a, b, n, e in powers:
        for i, (a0, b0, n0, e0) in enumerate(merged):
            if n0 == n and cancel(a * b0 - a0 * b) == 0:
                merged[i] = (a0, b0, n0, e0 + e)
                break
        else:
            merged.append((a, b, n, e))
    remaining = [power for power in merged if power[3] != 0]
    if len(remaining) != 1:
        return None
    a, b, n, p = remaining[0]
    if not all(_nonzero_coefficient(c) for c in (a, b)):
        return None
    return BinomialDifferential(_number(m), a, b, _number(n), _number(p))


def _nonzero_coefficient(c: Expr) -> bool:
    """Whether ``c`` is a coefficient of a binomial: a nonzero finite
    number, or an expression in parameters not known to vanish or to be
    infinite (zero at isolated values of the parameters only, which are
    the callers' cases)."""
    if c.is_number:
        return c.is_zero is False and c.is_finite is True
    return c.is_zero is not True and c.is_finite is not False


def chebyshev_elementary(binomial: BinomialDifferential) -> bool:
    """Chebyshev's criterion: the integral of ``x**m*(a + b*x**n)**p``
    (``m``, ``n``, ``p`` rational, ``a``, ``b`` nonzero) is elementary if
    and only if ``p``, ``(m + 1)/n`` or ``(m + 1)/n + p`` is an integer.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.algebraic import binomial_exponents, chebyshev_elementary
    >>> x = Symbol('x')
    >>> chebyshev_elementary(binomial_exponents(x**3*sqrt(x**2 + 1), x))
    True
    >>> chebyshev_elementary(binomial_exponents(sqrt(x**3 + 1), x))
    False
    """
    p = _fraction(binomial.p)
    q = (_fraction(binomial.m) + 1) / _fraction(binomial.n)
    return p.denominator == 1 or q.denominator == 1 or (q + p).denominator == 1


def _scaled_power(a: Expr, b: Expr, w: Expr, p: Rational) -> Expr:
    """``(1 + b*w/a)**(-p)``, written ``|a|**p*(±(a + b*w))**(-p)`` when the
    sign of ``a`` is known (an identity: a positive factor leaves a
    principal power), so that it cancels against a power of ``a + b*w`` in
    the integrand."""
    if a.is_extended_positive is True:
        return a**p * (a + b * w)**(-p)
    if a.is_extended_negative is True:
        return (-a)**p * (-a - b * w)**(-p)
    return (1 + b * w / a)**(-p)


def _real_at(f: Expr, x: Symbol, point: Expr) -> bool:
    """Whether ``f`` is exactly real at the rational ``point`` (``False``
    when it cannot be told)."""
    value = attempt(lambda: _expr(f._xreplace_evaluated({x: point})), settings.timeout, native=True)
    return value is not None and value.is_extended_real is True


def binomial_hypergeometric_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An antiderivative of a locally constant multiple of the binomial
    differential ``x**m*(a + b*x**n)**p`` (:func:`binomial_exponents`) whose
    integral Chebyshev's criterion proves non-elementary, by the Gauss
    hypergeometric function; ``None`` otherwise.

    With ``q = (m + 1)/n`` and ``z = -b*x**n/a``, ``H = x*(1 - z)**(-p)*
    2F1(-p, q; q + 1; z)/(m + 1)`` solves ``H' + (m/x + p*n*b*x**(n - 1)/(a +
    b*x**n))*H = 1`` (the series of ``2F1`` term by term, then analytic
    continuation), so ``f*H`` is an antiderivative of ``f`` for every
    branch of ``f`` off the cut ``z > 1`` of ``2F1``; where ``f`` is
    real on ``z > 1`` the same construction on ``x**(m + n*p)*(b +
    a*x**(-n))**p``, in ``1/z``, gives a real antiderivative, returned as
    a case of a ``Piecewise``, shifted by the constant which joins the
    two cases at ``z = 1`` when ``f`` is real and integrable on both
    sides (Gauss's sum of ``2F1`` at 1).

    With parameters in ``a`` and ``b`` (the exponents stay rational
    numbers), the case in ``1/z`` holds where ``z > 1``, a condition on
    ``x`` and the parameters which covers every sign of them (taken
    real), and the form in ``z`` elsewhere; each is ``f`` times a
    function real there. These cases are not joined at ``z = 1``. The
    isolated values ``a = 0`` and ``b = 0``, where the integrand is a
    monomial, are the callers' cases: the form in ``z`` holds at ``b =
    0`` and the case in ``1/z`` at ``a = 0``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.algebraic import binomial_hypergeometric_antiderivative
    >>> x = Symbol('x')
    >>> binomial_hypergeometric_antiderivative(sqrt(x**3 + 1), x)
    x*hyper((-1/2, 1/3), (4/3,), -x**3)
    >>> binomial_hypergeometric_antiderivative(x**3*sqrt(x**2 + 1), x) is None
    True
    >>> c = Symbol('c')
    >>> binomial_hypergeometric_antiderivative(sqrt(1 + c*x**3), x)
    Piecewise((2*x*sqrt(c*x**3 + 1)*hyper((-5/6, -1/2), (1/6,), -1/(c*x**3))/(5*sqrt(1 + 1/(c*x**3))), c*x**3 < -1), (x*hyper((-1/2, 1/3), (4/3,), -c*x**3), True))
    """
    f_ = to_expr(f)
    binomial = binomial_exponents(f_, x)
    if binomial is None or chebyshev_elementary(binomial):
        return None
    m, a, b, n, p = binomial.m, binomial.a, binomial.b, binomial.n, binomial.p
    m_, n_, p_ = _fraction(m), _fraction(n), _fraction(p)
    q = _number((m_ + 1) / n_)
    ratio = -b / a
    z = ratio * x**n
    # x**m*(a + b*x**n)**p is x**(m + n*p)*(b + a*x**(-n))**p up to a locally
    # constant factor: exponents m + n*p, -n, and q' = -(q + p), not an integer
    m2, q2 = _number(m_ + n_ * p_), _number(-((m_ + 1) / n_ + p_))
    inner_factor = f_ * _scaled_power(a, b, x**n, p)
    outer_factor = f_ * _scaled_power(b, a, x**(-n), p)
    # m + 1 and q + 1 are not zero: q is not an integer
    inside = x * inner_factor / (m + 1) * hyper([-p, q], [q + 1], z)
    outside = x * outer_factor / (m2 + 1) * hyper([-p, q2], [q2 + 1], 1 / z)
    if not ratio.is_number:
        return _parametric_cases(inside, outside, ratio, x, n)
    if ratio.is_extended_real is not True:
        return inside
    pieces: list[tuple[Expr, Boolean]] = []
    for side in (1, -1):
        if side == -1 and n.q != 1:
            # a principal power of a negative x: z is not real
            continue
        if (ratio * Integer(side)**n).is_extended_positive is not True:
            continue
        boundary = abs_(ratio)**(-1 / n)
        beyond = side * ceiling(2 * boundary)
        within = side * Rational(1, int(ceiling(2 / boundary)))
        if not _real_at(f_, x, beyond):
            # the first form holds there too, on the cut of 2F1 (with a
            # constant imaginary part where f is not real)
            continue
        piece = outside
        if _real_at(f_, x, within) and p_ > -1:
            # f real and integrable on both sides of z = 1: the case in 1/z
            # shifted to join the first form there
            joined = _joined(x, side * boundary, side, inner_factor, outer_factor, m, q, m2, q2, p)
            if joined is None:
                continue
            piece = outside + joined
        pieces.append((piece, x > boundary if side == 1 else x < -boundary))
    if not pieces:
        return inside
    return piecewise(*pieces, (inside, true))


def _parametric_cases(inside: Expr, outside: Expr, ratio: Expr, x: Symbol, n: Rational) -> Expr:
    """The antiderivative when ``z = ratio*x**n`` has parameters: the case in
    ``1/z`` where ``z > 1`` (for real values of the parameters, whatever
    their signs), the form in ``z`` elsewhere. Both are antiderivatives
    wherever they are analytic, which the first is for ``z > 1`` and the
    second off ``z >= 1``, and each is ``f`` times a function real there,
    so real where ``f`` is. The cases are not joined at ``z = 1`` (the
    constant of Gauss's sum would need the one-sided limits of ``f``
    there, in the parameters): an antiderivative on each interval, which
    jumps at ``z = 1``. ``ratio`` not real for real parameters: the form
    in ``z`` alone."""
    real: dict[Basic, Basic] = {s: _fresh(s.name, real=True, nonzero=True)
                                for s in sorted(ratio.free_symbols, key=lambda s: s.name)}
    if _expr(ratio._xreplace_evaluated(real)).is_extended_real is not True:
        return inside
    condition: Boolean
    if n.q == 1:
        # z real for every real x: z > 1 on one side (n odd) or on both (n even)
        if n.p % 2 == 0 and ratio.is_extended_nonpositive is True:
            return inside
        condition = ratio * x**n > 1
    else:
        # z not real for x < 0 (a principal power of a negative x); written
        # with Abs(x) so that no case compares a complex number, and false
        # at ratio = 0 (where the form in z is defined)
        if ratio.is_extended_nonpositive is True:
            return inside
        condition = and_(x > 0, ratio * abs_(x)**n > 1)
    if condition == false:
        return inside
    return piecewise((outside, condition), (inside, true))


def _joined(x: Symbol, point: Expr, side: int, inner_factor: Expr, outer_factor: Expr,
            m: Rational, q: Rational, m2: Rational, q2: Rational, p: Rational) -> Optional[Expr]:
    """The constant which makes ``x*outer_factor*2F1(-p, q2; q2 + 1;
    1/z)/(m2 + 1)`` beyond ``point`` (where ``z = 1``) continue ``x*
    inner_factor*2F1(-p, q; q + 1; z)/(m + 1)`` before it, for ``p > -1``:
    both series converge at 1 (Gauss's sum ``2F1(-p, q; q + 1; 1) =
    Gamma(q + 1)*Gamma(p + 1)/Gamma(q + p + 1)``), the factors have finite
    one-sided limits; ``None`` when a limit is not found."""
    before, after = ('-', '+') if side == 1 else ('+', '-')
    inner = attempt(lambda: limit(inner_factor, x, point, before), settings.timeout, native=True)
    outer = attempt(lambda: limit(outer_factor, x, point, after), settings.timeout, native=True)
    if inner is None or outer is None or not inner.is_number or not outer.is_number \
            or inner.has(nan, zoo, Limit) or outer.has(nan, zoo, Limit) \
            or inner.is_finite is not True or outer.is_finite is not True:
        return None
    first = point * inner * gamma(q + 1) * gamma(p + 1) / (gamma(q + p + 1) * (m + 1))
    second = point * outer * gamma(q2 + 1) * gamma(p + 1) / (gamma(q2 + p + 1) * (m2 + 1))
    return first - second


# ---------------------------------------------------------------------------
# The transformed integral

def _bound(t_of_x: Expr, x: Symbol, point: Expr, assumptions: Assumptions) -> Optional[Expr]:
    value = attempt(lambda: limit(t_of_x, x, point, assumptions=assumptions), settings.timeout, native=True)
    if value is None or value.has(nan, zoo, Limit) or value.free_symbols - t_of_x.free_symbols:
        return None
    return value


def _transformed(f: Expr, x: Symbol, a: Expr, b: Expr, substitution: Rationalisation,
                 assumptions: Assumptions, g: Optional[Expr] = None) -> Optional[ConditionalValue]:
    """The integral of the transformed integrand ``g`` (built from the
    substitution unless given) over the image of the range, by
    :func:`~sympy_extras.integrals.definite.conditional_integral`."""
    t = substitution.t
    if g is None:
        rule: dict[Basic, Basic] = {k: v for k, v in substitution.radicals.items()}
        replaced = _expr(f._xreplace_evaluated(rule))
        if _radical_atoms(replaced, x):
            return None
        g = replaced.subs(x, substitution.x_of_t) * substitution.x_of_t.diff(t)
    g = cancel(g)
    if not _is_rational_in(g, t):
        return None
    lower = _bound(substitution.t_of_x, x, a, assumptions)
    upper = _bound(substitution.t_of_x, x, b, assumptions)
    if lower is None or upper is None or lower == upper:
        return None
    from .definite import conditional_integral, verify_numerically
    found = conditional_integral(g, t, lower, upper, assumptions)
    if found is None:
        return None
    value = tidy(found.value, assumptions, found.condition)
    if settings.numerical_checks and verify_numerically(value, f, x, a, b, assumptions) is False:
        return None
    return ConditionalValue(value, found.condition)


def rationalised_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike, substitution: Rationalisation,
                          assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` through a rationalising substitution."""
    return _transformed(f, x, to_expr(a), to_expr(b), substitution, assumptions)


def algebraic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` for an algebraic integrand of genus zero:
    the roots of a Möbius function, Euler's substitutions for a square
    root of a polynomial of degree at most two, and Chebyshev's binomial
    differentials, in this order; ``None`` when none applies. The
    rational integral is computed by
    :func:`~sympy_extras.integrals.definite.conditional_integral`.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.algebraic import algebraic_integral
    >>> x = Symbol('x')
    >>> algebraic_integral(exp(x), x, 0, 1) is None
    True
    """
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    if not _radical_atoms(f_, x):
        return None
    substitution = moebius_root(f_, x)
    if substitution is not None:
        found = _transformed(f_, x, a_, b_, substitution, assumptions)
        if found is not None:
            return found
    best: Optional[ConditionalValue] = None
    for candidate in euler_substitutions(f_, x, a_, b_, assumptions):
        # every applicable Euler substitution, the shortest value kept
        found = _transformed(f_, x, a_, b_, candidate, assumptions)
        if found is not None and (best is None or _operations(found.value) < _operations(best.value)):
            best = found
    if best is not None:
        return best
    return _binomial_integral(f_, x, a_, b_, assumptions)
