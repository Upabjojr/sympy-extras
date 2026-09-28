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
:func:`sympy_extras.integrals.definite_integral` and confirmed numerically
when ``settings.numerical_checks`` is on.

Examples
========

>>> from sympy import symbols, sqrt, oo
>>> from sympy_extras.integrals.algebraic import algebraic_integral
>>> x = symbols('x')
>>> algebraic_integral(1/(x*sqrt(x**2 - 1)), x, 1, 2)
ConditionalValue(pi/3)
>>> algebraic_integral(sqrt(x)/(1 + x)**2, x, 0, oo)
ConditionalValue(pi/2)
>>> algebraic_integral(sqrt((1 - x)/(1 + x)), x, 0, 1)
ConditionalValue(-1 + pi/2)

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

from math import lcm
from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, nan, zoo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.logic.boolalg import And, Boolean, false, true
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel
from sympy.series.limits import Limit

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.limits import limit
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .marichev import tidy

__all__ = ['algebraic_integral', 'euler_substitution', 'euler_substitutions', 'binomial_differential', 'moebius_root',
           'rationalised_integral', 'Rationalisation', 'BinomialDifferential', 'binomial_exponents',
           'chebyshev_elementary', 'binomial_hypergeometric_antiderivative']


class Rationalisation:
    """A rationalising substitution ``x = X(t)``: the new variable, the
    map, the inverse ``t(x)`` (to carry the bounds) and the replacement
    of every radical atom of the integrand by a rational function of
    ``t``."""

    def __init__(self, t: Symbol, x_of_t: Expr, t_of_x: Expr, radicals: dict[Expr, Expr]) -> None:
        self.t = t
        self.x_of_t = x_of_t
        self.t_of_x = t_of_x
        self.radicals = radicals

    def __repr__(self) -> str:
        return "Rationalisation(x = %s, t = %s)" % (self.x_of_t, self.t_of_x)


def _is_rational_in(g: Expr, t: Symbol) -> bool:
    for node in g.atoms(Pow):
        if node.has(t) and not isinstance(node.exp, Integer):
            return False
    return g.is_rational_function(t)


def _polynomial(e: Expr, x: Symbol) -> Optional[Poly]:
    try:
        p = Poly(e, x)
    except PolynomialError:
        return None
    if p.degree() < 0:
        return None
    return p


def _radical_atoms(f: Expr, x: Symbol) -> list[Pow]:
    """The powers of ``f`` with a rational non-integer exponent whose base
    depends on ``x``."""
    found: list[Pow] = []
    for node in f.atoms(Pow):
        if node.has(x) and isinstance(node.exp, Rational) and not isinstance(node.exp, Integer):
            found.append(node)
    return found


# ---------------------------------------------------------------------------
# Euler's substitutions

def euler_substitutions(f: Expr, x: Symbol, a: Expr, b: Expr,
                        assumptions: Assumptions = None) -> list[Rationalisation]:
    """Euler's substitutions applicable to an integrand whose radicals are
    half-integer powers of one polynomial ``Q`` of degree one or two
    (bases proportional to ``Q`` with a positive ratio are allowed), in
    the order ``a > 0``, ``c > 0``, real roots; empty when the shape does
    not allow any.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic import euler_substitutions
    >>> x = symbols('x')
    >>> euler_substitutions(1/sqrt(x**2 + 1), x, 0, 1)
    [Rationalisation(x = (1 - _u**2)/(2*_u), t = -x + sqrt(x**2 + 1)), Rationalisation(x = -2*_u/(_u**2 - 1), t = (sqrt(x**2 + 1) - 1)/x)]
    """
    atoms = _radical_atoms(f, x)
    if not atoms or any(not isinstance(atom.exp, Rational) or atom.exp.q != 2 for atom in atoms):
        return []
    bases: list[Poly] = []
    for atom in atoms:
        p = _polynomial(as_expr(atom.base), x)
        if p is None or p.degree() > 2:
            return []
        bases.append(p)
    Q = bases[0]
    ratios: list[Expr] = []
    for p in bases:
        ratio = as_expr(cancel(p.as_expr() / Q.as_expr()))
        if ratio.has(x) or ask(as_boolean(ratio > 0), assumptions) is not True:
            return []
        ratios.append(ratio)
    u = Dummy('u')
    coefficients = [as_expr(c) for c in Q.all_coeffs()]
    # (x(u), u(x), y(u)) for each applicable substitution, y = sqrt(Q)
    maps: list[tuple[Expr, Expr, Expr]] = []
    if Q.degree() == 1:
        b1, c1 = coefficients
        maps.append((as_expr((u**2 - c1) / b1), as_expr(sqrt(Q.as_expr())), as_expr(u)))
    else:
        a2, b2, c2 = coefficients
        if ask(as_boolean(a2 > 0), assumptions) is True:
            x_of_u = as_expr((u**2 - c2) / (b2 - 2 * sqrt(a2) * u))
            maps.append((x_of_u, as_expr(sqrt(Q.as_expr()) - sqrt(a2) * x), as_expr(sqrt(a2) * x_of_u + u)))
        if ask(as_boolean(c2 > 0), assumptions) is True:
            x_of_u = as_expr((2 * sqrt(c2) * u - b2) / (a2 - u**2))
            maps.append((x_of_u, as_expr((sqrt(Q.as_expr()) - sqrt(c2)) / x), as_expr(x_of_u * u + sqrt(c2))))
        if all(isinstance(c, Rational) for c in coefficients):
            roots = [as_expr(r) for r in Q.all_roots()]
            if len(roots) == 2 and all(r.is_extended_real is True for r in roots) and roots[0] != roots[1]:
                r1, r2 = roots
                x_of_u = as_expr((a2 * r2 - r1 * u**2) / (a2 - u**2))
                maps.append((x_of_u, as_expr(sqrt(Q.as_expr()) / (x - r1)), as_expr((x_of_u - r1) * u)))
    found: list[Rationalisation] = []
    for x_of_u, u_of_x, y in maps:
        y_ = as_expr(cancel(y))
        radicals: dict[Expr, Expr] = {}
        for atom, ratio in zip(atoms, ratios):
            # (ratio Q)^(k/2) = ratio^(k/2) y^k
            k = int(as_expr(atom.exp) * 2)
            radicals[as_expr(atom)] = as_expr(ratio**as_expr(atom.exp) * y_**k)
        found.append(Rationalisation(u, as_expr(cancel(x_of_u)), u_of_x, radicals))
    return found


def euler_substitution(f: Expr, x: Symbol, a: Expr, b: Expr,
                       assumptions: Assumptions = None) -> Optional[Rationalisation]:
    """The first of :func:`euler_substitutions`, or ``None``."""
    found = euler_substitutions(f, x, a, b, assumptions)
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Roots of a Möbius function

def moebius_root(f: Expr, x: Symbol) -> Optional[Rationalisation]:
    """The substitution ``t**n = M(x)`` for an integrand whose radicals
    are rational powers of one Möbius (or linear) function ``M`` of ``x``
    (or of its reciprocal); ``None`` otherwise.

    >>> from sympy import symbols, cbrt
    >>> from sympy_extras.integrals.algebraic import moebius_root
    >>> x = symbols('x')
    >>> moebius_root(cbrt(x + 1)/x, x)
    Rationalisation(x = _t**3 - 1, t = (x + 1)**(1/3))
    """
    atoms = _radical_atoms(f, x)
    if not atoms:
        return None
    base0 = as_expr(atoms[0].base)
    numerator, denominator = [as_expr(e) for e in cancel(base0).as_numer_denom()]
    pn, pd = _polynomial(numerator, x), _polynomial(denominator, x)
    if pn is None or pd is None or pn.degree() > 1 or pd.degree() > 1:
        return None
    n = 1
    powers: list[tuple[Pow, Rational]] = []
    for atom in atoms:
        base = as_expr(atom.base)
        ratio = as_expr(cancel(base / base0))
        sign: Expr
        if ratio == 1:
            sign = S.One
        elif as_expr(cancel(base * base0)) == 1:
            sign = S.NegativeOne
        else:
            return None
        exponent = as_expr(atom.exp)
        if not isinstance(exponent, Rational):
            return None
        powers.append((atom, Rational(sign * exponent)))
        n = lcm(n, exponent.q)
    t = Dummy('t', positive=True)
    a1, b1 = (as_expr(pn.coeff_monomial(x)), as_expr(pn.coeff_monomial(1)))
    c1, d1 = (as_expr(pd.coeff_monomial(x)), as_expr(pd.coeff_monomial(1)))
    if as_expr(a1 - c1 * t**n) == 0:
        return None
    x_of_t = as_expr(cancel((d1 * t**n - b1) / (a1 - c1 * t**n)))
    if not x_of_t.has(t):
        return None
    radicals: dict[Expr, Expr] = {}
    for atom, exponent in powers:
        radicals[as_expr(atom)] = as_expr(t**int(exponent * n))
    return Rationalisation(t, x_of_t, as_expr(base0**Rational(1, n)), radicals)


# ---------------------------------------------------------------------------
# Binomial differentials

def binomial_differential(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr, Expr, Expr]]:
    """``(C, m, a, b, n, p)`` with ``f == C * x**m * (a + b*x**n)**p``,
    ``m``, ``n``, ``p`` rational and the binomial power not an integer
    power of a polynomial ``f`` could be expanded to; ``None`` otherwise.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic import binomial_differential
    >>> x = symbols('x')
    >>> binomial_differential(3*x**3*sqrt(x**2 + 1), x)
    (3, 3, 1, 1, 2, 1/2)
    """
    constant, rest = f.as_independent(x, as_Add=False)
    C = as_expr(constant)
    m: Expr = S.Zero
    binomial: Optional[tuple[Expr, Expr, Expr, Expr]] = None
    for factor in Mul.make_args(as_expr(rest)):
        factor_ = as_expr(factor)
        if factor_ == x:
            m = m + 1
            continue
        if isinstance(factor_, Pow) and factor_.base == x and isinstance(factor_.exp, Rational):
            m = m + as_expr(factor_.exp)
            continue
        if isinstance(factor_, Pow) and isinstance(factor_.exp, Rational) and isinstance(factor_.base, Add) \
                and binomial is None:
            terms = [as_expr(term) for term in factor_.base.args]
            if len(terms) != 2:
                return None
            constant_terms = [term for term in terms if not term.has(x)]
            if len(constant_terms) != 1:
                return None
            a_ = constant_terms[0]
            other = terms[0] if terms[1] is a_ else terms[1]
            coefficient, monomial = other.as_independent(x, as_Add=False)
            monomial_ = as_expr(monomial)
            if monomial_ == x:
                n_: Expr = S.One
            elif isinstance(monomial_, Pow) and monomial_.base == x and isinstance(monomial_.exp, Rational):
                n_ = as_expr(monomial_.exp)
            else:
                return None
            binomial = (a_, as_expr(coefficient), n_, as_expr(factor_.exp))
            continue
        return None
    if binomial is None:
        return None
    a_, b_, n_, p_ = binomial
    if isinstance(p_, Integer) and isinstance(m, Integer) and isinstance(n_, Integer):
        return None
    return (C, m, a_, b_, n_, p_)


def _binomial_integral(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """Chebyshev's three cases."""
    found = binomial_differential(f, x)
    if found is None:
        return None
    C, m, a_, b_, n, p = found
    if not all(isinstance(e, Rational) for e in (m, n, p)) or n == 0:
        return None
    m_, n_, p_ = Rational(m), Rational(n), Rational(p)
    t = Dummy('t', positive=True)
    q = (m_ + 1) / n_
    if isinstance(p_, Integer) or p_ == 0:
        # x = t**k rationalises the powers of x
        k = lcm(m_.q, n_.q)
        g = as_expr(C * t**(m_ * k) * (a_ + b_ * t**(n_ * k))**p_ * k * t**(k - 1))
        t_of_x = as_expr(x**Rational(1, k))
    elif q.is_integer:
        # t**s = a + b x**n
        s = p_.q
        g = as_expr(C * Rational(s, 1) / (n_ * b_) * t**(s - 1) * t**(s * p_) * ((t**s - a_) / b_)**int(q - 1))
        t_of_x = as_expr((a_ + b_ * x**n_)**Rational(1, s))
    elif (q + p_).is_integer:
        # t**s = a x**(-n) + b
        s = p_.q
        exponent = -int(q + p_) - 1
        g = as_expr(-C * Rational(s, 1) / (n_ * a_) * t**(s - 1 + s * p_) * ((t**s - b_) / a_)**exponent)
        t_of_x = as_expr((a_ * x**(-n_) + b_)**Rational(1, s))
    else:
        # Chebyshev: not elementary
        return None
    return _transformed(f, x, a, b, Rationalisation(t, S.Zero, as_expr(t_of_x), {}), assumptions, g)


class BinomialDifferential:
    """The exponents and coefficients of ``x**m*(a + b*x**n)**p``, ``n > 0``,
    of which an integrand is a locally constant multiple (see
    :func:`binomial_exponents`); ``a`` and ``b`` as the integrand writes
    them (``5*x**2 - 1`` gives ``a = -1``, ``b = 5``)."""

    def __init__(self, m: Rational, a: Expr, b: Expr, n: Rational, p: Rational) -> None:
        self.m = m
        self.a = a
        self.b = b
        self.n = n
        self.p = p

    def __repr__(self) -> str:
        return "BinomialDifferential(x**(%s)*(%s + %s*x**(%s))**(%s))" % (self.m, self.a, self.b, self.n, self.p)


#: a binomial ``a + b*x**n`` raised to ``e``, as ``(a, b, n, e)``
_BinomialPower = tuple[Expr, Expr, Rational, Rational]


def _binomial_powers(e: Expr, x: Symbol) -> Optional[tuple[Rational, list[_BinomialPower]]]:
    """``(m, [(a, b, n, e), ...])`` with the logarithmic derivative of ``e``
    that of ``x**m*prod((a + b*x**n)**e)``, ``n > 0``; ``None`` when ``e``
    is not a product of rational powers of ``x`` and of binomials."""
    if not e.has(x):
        return (Rational(0), [])
    if e == x:
        return (Rational(1), [])
    if isinstance(e, Pow):
        exponent = as_expr(e.exp)
        if not isinstance(exponent, Rational):
            return None
        inner = _binomial_powers(as_expr(e.base), x)
        if inner is None:
            return None
        # (u**r)'/u**r = r*u'/u for the principal power, wherever u is off its cut
        m, powers = inner
        return (Rational(m * exponent), [(a, b, n, Rational(k * exponent)) for a, b, n, k in powers])
    if isinstance(e, Mul):
        total = Rational(0)
        collected: list[_BinomialPower] = []
        for factor in e.args:
            found = _binomial_powers(as_expr(factor), x)
            if found is None:
                return None
            total = Rational(total + found[0])
            collected.extend(found[1])
        return (total, collected)
    if isinstance(e, Add):
        # the terms c*x**k, grouped by k
        terms: dict[Rational, Expr] = {}
        for term in e.args:
            coefficient, monomial = as_expr(term).as_independent(x, as_Add=False)
            c, monomial_ = as_expr(coefficient), as_expr(monomial)
            k: Expr
            if monomial_ == 1:
                k = S.Zero
            elif monomial_ == x:
                k = S.One
            elif isinstance(monomial_, Pow) and monomial_.base == x and isinstance(monomial_.exp, Rational):
                k = as_expr(monomial_.exp)
            else:
                return None
            if not isinstance(k, Rational):
                return None
            terms[k] = as_expr(terms.get(k, S.Zero) + c)
        present = sorted(k for k, c in terms.items() if c != 0)
        if len(present) == 1:
            return (present[0], [])
        if len(present) != 2:
            return None
        # c1*x**k1 + c2*x**k2 = x**k1*(c1 + c2*x**(k2 - k1)), k1 < k2
        k1, k2 = present
        return (k1, [(terms[k1], terms[k2], Rational(k2 - k1), Rational(1))])
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

    >>> from sympy import symbols, sqrt, cbrt
    >>> from sympy_extras.integrals.algebraic import binomial_exponents
    >>> x = symbols('x')
    >>> binomial_exponents(sqrt(x**7/(1 - 5*x**2)), x)
    BinomialDifferential(x**(7/2)*(1 + -5*x**(2))**(-1/2))
    >>> binomial_exponents(cbrt(x**2*(x**3 + 1))/x, x)
    BinomialDifferential(x**(-1/3)*(1 + 1*x**(3))**(1/3))
    >>> binomial_exponents(sqrt(1 + 1/x**2), x)
    BinomialDifferential(x**(-1)*(1 + 1*x**(2))**(1/2))
    >>> binomial_exponents(sqrt(1 + x**2)*sqrt(1 - x**2), x) is None
    True
    >>> a, b = symbols('a b')
    >>> binomial_exponents(sqrt(x**7/(a - b*x**2)), x)
    BinomialDifferential(x**(7/2)*(a + -b*x**(2))**(-1/2))
    """
    f_ = as_expr(f)
    found = _binomial_powers(f_, x)
    if found is None:
        return None
    m, powers = found
    # the binomials proportional to each other are one: (1 - 5*x**2) and (5*x**2 - 1)
    merged: list[_BinomialPower] = []
    for a, b, n, e in powers:
        for i, (a0, b0, n0, e0) in enumerate(merged):
            if n0 == n and as_expr(cancel(a * b0 - a0 * b)) == 0:
                merged[i] = (a0, b0, n0, Rational(e0 + e))
                break
        else:
            merged.append((a, b, n, e))
    remaining = [power for power in merged if power[3] != 0]
    if len(remaining) != 1:
        return None
    a, b, n, p = remaining[0]
    if not all(_nonzero_coefficient(c) for c in (a, b)):
        return None
    return BinomialDifferential(m, a, b, n, p)


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

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic import binomial_exponents, chebyshev_elementary
    >>> x = symbols('x')
    >>> chebyshev_elementary(binomial_exponents(x**3*sqrt(x**2 + 1), x))
    True
    >>> chebyshev_elementary(binomial_exponents(sqrt(x**3 + 1), x))
    False
    """
    q = Rational(binomial.m + 1) / binomial.n
    return binomial.p.q == 1 or q.q == 1 or Rational(q + binomial.p).q == 1


def _scaled_power(a: Expr, b: Expr, w: Expr, p: Rational) -> Expr:
    """``(1 + b*w/a)**(-p)``, written ``|a|**p*(±(a + b*w))**(-p)`` when the
    sign of ``a`` is known (an identity: a positive factor leaves a
    principal power), so that it cancels against a power of ``a + b*w`` in
    the integrand."""
    if a.is_extended_positive is True:
        return as_expr(a**p * (a + b * w)**(-p))
    if a.is_extended_negative is True:
        return as_expr((-a)**p * (-a - b * w)**(-p))
    return as_expr((1 + b * w / a)**(-p))


def _real_at(f: Expr, x: Symbol, point: Expr) -> bool:
    """Whether ``f`` is exactly real at the rational ``point`` (``False``
    when SymPy cannot tell)."""
    value = attempt(lambda: as_expr(f.xreplace({x: point})), settings.timeout)
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

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic import binomial_hypergeometric_antiderivative
    >>> x = symbols('x')
    >>> binomial_hypergeometric_antiderivative(sqrt(x**3 + 1), x)
    x*hyper((-1/2, 1/3), (4/3,), -x**3)
    >>> binomial_hypergeometric_antiderivative(x**3*sqrt(x**2 + 1), x) is None
    True
    >>> binomial_hypergeometric_antiderivative(sqrt(x**7/(1 - 5*x**2)), x)
    Piecewise((2*sqrt(5)*x*sqrt(x**7/(1 - 5*x**2))*sqrt(5 - 1/x**2)*hyper((-7/4, 1/2), (-3/4,), 1/(5*x**2))/35, x < -sqrt(5)/5), (2*x*sqrt(x**7/(1 - 5*x**2))*sqrt(1 - 5*x**2)*hyper((1/2, 9/4), (13/4,), 5*x**2)/9, True))
    >>> c = symbols('c')
    >>> binomial_hypergeometric_antiderivative(sqrt(1 + c*x**3), x)
    Piecewise((2*x*sqrt(c*x**3 + 1)*hyper((-5/6, -1/2), (1/6,), -1/(c*x**3))/(5*sqrt(1 + 1/(c*x**3))), c*x**3 < -1), (x*hyper((-1/2, 1/3), (4/3,), -c*x**3), True))
    """
    f_ = as_expr(f)
    binomial = binomial_exponents(f_, x)
    if binomial is None or chebyshev_elementary(binomial):
        return None
    m, a, b, n, p = binomial.m, binomial.a, binomial.b, binomial.n, binomial.p
    q = Rational(m + 1) / n
    ratio = as_expr(-b / a)
    z = as_expr(ratio * x**n)
    # x**m*(a + b*x**n)**p is x**(m + n*p)*(b + a*x**(-n))**p up to a locally
    # constant factor: exponents m + n*p, -n, and q' = -(q + p), not an integer
    m2, q2 = Rational(m + n * p), Rational(-(q + p))
    inner_factor = as_expr(f_ * _scaled_power(a, b, as_expr(x**n), p))
    outer_factor = as_expr(f_ * _scaled_power(b, a, as_expr(x**(-n)), p))
    # m + 1 and q + 1 are not zero: q is not an integer
    inside = as_expr(x * inner_factor / (m + 1) * hyper([-p, q], [q + 1], z))
    outside = as_expr(x * outer_factor / (m2 + 1) * hyper([-p, q2], [q2 + 1], as_expr(1 / z)))
    if not ratio.is_number:
        return _parametric_cases(inside, outside, ratio, x, n)
    if ratio.is_extended_real is not True:
        return inside
    pieces: list[tuple[Expr, Boolean]] = []
    for side in (1, -1):
        if side == -1 and n.q != 1:
            # a principal power of a negative x: z is not real
            continue
        if as_expr(ratio * side**n).is_extended_positive is not True:
            continue
        boundary = as_expr(Abs(ratio)**(-1 / n))
        beyond = as_expr(side * ceiling(2 * boundary))
        within = as_expr(side * Rational(1, int(ceiling(2 / boundary))))
        if not _real_at(f_, x, beyond):
            # the first form holds there too, on the cut of 2F1 (with a
            # constant imaginary part where f is not real)
            continue
        piece = outside
        if _real_at(f_, x, within) and p > -1:
            # f real and integrable on both sides of z = 1: the case in 1/z
            # shifted to join the first form there
            joined = _joined(x, as_expr(side * boundary), side, inner_factor, outer_factor, m, q, m2, q2, p)
            if joined is None:
                continue
            piece = as_expr(outside + joined)
        pieces.append((piece, as_boolean(x > boundary if side == 1 else x < -boundary)))
    if not pieces:
        return inside
    return as_expr(Piecewise(*pieces, (inside, true)))


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
    real: dict[Symbol, Expr] = {s: Dummy(s.name, real=True, nonzero=True) for s in sorted_symbols(free_symbols(ratio))}
    if as_expr(ratio.xreplace(real)).is_extended_real is not True:
        return inside
    condition: Boolean
    if n.q == 1:
        # z real for every real x: z > 1 on one side (n odd) or on both (n even)
        if n.p % 2 == 0 and ratio.is_extended_nonpositive is True:
            return inside
        condition = as_boolean(ratio * x**n > 1)
    else:
        # z not real for x < 0 (a principal power of a negative x); written
        # with Abs(x) so that no case compares a complex number, and false
        # at ratio = 0 (where the form in z is defined)
        if ratio.is_extended_nonpositive is True:
            return inside
        condition = as_boolean(And(x > 0, ratio * Abs(x)**n > 1))
    if condition == false:
        return inside
    return as_expr(Piecewise((outside, condition), (inside, true)))


def _joined(x: Symbol, point: Expr, side: int, inner_factor: Expr, outer_factor: Expr,
            m: Rational, q: Rational, m2: Rational, q2: Rational, p: Rational) -> Optional[Expr]:
    """The constant which makes ``x*outer_factor*2F1(-p, q2; q2 + 1;
    1/z)/(m2 + 1)`` beyond ``point`` (where ``z = 1``) continue ``x*
    inner_factor*2F1(-p, q; q + 1; z)/(m + 1)`` before it, for ``p > -1``:
    both series converge at 1 (Gauss's sum ``2F1(-p, q; q + 1; 1) =
    Gamma(q + 1)*Gamma(p + 1)/Gamma(q + p + 1)``), the factors have finite
    one-sided limits; ``None`` when a limit is not found."""
    before, after = ('-', '+') if side == 1 else ('+', '-')
    inner = attempt(lambda: limit(inner_factor, x, point, before), settings.timeout)
    outer = attempt(lambda: limit(outer_factor, x, point, after), settings.timeout)
    if inner is None or outer is None or not inner.is_number or not outer.is_number \
            or inner.has(nan, zoo, Limit) or outer.has(nan, zoo, Limit) \
            or inner.is_finite is not True or outer.is_finite is not True:
        return None
    first = point * inner * gamma(q + 1) * gamma(p + 1) / (gamma(q + p + 1) * (m + 1))
    second = point * outer * gamma(q2 + 1) * gamma(p + 1) / (gamma(q2 + p + 1) * (m2 + 1))
    return as_expr(first - second)


# ---------------------------------------------------------------------------
# The transformed integral

def _bound(t_of_x: Expr, x: Symbol, point: Expr, assumptions: Assumptions) -> Optional[Expr]:
    value = attempt(lambda: limit(t_of_x, x, point, assumptions=assumptions), settings.timeout)
    if value is None or value.has(nan, zoo, Limit) or value.free_symbols - t_of_x.free_symbols:
        return None
    return value


def _transformed(f: Expr, x: Symbol, a: Expr, b: Expr, substitution: Rationalisation,
                 assumptions: Assumptions, g: Optional[Expr] = None) -> Optional[ConditionalValue]:
    """The integral of the transformed integrand ``g`` (built from the
    substitution unless given) over the image of the range."""
    from .definite import conditional_integral, verify_numerically
    t = substitution.t
    if g is None:
        replaced = as_expr(f.xreplace(substitution.radicals))
        if _radical_atoms(replaced, x):
            return None
        g = as_expr(replaced.subs(x, substitution.x_of_t) * substitution.x_of_t.diff(t))
    g = as_expr(cancel(g))
    if not _is_rational_in(g, t):
        return None
    lower = _bound(substitution.t_of_x, x, a, assumptions)
    upper = _bound(substitution.t_of_x, x, b, assumptions)
    if lower is None or upper is None or lower == upper:
        return None
    found = conditional_integral(g, t, lower, upper, assumptions)
    if found is None:
        return None
    value = tidy(found.value, assumptions, found.condition)
    if settings.numerical_checks and verify_numerically(value, f, x, a, b, assumptions) is False:
        return None
    return ConditionalValue(value, found.condition)


def rationalised_integral(f: Expr, x: Symbol, a: Expr, b: Expr, substitution: Rationalisation,
                          assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` through a rationalising substitution."""
    return _transformed(f, x, a, b, substitution, assumptions)


def algebraic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` for an algebraic integrand of genus zero:
    the roots of a Möbius function, Euler's substitutions for a square
    root of a polynomial of degree at most two, and Chebyshev's binomial
    differentials, in this order; ``None`` when none applies.

    Examples
    ========

    >>> from sympy import symbols, sqrt, cbrt, oo
    >>> from sympy_extras.integrals.algebraic import algebraic_integral
    >>> x = symbols('x')
    >>> algebraic_integral(1/sqrt(x**2 + 1), x, 0, 1)
    ConditionalValue(-log(-1 + sqrt(2)))
    >>> algebraic_integral(x**3*sqrt(x**2 + 1), x, 0, 1)
    ConditionalValue(2*(1 + sqrt(2))/15)
    >>> algebraic_integral(cbrt(x)/(1 + x), x, 0, 1)
    ConditionalValue(-sqrt(3)*pi/3 - log(2) + 3)
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    if not _radical_atoms(f_, x):
        return None
    substitution = moebius_root(f_, x)
    if substitution is not None:
        found = _transformed(f_, x, a_, b_, substitution, assumptions)
        if found is not None:
            return found
    best: Optional[ConditionalValue] = None
    for substitution in euler_substitutions(f_, x, a_, b_, assumptions):
        # every applicable Euler substitution, the shortest value kept
        found = _transformed(f_, x, a_, b_, substitution, assumptions)
        if found is not None and (best is None or found.value.count_ops() < best.value.count_ops()):
            best = found
    if best is not None:
        return best
    return _binomial_integral(f_, x, a_, b_, assumptions)
