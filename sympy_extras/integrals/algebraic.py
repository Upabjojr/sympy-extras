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
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel
from sympy.series.limits import Limit

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.limits import limit
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .marichev import tidy

__all__ = ['algebraic_integral', 'euler_substitution', 'euler_substitutions', 'binomial_differential', 'moebius_root',
           'rationalised_integral', 'Rationalisation']


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
    ConditionalValue(log(1/(-1 + sqrt(2))))
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
