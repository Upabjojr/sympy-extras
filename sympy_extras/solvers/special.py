"""Second order linear equations solved by special functions: Bessel,
Whittaker (confluent hypergeometric) and Gauss hypergeometric equations
recognised through the invariant of the normal form.

The normal form `z'' = r(x) z` of `y'' + p y' + q y = 0` (`r = p^2/4 +
p'/2 - q`, `y = z e^{-\\int p/2}`) is invariant under changes of the
dependent variable; a change of the independent variable `t = \\phi(x)`
turns `z'' = R(t) z` into `w'' = [\\phi'^2 R(\\phi) - \\tfrac12 \\{\\phi, x\\}] w`
for `w(x) = z(\\phi(x)) \\phi'(x)^{-1/2}` (`\\{\\phi, x\\}` the Schwarzian
derivative). For `\\phi = a (x - c)^k` the Schwarzian is `(1 - k^2)/(2 (x -
c)^2)`, so the invariants of

* the **Bessel** equation, `R(t) = (\\nu^2 - 1/4)/t^2 - 1`, become
  `r = (k^2 \\nu^2 - 1/4)/(x - c)^2 - a^2 k^2 (x - c)^{2k - 2}`;
* the **Whittaker** equation, `R(t) = 1/4 - \\kappa/t + (\\mu^2 - 1/4)/t^2`,
  become `r = a^2 k^2/4\\, (x - c)^{2k-2} - a k^2 \\kappa\\, (x - c)^{k-2} +
  (k^2 \\mu^2 - 1/4)/(x - c)^2`;

and a rational `r` whose expansion in powers of `x - c` has exponents of
those shapes is matched to the family, `c, k, a` and the parameters read
off, and the solutions written with :func:`~sympy.besselj`,
:func:`~sympy.bessely` (or :func:`~sympy.besseli`, :func:`~sympy.besselk`)
and with Whittaker's `M_{\\kappa, \\pm\\mu}` expressed through
:class:`~sympy.hyper`. Airy's equation is the case `k = 3/2`, parabolic
cylinder functions `k = 2`. For `\\phi` a Möbius transformation (Schwarzian
zero) sending two poles of `r` and a third pole or infinity to `0, 1,
\\infty`, the invariant of **Riemann's equation** `(1 - \\lambda^2)/(4 z^2)
+ (1 - \\mu^2)/(4 (z - 1)^2) + (\\lambda^2 + \\mu^2 - \\nu^2 - 1)/(4 z (z - 1))`
gives the exponent differences and the solutions `z^\\alpha (1 - z)^\\beta
{}_2F_1(a, b; c; z)`.

This is the role played by Mellin transforms in Mathematica's ``DSolve``
(solutions in terms of special functions); SymPy's ``dsolve`` has hints
for a few of these equations in their standard forms only.
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral, integrate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr
from sympy_extras.settings import settings

from .kovacic import normal_form
from .linear_ode import LinearOperator

__all__ = ['special_solutions', 'bessel_solutions', 'whittaker_solutions', 'hypergeometric_solutions',
           'whittaker_m']


def whittaker_m(kappa: Expr, mu: Expr, t: Expr) -> Expr:
    """Whittaker's function ``M_{kappa, mu}(t) = exp(-t/2) t**(mu + 1/2)
    1F1(mu - kappa + 1/2; 1 + 2 mu; t)``.

    >>> from sympy.abc import t
    >>> from sympy_extras.solvers.special import whittaker_m
    >>> whittaker_m(0, S(1)/2, t)
    t*exp(-t/2)*hyper((1,), (2,), t)
    """
    return as_expr(exp(-t/2)*t**(mu + S.Half)*hyper([mu - kappa + S.Half], [1 + 2*mu], t))


def _shifted_terms(r: Expr, c: Expr, x: Symbol) -> Optional[dict[int, Expr]]:
    """The coefficients of ``(x - c)**e`` in ``r`` when ``r`` is a
    Laurent polynomial in ``x - c`` (poles only at ``c``)."""
    s = Dummy('s')
    shifted = as_expr(cancel(r.subs(x, c + s)))
    num, den = shifted.as_numer_denom()
    try:
        den_poly = Poly(den, s)
        num_poly = Poly(num, s)
    except PolynomialError:
        return None
    if not den_poly.is_monomial:
        return None
    j = den_poly.degree()
    lead = as_expr(den_poly.LC())
    result: dict[int, Expr] = {}
    for (e,), coefficient in num_poly.terms():
        result[int(e) - j] = as_expr(coefficient/lead)
    return result


def _candidates(r: Expr, x: Symbol) -> list[Expr]:
    """Shifts ``c`` around which ``r`` may be a Laurent polynomial."""
    num, den = cancel(r).as_numer_denom()
    try:
        den_poly = Poly(den, x)
        num_poly = Poly(num, x)
    except PolynomialError:
        return []
    if den_poly.degree() > 0:
        found = roots(den_poly)
        if sum(found.values()) != den_poly.degree() or len(found) != 1:
            return []
        return [as_expr(c) for c in found]
    m = num_poly.degree()
    if m == 0:
        return [S.Zero]
    coefficients = num_poly.all_coeffs()
    centre = as_expr(-coefficients[1]/(m*coefficients[0]))
    return [centre] if m > 1 else [centre, S.Zero]


class _Family:
    """A normal-form equation ``z'' = r z`` matched to a family."""

    def __init__(self, c: Expr, k: Rational, a: Expr, parameters: dict[str, Expr]) -> None:
        self.c = c
        self.k = k
        self.a = a
        self.parameters = parameters


def _match_bessel(terms: dict[int, Expr], c: Expr) -> Optional[_Family]:
    others = {e: v for e, v in terms.items() if e != -2}
    if len(others) != 1:
        return None
    (m, B), = others.items()
    k = Rational(m + 2, 2)
    if k == 0:
        return None
    A = terms.get(-2, S.Zero)
    a = sqrt(-B)/k
    nu2 = as_expr((A + Rational(1, 4))/k**2)
    return _Family(c, k, as_expr(a), {'nu2': nu2, 'B': B})


def _match_whittaker(terms: dict[int, Expr], c: Expr) -> Optional[_Family]:
    others = sorted(e for e in terms if e != -2)
    if len(others) != 2:
        return None
    e1, e2 = others
    if e2 != 2*e1 + 2:
        return None
    k = Rational(e1 + 2)
    if k == 0:
        return None
    C, D, E = terms[e2], terms[e1], terms.get(-2, S.Zero)
    a = as_expr(2*sqrt(C)/k)
    kappa = as_expr(-D/(a*k**2))
    mu2 = as_expr((E + Rational(1, 4))/k**2)
    return _Family(c, k, a, {'kappa': kappa, 'mu2': mu2})


def _prefactor(p: Expr, x: Symbol) -> Optional[Expr]:
    integral = attempt(lambda: integrate(p, x, conds='none'), settings.timeout)
    if integral is None or integral.has(Integral):
        return None
    return as_expr(exp(-as_expr(integral)/2))


def _sqrt_parameter(value2: Expr) -> Expr:
    """``sqrt(value2)`` with perfect squares recognised: the parameters
    are exponent differences, which may be taken with either sign."""
    from sympy.polys.polytools import factor, sqf_list
    factored = as_expr(factor(value2))
    if factored.is_number:
        return as_expr(sqrt(factored))
    symbols = sorted((v for v in factored.free_symbols if isinstance(v, Symbol)), key=lambda v: v.name)
    if factored.is_polynomial(*symbols):
        constant, factors = sqf_list(factored, *symbols)
        if all(m % 2 == 0 for _, m in factors):
            root: Expr = sqrt(constant)
            for f, m in factors:
                root = as_expr(root*as_expr(f)**(m//2))
            return root
    return as_expr(sqrt(factored))


def _without_constant(e: Expr, x: Symbol) -> Expr:
    constant, rest = e.as_independent(x, as_Add=False)
    return as_expr(rest) if constant != 0 and rest != 0 else e


def _tidy(e: Expr) -> Expr:
    """A light simplification which does not choke on hypergeometric
    parameters with radicals."""
    from sympy.simplify.powsimp import powsimp
    from sympy.polys.polyerrors import PolynomialError as _PolynomialError
    try:
        return as_expr(powsimp(e, combine='exp'))
    except (_PolynomialError, ValueError, TypeError, NotImplementedError):
        return e


def bessel_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` in Bessel functions, or ``None``.

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.special import bessel_solutions
    >>> bessel_solutions(-x, x)   # Airy
    [sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
    """
    for c in _candidates(r, x):
        terms = _shifted_terms(r, c, x)
        if terms is None:
            continue
        family = _match_bessel(terms, c)
        if family is None:
            continue
        k, a = family.k, family.a
        nu = _sqrt_parameter(family.parameters['nu2'])
        u = as_expr(x - c)
        B = family.parameters['B']
        if B.is_positive:
            # a is imaginary: modified Bessel functions of sqrt(B)/k u**k
            t = as_expr(sqrt(B)/k*u**k)
            first, second = besseli(nu, t), besselk(nu, t)
        else:
            t = as_expr(a*u**k)
            first, second = besselj(nu, t), bessely(nu, t)
            if nu.is_integer is False:
                second = besselj(-nu, t)
        return [as_expr(sqrt(u)*first), as_expr(sqrt(u)*second)]
    return None


def whittaker_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` in Whittaker functions (through
    ``1F1``), or ``None``.

    >>> from sympy.abc import x, a
    >>> from sympy_extras.solvers.special import whittaker_solutions
    >>> whittaker_solutions(x**2 + a, x)[0]
    (x**2)**(3/4)*exp(-x**2/2)*hyper((a/4 + 3/4,), (3/2,), x**2)/sqrt(x)
    """
    for c in _candidates(r, x):
        terms = _shifted_terms(r, c, x)
        if terms is None:
            continue
        family = _match_whittaker(terms, c)
        if family is None:
            continue
        k, a = family.k, family.a
        kappa = family.parameters['kappa']
        mu = _sqrt_parameter(family.parameters['mu2'])
        u = as_expr(x - c)
        t = as_expr(a*u**k)
        # w = z(t) phi'**(-1/2), phi' = a k u**(k-1): the constant is dropped
        factor = as_expr(u**((1 - k)/2))
        solutions = [as_expr(factor*whittaker_m(kappa, mu, t))]
        if (2*mu).is_integer is not True:
            solutions.append(as_expr(factor*whittaker_m(kappa, -mu, t)))
        return solutions
    return None


def _moebius(c1: Expr, c2: Expr, c3: Optional[Expr], x: Symbol) -> tuple[Expr, Expr]:
    """``phi`` sending ``c1, c2, c3`` to ``0, 1, oo`` (``c3 = None`` for
    infinity) and its inverse ``x(z)`` (as an expression in the symbol
    returned as the second element's free symbol ``z``)."""
    z = Dummy('z')
    if c3 is None:
        phi = as_expr((x - c1)/(c2 - c1))
        inverse = as_expr(c1 + (c2 - c1)*z)
    else:
        phi = as_expr((x - c1)*(c2 - c3)/((x - c3)*(c2 - c1)))
        # z (x - c3)(c2 - c1) = (x - c1)(c2 - c3)
        inverse = as_expr((c1*(c2 - c3) - z*c3*(c2 - c1))/((c2 - c3) - z*(c2 - c1)))
    return phi, inverse


def hypergeometric_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` through Gauss's hypergeometric function
    when ``r`` has two or three regular singular points (counting
    infinity), or ``None``.

    >>> from sympy import Rational
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.special import hypergeometric_solutions
    >>> hypergeometric_solutions(-1/(4*x**2) - 1/(4*(x - 1)**2) + 1/(x*(x - 1)), x)[0]
    sqrt(x)*sqrt(1 - x)*hyper((1/2 - sqrt(3)/2, 1/2 + sqrt(3)/2), (1,), 1 - x)
    """
    num, den = cancel(r).as_numer_denom()
    try:
        den_poly = Poly(den, x)
        num_poly = Poly(num, x)
    except PolynomialError:
        return None
    found = roots(den_poly)
    if sum(found.values()) != den_poly.degree() or any(m > 2 for m in found.values()):
        return None
    poles = [as_expr(c) for c in found]
    order_at_infinity = den_poly.degree() - num_poly.degree()
    if len(poles) == 2 and order_at_infinity >= 2:
        c1, c2, c3 = poles[0], poles[1], None
    elif len(poles) == 3 and order_at_infinity >= 4:
        # infinity is not singular: phi'**2 R(phi) with phi Moebius
        c1, c2, c3 = poles[0], poles[1], poles[2]
    else:
        return None
    phi, inverse = _moebius(c1, c2, c3, x)
    [z] = [s for s in inverse.free_symbols if isinstance(s, Dummy)] if isinstance(inverse, Expr) and inverse.free_symbols else [Dummy('z')]
    dphi = as_expr(phi.diff(x))
    R = as_expr(cancel((r/dphi**2).subs(x, inverse)))
    # R = (lambda^2 - 1)/(4 z^2) + (mu^2 - 1)/(4 (z-1)^2) + K/(z (z - 1)), the
    # exponents at 0 being (1 +- lambda)/2 (roots of rho (rho - 1) = A0)
    A0 = as_expr(cancel(R*z**2).subs(z, 0))
    A1 = as_expr(cancel(R*(z - 1)**2).subs(z, 1))
    lam2 = as_expr(1 + 4*A0)
    mu2 = as_expr(1 + 4*A1)
    remainder = as_expr(cancel(R - A0/z**2 - A1/(z - 1)**2))
    K = as_expr(cancel(remainder*z*(z - 1)))
    if K.has(z):
        return None
    nu2 = as_expr(lam2 + mu2 - 1 + 4*K)
    lam, mu, nu = _sqrt_parameter(lam2), _sqrt_parameter(mu2), _sqrt_parameter(nu2)
    alpha, alpha_ = (1 + lam)/2, (1 - lam)/2
    beta = (1 + mu)/2
    a1 = as_expr((1 + lam + mu + nu)/2)
    b1 = as_expr((1 + lam + mu - nu)/2)
    c_ = as_expr(1 + lam)
    w = as_expr(phi**alpha*(1 - phi)**beta*hyper([a1, b1], [c_], phi))
    solutions = [_without_constant(as_expr(w/sqrt(dphi)), x)]
    if lam.is_integer is not True:
        a2 = as_expr((1 - lam + mu + nu)/2)
        b2 = as_expr((1 - lam + mu - nu)/2)
        w2 = as_expr(phi**alpha_*(1 - phi)**beta*hyper([a2, b2], [1 - lam], phi))
        solutions.append(_without_constant(as_expr(w2/sqrt(dphi)), x))
    return solutions


def special_solutions(equation: Basic, f: AppliedUndef) -> Optional[list[Expr]]:
    """Solutions of a second order linear equation with rational
    coefficients in Bessel, Whittaker or hypergeometric functions, or
    ``None`` when the equation is not recognised.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x, a, n
    >>> from sympy_extras.solvers.special import special_solutions
    >>> y = Function('y')(x)
    >>> special_solutions(y.diff(x, 2) + x*y, y)
    [sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
    >>> special_solutions((1 - x**2)*y.diff(x, 2) - 2*x*y.diff(x) + n*(n + 1)*y, y)
    [sqrt(1/2 - x/2)*sqrt(x/2 + 1/2)*hyper((-n, n + 1), (1,), 1/2 - x/2)/sqrt(x**2 - 1)]
    >>> special_solutions(x*y.diff(x, 2) + y.diff(x) + a*y, y)
    [besselj(0, 2*sqrt(a)*sqrt(x)), bessely(0, 2*sqrt(a)*sqrt(x))]
    """
    L = LinearOperator.from_equation(equation, f)
    x_ = f.args[0]
    if not isinstance(x_, Symbol) or L.order != 2:
        return None
    x = x_
    a2, a1, a0 = L.coefficients[2], L.coefficients[1], L.coefficients[0]
    p = as_expr(cancel(a1/a2))
    q = as_expr(cancel(a0/a2))
    r = normal_form(p, q, x)
    prefactor = _prefactor(p, x)
    if prefactor is None:
        return None
    for method in (bessel_solutions, whittaker_solutions, hypergeometric_solutions):
        found = method(r, x)
        if found:
            return [_without_constant(_tidy(as_expr(prefactor*z)), x) for z in found]
    return None
