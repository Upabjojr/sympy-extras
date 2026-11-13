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
off, and the solutions written with ``besselj``, ``bessely`` (or
``besseli``, ``besselk``) and with Whittaker's `M_{\\kappa, \\pm\\mu}`
expressed through ``hyper``. Airy's equation is the case `k = 3/2`,
parabolic cylinder functions `k = 2`. For `\\phi` a Möbius transformation
(Schwarzian zero) sending two poles of `r` and a third pole or infinity to
`0, 1, \\infty`, the invariant of **Riemann's equation** `(1 -
\\lambda^2)/(4 z^2) + (1 - \\mu^2)/(4 (z - 1)^2) + (\\lambda^2 + \\mu^2 -
\\nu^2 - 1)/(4 z (z - 1))` gives the exponent differences and the
solutions `z^\\alpha (1 - z)^\\beta {}_2F_1(a, b; c; z)`.

This is the role played by Mellin transforms in Mathematica's ``DSolve``
(solutions in terms of special functions); SymPy's ``dsolve`` has hints
for a few of these equations in their standard forms only.

The module works on the objects of the core and imports no SymPy (its
face on SymPy objects is :mod:`sympy_extras.solvers.special_sympy`): the
special functions are those of :mod:`sympy_extras.functions`; the factor
``exp(-Integral(p)/2)`` comes from the native indefinite integration
(:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.solvers.special import bessel_solutions
>>> x = Symbol('x')
>>> bessel_solutions(-x, x)   # Airy
[sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
"""
from __future__ import annotations

from typing import Final, Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import sqrt
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy_extras.functions.special.hyper import hyper
from sympy_extras.polys.polytools import factor, sqf_list
from sympy_extras.simplify.powsimp import powsimp

from .kovacic import normal_form
from .linear_ode import _FAILURES, LinearOperator, UPoly, _normal, _replaced, _root_multiplicities

__all__ = ['special_solutions', 'bessel_solutions', 'whittaker_solutions', 'hypergeometric_solutions',
           'whittaker_m']

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)
_HALF: Final[Expr] = Rational(1, 2)
_QUARTER: Final[Expr] = Rational(1, 4)


def whittaker_m(kappa: Expr, mu: Expr, t: Expr) -> Expr:
    """Whittaker's function ``M_{kappa, mu}(t) = exp(-t/2) t**(mu + 1/2)
    1F1(mu - kappa + 1/2; 1 + 2 mu; t)``.

    >>> from sympy_extras.core import Integer, Rational, Symbol
    >>> from sympy_extras.solvers.special import whittaker_m
    >>> t = Symbol('t')
    >>> whittaker_m(Integer(0), Rational(1, 2), t)
    t*exp(-t/2)*hyper((1,), (2,), t)
    """
    return exp(-t/2)*t**(mu + _HALF)*hyper([mu - kappa + _HALF], [1 + 2*mu], t)


def _shifted_terms(r: Expr, c: Expr, x: Symbol) -> Optional[dict[int, Expr]]:
    """The coefficients of ``(x - c)**e`` in ``r`` when ``r`` is a
    Laurent polynomial in ``x - c`` (poles only at ``c``)."""
    s = Dummy('s')
    shifted = _normal(_replaced(r, {x: c + s}))
    num, den = shifted.as_numer_denom()
    try:
        den_poly = UPoly.of(den, s)
        num_poly = UPoly.of(num, s)
    except ValueError:
        return None
    if len(den_poly.terms_) != 1:
        return None
    j = den_poly.degree()
    lead = den_poly.LC()
    result: dict[int, Expr] = {}
    for e, coefficient in sorted(num_poly.terms_.items(), reverse=True):
        result[e - j] = coefficient/lead
    return result


def _candidates(r: Expr, x: Symbol) -> list[Expr]:
    """Shifts ``c`` around which ``r`` may be a Laurent polynomial."""
    num, den = _normal(r).as_numer_denom()
    try:
        den_poly = UPoly.of(den, x)
        num_poly = UPoly.of(num, x)
    except ValueError:
        return []
    if den_poly.degree() > 0:
        found = _root_multiplicities(den_poly)
        if sum(found.values()) != den_poly.degree() or len(found) != 1:
            return []
        return list(found)
    if num_poly.is_zero:
        # r = 0: the equation is z'' = 0, with no singular point to
        # centre a Bessel or Whittaker form on (sympy-extras#42); its
        # degree would be -oo and its coefficient list [0]
        return []
    m = num_poly.degree()
    if m == 0:
        return [_ZERO]
    coefficients = num_poly.all_coeffs()
    centre = -coefficients[1]/(m*coefficients[0])
    return [centre] if m > 1 else [centre, _ZERO]


class _Family:
    """A normal-form equation ``z'' = r z`` matched to a family."""

    def __init__(self, c: Expr, k: Rational, a: Expr, parameters: dict[str, Expr]) -> None:
        self.c: Final[Expr] = c
        self.k: Final[Rational] = k
        self.a: Final[Expr] = a
        self.parameters: Final[dict[str, Expr]] = parameters


def _match_bessel(terms: dict[int, Expr], c: Expr) -> Optional[_Family]:
    others = {e: v for e, v in terms.items() if e != -2}
    if len(others) != 1:
        return None
    (m, B), = others.items()
    k = Rational(m + 2, 2)
    if k == _ZERO:
        return None
    A = terms.get(-2, _ZERO)
    a = sqrt(-B)/k
    nu2 = (A + _QUARTER)/k**2
    return _Family(c, k, a, {'nu2': nu2, 'B': B})


def _match_whittaker(terms: dict[int, Expr], c: Expr) -> Optional[_Family]:
    others = sorted(e for e in terms if e != -2)
    if len(others) != 2:
        return None
    e1, e2 = others
    if e2 != 2*e1 + 2:
        return None
    k = Rational(e1 + 2, 1)
    if k == _ZERO:
        return None
    C, D, E = terms[e2], terms[e1], terms.get(-2, _ZERO)
    a = 2*sqrt(C)/k
    kappa = -D/(a*k**2)
    mu2 = (E + _QUARTER)/k**2
    return _Family(c, k, a, {'kappa': kappa, 'mu2': mu2})


def _prefactor(p: Expr, x: Symbol) -> Optional[Expr]:
    from sympy_extras.integrals.indefinite import closed_antiderivative
    integral = closed_antiderivative(p, x)
    if integral is None:
        return None
    return exp(-integral/2)


def _sqrt_parameter(value2: Expr) -> Expr:
    """``sqrt(value2)`` with perfect squares recognised: the parameters
    are exponent differences, which may be taken with either sign."""
    try:
        factored = factor(value2)
    except _FAILURES:
        factored = value2
    if factored.is_number:
        return sqrt(factored)
    symbols = sorted((v for v in factored.free_symbols if isinstance(v, Symbol)), key=lambda v: v.name)
    if factored.is_polynomial(*symbols):
        try:
            constant, factors = sqf_list(factored, *symbols)
        except _FAILURES:
            return sqrt(factored)
        multiplicities = [m for _, m in factors]
        if all(isinstance(m, Integer) and int(m) % 2 == 0 for m in multiplicities):
            root: Expr = sqrt(constant)
            for f, m in factors:
                if isinstance(m, Integer):
                    root = root*f**(int(m)//2)
            return root
    return sqrt(factored)


def _without_constant(e: Expr, x: Symbol) -> Expr:
    constant, rest = e.as_independent(x, as_Add=False)
    return rest if constant != _ZERO and rest != _ZERO else e


def _tidy(e: Expr) -> Expr:
    """A light simplification which does not choke on hypergeometric
    parameters with radicals."""
    try:
        return powsimp(e, combine='exp')
    except _FAILURES:
        return e


def bessel_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` in Bessel functions, or ``None``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.special import bessel_solutions
    >>> x = Symbol('x')
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
        u = x - c
        B = family.parameters['B']
        if B.is_positive:
            # a is imaginary: modified Bessel functions of sqrt(B)/k u**k
            t = sqrt(B)/k*u**k
            first, second = besseli(nu, t), besselk(nu, t)
        else:
            t = a*u**k
            first, second = besselj(nu, t), bessely(nu, t)
            if nu.is_integer is False:
                second = besselj(-nu, t)
        return [sqrt(u)*first, sqrt(u)*second]
    return None


def whittaker_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` in Whittaker functions (through
    ``1F1``), or ``None``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.special import whittaker_solutions
    >>> x, a = symbols('x a')
    >>> found = whittaker_solutions(x**2 + a, x)
    >>> len(found), found[0].has(x)
    (2, True)
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
        u = x - c
        t = a*u**k
        # w = z(t) phi'**(-1/2), phi' = a k u**(k-1): the constant is dropped
        factor_ = u**((1 - k)/2)
        solutions = [factor_*whittaker_m(kappa, mu, t)]
        if (2*mu).is_integer is not True:
            solutions.append(factor_*whittaker_m(kappa, -mu, t))
        return solutions
    return None


def _moebius(c1: Expr, c2: Expr, c3: Optional[Expr], x: Symbol) -> tuple[Expr, Expr, Symbol]:
    """``phi`` sending ``c1, c2, c3`` to ``0, 1, oo`` (``c3 = None`` for
    infinity), its inverse ``x(z)`` and the symbol ``z``."""
    z = Dummy('z')
    if c3 is None:
        phi = (x - c1)/(c2 - c1)
        inverse = c1 + (c2 - c1)*z
    else:
        phi = (x - c1)*(c2 - c3)/((x - c3)*(c2 - c1))
        # z (x - c3)(c2 - c1) = (x - c1)(c2 - c3)
        inverse = (c1*(c2 - c3) - z*c3*(c2 - c1))/((c2 - c3) - z*(c2 - c1))
    return phi, inverse, z


def hypergeometric_solutions(r: Expr, x: Symbol) -> Optional[list[Expr]]:
    """Solutions of ``z'' = r z`` through Gauss's hypergeometric function
    when ``r`` has two or three regular singular points (counting
    infinity), or ``None``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.special import hypergeometric_solutions
    >>> x = Symbol('x')
    >>> hypergeometric_solutions(-1/(4*x**2) - 1/(4*(x - 1)**2) + 1/(x*(x - 1)), x)[0]
    sqrt(x)*sqrt(1 - x)*hyper((1/2 - sqrt(3)/2, 1/2 + sqrt(3)/2), (1,), 1 - x)
    """
    num, den = _normal(r).as_numer_denom()
    try:
        den_poly = UPoly.of(den, x)
        num_poly = UPoly.of(num, x)
    except ValueError:
        return None
    found = _root_multiplicities(den_poly)
    if sum(found.values()) != den_poly.degree() or any(m > 2 for m in found.values()):
        return None
    poles = list(found)
    order_at_infinity = den_poly.degree() - num_poly.degree()
    c3: Optional[Expr]
    if len(poles) == 2 and order_at_infinity >= 2:
        c1, c2, c3 = poles[0], poles[1], None
    elif len(poles) == 3 and order_at_infinity >= 4:
        # infinity is not singular: phi'**2 R(phi) with phi Moebius
        c1, c2, c3 = poles[0], poles[1], poles[2]
    else:
        return None
    phi, inverse, z = _moebius(c1, c2, c3, x)
    dphi = phi.diff(x)
    R = _normal(_replaced(r/dphi**2, {x: inverse}))
    # R = (lambda^2 - 1)/(4 z^2) + (mu^2 - 1)/(4 (z-1)^2) + K/(z (z - 1)), the
    # exponents at 0 being (1 +- lambda)/2 (roots of rho (rho - 1) = A0)
    A0 = _replaced(_normal(R*z**2), {z: _ZERO})
    A1 = _replaced(_normal(R*(z - 1)**2), {z: _ONE})
    lam2 = 1 + 4*A0
    mu2 = 1 + 4*A1
    remainder = _normal(R - A0/z**2 - A1/(z - 1)**2)
    K = _normal(remainder*z*(z - 1))
    if K.has(z):
        return None
    nu2 = lam2 + mu2 - 1 + 4*K
    lam, mu, nu = _sqrt_parameter(lam2), _sqrt_parameter(mu2), _sqrt_parameter(nu2)
    alpha, alpha_ = (1 + lam)/2, (1 - lam)/2
    beta = (1 + mu)/2
    a1 = (1 + lam + mu + nu)/2
    b1 = (1 + lam + mu - nu)/2
    c_ = 1 + lam
    w = phi**alpha*(1 - phi)**beta*hyper([a1, b1], [c_], phi)
    solutions = [_without_constant(w/sqrt(dphi), x)]
    if lam.is_integer is not True:
        a2 = (1 - lam + mu + nu)/2
        b2 = (1 - lam + mu - nu)/2
        w2 = phi**alpha_*(1 - phi)**beta*hyper([a2, b2], [1 - lam], phi)
        solutions.append(_without_constant(w2/sqrt(dphi), x))
    return solutions


def special_solutions(equation: Basic, f: AppliedUndef) -> Optional[list[Expr]]:
    """Solutions of a second order linear equation with rational
    coefficients in Bessel, Whittaker or hypergeometric functions, or
    ``None`` when the equation is not recognised.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.special import special_solutions
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> special_solutions(y.diff(x, 2) + x*y, y)
    [sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
    """
    L = LinearOperator.from_equation(equation, f)
    x_ = f.args[0]
    if not isinstance(x_, Symbol) or L.order != 2:
        return None
    x = x_
    a2, a1, a0 = L.coefficients[2], L.coefficients[1], L.coefficients[0]
    p = _normal(a1/a2)
    q = _normal(a0/a2)
    r = normal_form(p, q, x)
    prefactor = _prefactor(p, x)
    if prefactor is None:
        return None
    for method in (bessel_solutions, whittaker_solutions, hypergeometric_solutions):
        found = method(r, x)
        if found:
            return [_without_constant(_tidy(prefactor*z), x) for z in found]
    return None
