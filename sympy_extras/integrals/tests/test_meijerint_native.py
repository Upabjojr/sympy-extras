"""Tests of the native Meijer G-function integration
(:mod:`sympy_extras.integrals.meijerint`): the cases of SymPy's
``test_meijerint.py``, the antiderivatives checked by differentiation and
the definite integrals against their known values (numerically, at
sample values of the parameters)."""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/tests/test_meijerint.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: on the core; the values compared numerically at sample points
# of the parameters instead of their printed forms.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras._numeric import reliable_value
from sympy_extras.core import I, Integer, Rational, S, Symbol, oo, pi
from sympy_extras.core.basic import MatchDict
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, true
from sympy_extras.functions import (Chi, Ci, Shi, Si, besseli, besselj, cos, cosh, erf, erfc, exp, expint, gamma,
                                    log, meijerg, polygamma, sin, sinc, sqrt)
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.elementary.exponential import exp_polar
from sympy_extras.functions.special.hyper import MeijerG
from sympy_extras.integrals.indefinite import is_antiderivative
from sympy_extras.integrals.meijerint import (_check_antecedents, _condsimp, _g, _get_coeff_exp, _inflate_g,
                                              _match, _rewrite1, _rewrite_saxena, lookup_table,
                                              meijer_antiderivative, meijerint_definite, meijerint_indefinite,
                                              rewrite_single, z as z_dummy)

ZERO = Integer(0)


def _syms(names: str, **assumptions: bool) -> list[Symbol]:
    return [Symbol(name, **assumptions) for name in names.split()]


x, y, a, b, c, d, s, t = _syms('x y a b c d s t')


def _values(e: Expr, extra: Optional[dict[Symbol, Expr]] = None) -> dict[Symbol, Expr]:
    """Sample values of the symbols of ``e`` (positive rationals)."""
    values: dict[Symbol, Expr] = {}
    for i, sym in enumerate(sorted((v for v in e.free_symbols if isinstance(v, Symbol)), key=str)):
        values[sym] = Integer(2 + i) if sym.is_integer else Rational(7 + 3 * i, 10)
    if extra:
        values.update(extra)
    return values


def _same(u: Expr, v: Expr, extra: Optional[dict[Symbol, Expr]] = None) -> bool:
    values = _values(u + v, extra)
    first = reliable_value(u, 15, values)
    second = reliable_value(v, 15, values)
    assert first is not None and second is not None
    return abs(complex(first) - complex(second)) < 1e-10 * max(1.0, abs(complex(second)))


def _derivative_is(F: Expr, f: Expr, var: Symbol) -> bool:
    """``F' = f`` at two positive points (the polar antiderivatives of the
    route hold on the positive axis)."""
    return all(_same(F.diff(var).subs(var, point), f.subs(var, point))
               for point in (Rational(3, 10), Rational(13, 10)))


def _antiderivative(f: Expr, var: Symbol) -> Expr:
    found = meijerint_indefinite(f, var)
    assert found is not None
    assert found[1] is true
    return found[0]


def _definite(f: Expr, var: Symbol, lower: Expr, upper: Expr) -> tuple[Expr, Boolean]:
    found = meijerint_definite(f, var, lower, upper)
    assert found is not None
    return found


def test_rewrite_single() -> None:
    def t1(expr: Expr, coeff: Expr, m: Expr) -> None:
        e = rewrite_single(meijerg([a], [b], [c], [d], expr), x)
        assert e is not None
        g = e[0][0][2]
        assert isinstance(g, MeijerG)
        assert g.argument == coeff * m

    def tn(expr: Expr) -> None:
        assert rewrite_single(meijerg([a], [b], [c], [d], expr), x) is None
    t1(x, Integer(1), x)
    t1(x**2, Integer(1), x**2)
    t1(x**2 + y * x**2, y + 1, x**2)
    tn(x**2 + x)
    tn(x**y)


def test_rewrite1() -> None:
    assert _rewrite1(x**3 * meijerg([a], [b], [c], [d], x**2 + y * x**2) * 5, x) == \
        (5, x**3, [(1, 0, _g([a], [b], [c], [d], x**2 * (y + 1)))], true)


def test_recursive_mellin_transform() -> None:
    # a product which the table does not have, by its Mellin transform
    # (the theorem for two G-functions) inverted as a G-function
    found = rewrite_single(exp(-x) * sin(x), x)
    assert found is not None and found[1] is true
    total: Expr = ZERO
    for coeff, power, g in found[0]:
        total = total + coeff * x**power * g
    assert _same(total, exp(-x) * sin(x), {x: Rational(7, 10)})
    F = _antiderivative(exp(-x) * sin(x), x)
    assert _derivative_is(F, exp(-x) * sin(x), x)


def test_recursive() -> None:
    ap, bp, cp = _syms('a b c', positive=True)
    for f, expected in [
            (exp(-(x - ap)**2) * exp(-(x - bp)**2),
             sqrt(2) * sqrt(pi) * (erf(sqrt(2) * (ap + bp) / 2) + 1) * exp(-ap**2 / 2 + ap * bp - bp**2 / 2) / 4),
            (exp(-(x - ap)**2) * exp(-(x - bp)**2) * exp(cp * x),
             sqrt(2) * sqrt(pi) * (erf(sqrt(2) * (2 * ap + 2 * bp + cp) / 4) + 1)
             * exp(-ap**2 - bp**2 + (2 * ap + 2 * bp + cp)**2 / 8) / 4),
            (exp(-(x - ap - bp - cp)**2), sqrt(pi) / 2 * (1 + erf(ap + bp + cp))),
            (exp(-(x + ap + bp + cp)**2), sqrt(pi) / 2 * (1 - erf(ap + bp + cp)))]:
        value, cond = _definite(f, x, ZERO, oo)
        assert cond is true and _same(value, expected)


def test_meijerint_indefinite_numerically() -> None:
    values = {a: Rational(1, 10), b: Rational(3, 10), c: Rational(1, 3), d: Rational(2, 7)}

    def check(fac: Expr, argument: Expr) -> None:
        g = (meijerg([a], [b], [c], [d], argument) * fac).subs(values)
        assert is_antiderivative(_antiderivative(g, x), g, x) is True
    check(Integer(1), x)
    check(Integer(2), x)
    check(Integer(1), 2 * x)
    check(Integer(1), x**2)
    check(Integer(5), x**Rational(3, 2))
    check(x**3, x)
    check(3 * x**Rational(3, 2), 4 * x**Rational(7, 3))


def test_meijerint_definite_empty_range() -> None:
    assert meijerint_definite(x, x, ZERO, ZERO) == (0, true)
    assert meijerint_definite(x, x, oo, oo) == (0, true)


def test_inflate() -> None:
    values = {a: Rational(1, 10), b: Rational(3, 10), c: Rational(1, 3), d: Rational(2, 7), y: Rational(1, 5)}
    for g, n in [(_g([a], [b], [c], [d], x), 3), (_g([a, y], [b], [c], [d], x), 3),
                 (_g([a], [b], [c, y], [d], 2 * x**3), 3)]:
        C, h = _inflate_g(g, n)
        assert _same(g.subs(values), (C * h).subs(values), {x: Rational(1, 2)})


def test_meijerint() -> None:
    sp = Symbol('s', positive=True)
    assert _definite(x**sp * meijerg([], [], [0], [], x), x, ZERO, oo) == (gamma(sp + 1), true)
    assert meijerint_indefinite(exp(x), x) == (exp(x), true)
    ap, bp = _syms('a b', positive=True)
    value, cond = _definite(x**ap, x, ZERO, bp)
    assert cond is true and _same(value, bp**(ap + 1) / (ap + 1))
    assert _definite((x + 1)**3 * exp(-x), x, ZERO, oo) == (16, true)
    sigma, mu = _syms('sigma mu', positive=True)
    value, cond = _definite(exp(-((x - mu) / (2 * sigma))**2), x, ZERO, oo)
    assert cond is true and _same(value, sqrt(pi) * sigma * (2 - erfc(mu / (2 * sigma))))
    value, _ = _definite(exp(-mu * x) * exp(sigma * x), x, ZERO, oo)
    assert _same(value, 1 / (mu - sigma), {mu: Integer(3), sigma: Integer(1)})
    # the substitutions which change the limits
    assert _definite(exp(x), x, -oo, Integer(2)) == (exp(2), true)
    value, cond = _definite(exp(x), x, ZERO, I)
    assert cond is true and _same(value, exp(I) - 1)
    # over (-oo, oo)
    assert _definite(exp(-x**2), x, -oo, oo) == (sqrt(pi), true)
    assert _definite(exp(-abs_(x)), x, -oo, oo) == (2, true)
    assert _definite(exp(-(2 * x - 3)**2), x, -oo, oo) == (sqrt(pi) / 2, true)
    assert _definite(exp(-abs_(2 * x - 3)), x, -oo, oo) == (1, true)
    assert _definite(exp(-((x - mu) / sigma)**2 / 2) / sqrt(2 * pi * sigma**2), x, -oo, oo) == (1, true)
    assert _definite(sinc(x)**2, x, -oo, oo) == (pi, true)
    # one of the extra conditions for two G-functions
    assert _definite(exp(-x) * sin(x), x, ZERO, oo) == (S.Half, true)
    for n in range(4):
        expected = ((1 / (1 + x**2)).diff((x, n))).subs(x, Integer(1)) * Integer(-1)**n
        value, cond = _definite(exp(-x) * sin(x) * x**n, x, ZERO, oo)
        assert cond is true and _same(value, expected)
    value, cond = _definite(exp(-x) * sin(x + a), x, ZERO, oo)
    assert cond is true and _same(value, sqrt(2) * sin(a + pi / 4) / 2)
    # the condition 14 of Prudnikov (besselj*besselj in disguise)
    value, cond = _definite(meijerg([], [], [a / 2], [-a / 2], x / 4) * meijerg([], [], [b / 2], [-b / 2], x / 4)
                            * x**(s - 1), x, ZERO, oo)
    expected = 4 * 2**(2 * s - 2) * gamma(-2 * s + 1) * gamma(a / 2 + b / 2 + s) \
        / (gamma(-a / 2 + b / 2 - s + 1) * gamma(a / 2 - b / 2 - s + 1) * gamma(a / 2 + b / 2 - s + 1))
    assert _same(value, expected, {s: Rational(1, 4)})
    assert meijerint_definite(sin(x**a) * sin(x**b), x, ZERO, oo) is None
    value, cond = _definite(exp(-x**2) * log(x), x, ZERO, oo)
    assert cond is true and _same(value, sqrt(pi) * polygamma(0, S.Half) / 4)
    k = Symbol('k', positive=True)
    value, cond = _definite(x**k * exp(-ap * x**2), x, -oo, oo)
    assert cond is true and _same(value, ap**(-k / 2 - S.Half) * ((-1)**k + 1) * gamma(k / 2 + S.Half) / 2,
                                  {k: Integer(2)})


def test_bessel() -> None:
    assert meijer_antiderivative(x * besselj(0, x), x) == x * besselj(1, x)
    assert meijer_antiderivative(x * besseli(0, x), x) == x * besseli(1, x)
    assert meijer_antiderivative(besselj(1, x), x) == -besselj(0, x)
    for f in [besselj(1, x)**2 / x, besselj(0, x)**2 / x**2, besselj(0, x) * besselj(1, x),
              x**2 * besselj(0, x) * besselj(1, x), besselj(0, x) * besselj(1, x) / x, besselj(1, x**2) * x]:
        F = meijer_antiderivative(f, x)
        assert F is not None and is_antiderivative(F, f, x) is True
    z = Symbol('z')
    value, _ = _definite(besselj(a, z) * besselj(b, z) / z, z, ZERO, oo)
    assert _same(value, 2 * sin(pi * (a / 2 - b / 2)) / (pi * (a - b) * (a + b)), {a: Integer(2), b: Integer(1)})


def test_branch_bug() -> None:
    F = _antiderivative(erf(x**3), x)
    assert is_antiderivative(F, erf(x**3), x) is True


def test_linear_subs() -> None:
    assert _antiderivative(sin(x - 1), x) == -cos(x - 1)
    assert _antiderivative(besselj(1, x - 1), x) == -besselj(0, x - 1)


def test_probability() -> None:
    mu1 = Symbol('mu1', nonzero=True)
    sigma1 = Symbol('sigma1', positive=True)
    rate = Symbol('lambda', positive=True)

    def normal(var: Expr, m: Expr, sd: Expr) -> Expr:
        return 1 / sqrt(2 * pi * sd**2) * exp(-(var - m)**2 / 2 / sd**2)
    for n, expected in [(0, Integer(1)), (1, mu1), (2, mu1**2 + sigma1**2), (3, mu1**3 + 3 * mu1 * sigma1**2)]:
        value, cond = _definite(x**n * normal(x, mu1, sigma1), x, -oo, oo)
        assert cond is true and _same(value, expected)
    for n, expected in [(0, Integer(1)), (1, 1 / rate), (2, 2 / rate**2)]:
        value, cond = _definite(x**n * rate * exp(-rate * x), x, ZERO, oo)
        assert cond is true and _same(value, expected)
    alpha, beta = _syms('alpha beta', positive=True)
    betadist = x**(alpha - 1) * (1 + x)**(-alpha - beta) * gamma(alpha + beta) / gamma(alpha) / gamma(beta)
    assert _definite(betadist, x, ZERO, oo) == (1, true)
    value, cond = _definite(x * betadist, x, ZERO, oo)
    assert _same(value, alpha / (beta - 1), {beta: Integer(3)}) and cond == (beta > 1)
    ap, bp = _syms('a b', positive=True)
    betad = x**(ap - 1) * (-x + 1)**(bp - 1) * gamma(ap + bp) / (gamma(ap) * gamma(bp))
    value, _ = _definite(x * betad, x, ZERO, Integer(1))
    assert _same(value, ap / (ap + bp))
    k = Symbol('k', integer=True, positive=True)
    chi = 2**(1 - k / 2) * x**(k - 1) * exp(-x**2 / 2) / gamma(k / 2)
    value, _ = _definite(x**2 * chi, x, ZERO, oo)
    assert _same(value, k)
    lamda, mu = _syms('lamda mu', positive=True)
    dist = sqrt(lamda / 2 / pi) * x**Rational(-3, 2) * exp(-lamda * (x - mu)**2 / x / 2 / mu**2)
    value, _ = _definite(x * dist, x, ZERO, oo)
    assert _same(value, mu)
    nu, sigma = _syms('nu sigma', positive=True)
    rice = x / sigma**2 * exp(-(x**2 + nu**2) / 2 / sigma**2) * besseli(0, x * nu / sigma**2)
    assert _definite(rice, x, ZERO, oo) == (1, true)
    mur = Symbol('mu', real=True)
    laplace = exp(-abs_(x - mur) / bp) / 2 / bp
    value, cond = _definite(x**2 * laplace, x, -oo, oo)
    assert cond is true and _same(value, 2 * bp**2 + mur**2)


def test_expint() -> None:
    for f, expected in [(Si(x), x * Si(x) + cos(x)), (Shi(x), x * Shi(x) - cosh(x))]:
        assert _antiderivative(f, x) == expected
    for f in [expint(1, x), expint(2, x), Ci(x), Chi(x), exp(-x) / x, exp(-x) / x**2, cos(x) / x, cosh(x) / x]:
        assert _derivative_is(_antiderivative(f, x), f, x)
    assert _definite(Si(x) * exp(-x), x, ZERO, oo) == (pi / 4, true)
    assert _definite(expint(1, x) * sin(x), x, ZERO, oo) == (log(2) / 2, true)
    assert _definite(expint(1, x) * besselj(0, x), x, ZERO, oo) == (log(1 + sqrt(2)), true)


def test_issue_6122() -> None:
    assert _definite(exp(-I * x**2), x, -oo, oo) == (-I * sqrt(pi) * exp(I * pi / 4), true)


def test_fresnel_and_others() -> None:
    for f in [sin(pi * x**2 / 2), cos(pi * x**2 / 2), x * sqrt(2 * x + 3), (b + t)**(-a)]:
        var = t if f.has(t) else x
        assert is_antiderivative(_antiderivative(f, var), f, var) is True
    assert meijerint_indefinite(x**x**x, x) is None


def test_parameters_are_unpolarified() -> None:
    # the parameter 1 - a of (b + t)**(-a) at a = -1/2 was the polar
    # 1 - exp_polar(I*pi)/2 (SymPy's meijerg unpolarifies its parameters):
    # the antiderivative of x*sqrt(2*x + 3) came out wrong,
    # 4042*x**2*sqrt(2*x + 3)/81 + ...
    F = _antiderivative(x * sqrt(2 * x + 3), x)
    assert is_antiderivative(F, x * sqrt(2 * x + 3), x) is True
    g = _g([1 - exp_polar(I * pi) / 2], [], [0], [], x)
    assert g.an == (Rational(3, 2),)


def test_polar_number_argument_is_positive() -> None:
    # exp_polar(0) was not known positive: the condition of the integral
    # of exp(x) over (0, I) (a G-function of argument x*exp_polar(0)) came
    # out False, and SymPy's exp(I) - 1 was lost
    assert exp_polar(ZERO).is_positive is True
    value, cond = _definite(exp(x), x, ZERO, I)
    assert cond is true and _same(value, exp(I) - 1)


def test_product_matching() -> None:
    # the matcher of the core gives no binding for Heaviside(z - 1)*sqrt(z - 1)
    # against Heaviside(p*z**q - b)*(p*z**q - b)**(a - 1) (it misses z against
    # p*z**q once p is bound to 1): the factors are matched one by one
    from sympy_extras.core.symbol import Wild
    from sympy_extras.functions import heaviside
    p, q, w, v = (Wild(n, exclude=[z_dummy]) for n in 'pqab')
    pattern = heaviside(p * z_dummy**q - v) * (p * z_dummy**q - v)**(w - 1)
    found = _match(heaviside(z_dummy - 1) * sqrt(z_dummy - 1), pattern)
    assert found is not None and found[w] == Rational(3, 2) and found[v] == 1


def test_lookup_table() -> None:
    # every formula of the table against its G-functions, at a sample point
    # (SymPy's test_lookup_table)
    for entries in lookup_table().values():
        for entry in entries:
            values: dict[Symbol, Expr] = {}
            for w in sorted((v for v in entry.formula.free_symbols if isinstance(v, Symbol)), key=str):
                values[w] = Integer(3) if _has_properties(w) else Rational(17, 10)
            values[z_dummy] = Rational(17, 10)
            bindings: MatchDict = {k: v for k, v in values.items()}
            terms = entry.terms(bindings)
            expanded: Expr = ZERO
            for fac, g in terms:
                expanded = expanded + fac * g
            value = expanded.subs(values)
            if reliable_value(value, 15) is None:
                # exp_polar(-I*pi) in the argument of the entire G-function of exp
                value = value.subs(exp_polar(-I * pi), Integer(-1))
            assert _same(entry.formula.subs(values), value)


def _has_properties(w: Symbol) -> bool:
    """Whether ``w`` is a wild with properties (the positive integers)."""
    from sympy_extras.core.symbol import Wild
    return isinstance(w, Wild) and bool(w.properties)


def test_condsimp() -> None:
    from sympy_extras.core import Or, Eq
    assert _condsimp(Or(x < y, Eq(x, y))) == (x <= y)


def test_get_coeff_exp() -> None:
    assert _get_coeff_exp(a * x**b, x) == (a, b)
    assert _get_coeff_exp(2 * x, x) == (2, 1)


def test_saxena_and_antecedents() -> None:
    sr, tr, mu = _syms('s t mu', real=True)
    g1 = _g([], [], [0], [], sr * tr)
    g2 = _g([], [], [mu / 2], [-mu / 2], tr**2 / 4)
    r = _rewrite_saxena(Integer(1), Integer(1), g1, g2, tr)
    assert r is not None
    C, f1, f2 = r
    assert _same(C, sr / (4 * sqrt(pi)), {sr: Integer(2)})
    assert f1.argument == sr**2 * tr / 4 and f2.argument == tr / 4
    assert _check_antecedents(f1, f2, tr) != False
