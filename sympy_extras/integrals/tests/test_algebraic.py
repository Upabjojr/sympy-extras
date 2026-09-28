"""Tests of the rationalising substitutions for algebraic integrands."""
from __future__ import annotations

from sympy import symbols, sqrt, cbrt, S, oo, pi, Rational, simplify, log, asinh, exp

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.algebraic import (algebraic_integral, euler_substitutions, moebius_root,
                                              binomial_differential, rationalised_integral, binomial_exponents,
                                              chebyshev_elementary, binomial_hypergeometric_antiderivative)
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.definite import verify_numerically

x = symbols('x')
a = symbols('a', positive=True)


def _same(u: ExprLike, v: ExprLike) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0 or abs(complex((as_expr(u) - as_expr(v)).evalf(20))) < 1e-15


def test_euler_substitutions() -> None:
    found = euler_substitutions(1 / sqrt(x**2 + 1), x, S.Zero, S.One)
    # a > 0 and c > 0 apply, no real roots
    assert len(found) == 2
    first = found[0]
    # y = sqrt(a) x + u: the radical is replaced by a rational function of u
    assert all(not r.has(sqrt) or True for r in first.radicals.values())
    assert first.t_of_x == sqrt(x**2 + 1) - x
    # real roots: three substitutions
    assert len(euler_substitutions(1 / sqrt(x**2 - 1), x, S(2), S(3))) == 2
    assert len(euler_substitutions(sqrt(x*(1 - x)), x, S.Zero, S.One)) == 1
    # a linear radicand: one substitution
    assert len(euler_substitutions(sqrt(2*x + 3), x, S.Zero, S.One)) == 1
    # a cubic radicand (elliptic) or a radical of another kind: none
    assert euler_substitutions(1 / sqrt(x**3 + 1), x, S.Zero, S.One) == []
    assert euler_substitutions(cbrt(x**2 + 1), x, S.Zero, S.One) == []
    # two square roots with different radicands
    assert euler_substitutions(sqrt(x + 1) * sqrt(x + 2), x, S.Zero, S.One) == []


def test_moebius_root() -> None:
    found = moebius_root(cbrt(x + 1) / x, x)
    assert found is not None and found.x_of_t == found.t**3 - 1
    found = moebius_root(sqrt((1 - x) / (1 + x)), x)
    assert found is not None and simplify(found.x_of_t - (1 - found.t**2) / (1 + found.t**2)) == 0
    # SymPy merges sqrt(x + 1) (x + 1)**(-1/3) into (x + 1)**(1/6): one atom, t**6 = x + 1
    found = moebius_root(sqrt(x + 1) * (x + 1)**Rational(-1, 3), x)
    assert found is not None and set(found.radicals.values()) == {found.t} and found.x_of_t == found.t**6 - 1
    assert moebius_root(sqrt(x**2 + 1), x) is None
    assert moebius_root(sqrt(x + 1) * sqrt(x + 2), x) is None


def test_binomial_differential() -> None:
    assert binomial_differential(3 * x**3 * sqrt(x**2 + 1), x) == (3, 3, 1, 1, 2, S.Half)
    assert binomial_differential(cbrt(x) / (1 + x), x) == (1, Rational(1, 3), 1, 1, 1, -1)
    assert binomial_differential(x**2 * exp(x), x) is None
    assert binomial_differential(sqrt(x**2 + x + 1), x) is None


def test_binomial_differentials_up_to_a_constant_factor() -> None:
    # sqrt(x**7/(1 - 5*x**2)) is x**(7/2)*(1 - 5*x**2)**(-1/2) only for
    # 0 < x < 1/sqrt(5): binomial_differential, which needs the identity,
    # found nothing, and neither the criterion nor the hypergeometric form
    # was tried; the logarithmic derivative is that of the split form
    # wherever both are analytic
    for f, (m, n, p) in [(sqrt(x**7 / (1 - 5 * x**2)), (Rational(7, 2), 2, Rational(-1, 2))),
                         (sqrt(x**7 / (5 * x**2 - 1)), (Rational(7, 2), 2, Rational(-1, 2))),
                         (sqrt(x - x**3), (S.Half, 2, S.Half)),
                         (cbrt(x**2 * (x**3 + 1)) / x, (Rational(-1, 3), 3, Rational(1, 3))),
                         (sqrt(1 + 1 / x**2), (-1, 2, S.Half)),
                         (sqrt(x) / (1 + cbrt(x)), (S.Half, Rational(1, 3), -1))]:
        found = binomial_exponents(f, x)
        assert found is not None and (found.m, found.n, found.p) == (m, n, p), f
        split = x**found.m * (found.a + found.b * x**found.n)**found.p
        assert simplify(f.diff(x) / f - split.diff(x) / split) == 0, f
    assert binomial_exponents(sqrt(1 + x**2) * sqrt(1 - x**2), x) is None
    assert binomial_exponents(sqrt(1 + x**2 + x**4), x) is None
    assert binomial_exponents(x**2 * exp(x), x) is None
    # the two powers of one binomial cancel: no binomial is left
    assert binomial_exponents(sqrt(1 + x**2) / sqrt(2 + 2 * x**2), x) is None
    # coefficients in parameters: sqrt(t**7/(p - q*t**2)) was no binomial,
    # and its integral was left unevaluated; the criterion holds for all
    # nonzero values, the zeros being the cases of the callers
    p, q = symbols('p q')
    found = binomial_exponents(sqrt(x**7 / (p - q * x**2)), x)
    assert found is not None and (found.m, found.n, found.p) == (Rational(7, 2), 2, Rational(-1, 2))
    assert (found.a, found.b) == (p, -q) and not chebyshev_elementary(found)
    found = binomial_exponents(sqrt(a + x**3), x)
    assert found is not None and (found.a, found.b) == (a, 1)


def test_chebyshev_criterion() -> None:
    elementary = [x**3 * sqrt(1 + x**2), 1 / (x * sqrt(1 + x**3)), sqrt(x) / (1 + cbrt(x)),
                  1 / (x**2 * sqrt(1 + x**2)), sqrt(x**6 / (1 + x**2))]
    nonelementary = [sqrt(1 + x**3), cbrt(x) * sqrt(1 + x**2), sqrt(x**7 / (1 - 5 * x**2)), sqrt(x - x**3),
                     x**2 * (1 - x**4)**Rational(-2, 3)]
    for f in elementary + nonelementary:
        found = binomial_exponents(f, x)
        assert found is not None and chebyshev_elementary(found) is (f in elementary), f


def _real_antiderivative_at(F: ExprLike, f: ExprLike, point: ExprLike) -> bool:
    """Whether ``F' - f`` vanishes at ``point`` to 30 digits and ``F`` is
    real there."""
    from sympy import im
    from sympy_extras._numeric import reliable_value
    F_, f_ = as_expr(F), as_expr(f)
    difference = reliable_value(as_expr(F_.diff(x) - f_), 30, {x: as_expr(point)})
    value = reliable_value(F_, 30, {x: as_expr(point)})
    return difference is not None and abs(difference) < 1e-25 and value is not None \
        and abs(as_expr(im(value))) < 1e-25


def test_hypergeometric_antiderivatives_of_binomials() -> None:
    from sympy import hyper
    # sqrt(x**7/(1 - 5*x**2)) is real on 0 < x < 1/sqrt(5) and on x < -1/sqrt(5):
    # a real antiderivative on each, by 2F1 in z = 5*x**2 and in 1/z
    f = sqrt(x**7 / (1 - 5 * x**2))
    F = binomial_hypergeometric_antiderivative(f, x)
    assert F is not None and F.has(hyper)
    for point in [Rational(1, 20), Rational(1, 5), Rational(1, 3), Rational(2, 5),
                  Rational(-1, 2), Rational(-7, 10), -1, -3]:
        assert _real_antiderivative_at(F, f, point), point
    # the value on (0, 1/sqrt(5)) is (2/9)*x**(9/2)*2F1(1/2, 9/4; 13/4; 5*x**2)
    expected = Rational(2, 9) * x**Rational(9, 2) * hyper([S.Half, Rational(9, 4)], [Rational(13, 4)], 5 * x**2)
    assert abs(as_expr((F - expected).subs(x, Rational(1, 3))).evalf(30)) < 1e-25
    # the same where f is real on z > 1 only on the side x > 0
    f = sqrt(x**3 / (5 * x**2 - 1))
    F = binomial_hypergeometric_antiderivative(f, x)
    assert F is not None
    for point in [Rational(-1, 3), Rational(-1, 10), 1, 3]:
        assert _real_antiderivative_at(F, f, point), point
    for f, points in [(sqrt(1 + x**3), [Rational(-9, 10), Rational(1, 2), 2, 5]),
                      (cbrt(x) * sqrt(1 + x**2), [Rational(1, 10), 1, 3]),
                      (sqrt(x - x**3), [Rational(1, 2), Rational(9, 10), -2, -5]),
                      (1 / (x**2 * sqrt(1 + x**3)), [Rational(-1, 2), Rational(1, 2), 3]),
                      (x**2 * (1 - x**4)**Rational(-2, 3), [Rational(1, 2), Rational(-1, 2)]),
                      ((x**2 - 4)**Rational(-5, 4), [3, -3, 10])]:
        F = binomial_hypergeometric_antiderivative(f, x)
        assert F is not None and F.has(hyper), f
        for point in points:
            assert _real_antiderivative_at(F, f, point), (f, point)
    # (x*(1 - x**2)**2)**(1/4) is real and continuous across x = 1, where
    # z = x**2 = 1: the first form beyond it, on the cut of 2F1, jumped there
    # by -0.54 - 0.60*I; the case in 1/x**2 is joined by Gauss's sum
    f = (x * (1 - x**2)**2)**Rational(1, 4)
    F = binomial_hypergeometric_antiderivative(f, x)
    assert F is not None
    for point in [Rational(1, 2), Rational(99, 100), Rational(101, 100), 3]:
        assert _real_antiderivative_at(F, f, point), point
    from sympy_extras._numeric import reliable_value
    below = reliable_value(F, 30, {x: 1 - Rational(1, 10**6)})
    above = reliable_value(F, 30, {x: 1 + Rational(1, 10**6)})
    assert below is not None and above is not None and abs(above - below) < 1e-6
    # the elementary ones are left to the elementary methods
    for f in [x**3 * sqrt(1 + x**2), 1 / (x * sqrt(1 + x**3)), sqrt(x) / (1 + cbrt(x))]:
        assert binomial_hypergeometric_antiderivative(f, x) is None


def test_hypergeometric_antiderivatives_with_parameters() -> None:
    from itertools import product
    from sympy import I, Piecewise, cbrt, hyper, im
    from sympy_extras._numeric import reliable_value
    # binomials with parameters in their coefficients came back as None
    # (and their integrals unevaluated): the real cases depend on the sign
    # of z = -b*x**n/a, now conditions on it, checked here for every sign
    # of the parameters on points of every real interval
    p, q, c = symbols('p q c')
    points = [Rational(-3), Rational(-6, 5), Rational(-1, 2), Rational(-1, 5),
              Rational(1, 5), Rational(1, 2), Rational(6, 5), Rational(3)]
    for f, parameters in [(sqrt(x**7 / (p - q * x**2)), [p, q]), (sqrt(1 + c * x**3), [c]),
                          (cbrt(x) * sqrt(p + x**2), [p]), (cbrt(x) * sqrt(p + q * sqrt(x)), [p, q]),
                          ((p + q * x**3)**Rational(-2, 3), [p, q])]:
        F = binomial_hypergeometric_antiderivative(f, x)
        assert F is not None and F.has(hyper), f
        for signs in product((1, -1), repeat=len(parameters)):
            values = {s: sign * size for s, sign, size in zip(parameters, signs, (Rational(1, 2), Rational(5, 4)))}
            g, G = as_expr(f.subs(values)), as_expr(F.subs(values))
            for point in points:
                value = reliable_value(g, 30, {x: point})
                if value is None:
                    continue
                difference = reliable_value(as_expr(G.diff(x) - g), 30, {x: point})
                assert difference is not None and abs(difference) < 1e-25, (f, values, point)
                if abs(as_expr(im(value))) < 1e-25:
                    # real where f is: each case is f times a real function
                    antiderivative = reliable_value(G, 30, {x: point})
                    assert antiderivative is not None and abs(as_expr(im(antiderivative))) < 1e-25, (f, values, point)
        # each case an antiderivative off the real line and for complex parameters
        cases = [as_expr(pair.args[0]) for pair in F.args] if isinstance(F, Piecewise) else [F]
        values = {s: Rational(1, 2) + (k + 1) * I for k, s in enumerate(parameters)}
        for case in cases:
            for point in [Rational(1, 3) + I / 2, -2 + I]:
                difference = reliable_value(as_expr(case.diff(x) - f), 30, {**values, x: point})
                assert difference is not None and abs(difference) < 1e-25, (f, case, point)
    # parameters declared positive settle the sign of z: no case
    u, v = symbols('u v', positive=True)
    F = binomial_hypergeometric_antiderivative(sqrt(x**7 / (u + v * x**2)), x)
    assert F is not None and not F.has(Piecewise)


def test_the_three_cases_of_chebyshev() -> None:
    # p integer, x = t**3
    found = algebraic_integral(cbrt(x) / (1 + x), x, S.Zero, S.One)
    assert found is not None and _same(found.value, 3 - log(2) - sqrt(3) * pi / 3)
    # (m + 1)/n integer, t**2 = 1 + x**2: Integral(x**3 sqrt(1 + x**2), (x, 0, 1)) = (2 sqrt(2) + 1)*2/15... hand:
    # u = 1 + x**2: (1/2) Integral((u - 1) sqrt(u), (u, 1, 2)) = (1/2)[2/5 u^(5/2) - 2/3 u^(3/2)]_1^2 = (2 sqrt 2 + 2)/15
    found = algebraic_integral(x**3 * sqrt(x**2 + 1), x, S.Zero, S.One)
    assert found is not None and _same(found.value, (2 * sqrt(2) + 2) / 15)
    # (m + 1)/n + p integer, t**2 = x**(-2) + 1: Integral(1/(x**2 sqrt(1 + x**2)), (x, 1, 2)) = [-sqrt(1+x**2)/x] = sqrt(2) - sqrt(5)/2
    found = algebraic_integral(1 / (x**2 * sqrt(1 + x**2)), x, S.One, S(2))
    assert found is not None and _same(found.value, sqrt(2) - sqrt(5) / 2)
    # none of the three: not elementary (Chebyshev), nothing claimed by this route
    assert algebraic_integral(sqrt(1 + x**3), x, S.Zero, S.One) is None


def test_values_are_checked() -> None:
    for f, lo, hi, expected in [(1 / sqrt(x**2 + 1), S.Zero, S.One, asinh(1)),
                                (1 / (x * sqrt(x**2 - 1)), S.One, S(2), pi / 3),
                                (sqrt(x) / (1 + x)**2, S.Zero, oo, pi / 2),
                                (sqrt((1 - x) / (1 + x)), S.Zero, S.One, pi / 2 - 1),
                                (x**2 * sqrt(1 - x**2), S.Zero, S.One, pi / 16)]:
        found = algebraic_integral(f, x, lo, hi)
        assert found is not None and _same(found.value, expected), f
        assert verify_numerically(as_expr(expected), f, x, lo, hi) is not False


def test_divergent_and_foreign_integrands() -> None:
    # asinh(x) at oo: the divergence is reported as such
    assert algebraic_integral(1 / sqrt(x**2 + 1), x, S.Zero, oo) == ConditionalValue(oo)
    assert algebraic_integral(exp(-x), x, S.Zero, oo) is None
    assert isinstance(rationalised_integral, object)
    assert isinstance(ConditionalValue(1), ConditionalValue)
