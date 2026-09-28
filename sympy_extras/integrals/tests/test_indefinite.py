"""Tests of the verified indefinite integration driver."""
from __future__ import annotations

from sympy import symbols, sqrt, exp, sin, cos, log, tan, atan, asin, Integral, I, S, diff, simplify, erf

from sympy_extras._typing import as_expr
from sympy_extras.integrals import indefinite_integral, verified_antiderivative, is_antiderivative
from sympy_extras.integrals import indefinite as module

x = symbols('x')


def test_classic_integrands_by_the_typed_methods() -> None:
    # the typed methods answer these, SymPy's routes are never reached
    expected = {
        sqrt(1 - x**2): 'radicals', x / (x**2 + 1): 'rational', x * exp(x): 'exponential',
        tan(x)**3: 'trigonometric', 1 / (x * (log(x)**2 + 1)): 'risch', 1 / (x**3 + 1): 'rational',
        sqrt(x**2 + 1) / x: 'radicals', x**2 * atan(x): 'trigonometric', 1 / sqrt(x**2 + 1): 'radicals',
        x / sqrt(x**4 + 1): 'trager', exp(x) * sin(x): 'trigonometric', log(x)**2: 'risch'}
    for f, method in expected.items():
        found = verified_antiderivative(f, x)
        assert found is not None and found[1] == method, (f, found)
        assert simplify(diff(found[0], x) - f) == 0 or is_antiderivative(found[0], f, x) is True
    assert indefinite_integral(sqrt(1 - x**2), x) == x * sqrt(1 - x**2) / 2 + asin(x) / 2
    # the error function from the exponential table
    found = verified_antiderivative(exp(-x**2), x)
    assert found is not None and found[1] == 'exponential' and found[0].has(erf)
    # SymPy's methods stay the last resort, still checked
    found = verified_antiderivative(sin(x), x, methods=['sympy'])
    assert found == (-cos(x), 'sympy')


def test_is_antiderivative() -> None:
    assert is_antiderivative(log(x**2 + 1) / 2, x / (x**2 + 1), x) is True
    assert is_antiderivative(sin(x), sin(x), x) is False
    assert is_antiderivative(Integral(exp(-x**2), x), exp(-x**2), x) is None
    # the numerical fallback: an identity cancel does not see
    assert is_antiderivative(x * sqrt(1 - x**2) / 2 + asin(x) / 2, sqrt(1 - x**2), x) is True


def test_wrong_candidates_are_refused(monkeypatch: object) -> None:
    # a method returning a wrong antiderivative is not believed: the
    # driver moves on, and returns the Integral when nothing checks
    import pytest
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    wrong = [('wrong', lambda f, v, assumptions: as_expr(cos(v)))]
    monkeypatch.setattr(module, 'METHODS', wrong)
    assert indefinite_integral(sin(x), x) == Integral(sin(x), x)
    assert verified_antiderivative(sin(x), x) is None


def test_real_forms_are_preferred() -> None:
    # a candidate with I for a real integrand is rewritten when a real
    # antiderivative checks (the logarithms of conjugates as an arctangent)
    F = module.real_form(-I * log(x - I) / 2 + I * log(x + I) / 2, 1 / (x**2 + 1), x)
    assert F == atan(x)
    # the sign matters: this pair is an antiderivative of -1/(x**2 + 1), and
    # no real rewriting checks against 1/(x**2 + 1), so the form is kept
    kept = module.real_form(I * log(x - I) / 2 - I * log(x + I) / 2, 1 / (x**2 + 1), x)
    assert kept.has(I)
    assert not indefinite_integral(1 / (x**2 + 1), x).has(I)
    assert not indefinite_integral(1 / (x**4 + 1), x).has(I)


def test_antiderivatives_wrong_for_negative_x_are_refused() -> None:
    # the census of SymPy's mistakes: -asinh(1/x) is an antiderivative of
    # 1/(x*sqrt(x**2 + 1)) for x > 0 only (Maxima's reference is
    # -asinh(1/Abs(x))), and the sampler of the check drew positive
    # points only
    from sympy import asinh
    assert is_antiderivative(-asinh(1 / x), 1 / (x * sqrt(x**2 + 1)), x) is False
    assert is_antiderivative(-asinh(1 / x), 1 / (x * sqrt(x**2 + 1)), x, [x > 0]) is not False
    found = indefinite_integral(1 / (x * sqrt(x**2 + 1)), x)
    assert found.has(Integral) or is_antiderivative(found, 1 / (x * sqrt(x**2 + 1)), x) is True


def test_complex_form_candidates_need_both_signs() -> None:
    # the Meijer route gives exp(a*z**r) a polar incomplete gamma which is
    # an antiderivative for z > 0 and the negative of one for z < 0 (with
    # r = 2 both sides are real); the check draws r non-integer, so z < 0
    # is complex and skipped, and the candidate passed on z > 0 alone
    from sympy import exp_polar, gamma, lowergamma, pi
    a, r, z = symbols('a r z')
    polar = exp(-I * pi / r) * gamma(1 / r) * lowergamma(1 / r, a * z**r * exp_polar(I * pi)) \
        / (a**(1 / r) * r**2 * gamma(1 + 1 / r))
    assert is_antiderivative(polar, exp(a * z**r), z) is not True
    assert is_antiderivative(polar.subs(r, 2), exp(a * z**2), z) is False
    found = indefinite_integral(exp(a * z**r), z)
    assert not found.has(exp_polar)
    # a real-form candidate right on one side of a one-sided integrand is still accepted
    assert is_antiderivative(x * log(x) - x, log(x), x) is True


def test_the_census_cases_are_never_returned_wrong() -> None:
    # the census of SymPy 1.14 on 1165 published indefinite integrals
    # (sympy-extras-benchmarks, indefinite_integrals): nine wrong answers
    # from the Meijer G route, right for x > 0 and wrong for x < 0, and a
    # nan from manualintegrate; the driver either verifies or declines
    from sympy import besselk, asin, nan
    cases = [exp(-x**3), x * exp(-x**3), 1 / (x * sqrt(x**2 + 1)), 1 / (x * sqrt(x**2 - 1)), besselk(7, x),
             x**2 * asin(sqrt(1 - x**2) / (3 * sqrt(S(1) / 9 - x**2 / 9)))]
    for f in cases:
        found = indefinite_integral(f, x)
        assert not found.has(nan), f
        assert found.has(Integral) or is_antiderivative(found, f, x) is True, (f, found)


def test_rewriting_and_substitutions_are_tried() -> None:
    # fractional powers of x go through x = t**k, powers of positive bases
    # and hyperbolic functions through exponentials, all by the typed
    # methods and verified against the original integrand
    from sympy import cosh, S
    # (Trager's algorithm, tried first, integrates it on the curve y**6 = x
    # since the logarithmic part of the roots of index n; the rewriting
    # route is the one which does it without that)
    found = verified_antiderivative(1 / (x**(S(1) / 3) + sqrt(x)), x)
    assert found is not None and found[1] in ('trager', 'rewriting')
    assert is_antiderivative(found[0], 1 / (x**(S(1) / 3) + sqrt(x)), x) is True
    found = verified_antiderivative(1 / (x**(S(1) / 3) + sqrt(x)), x, methods=['rewriting'])
    assert found is not None and found[1] == 'rewriting' and is_antiderivative(found[0], 1 / (x**(S(1) / 3) + sqrt(x)), x) is True
    found = verified_antiderivative(2**x * cosh(x), x)
    assert found is not None and found[1] in ('heurisch', 'rewriting')
    assert is_antiderivative(found[0], 2**x * cosh(x), x) is True


def test_the_numerical_check_survives_huge_values() -> None:
    # exp(A*x**r) at a sampled point is 1e300 and more: abs() of a Python
    # complex overflowed in numerically_equal, and a right antiderivative
    # of the incomplete gamma family was refused
    from sympy import Float
    from sympy_extras.integrals.conditions import numerically_equal
    huge = Float(3e307) * (1 + I)
    assert numerically_equal(huge, huge) is True
    assert numerically_equal(huge, Float(2e307) * (1 + I)) is False


def test_the_assumptions_reach_the_typed_methods() -> None:
    # Maxima's rtestint 20 and 29 record their signs as facts, not as
    # assumptions on the symbols; the radical table decides its cases from them
    a, b, c = symbols('a b c')
    Q = a + b * x + c * x**2
    facts = [a > 0, b > 0, c > 0, 4 * a * c - b**2 > 0]
    for f in (sqrt(Q) / x, 1 / (x**2 * sqrt(Q))):
        found = verified_antiderivative(f, x, facts)
        assert found is not None and found[1] == 'radicals', f
        assert is_antiderivative(found[0], f, x, facts) is True
    # without the facts Trager's algorithm answers, over the parameters
    # made rational (a = alpha**2, c = gamma**2), in a form complex where
    # a or c is negative and right there too
    F = indefinite_integral(sqrt(Q) / x, x)
    assert not F.has(Integral) and is_antiderivative(F, sqrt(Q) / x, x, facts) is True


def test_nonelementary_binomial_differentials() -> None:
    from sympy import Rational, cbrt, hyper, im
    from sympy_extras._numeric import reliable_value
    # sqrt(t**7/(1 - 5*t**2)) came back unevaluated: its integral is not
    # elementary (Chebyshev), and the hypergeometric form was not tried
    f = sqrt(x**7 / (1 - 5 * x**2))
    F = indefinite_integral(f, x)
    assert not F.has(Integral) and F.has(hyper)
    # real and an antiderivative on both intervals where f is real
    for point in [Rational(1, 5), Rational(1, 3), Rational(-1, 2), -2]:
        difference = reliable_value(as_expr(F.diff(x) - f), 30, {x: point})
        value = reliable_value(F, 30, {x: point})
        assert difference is not None and abs(difference) < 1e-25, point
        assert value is not None and abs(as_expr(im(value))) < 1e-25, point
    found = verified_antiderivative(sqrt(1 + x**3), x)
    assert found is not None and found[1] == 'binomial'
    # the elementary binomial differentials keep their elementary antiderivatives
    for g in [x**3 * sqrt(1 + x**2), 1 / (x * sqrt(1 + x**3)), sqrt(x) / (1 + cbrt(x))]:
        G = indefinite_integral(g, x)
        assert not G.has(Integral) and not G.has(hyper), g


def test_binomial_differentials_with_parameters_and_their_special_values() -> None:
    from sympy import Eq, Piecewise, Rational, cbrt, hyper
    from sympy_extras._numeric import reliable_value
    a, b = symbols('a b')
    # sqrt(t**7/(a - b*t**2)) and x**(1/3)*sqrt(a + x**2) came back
    # unevaluated (the binomial coefficients had to be numbers); at a = 0,
    # where z = b*x**2/a is undefined, the integrand is a monomial and gets
    # its own case, and at b = 0 the generic form holds (z = 0)
    for f in [sqrt(x**7 / (a - b * x**2)), cbrt(x) * sqrt(a + x**2)]:
        F = indefinite_integral(f, x)
        assert isinstance(F, Piecewise) and F.has(hyper), f
        special, condition = as_expr(F.args[0].args[0]), F.args[0].args[1]
        assert condition == Eq(a, 0), F
        assert not special.has(hyper)
        at_zero = as_expr(f.subs(a, 0))
        for values in [{a: Rational(1, 2), b: Rational(5, 4)}, {a: Rational(-1, 2), b: Rational(5, 4)},
                       {a: Rational(1, 2), b: Rational(-5, 4)}, {a: Rational(-3), b: Rational(-5, 4)},
                       {a: Rational(3), b: S.Zero}]:
            g, G = as_expr(f.subs(values)), as_expr(F.subs(values))
            for point in [Rational(-2), Rational(-1, 3), Rational(1, 3), Rational(2)]:
                if reliable_value(g, 30, {x: point}) is None:
                    continue
                difference = reliable_value(as_expr(G.diff(x) - g), 30, {x: point})
                assert difference is not None and abs(difference) < 1e-25, (f, values, point)
        # (where the integrand at a = 0 is real, as the checks of the
        # package: x**(1/3)*sqrt(x**2) is not for x < 0)
        for point in [Rational(1, 3), Rational(2)]:
            difference = reliable_value(as_expr(special.diff(x) - at_zero), 30, {b: Rational(-5, 4), x: point})
            assert difference is not None and abs(difference) < 1e-25, (f, point)


def test_an_enclosing_limit_stops_the_integration() -> None:
    # the bug: under an enclosing limit of half a second (the census runs
    # every integral under one), each inner attempt() took the expiry for
    # its method failing, and indefinite_integral returned the integral
    # unevaluated -- a verdict of "cannot" for an integral it solves in
    # twenty seconds -- instead of stopping; the expiry now goes to the
    # owner of the enclosing limit
    import time
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    a, b, c = symbols('a b c')
    f = sqrt((a + b*x + c*x**2)**3) / x**2
    facts = [4*a*c - b**2 > 0, c > 0, a > 0, b > 0]
    started = time.monotonic()
    assert attempt(lambda: indefinite_integral(f, x, facts), 0.5) is None
    assert time.monotonic() - started < 3 * settings.time_scale
