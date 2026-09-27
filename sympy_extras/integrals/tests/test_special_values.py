"""The isolated values of the parameters of definite and indefinite
integrals get cases of their own (:mod:`sympy_extras._special_values`):
each is checked against the integral computed at the point, by the
quadrature or by hand, and the generic branch against the quadrature."""
from __future__ import annotations

from sympy import (Abs, DiracDelta, Eq, Integral, Piecewise, Rational, S, cos, diff, exp, limit, log, oo, pi, sin,
                   simplify, symbols)
from sympy.core.expr import Expr

from sympy_extras._typing import as_expr
from sympy_extras.integrals import definite_integral, indefinite_integral, verify_numerically

x, y, k, a, b, c, n = symbols('x y k a b c n')


def _has_case(value: Expr, condition: object) -> bool:
    return isinstance(value, Piecewise) and any(pair.args[1] == condition for pair in value.args)


def test_the_integral_of_a_power_of_y_at_y_1() -> None:
    # the bug: (y**2 - 1)/log(y), which is 0/0 at y = 1, where the
    # integral is 2
    value = definite_integral(y**x, (x, 0, 2))
    assert value.subs(y, 1) == 2
    assert _has_case(value, Eq(y, 1))
    assert value.subs(y, 3) == Rational(8) / log(3)
    assert verify_numerically(value, y**x, x, S.Zero, S(2)) is True


def test_an_exponential_at_a_vanishing_rate() -> None:
    # the bug: (exp(k) - 1)/k, undefined at k = 0, where the integral is 1
    value = definite_integral(exp(k * x), (x, 0, 1))
    assert value.subs(k, 0) == 1
    assert value.subs(k, 1) == exp(1) - 1
    value = definite_integral(x * exp(k * x), (x, 0, 1))
    assert value.subs(k, 0) == Rational(1, 2)
    # the limit agrees here, as a cross-check of the value at the point
    generic = [pair.args[0] for pair in value.args if pair.args[1] == S.true][0]
    assert limit(generic, k, 0) == Rational(1, 2)
    # the assumptions exclude the point: no case
    assert not definite_integral(exp(k * x), (x, 0, 1), k > 0).has(Piecewise)


def test_two_parameters() -> None:
    # sin(a*x)*sin(b*x) over (0, pi): SymPy's cases a = b and a = -b are
    # kept, and b = 0, where their values are 0/0, gets its own
    f = sin(a * x) * sin(b * x)
    value = definite_integral(f, (x, 0, pi))
    assert value.subs({a: 0, b: 0}) == 0
    assert value.subs({a: 2, b: 0}) == 0
    assert value.subs({a: 1, b: 1}) == pi / 2
    assert value.subs({a: -1, b: 1}) == -pi / 2
    assert verify_numerically(value, f, x, S.Zero, pi) is True
    # a hyperplane found by the package: exp((a - b)*x) at a = b
    value = definite_integral(exp(a * x) * exp(-b * x), (x, 0, 1))
    assert value.subs({a: 2, b: 2}) == 1


def test_the_value_at_the_point_is_not_the_limit() -> None:
    # k/(k**2 + x**2) over (0, 1) is atan(1/k) for k > 0 and -atan(1/k)
    # for k < 0: the limits at k = 0 are pi/2 and -pi/2, the integral
    # of 0 there is 0
    value = definite_integral(k / (k**2 + x**2), (x, 0, 1))
    assert value.subs(k, 0) == 0
    generic = [pair.args[0] for pair in value.args if not pair.args[1].has(Eq) and not pair.args[0].has(Integral)][0]
    assert limit(generic, k, 0, '+') == pi / 2


def test_points_between_the_cases_of_a_singularity() -> None:
    # the bug: log(Abs(x - c)) over (0, 1) was unevaluated at c = 0 and
    # c = 1, between the cases c < 0, 0 < c < 1 and c > 1 (the values
    # there, c*log(-c) and the like, are 0*log(0))
    value = definite_integral(log(Abs(x - c)), (x, 0, 1))
    assert value.subs(c, 0) == -1 and value.subs(c, 1) == -1


def test_divergence_beyond_the_condition() -> None:
    # the bug: SymPy's gamma(n + 1)/gamma(n + 2), unconditional, replaced
    # 1/(n + 1) for n > -1 (checked at n > 0 only): a finite value for
    # n < -1, where the integral diverges
    value = definite_integral(x**n, (x, 0, 1))
    assert value.subs(n, Rational(-3, 2)).has(Integral)
    # n = -1 lies on the boundary of the divergence n <= -1, not between
    # cases: no case of its own
    assert not value.has(Eq) and value.subs(n, -1).has(Integral)
    assert value.subs(n, 1) == Rational(1, 2)


def test_the_cases_can_be_turned_off() -> None:
    assert definite_integral(y**x, (x, 0, 2), special_values=False) == (y**2 - 1) / log(y)
    assert indefinite_integral(exp(k * x), x, special_values=False) == exp(k * x) / k


def test_numerical_verification_checks_the_isolated_values() -> None:
    # a wrong value at the point, which no sample of a region reaches
    wrong = Piecewise((3, Eq(y, 1)), ((y**2 - 1) / log(y), True))
    assert verify_numerically(as_expr(wrong), y**x, x, S.Zero, S(2)) is False
    # the value of sin(a*x)*sin(b*x) wrong at a = b = 0 only
    wrong = Piecewise((1, Eq(a, 0) & Eq(b, 0)), (pi / 2 - sin(2 * pi * b) / (4 * b), Eq(a, b)),
                      ((-a * sin(pi * b) * cos(pi * a) + b * sin(pi * a) * cos(pi * b)) / (a**2 - b**2), True))
    assert verify_numerically(as_expr(wrong), sin(a * x) * sin(b * x), x, S.Zero, pi) is False


def test_indefinite_integrals_in_sympys_form() -> None:
    F = indefinite_integral(exp(k * x), x)
    # SymPy's form: Piecewise((exp(k*x)/k, Ne(k, 0)), (x, True))
    assert isinstance(F, Piecewise) and F.args[1].args == (x, True)
    assert F.subs(k, 0) == x and diff(F.subs(k, 2), x) == exp(2 * x)
    F = indefinite_integral(x * exp(k * x), x)
    assert F.subs(k, 0) == x**2 / 2
    # several cases, each under its equations, the most specific first
    f = sin(a * x) * sin(b * x)
    F = indefinite_integral(f, x)
    assert isinstance(F, Piecewise) and F.args[0].args[1] == Eq(a, 0) & Eq(b, 0)
    for point in ({a: 0, b: 0}, {a: 1, b: 1}, {a: -1, b: 1}, {a: 2, b: 3}):
        assert simplify(diff(F.subs(point), x) - f.subs(point)) == 0


def test_no_case_where_the_problem_is_undefined() -> None:
    z = symbols('z')
    # the bug: Piecewise((Ei(b*z*log(a)), Ne(a, 0)), (Integral(0**(b*z)/z, z), True)):
    # at a = 0 the integrand 0**(b*z) is 0 or zoo according to the sign of
    # b*z, no function of z to integrate, and the unevaluated case made the
    # whole answer unevaluated
    F = indefinite_integral(a**(b * z) / z, z)
    assert not F.has(Integral)
    assert simplify((diff(F, z) - a**(b * z) / z).subs({a: 2, b: 3}).rewrite(exp)) == 0
    # the bug: a case for DiracDelta(0), the integrand at a = 0, which is
    # no function either: Piecewise((Integral(exp(-u*v)*DiracDelta(0), (u, 0, oo)), Eq(a, 0)), ...)
    u, v = symbols('u v')
    value = definite_integral(exp(-u * v) * DiracDelta(a * u), (u, 0, oo))
    assert not value.has(Integral)


def test_the_cases_of_an_antiderivative_are_computed() -> None:
    e = symbols('e')
    # the bug: a case Integral(0**(1/x)/x**3, x) at e = 0, where e**(1/x) is
    # undefined for x < 0, computed for seventeen seconds, and under the
    # twenty seconds of the census Piecewise((..., Ne(e, 1)),
    # (Integral(x**(-3), x), True)): e = 0 had taken the time of e = 1
    F = indefinite_integral(e**(1 / x) / x**3, x)
    assert not F.has(Integral)
    assert F.subs(e, 1) == -1 / (2 * x**2)
    assert simplify(diff(F.subs(e, 2), x) - 2**(1 / x) / x**3) == 0

