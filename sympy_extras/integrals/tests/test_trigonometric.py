"""Tests of the trigonometric and hyperbolic integrator."""
from __future__ import annotations

from sympy import (symbols, sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch, exp, log,
                   asin, acos, atan, acot, asinh, acosh, atanh, sqrt, diff, cancel, simplify, Piecewise, Ne, S,
                   Rational, Expr, Integral, integrate)
from sympy.logic.boolalg import Boolean

from sympy_extras._typing import as_expr
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.trigonometric import trigonometric_antiderivative, _sin_cos, _sinh_cosh

x = symbols('x')
a, b = symbols('a b', positive=True)


def _same(u: object, v: object) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0


def _checks(f: Expr, facts: list[Boolean] = []) -> bool:
    F = trigonometric_antiderivative(f, x)
    if F is None:
        return False
    if isinstance(F, Piecewise):
        F = as_expr(F.args[0].args[0])
    difference = as_expr(cancel(diff(F, x) - f))
    return difference == 0 or simplify(difference) == 0 or numerically_equal(diff(F, x), f, facts)


def test_powers_of_sine_and_cosine() -> None:
    assert _same(trigonometric_antiderivative(sin(x)**2, x), x / 2 - sin(x) * cos(x) / 2)
    assert _same(trigonometric_antiderivative(sin(x)**3 * cos(x)**2, x), cos(x)**5 / 5 - cos(x)**3 / 3)
    assert trigonometric_antiderivative(sin(x)**(-2), x) == -cot(x)
    for n in range(-4, 5):
        for m in range(-4, 5):
            if (n, m) != (0, 0):
                assert _checks(sin(x)**n * cos(x)**m), (n, m)
    # a frequency and a coefficient
    assert _checks(3 * sin(2 * x)**4 * cos(2 * x)**2) and _checks(sin(a * x)**3 / cos(a * x))
    # the raising and lowering formulas from d(sin**(n+1) cos**(m+1))
    u = symbols('u')
    for n, m in [(-2, -2), (-4, 2), (2, -4), (4, 4), (0, -6)]:
        F = _sin_cos(n, m, u)
        # simplify does not always see the identity: sin**6 - 3 sin**4 + 3 sin**2 is 1 - cos**6
        assert F is not None and numerically_equal(diff(F, u), sin(u)**n * cos(u)**m, [u > 0, u < 1]), (n, m)


def test_tangent_secant_families() -> None:
    assert trigonometric_antiderivative(tan(x)**3, x) == log(cos(x)) + tan(x)**2 / 2
    assert trigonometric_antiderivative(sec(x), x) == log(tan(x) + sec(x))
    assert trigonometric_antiderivative(csc(x), x) == log(tan(x / 2))
    assert _same(trigonometric_antiderivative(sec(x)**3, x), log(tan(x) + sec(x)) / 2 + tan(x) * sec(x) / 2)
    assert _same(trigonometric_antiderivative(cot(x)**4, x), x - cot(x)**3 / 3 + cot(x))
    for k in range(1, 6):
        for g in (tan, cot, sec, csc):
            assert _checks(g(x)**k), (g, k)
    assert _checks(tan(2 * x)**2 * sec(2 * x)**2) and _checks(sin(x) * tan(x))


def test_hyperbolic_families() -> None:
    assert _same(trigonometric_antiderivative(sinh(x)**2, x), sinh(x) * cosh(x) / 2 - x / 2)
    assert _same(trigonometric_antiderivative(tanh(x)**3, x), log(cosh(x)) - tanh(x)**2 / 2)
    assert trigonometric_antiderivative(sech(x), x) == atan(sinh(x))
    assert trigonometric_antiderivative(csch(x), x) == log(tanh(x / 2))
    assert _same(trigonometric_antiderivative(sinh(x)**3 * cosh(x)**2, x), cosh(x)**5 / 5 - cosh(x)**3 / 3)
    for n in range(-3, 4):
        for m in range(-3, 4):
            if (n, m) != (0, 0):
                assert _checks(sinh(x)**n * cosh(x)**m, [x > 0]), (n, m)
    for k in range(1, 5):
        for g in (tanh, coth, sech, csch):
            assert _checks(g(x)**k, [x > 0]), (g, k)
    u = symbols('u')
    for n, m in [(-2, -2), (-4, 2), (2, -4), (4, 4)]:
        F = _sinh_cosh(n, m, u)
        assert F is not None and numerically_equal(diff(F, u), sinh(u)**n * cosh(u)**m, [u > 0, u < 1]), (n, m)


def test_polynomials_and_exponentials() -> None:
    assert _same(trigonometric_antiderivative(x**2 * cos(2 * x), x), x**2 * sin(2 * x) / 2 + x * cos(2 * x) / 2 - sin(2 * x) / 4)
    for f in [x * exp(x) * sin(x), x**2 * exp(-x) * cos(x), x**3 * sin(a * x), x * exp(b * x) * cos(a * x),
              x * sinh(2 * x), x**2 * exp(x) * cosh(3 * x), x * sin(x)**2, x**2 * sin(x) * cos(2 * x),
              exp(2 * x) * sin(3 * x) * cos(x), x * exp(a * x) * sin(a * x)**2, exp(-x) * sinh(x)**2]:
        assert _checks(f), f


def test_products_of_different_frequencies() -> None:
    assert _same(trigonometric_antiderivative(sin(2 * x) * cos(3 * x), x), cos(x) / 2 - cos(5 * x) / 10)
    for f in [sin(a * x) * cos(b * x), sin(a * x) * sin(b * x), cos(a * x) * cos(b * x)]:
        F = trigonometric_antiderivative(f, x)
        assert isinstance(F, Piecewise) and F.args[0].args[1] == Ne(a - b, 0)
        generic, special = as_expr(F.args[0].args[0]), as_expr(F.args[1].args[0])
        assert simplify(diff(generic, x) - f) == 0
        assert simplify(diff(special, x) - f.subs(b, a)) == 0
    assert _checks(sin(2 * x) * sin(5 * x) * cos(x))


def test_rational_functions_of_sine_and_cosine() -> None:
    assert _same(trigonometric_antiderivative(1 / (2 + cos(x)), x), 2 * sqrt(3) * atan(sqrt(3) * tan(x / 2) / 3) / 3)
    facts = [x > S(1) / 10, x < S(1) / 2]
    for f in [sin(x) / (1 + cos(x)**2), (1 + sin(x)) / (1 + cos(x)), 1 / (3 + 5 * cos(2 * x)),
              tan(x) / (1 + sin(x)), cos(x)**2 / (1 + sin(x)**2), 1 / (1 + tan(x)), 1 / (a + cos(x)),
              1 / (sinh(x) + 2), cosh(x) / (1 + sinh(x)**2), 1 / (tanh(x) + 2)]:
        assert _checks(f, facts), f


def test_inverse_functions() -> None:
    assert _same(trigonometric_antiderivative(asin(x), x), x * asin(x) + sqrt(1 - x**2))
    assert _same(trigonometric_antiderivative(x**2 * atan(x), x), x**3 * atan(x) / 3 - x**2 / 6 + log(x**2 + 1) / 6)
    assert _same(trigonometric_antiderivative(asin(x)**2, x), x * asin(x)**2 - 2 * x + 2 * sqrt(1 - x**2) * asin(x))
    inside = [x > Rational(1, 10), x < Rational(1, 2)]
    for f in [x**3 * acos(x), x * asinh(2 * x), atan(2 * x), x**2 * acot(x), x * atanh(x), x * asin(a * x),
              acos(x)**2, asinh(x)**2, x**2 * asin(x)]:
        assert _checks(f, inside), f
    assert _checks(acosh(x)**2, [x > 2]) and _checks(x * acosh(x), [x > 2])
    # not elementary, and not of the families
    assert trigonometric_antiderivative(atan(x)**2, x) is None
    assert trigonometric_antiderivative(x**2, x) is None
    assert trigonometric_antiderivative(sqrt(x) * sin(x), x) is None
    assert trigonometric_antiderivative(sin(x**2), x) is None
    assert trigonometric_antiderivative(sin(x) / x, x) is None


def test_every_result_is_verified() -> None:
    # the check by differentiation refuses a wrong candidate: a Piecewise
    # whose special branch does not check keeps the generic one only
    from sympy_extras.integrals.trigonometric import _verified
    assert _verified(-cos(x), sin(x), x) and not _verified(cos(x), sin(x), x)
    assert not _verified(x * asin(x), asin(x), x)


def test_sympy_leaves_these_unevaluated() -> None:
    # SymPy's integrate returns the integral unevaluated for the first and
    # spends its time limit on the second (SymPy 1.14); the Weierstrass
    # substitution does both in a second
    for f in [tan(x) / (1 + sin(x)), cos(x)**2 / (1 + sin(x)**2)]:
        assert _checks(f, [x > S(1) / 10, x < S(1) / 2]), f
    assert integrate(tan(x) / (1 + sin(x)), x).has(Integral)
