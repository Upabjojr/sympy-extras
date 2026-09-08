from __future__ import annotations

from sympy import exp, oo, log, Piecewise, Eq, S, sqrt, sin, O, Rational
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.abc import a, b, x

from sympy_extras.assumptions import limit, series, element


def test_limit_with_assumptions() -> None:
    assert limit(exp(a*x), x, oo, assumptions=a < 0) == 0
    assert limit(exp(a*x), x, oo, assumptions=a > 0) is oo
    assert limit(exp(a*x), x, oo, assumptions=Eq(a, 0)) == 1
    assert limit(x**a, x, oo, assumptions=a > 0) is oo
    assert limit(x**a, x, oo, assumptions=a < 0) == 0
    assert limit(x**a*log(x), x, 0, assumptions=a > 0) == 0
    assert limit(sin(a*x)/x, x, 0) == a
    assert limit((1 + a/x)**x, x, oo) == exp(a)
    assert limit(a**x, x, oo, assumptions=(a > 0) & (a < 1)) == 0
    assert limit(a**x, x, oo, assumptions=a > 1) is oo
    assert limit(x**b, x, oo, assumptions=element(b, S.Integers) & (b > 0)) is oo


def test_limit_cases() -> None:
    result = limit(exp(a*x), x, oo)
    assert result == Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))
    result = limit(x**a, x, oo)
    assert isinstance(result, Piecewise) and result.subs(a, 2) is oo and result.subs(a, -1) == 0
    result = limit(exp(a*x), x, oo, assumptions=a >= 0)
    assert result == Piecewise((oo, a > 0), (1, Eq(a, 0)))
    # nothing to decide: SymPy's answer
    assert limit(exp(-x**2)*a, x, oo) == 0


def test_series_with_assumptions() -> None:
    assert series(sqrt(a**2 + x), x, 0, 2, assumptions=a > 0) == a + x/(2*a) + O(x**2)
    assert series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0) == -a - x/(2*a) + O(x**2)
    expansion = series(log(a*x), x, 1, 2, assumptions=a > 0)
    assert expansion.removeO() == log(a) + x - 1
    assert series(exp(a*x), x, 0, 3) == 1 + a*x + a**2*x**2/2 + O(x**3)


def test_limit_seq_with_assumptions() -> None:
    from sympy import factorial, harmonic, EulerGamma, Limit, pi
    from sympy.abc import n
    from sympy_extras.assumptions import limit_seq
    assert limit_seq(x**n/factorial(n), n) == 0
    assert limit_seq((1 + x/n)**n, n) == exp(x)
    assert limit_seq(harmonic(n) - log(n), n) == EulerGamma
    assert limit_seq((-1)**n/n, n) == 0
    assert limit_seq(sin(pi*n), n) == 0
    assert limit_seq(a**n, n, assumptions=(a > 0) & (a < 1)) == 0
    assert limit_seq(a**n, n, assumptions=a > 1) is oo
    assert limit_seq(a**n, n, assumptions=Eq(a, 1)) == 1
    assert limit_seq(n*a**n, n, assumptions=(a > 0) & (a < 1)) == 0
    assert limit_seq(a**n/n, n, assumptions=a > 1) is oo
    # a negative base oscillates unless the magnitude tends to zero
    assert limit_seq(a**n, n, assumptions=(a > -1) & (a < 0)) == 0
    assert limit_seq(a**n, n, assumptions=a < -1) == Limit(a**n, n, oo, '-')
    assert limit_seq(a**n, n, assumptions=a > 0) == Piecewise((oo, a > 1), (1, Eq(a, 1)), (0, a < 1))
    result = limit_seq(a**n, n)
    assert result.subs(a, 2) is oo and result.subs(a, S.Half) == 0 and result.subs(a, -S.Half) == 0
    assert isinstance(result.subs(a, -2), Limit)
    assert limit_seq(n**a, n, assumptions=a < 0) == 0
    assert limit_seq(n**a, n) == Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))


def test_limit_of_a_power_with_a_negative_base() -> None:
    # the sign of log(a) decides limit(a**x, x, oo) only for a > 0: the
    # case split used to claim the limit is 0 for every a < 1, which is
    # wrong for a <= -1, where |a**x| does not tend to zero
    assert limit(a**x, x, oo, assumptions=a < -1) is S.ComplexInfinity
    assert limit(a**x, x, oo, assumptions=Eq(a, -1)) is S.NaN
    assert limit(a**x, x, oo, assumptions=(a > -1) & (a < 0)) == 0
    assert limit(a**x, x, oo, assumptions=Eq(a, 0)) == 0
    assert limit(2*a**x, x, oo, assumptions=a < -1) is S.ComplexInfinity
    general = limit(a**x, x, oo)
    assert general.subs(a, -3) is S.ComplexInfinity
    assert general.subs(a, Rational(-1, 2)) == 0
    assert general.subs(a, -1) is S.NaN
    assert general.subs(a, 2) is oo
    assert general.subs(a, Rational(1, 2)) == 0
    # the exponent decides which way the modulus goes
    assert limit(a**x, x, -oo, assumptions=a < -1) == 0
    assert limit(a**(-x), x, oo, assumptions=a < -1) == 0
    assert limit(a**(-x), x, oo, assumptions=(a > -1) & (a < 0)) is S.ComplexInfinity
    assert limit(a**(2*x), x, oo, assumptions=a < -1) is S.ComplexInfinity
    # asking SymPy for the sign of log(a) with a < 0 gives a sign of a
    # non-real number, which raised TypeError: Invalid comparison of
    # non-real I instead of leaving the limit unevaluated
    unevaluated = limit(a**x*x, x, oo)
    assert unevaluated.subs(a, 2) is oo and unevaluated.subs(a, Rational(1, 2)) == 0


def test_limit_of_an_oscillating_expression() -> None:
    # the sign of the bounds of an oscillating factor is not a case to
    # split on: asking the CAD for it raised PolynomialError, and the
    # conditions of the answer compared an AccumBounds with zero
    result = limit(sin(x)*x**a, x, oo)
    assert isinstance(result, Piecewise)
    assert not any(pair.args[1].has(AccumBounds) for pair in result.args)
    assert result.subs(a, -1) == 0
    assert result.subs(a, 0) == AccumBounds(-1, 1)
    assert limit(sin(x)/x, x, oo) == 0
