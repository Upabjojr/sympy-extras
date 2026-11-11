"""The native limit, limit_seq and series on objects of the core, checked
against hand computations (the SymPy face is tested in test_limits.py)."""
from __future__ import annotations

from sympy_extras.core import Eq, I, Integer, Rational, Symbol, oo, sqrt, zoo, nan
from sympy_extras.core.limits import Limit
from sympy_extras.core.order import Order
from sympy_extras.core.logic import and_
from sympy_extras.functions import exp, factorial, log, sin, atan
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.sets import Integers
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.limits import _sign_dependency, limit, limit_seq, series

a, b, x, n = Symbol('a'), Symbol('b'), Symbol('x'), Symbol('n')


def test_native_limit_under_assumptions() -> None:
    assert limit(exp(a*x), x, oo, assumptions=a < 0) == 0
    assert limit(exp(a*x), x, oo, assumptions=a > 0) == oo
    assert limit(exp(a*x), x, oo, assumptions=Eq(a, 0)) == 1
    assert limit(x**a, x, oo, assumptions=a < 0) == 0
    assert limit(x**a*log(x), x, Integer(0), assumptions=a > 0) == 0
    assert limit(a**x, x, oo, assumptions=and_(a > 0, a < 1)) == 0
    assert limit(a**x, x, oo, assumptions=a > 1) == oo
    assert limit(x**b, x, oo, assumptions=and_(element(b, Integers), b > 0)) == oo
    assert limit(sin(a*x)/x, x, Integer(0)) == a
    assert limit((1 + a/x)**x, x, oo) == exp(a)


def test_native_limit_cases() -> None:
    result = limit(exp(a*x), x, oo)
    assert isinstance(result, Piecewise)
    assert [p.expr for p in result.args] == [oo, Integer(1), Integer(0)]
    assert result.subs(a, 2) == oo and result.subs(a, 0) == 1 and result.subs(a, -3) == 0
    value = limit(atan(a*x), x, oo)
    assert value.subs(a, 1) == limit(atan(x), x, oo) and value.subs(a, 0) == 0
    assert limit(exp(a*x), x, oo, assumptions=a >= 0).subs(a, 0) == 1


def test_native_negative_base() -> None:
    # |a**x| = |a|**x: zoo for a < -1, nan on the unit circle, 0 inside
    assert limit(a**x, x, oo, assumptions=a < -1) == zoo
    assert limit(a**x, x, oo, assumptions=Eq(a, -1)) == nan
    assert limit(a**x, x, oo, assumptions=and_(a > -1, a < 0)) == 0
    assert limit(a**x, x, -oo, assumptions=a < -1) == 0
    general = limit(a**x, x, oo)
    for value, expected in ((Integer(-3), zoo), (Rational(-1, 2), Integer(0)), (Integer(-1), nan),
                            (Integer(2), oo), (Rational(1, 2), Integer(0))):
        assert general.subs(a, value) == expected, value


def test_native_directed_infinity() -> None:
    # sympy-extras#52: the argument winds, so the limit is zoo, not oo
    assert limit(exp(x**2 + I*x), x, oo) == zoo
    assert limit(x*exp(-x), x, oo) == 0


def test_native_degenerate_parameter() -> None:
    # sympy-extras#30: a/b holds for b != 0 only
    value = limit((a*x + 1)/(b*x + 2), x, oo)
    assert value.subs({a: 3, b: 2}) == Rational(3, 2)
    at_zero = value.subs({a: 3, b: 0})
    assert at_zero == oo or isinstance(at_zero, Limit)


def test_sign_dependency_from_the_message() -> None:
    from sympy_extras.core import Dummy
    d = Dummy('a', real=True)
    assert _sign_dependency('Result depends on the sign of sign(%s)' % d, {d: a}) == a
    assert _sign_dependency('Result depends on the sign of -%s + 1' % d, {d: a}) == 1 - a
    assert _sign_dependency('no sign here', {d: a}) is None


def test_native_limit_seq() -> None:
    assert limit_seq(x**n/factorial(n), n) == 0
    assert limit_seq((1 + x/n)**n, n) == exp(x)
    assert limit_seq(a**n, n, assumptions=and_(a > 0, a < 1)) == 0
    assert limit_seq(a**n, n, assumptions=a > 1) == oo
    assert limit_seq(a**n, n, assumptions=and_(a > -1, a < 0)) == 0
    assert isinstance(limit_seq(a**n, n, assumptions=a < -1), Limit)
    value = limit_seq(n**a, n)
    assert value.subs(a, 2) == oo and value.subs(a, 0) == 1 and value.subs(a, -1) == 0


def test_native_series() -> None:
    assert series(sqrt(a**2 + x), x, 0, 2, assumptions=a > 0) == a + x/(2*a) + Order(x**2)
    assert series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0) == -a - x/(2*a) + Order(x**2)
    assert series(log(a*x), x, 1, 2, assumptions=a > 0).removeO() == log(a) + x - 1
    assert series(exp(a*x), x, 0, 3) == 1 + a*x + a**2*x**2/2 + Order(x**3)


def test_native_limit_of_logarithms_with_a_negative_parameter() -> None:
    # test_regions' agreeing cases: the boundary value at t = 0 came back as
    # I*a*log(-a - I*a)/2 - I*a*log(-a + I*a)/2, equal to -pi*a/4 but not
    # recognised as such, and the cases of the region integral no longer merged
    from sympy_extras.core import Dummy, pi
    t = Dummy('t', positive=True)
    r = sqrt(a**2 - t**2*a**2)/(t + 1)
    e = -a*(-I*log(-I*a + r)/2 + I*log(I*a + r)/2)
    assert limit(e, t, Integer(0), '+', assumptions=a < 0) == -pi*a/4
    assert limit(e, t, Integer(0), '+', assumptions=a > 0) == pi*a/4
