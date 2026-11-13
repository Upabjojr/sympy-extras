"""Tests of the port of SymPy's formal power series on the typed core:
SymPy's own tests (``sympy/series/tests/test_formal.py``; the expected
series are facts, hand computations), checked as properties where the
native form legitimately differs (the coefficients of a series, the
difference of two formulas at the integers), and the bugs of SymPy the
port does not have."""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/tests/test_formal.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the core's objects; the formulas compared at integers and
# the truncations by their polynomial parts and order terms.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from pytest import raises

from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.core import (Derivative, Expr, Function, I, Integer, O, Rational, S, Symbol, add, mul, oo, pi,
                               sqrt)
from sympy_extras.core.function import expand
from sympy_extras.functions import acos, acosh, airyai, asin, asinh, atan, cos, erf, exp, factorial, gamma, log, sin
from sympy_extras.polys.polytools import cancel
from sympy_extras.series.formal import (FormalPowerSeries, FormalPowerSeriesCompose, FormalPowerSeriesInverse,
                                        FormalPowerSeriesProduct, Formula, bell_incomplete, coeff_of, exp_re, fps,
                                        hyper_re, rational_algorithm, rational_independent, simpleDE)
from sympy_extras.series.sequences import SeqFormula

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
n, m, k = Symbol('n', integer=True), Symbol('m', integer=True), Symbol('k', integer=True)
f, r = Function('f'), Function('r')


def _series(e: Expr) -> FormalPowerSeries:
    assert isinstance(e, FormalPowerSeries), e
    return e


def _same(a: Expr, b: Expr) -> bool:
    """The same truncated series: equal order terms, polynomial parts with
    a zero difference."""
    if a.getO() != b.getO():
        return False
    return cancel(expand(a.removeO() - b.removeO())) == 0


def _same_formula(found: Optional[Formula], formula: Expr, ind: Expr, start: int) -> bool:
    """A formula of the coefficients equal to ``formula`` at the integers."""
    if found is None:
        return False
    if found[2] != start or cancel(expand(found[1] - ind)) != 0:
        return False
    return all(cancel(expand((found[0] - formula).subs(k, Integer(j)))) == 0 for j in range(start, start + 8))


def _d(i: int) -> Expr:
    return Derivative(f(x), (x, i))


def test_rational_algorithm() -> None:
    e = 1 / ((x - 1)**2 * (x - 2))
    assert _same_formula(rational_algorithm(e, x, k), -2**(-k - 1) + 1 - (factorial(k + 1) / factorial(k)),
                         S.Zero, 0)
    e = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    assert _same_formula(rational_algorithm(e, x, k), -15*2**(-k - 1) + 4, x + 4, 0)
    e = z / (y*m - m*x - y*x + x**2)
    assert _same_formula(rational_algorithm(e, x, k), ((-y**(-k - 1)*z) / (y - m)) + ((m**(-k - 1)*z) / (y - m)),
                         S.Zero, 0)
    e = x / (1 - x - x**2)
    assert rational_algorithm(e, x, k) is None
    assert _same_formula(rational_algorithm(e, x, k, full=True),
                         (Rational(-1, 2) + sqrt(5)/2)**(-k - 1) * (-sqrt(5)/10 + S.Half)
                         + (-sqrt(5)/2 - S.Half)**(-k - 1) * (sqrt(5)/10 + S.Half), S.Zero, 0)
    e = 1 / (x**2 + 2*x + 2)
    assert rational_algorithm(e, x, k) is None
    assert _same_formula(rational_algorithm(e, x, k, full=True),
                         (I*(-1 + I)**(-k - 1)) / 2 - (I*(-1 - I)**(-k - 1)) / 2, S.Zero, 0)
    assert _same_formula(rational_algorithm(log(1 + x), x, k), -(-1)**(-k) / k, S.Zero, 1)
    assert rational_algorithm(atan(x), x, k) is None
    assert _same_formula(rational_algorithm(atan(x), x, k, full=True), ((I*I**(-k)) / 2 - (I*(-I)**(-k)) / 2) / k,
                         S.Zero, 1)
    e = x*atan(x) - log(1 + x**2) / 2
    assert rational_algorithm(e, x, k) is None
    assert _same_formula(rational_algorithm(e, x, k, full=True),
                         ((I*I**(-k + 1)) / 2 - (I*(-I)**(-k + 1)) / 2) / (k*(k - 1)), S.Zero, 2)
    e = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert rational_algorithm(e, x, k) is None
    assert _same_formula(rational_algorithm(e, x, k, full=True),
                         (-(-1)**(-k) / 2 - (I*I**(-k)) / 2 + (I*(-I)**(-k)) / 2 + S.Half) / k, S.Zero, 1)
    assert rational_algorithm(cos(x), x, k) is None


def test_rational_algorithm_declines_quadratic_denominators() -> None:
    # SymPy's rational algorithm reads every partial fraction as a power of
    # a linear polynomial: 1/(x**2 + x + 1) gave the formula (-1)**k, the
    # series of 1/(1 + x) (1 - x + x**2 - x**3 + ...); the series is
    # 1 - x + x**3 - x**4 + ...
    assert rational_algorithm(1/(x**2 + x + 1), x, k) is None
    s = _series(fps(1/(x**2 + x + 1), x))
    assert _same(s.truncate(8), 1 - x + x**3 - x**4 + x**6 - x**7 + O(x**8))
    s = _series(fps(1/(x**2 + x + 1), x, full=True))
    assert _same(s.truncate(8), 1 - x + x**3 - x**4 + x**6 - x**7 + O(x**8))


def test_rational_independent() -> None:
    ri = rational_independent
    assert ri([], x) == []
    assert ri([cos(x), sin(x)], x) == [cos(x), sin(x)]
    assert ri([x**2, sin(x), x*sin(x), x**3], x) == [x**3 + x**2, x*sin(x) + sin(x)]
    assert ri([S.One, x*log(x), log(x), sin(x)/x, cos(x), sin(x), x], x) == \
        [x + 1, x*log(x) + log(x), sin(x)/x + sin(x), cos(x)]


def _first_de(e: Expr) -> tuple[Expr, int]:
    for DE in simpleDE(e, x, f):
        return DE
    raise AssertionError("no differential equation")


def _same_de(found: tuple[Expr, int], expected: Expr, order: int) -> bool:
    return found[1] == order and expand(found[0] - expected) == 0


def test_simpleDE() -> None:
    # the first equation found (collect leaves the derivatives of different
    # orders apart: the equations are compared expanded)
    assert _same_de(_first_de(exp(x)), -f(x) + _d(1), 1)
    assert _same_de(_first_de(sin(x)), f(x) + _d(2), 2)
    assert _same_de(_first_de(log(1 + x)), (x + 1)*_d(2) + _d(1), 2)
    assert _same_de(_first_de(asin(x)), x*_d(1) + (x**2 - 1)*_d(2), 2)
    assert _same_de(_first_de(exp(x)*sin(x)), 2*f(x) - 2*_d(1) + _d(2), 2)
    assert _same_de(_first_de(((1 + x)/(1 - x))**n), 2*n*f(x) + (x**2 - 1)*_d(1), 1)
    assert _same_de(_first_de(airyai(x)), -x*f(x) + _d(2), 2)


def test_exp_re() -> None:
    assert exp_re(-f(x) + _d(1), r, k) == -r(k) + r(k + 1)
    assert exp_re(f(x) + _d(2), r, k) == r(k) + r(k + 2)
    assert exp_re(f(x) + _d(1) + _d(2), r, k) == r(k) + r(k + 1) + r(k + 2)
    assert exp_re(_d(1) + _d(2), r, k) == r(k) + r(k + 1)
    assert exp_re(_d(3) + _d(4) + _d(1), r, k) == r(k) + r(k + 2) + r(k + 3)


def test_hyper_re() -> None:
    def same(a: Expr, b: Expr) -> bool:
        return expand(a - b) == 0
    assert same(hyper_re(f(x) + _d(2), r, k), r(k) + (k+1)*(k+2)*r(k + 2))
    assert same(hyper_re(-x*f(x) + _d(2), r, k), (k + 2)*(k + 3)*r(k + 3) - r(k))
    assert same(hyper_re(2*f(x) - 2*_d(1) + _d(2), r, k), (-2*k - 2)*r(k + 1) + (k + 1)*(k + 2)*r(k + 2) + 2*r(k))
    assert same(hyper_re(2*n*f(x) + (x**2 - 1)*_d(1), r, k), k*r(k) + 2*n*r(k + 1) + (-k - 2)*r(k + 2))
    assert same(hyper_re((x**10 + 4)*_d(1) + x*(x**10 - 1)*_d(2), r, k),
                (k*(k - 1) + k)*r(k) + (4*k - (k + 9)*(k + 10) + 40)*r(k + 10))
    assert same(hyper_re((x**2 - 1)*_d(3) + 3*x*_d(2) + _d(1), r, k),
                (k*(k - 2)*(k - 1) + 3*k*(k - 1) + k)*r(k) + (-k*(k + 1)*(k + 2))*r(k + 2))


def test_fps() -> None:
    assert fps(1) == 1
    assert fps(2, x) == 2
    assert fps(2, x, dir='+') == 2
    assert fps(2, x, dir='-') == 2
    assert fps(1/x + 1/x**2) == 1/x + 1/x**2
    assert fps(log(1 + x), hyper=False, rational=False) == log(1 + x)

    s = _series(fps(x**2 + x + 1))
    assert s.function == x**2 + x + 1
    assert s[0] == 1
    assert s[2] == x**2
    assert _same(s.truncate(4), x**2 + x + 1 + O(x**4))
    assert s.polynomial() == x**2 + x + 1

    s = _series(fps(log(1 + x)))
    assert s.function == log(1 + x)
    assert s.subs(x, y) == s
    assert s[:5] == [0, x, -x**2/2, x**3/3, -x**4/4]
    assert s.as_leading_term(x) == x
    assert s.polynomial(6) == x - x**2/2 + x**3/3 - x**4/4 + x**5/5

    kk = s.ak.variables[0]
    infinite = s.infinite
    assert isinstance(infinite, Sum)
    assert infinite.limits == ((kk, Integer(1), oo),)
    assert cancel(infinite.function - (-(-1)**(-kk)*x**kk)/kk) == 0

    ft, first = s.truncate(n=None), s[:5]
    for i, t in enumerate(ft):
        if i == 5:
            break
        assert first[i] == t

    s = _series(sin(x).fps(x))
    assert _same(s.truncate(), x - x**3/6 + x**5/120 + O(x**6))

    raises(NotImplementedError, lambda: fps(y*x))
    raises(ValueError, lambda: fps(x, dir=0))


def _truncated(e: Expr, n: int = 6) -> Expr:
    return _series(e).truncate(n)


def test_fps__rational() -> None:
    assert fps(1/x) == (1/x)
    assert fps((x**2 + x + 1) / x**3, dir=-1) == (x**2 + x + 1) / x**3

    assert _same(_truncated(fps(1 / ((x - 1)**2 * (x - 2)), x)),
                 Rational(-1, 2) - x*Rational(5, 4) - 17*x**2/8 - 49*x**3/16 - 129*x**4/32 - 321*x**5/64 + O(x**6))
    assert _same(_truncated(fps((1 + x + x**2 + x**3) / ((x - 1) * (x - 2)), x)),
                 S.Half + x*Rational(5, 4) + 17*x**2/8 + 49*x**3/16 + 113*x**4/32 + 241*x**5/64 + O(x**6))
    assert _same(_truncated(fps(x / (1 - x - x**2), x, full=True)), x + x**2 + 2*x**3 + 3*x**4 + 5*x**5 + O(x**6))
    assert _same(_truncated(fps(1 / (x**2 + 2*x + 2), x, full=True)),
                 S.Half - x/2 + x**2/4 - x**4/8 + x**5/8 + O(x**6))

    e = log(1 + x)
    assert _same(_truncated(fps(e, x)), x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6))
    assert _same(_truncated(fps(e, x, dir=1)), _truncated(fps(e, x, dir=-1)))
    assert _same(_truncated(fps(e, x, 2)),
                 log(3) - Rational(2, 3) - (x - 2)**2/18 + (x - 2)**3/81 - (x - 2)**4/324 + (x - 2)**5/1215 + x/3
                 + O((x - 2)**6, (x, 2)))
    assert _same(_truncated(fps(e, x, 2, dir=-1)),
                 log(3) - Rational(2, 3) - (-x + 2)**2/18 - (-x + 2)**3/81 - (-x + 2)**4/324 - (-x + 2)**5/1215
                 + x/3 + O((x - 2)**6, (x, 2)))

    e = atan(x)
    assert _same(_truncated(fps(e, x, full=True)), x - x**3/3 + x**5/5 + O(x**6))
    assert _same(_truncated(fps(e, x, full=True, dir=1)), _truncated(fps(e, x, full=True, dir=-1)))
    assert _same(_truncated(fps(e, x, 2, full=True)),
                 atan(2) - Rational(2, 5) - 2*(x - 2)**2/25 + 11*(x - 2)**3/375 - 6*(x - 2)**4/625
                 + 41*(x - 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))
    assert _same(_truncated(fps(e, x, 2, full=True, dir=-1)),
                 atan(2) - Rational(2, 5) - 2*(-x + 2)**2/25 - 11*(-x + 2)**3/375 - 6*(-x + 2)**4/625
                 - 41*(-x + 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))

    assert _same(_truncated(fps(x*atan(x) - log(1 + x**2) / 2, x, full=True)), x**2/2 - x**4/12 + O(x**6))
    assert _same(_truncated(fps(log((1 + x) / (1 - x)) / 2 - atan(x), x, full=True), 10),
                 2*x**3/3 + 2*x**7/7 + O(x**10))


def test_fps__hyper() -> None:
    assert _same(_truncated(fps(sin(x), x)), x - x**3/6 + x**5/120 + O(x**6))
    assert _same(_truncated(fps(cos(x), x)), 1 - x**2/2 + x**4/24 + O(x**6))
    assert _same(_truncated(fps(exp(x), x)), 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6))
    assert _same(_truncated(fps(atan(x), x)), x - x**3/3 + x**5/5 + O(x**6))
    assert _same(_truncated(fps(exp(acos(x)), x)),
                 exp(pi/2) - x*exp(pi/2) + x**2*exp(pi/2)/2 - x**3*exp(pi/2)/3 + 5*x**4*exp(pi/2)/24
                 - x**5*exp(pi/2)/6 + O(x**6))
    assert _same(_truncated(fps(exp(acosh(x)), x)), I + x - I*x**2/2 - I*x**4/8 + O(x**6))
    assert _same(_truncated(fps(atan(1/x), x)), pi/2 - x + x**3/3 - x**5/5 + O(x**6))
    assert _same(_truncated(fps(x*atan(x) - log(1 + x**2) / 2, x, rational=False)), x**2/2 - x**4/12 + O(x**6))
    assert _same(_truncated(fps(log(1 + x), x, rational=False)), x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6))
    assert _same(_truncated(fps(airyai(x**2), x)),
                 3**Rational(5, 6)*gamma(Rational(1, 3))/(6*pi) - 3**Rational(2, 3)*x**2/(3*gamma(Rational(1, 3)))
                 + O(x**6))
    assert _same(_truncated(fps(exp(x)*sin(x), x)), x + x**2 + x**3/3 - x**5/30 + O(x**6))
    assert _same(_truncated(fps(exp(x)*sin(x)/x, x)), 1 + x + x**2/3 - x**4/30 - x**5/90 + O(x**6))
    assert _same(_truncated(fps(sin(x) * cos(x), x)), x - 2*x**3/3 + 2*x**5/15 + O(x**6))


def test_fps_shift() -> None:
    assert _same(_truncated(fps(x**-5*sin(x), x)),
                 1/x**4 - 1/(6*x**2) + Rational(1, 120) - x**2/5040 + x**4/362880 + O(x**6))
    assert _same(_truncated(fps(x**2*atan(x), x, rational=False)), x**3 - x**5/3 + O(x**6))
    assert _same(_truncated(fps(cos(sqrt(x))*x, x)), x - x**2/2 + x**3/24 - x**4/720 + x**5/40320 + O(x**6))
    assert _same(_truncated(fps(x**2*cos(sqrt(x)), x)), x**2 - x**3/2 + x**4/24 - x**5/720 + O(x**6))


def test_fps__Add_expr() -> None:
    assert _same(_truncated(fps(x*atan(x) - log(1 + x**2) / 2, x)), x**2/2 - x**4/12 + O(x**6))
    assert _same(_truncated(fps(sin(x) + cos(x) - exp(x) + log(1 + x), x)),
                 x - 3*x**2/2 - x**4/4 + x**5/5 + O(x**6))
    assert _same(_truncated(fps(1/x + sin(x), x)), 1/x + x - x**3/6 + x**5/120 + O(x**6))
    assert _same(_truncated(fps(sin(x) - cos(x) + 1/(x - 1), x)),
                 -2 - x**2/2 - 7*x**3/6 - 25*x**4/24 - 119*x**5/120 + O(x**6))


def test_fps__asymptotic() -> None:
    assert fps(exp(x), x, oo) == exp(x)
    assert _same(_truncated(fps(exp(x), x, -oo)), O(1/x**6, (x, oo)))
    assert _same(_truncated(fps(erf(x), x, oo)), 1 + O(1/x**6, (x, oo)))
    assert _same(_truncated(fps(erf(x), x, -oo)), -1 + O(1/x**6, (x, oo)))
    assert _same(_truncated(fps(atan(x), x, oo, full=True)),
                 -1/(5*x**5) + 1/(3*x**3) - 1/x + pi/2 + O(1/x**6, (x, oo)))
    assert _same(_truncated(fps(atan(x), x, -oo, full=True)),
                 -1/(5*x**5) + 1/(3*x**3) - 1/x - pi/2 + O(1/x**6, (x, oo)))
    e = log(1 + x)
    assert fps(e, x, oo) != (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x - log(1/x)
                             + O(1/x**6, (x, oo)))
    assert fps(e, x, -oo) != (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x + I*pi - log(-1/x)
                              + O(1/x**6, (x, oo)))


def test_fps__fractional() -> None:
    assert _same(_truncated(fps(sin(sqrt(x)) / x, x)),
                 1/sqrt(x) - sqrt(x)/6 + x**Rational(3, 2)/120 - x**Rational(5, 2)/5040 + x**Rational(7, 2)/362880
                 - x**Rational(9, 2)/39916800 + x**Rational(11, 2)/6227020800 + O(x**6))
    assert _same(_truncated(fps(sin(sqrt(x)) * x, x)),
                 x**Rational(3, 2) - x**Rational(5, 2)/6 + x**Rational(7, 2)/120 - x**Rational(9, 2)/5040
                 + x**Rational(11, 2)/362880 + O(x**6))
    assert _same(_truncated(fps(atan(sqrt(x)) / x**2, x)),
                 x**Rational(-3, 2) - x**Rational(-1, 2)/3 + x**S.Half/5 - x**Rational(3, 2)/7 + x**Rational(5, 2)/9
                 - x**Rational(7, 2)/11 + x**Rational(9, 2)/13 - x**Rational(11, 2)/15 + O(x**6))
    assert _same(_truncated(fps(exp(sqrt(x)), x)).expand(),
                 1 + x/2 + x**2/24 + x**3/720 + x**4/40320 + x**5/3628800 + sqrt(x) + x**Rational(3, 2)/6
                 + x**Rational(5, 2)/120 + x**Rational(7, 2)/5040 + x**Rational(9, 2)/362880
                 + x**Rational(11, 2)/39916800 + O(x**6))
    assert _same(_truncated(fps(exp(sqrt(x))*x, x)).expand(),
                 x + x**2/2 + x**3/24 + x**4/720 + x**5/40320 + x**Rational(3, 2) + x**Rational(5, 2)/6
                 + x**Rational(7, 2)/120 + x**Rational(9, 2)/5040 + x**Rational(11, 2)/362880 + O(x**6))


def test_fps__logarithmic_singularity() -> None:
    e = log(1 + 1/x)
    expected = -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    assert fps(e, x) != expected
    assert fps(e, x, rational=False) != expected


def test_fps_symbolic() -> None:
    assert _same(_truncated(fps(x**n*sin(x**2), x), 8), x**(n + 2) - x**(n + 6)/6 + O(x**(n + 8), x))
    s = _series(fps(x**n*log(1 + x), x))
    kk = s.ak.variables[0]
    infinite = s.infinite
    assert isinstance(infinite, Sum) and infinite.limits == ((kk, Integer(1), oo),)
    assert cancel(infinite.function - (-(-1)**(-kk)*x**(kk + n))/kk) == 0
    assert _same(_truncated(fps((x - 2)**n*log(1 + x), x, 2)),
                 (x - 2)**n*log(3) + (x - 2)**(n + 1)/3 - (x - 2)**(n + 2)/18 + (x - 2)**(n + 3)/81
                 - (x - 2)**(n + 4)/324 + (x - 2)**(n + 5)/1215 + O((x - 2)**(n + 6), (x, 2)))
    assert _same(_truncated(fps(x**(n - 2)*cos(x), x)), x**(n - 2) - x**n/2 + x**(n + 2)/24 + O(x**(n + 4), x))
    assert _same(_truncated(fps(x**(n - 2)*sin(x) + x**n*exp(x), x)),
                 x**(n - 1) + x**(n + 1) + x**(n + 2)/2 + x**n + x**(n + 4)/24 + x**(n + 5)/60 + O(x**(n + 6), x))
    assert _same(_truncated(fps(x**n*atan(x), x, oo)),
                 -x**(n - 5)/5 + x**(n - 3)/3 + x**n*(pi/2 - 1/x) + O((1/x)**(-n)/x**6, (x, oo)))
    assert _same(_truncated(fps(x**(n/2)*cos(x), x)),
                 x**(n/2) - x**(n/2 + 2)/2 + x**(n/2 + 4)/24 + O(x**(n/2 + 6), x))
    assert _same(_truncated(fps(x**(n + m)*sin(x), x)),
                 x**(m + n + 1) - x**(m + n + 3)/6 + x**(m + n + 5)/120 + O(x**(m + n + 6), x))


def test_fps__slow() -> None:
    assert _same(_truncated(fps(x*exp(x)*sin(2*x), x)), 2*x**2 + 2*x**3 - x**4/3 - x**5 + O(x**6))


def test_fps__operations() -> None:
    f1, f2 = _series(fps(sin(x))), _series(fps(cos(x)))

    fsum = _series(f1 + f2)
    assert fsum.function == sin(x) + cos(x)
    assert _same(fsum.truncate(), 1 + x - x**2/2 - x**3/6 + x**4/24 + x**5/120 + O(x**6))
    fsum = _series(f1 + 1)
    assert fsum.function == sin(x) + 1
    assert _same(fsum.truncate(), 1 + x - x**3/6 + x**5/120 + O(x**6))
    fsum = _series(1 + f2)
    assert fsum.function == cos(x) + 1
    assert _same(fsum.truncate(), 2 - x**2/2 + x**4/24 + O(x**6))
    assert (f1 + x) == add(f1, x)
    assert _same(-f2.truncate(), -1 + x**2/2 - x**4/24 + O(x**6))
    assert (f1 - f1) == 0

    fsub = _series(f1 - f2)
    assert fsub.function == sin(x) - cos(x)
    assert _same(fsub.truncate(), -1 + x + x**2/2 - x**3/6 - x**4/24 + x**5/120 + O(x**6))
    fsub = _series(f1 - 1)
    assert fsub.function == sin(x) - 1
    assert _same(fsub.truncate(), -1 + x - x**3/6 + x**5/120 + O(x**6))
    fsub = _series(1 - f2)
    assert fsub.function == -cos(x) + 1
    assert _same(fsub.truncate(), x**2/2 - x**4/24 + O(x**6))
    raises(ValueError, lambda: f1 + fps(exp(x), dir=-1))
    raises(ValueError, lambda: f1 + fps(exp(x), x0=1))

    fm = _series(f1 * 3)
    assert fm.function == 3*sin(x)
    assert _same(fm.truncate(), 3*x - x**3/2 + x**5/40 + O(x**6))
    fm = _series(3 * f2)
    assert fm.function == 3*cos(x)
    assert _same(fm.truncate(), 3 - 3*x**2/2 + x**4/8 + O(x**6))
    assert (f1 * f2) == mul(f1, f2)
    assert (f1 * x) == mul(f1, x)

    fd = _series(f1.diff(x))
    assert fd.function == cos(x)
    assert _same(fd.truncate(), 1 - x**2/2 + x**4/24 + O(x**6))
    fd = _series(f2.diff(x))
    assert fd.function == -sin(x)
    assert _same(fd.truncate(), -x + x**3/6 - x**5/120 + O(x**6))
    fd = _series(f2.diff(x).diff(x))
    assert fd.function == -cos(x)
    assert _same(fd.truncate(), -1 + x**2/2 - x**4/24 + O(x**6))

    f3 = _series(fps(exp(sqrt(x))))
    assert _same(_series(f3.diff(x)).truncate().expand(),
                 1/(2*sqrt(x)) + S.Half + x/12 + x**2/240 + x**3/10080 + x**4/725760 + x**5/79833600 + sqrt(x)/4
                 + x**Rational(3, 2)/48 + x**Rational(5, 2)/1440 + x**Rational(7, 2)/80640
                 + x**Rational(9, 2)/7257600 + x**Rational(11, 2)/958003200 + O(x**6))

    assert f1.integrate((x, 0, 1)) == -cos(1) + 1
    fi = _series(f1.integrate(x))
    assert fi.function == -cos(x)
    assert _same(fi.truncate(), -1 + x**2/2 - x**4/24 + O(x**6))
    fi = _series(f2.integrate(x))
    assert fi.function == sin(x)
    assert _same(fi.truncate(), x - x**3/6 + x**5/120 + O(x**6))


def test_fps__product() -> None:
    f1, f2, f3 = _series(fps(sin(x))), _series(fps(exp(x))), _series(fps(cos(x)))
    raises(ValueError, lambda: f1.product(exp(x), x))
    raises(ValueError, lambda: f1.product(fps(exp(x), dir=-1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(x), x0=1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(y)), x, 4))

    fprod = f1.product(f2, x)
    assert isinstance(fprod, FormalPowerSeriesProduct)
    assert isinstance(fprod.ffps, FormalPowerSeries)
    assert isinstance(fprod.gfps, FormalPowerSeries)
    assert fprod.f == sin(x)
    assert fprod.g == exp(x)
    assert fprod.function == sin(x) * exp(x)
    assert fprod._eval_terms(4) == x + x**2 + x**3/3
    assert _same(fprod.truncate(4), x + x**2 + x**3/3 + O(x**4))
    assert fprod.polynomial(4) == x + x**2 + x**3/3

    raises(NotImplementedError, lambda: fprod._eval_term(Integer(5)))
    raises(NotImplementedError, lambda: fprod.infinite)
    raises(NotImplementedError, lambda: fprod._eval_derivative(x))
    raises(NotImplementedError, lambda: fprod.integrate(x))

    assert f1.product(f3, x)._eval_terms(4) == x - 2*x**3/3
    assert _same(f1.product(f3, x).truncate(4), x - 2*x**3/3 + O(x**4))


def test_fps__compose() -> None:
    f1, f2, f3 = _series(fps(exp(x))), _series(fps(sin(x))), _series(fps(cos(x)))
    raises(ValueError, lambda: f1.compose(sin(x), x))
    raises(ValueError, lambda: f1.compose(fps(sin(x), dir=-1), x, 4))
    raises(ValueError, lambda: f1.compose(fps(sin(x), x0=1), x, 4))
    raises(ValueError, lambda: f1.compose(fps(sin(y)), x, 4))
    raises(ValueError, lambda: f1.compose(f3, x))
    raises(ValueError, lambda: f2.compose(f3, x))

    fcomp = f1.compose(f2, x)
    assert isinstance(fcomp, FormalPowerSeriesCompose)
    assert isinstance(fcomp.ffps, FormalPowerSeries)
    assert isinstance(fcomp.gfps, FormalPowerSeries)
    assert fcomp.f == exp(x)
    assert fcomp.g == sin(x)
    assert fcomp.function == exp(sin(x))
    assert fcomp._eval_terms(6) == 1 + x + x**2/2 - x**4/8 - x**5/15
    assert _same(fcomp.truncate(), 1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6))
    assert _same(fcomp.truncate(5), 1 + x + x**2/2 - x**4/8 + O(x**5))

    raises(NotImplementedError, lambda: fcomp._eval_term(Integer(5)))
    raises(NotImplementedError, lambda: fcomp.infinite)
    raises(NotImplementedError, lambda: fcomp._eval_derivative(x))
    raises(NotImplementedError, lambda: fcomp.integrate(x))

    assert _same(f1.compose(f2, x).truncate(4), 1 + x + x**2/2 + O(x**4))
    assert _same(f1.compose(f2, x).truncate(8), 1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8))
    assert _same(f1.compose(f2, x).truncate(6), 1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6))
    assert _same(f2.compose(f2, x).truncate(4), x - x**3/3 + O(x**4))
    assert _same(f2.compose(f2, x).truncate(8), x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8))
    assert _same(f2.compose(f2, x).truncate(6), x - x**3/3 + x**5/10 + O(x**6))


def test_fps__inverse() -> None:
    f1, f2, f3 = _series(fps(sin(x))), _series(fps(exp(x))), _series(fps(cos(x)))
    raises(ValueError, lambda: f1.inverse(x))

    finv = f2.inverse(x)
    assert isinstance(finv, FormalPowerSeriesInverse)
    assert isinstance(finv.ffps, FormalPowerSeries)
    raises(ValueError, lambda: finv.gfps)
    assert finv.f == exp(x)
    assert finv.function == exp(-x)
    assert finv._eval_terms(5) == 1 - x + x**2/2 - x**3/6 + x**4/24
    assert _same(finv.truncate(), 1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6))
    assert _same(finv.truncate(5), 1 - x + x**2/2 - x**3/6 + x**4/24 + O(x**5))

    raises(NotImplementedError, lambda: finv._eval_term(Integer(5)))
    raises(ValueError, lambda: finv.g)
    raises(NotImplementedError, lambda: finv.infinite)
    raises(NotImplementedError, lambda: finv._eval_derivative(x))
    raises(NotImplementedError, lambda: finv.integrate(x))

    assert _same(f2.inverse(x).truncate(8),
                 1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + x**6/720 - x**7/5040 + O(x**8))
    assert _same(f3.inverse(x).truncate(), 1 + x**2/2 + 5*x**4/24 + O(x**6))
    assert _same(f3.inverse(x).truncate(8), 1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8))


def test_coefficients_agree_with_the_taylor_series() -> None:
    # an independent check: the truncations against the core's series (SymPy
    # finds no formula for log(1 + x)/(1 + x) and exp(x)/(1 - x): their
    # recurrences have d'Alembertian solutions, found by rsolve_linear)
    for e in [exp(x)*cos(x), log(1 + x)/(1 + x), exp(x)/(1 - x), 1/(1 - 3*x + 2*x**2), sin(x)**2, exp(-x**2),
              asinh(x), sqrt(1 + x), exp(2*x)*sin(3*x), (1 - cos(x))/x**2]:
        s = _series(fps(e, x))
        assert _same(s.truncate(7), e.series(x, 0, 7)), e


def test_coeff_and_bell() -> None:
    assert coeff_of(x**3 + 2*x + 5, x, 0) == 5
    assert coeff_of(x**3 + 2*x + 5, x, 3) == 1
    assert coeff_of(x**3 + 2*x + 5, x, -1) == 0
    x1, x2, x3 = Symbol('x1'), Symbol('x2'), Symbol('x3')
    assert bell_incomplete(4, 2, [x1, x2, x3]) == 4*x1*x3 + 3*x2**2
    assert bell_incomplete(6, 2, [x1, x2, x3, x1, x2]) == expand(6*x1*x2 + 15*x2*x1 + 10*x3**2)
    assert isinstance(_series(fps(x**3 + 1)).ak, SeqFormula)

