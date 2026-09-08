from __future__ import annotations

from sympy import (factorial, binomial, log, sqrt, sin, cos, exp, Abs, Eq, And, Or, S, Sum, oo,
    pi, Rational)
from sympy.abc import n, x, p

from sympy_extras.concrete.convergence import sum_convergence, product_convergence, is_convergent


def test_sum_convergence_numbers() -> None:
    # decided series, checked against SymPy's is_convergent and the textbooks
    assert sum_convergence(factorial(n)/n**n, n) is S.true            # ratio test, limit 1/e
    assert sum_convergence(binomial(2*n, n)/4**n, n) is S.false       # Raabe, rho = 1/2 (Knopp, p. 288)
    assert sum_convergence(1/(n**2 + 1), n) is S.true
    assert sum_convergence(1/n, n) is S.false
    assert sum_convergence(1/sqrt(n), n) is S.false
    assert sum_convergence(1/(n*log(n)), n, lower=2) is S.false      # integral test
    assert sum_convergence(1/(n*log(n)**2), n, lower=2) is S.true
    assert sum_convergence((n/(n + 1))**(n**2), n) is S.true          # root test, limit 1/e
    assert sum_convergence(n**3*exp(-n), n) is S.true
    assert sum_convergence(sin(n)/n**2, n) is S.true
    assert sum_convergence(1, n) is S.false
    assert sum_convergence(0, n) is S.true
    # alternating series
    assert sum_convergence((-1)**n/n, n) is S.true
    assert sum_convergence((-1)**n/log(n), n, lower=2) is S.true
    assert sum_convergence((-1)**n/sqrt(n), n) is S.true
    assert sum_convergence((-1)**n, n) is S.false
    assert sum_convergence(cos(n*S.Pi)/n, n) is S.true
    # terms which do not tend to zero
    assert sum_convergence(sin(n), n) is S.false
    assert sum_convergence(2**n/factorial(n), n) is S.true
    for term, start in ((factorial(n)/n**n, 1), (1/(n**2 + 1), 1), (binomial(2*n, n)/4**n, 1), (1/(n*log(n)), 2)):
        assert bool(Sum(term, (n, start, oo)).is_convergent()) == (sum_convergence(term, n, lower=start) is S.true)


def test_sum_convergence_parameters() -> None:
    # Mathematica: SumConvergence[x^n, n] is Abs[x] < 1
    assert sum_convergence(x**n, n) == And(x > -1, x < 1)
    assert sum_convergence(n*x**n, n) == And(x > -1, x < 1)
    # SumConvergence[x^n/n, n] is Abs[x] < 1 || x == -1
    assert sum_convergence(x**n/n, n) == And(x >= -1, x < 1)
    assert sum_convergence(x**n/n**2, n) == And(x >= -1, x <= 1)
    assert sum_convergence(x**n/n, n, x < 0) == (x >= -1)
    # the p-series: Raabe's test with the boundary p = 1
    assert sum_convergence(1/n**p, n) == (p > 1)
    assert sum_convergence(1/n**p, n, p > 2) is S.true
    assert sum_convergence(1/n**p, n, p < 1) is S.false
    assert sum_convergence(1/(2*n + 1)**p, n, lower=0) == (p > 1)
    assert sum_convergence(n**x, n) == (x < -1)
    assert sum_convergence(1/(n**p + 1), n, p > 0) == (p > 1)
    assert sum_convergence(1/n**(p + 1), n, p > 0) is S.true
    assert sum_convergence(x**n/factorial(n), n) is S.true
    assert sum_convergence(1/(n**2 + x**2), n) is S.true
    # two parameters: the boundary x = 1 is examined
    result = sum_convergence(n**p*x**n, n, x > 0)
    assert result is not None
    assert result == Or(x < 1, And(Eq(x, 1), p < -1)) or result.subs({x: 1, p: -2}) is S.true and \
        result.subs({x: 1, p: 0}) is S.false and result.subs({x: Rational(1, 2), p: 5}) is S.true
    # alternating with a parameter: (-1)**n * n**p converges exactly for p < 0
    assert sum_convergence((-1)**n*n**p, n) == (p < 0)
    assert sum_convergence((-1)**n/n**p, n, p > 0) is S.true
    assert sum_convergence(p**n/n**2, n, p > 0) == (p <= 1)


def test_is_convergent() -> None:
    assert is_convergent(x**n, n, (x > 0) & (x < 1)) is True
    assert is_convergent(x**n, n, x > 1) is False
    assert is_convergent(x**n, n) is None
    assert is_convergent(1/n**2, n) is True
    assert is_convergent(1/n, n) is False


def test_product_convergence() -> None:
    assert product_convergence(1 + 1/n**2, n) is S.true
    assert product_convergence(1 - 1/n**2, n, lower=2) is S.true
    assert product_convergence(1 + x/n**2, n) is S.true
    assert product_convergence(1 + 1/n, n) is S.false
    assert product_convergence(n/(n + 1), n) is S.false
    assert product_convergence(cos(1/n), n) is S.true
    assert product_convergence(1 + x**n, n) == And(x > -1, x < 1)
    assert product_convergence(2, n) is S.false
    assert product_convergence(1, n) is S.true
    assert product_convergence(1 + Abs(x)/n, n) == Eq(x, 0)


def test_dirichlet_test() -> None:
    # a bounded oscillating factor times a factor decreasing to zero
    # converges; sympy's Sum(sin(n)/n, (n, 1, oo)).is_convergent() says
    # that this series diverges, and the answer used to be taken from it
    assert sum_convergence(sin(n)/n, n) is S.true
    assert sum_convergence(cos(n)/n, n) is S.true
    assert sum_convergence(sin(2*n)/n, n) is S.true
    assert sum_convergence(sin(n)/sqrt(n), n) is S.true
    assert sum_convergence(sin(n)/log(n + 1), n) is S.true
    assert sum_convergence(sin(pi*n/3)/n, n) is S.true
    # the factor is constant when the frequency is a multiple of 2*pi
    assert sum_convergence(cos(2*pi*n)/n, n) is S.false
    # and the terms have to tend to zero
    assert sum_convergence(sin(n), n) is S.false
    assert sum_convergence(n*sin(n), n) is S.false
