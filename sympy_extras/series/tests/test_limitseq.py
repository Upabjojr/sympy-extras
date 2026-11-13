"""Tests of the port of SymPy's ``limit_seq`` on the typed core: the
known limits of SymPy's own tests (``sympy/series/tests/test_limitseq.py``,
the results are facts: hand computations), and the differences."""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/tests/test_limitseq.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the core's objects; the cases with sums whose closed
# forms need SymPy's summation are left out.
# ---------------------------------------------------------------------
from __future__ import annotations

from sympy_extras._testing import untyped
from typing import Callable

from sympy_extras.core import E, EulerGamma, GoldenRatio, I, Integer, Rational, Symbol, oo, pi, sqrt
from sympy_extras.core.accumbounds import accum_bounds
from sympy_extras.core.add import add
from sympy_extras.concrete.expr_with_limits import sum_
from sympy_extras.functions import binomial, cos, exp, factorial, fibonacci, gamma, log, sin, subfactorial
from sympy_extras.polys.polytools import cancel
from sympy_extras.series.limitseq import difference_delta as dd, dominant, limit_seq

n, m, k = Symbol('n', integer=True), Symbol('m', integer=True), Symbol('k', integer=True)


def _raises_value_error(f: Callable[..., object], *args: object) -> bool:
    try:
        untyped(f)(*args)
    except ValueError:
        return True
    return False


def test_difference_delta() -> None:
    e = n*(n + 1)
    assert dd(e, n) == 2*n + 2
    assert dd(e*k, n, 2) == k*(4*n + 6)
    assert _raises_value_error(dd, e*k, n, oo)


def test_difference_delta_of_sums() -> None:
    e = sum_(1/k, (k, 1, n))
    assert dd(e, n) == 1/(n + 1)
    assert dd(e, n, 5) == add(*[1/(i + n + 1) for i in range(5)])
    e = sum_(1/k, (k, 1, 3*n))
    assert dd(e, n) == add(*[1/(i + 3*n + 1) for i in range(3)])
    e = n*sum_(1/k, (k, 1, n))
    assert dd(e, n) == 1 + sum_(1/k, (k, 1, n))


def test_difference_delta_of_powers() -> None:
    assert dd(4**n, n) == 3*4**n and dd(4**n, n, 2) == 15*4**n
    assert dd(4**(2*n), n) == 15*4**(2*n) and dd(4**(2*n), n, 2) == 255*4**(2*n)
    assert dd(n**4, n) == (n + 1)**4 - n**4
    assert dd(n**n, n) == (n + 1)**(n + 1) - n**n
    assert dd(n + n*(n + 1), n) == 2*n + 3 and dd(n + n*(n + 1), n, 2) == 4*n + 8


def test_dominant() -> None:
    assert dominant(5*n**3 + 4*n**2 + n + 1, n) == 5*n**3
    assert dominant(2**n + 3**n + n**5, n) == 3**n


def test_limit_seq() -> None:
    assert limit_seq((5*n**3 + 3*n**2 + 4)/(3*n**3 + 4*n - 5), n) == Rational(5, 3)
    e = binomial(2*n, n)/4**n
    assert limit_seq(e, m) == e
    assert limit_seq(Integer(3), n) == 3


def test_alternating_sign() -> None:
    assert limit_seq((-1)**n/n**2, n) == 0
    assert limit_seq((-2)**(n + 1)/(n + 3**n), n) == 0
    assert limit_seq((2*n + (-1)**n)/(n + 1), n) == 2
    assert limit_seq(sin(pi*n), n) == 0
    assert limit_seq(cos(2*pi*n), n) == 1
    assert limit_seq(Rational(-1, 5)**n, n) == 0
    assert limit_seq((I/3)**n, n) == 0
    assert limit_seq(sqrt(n)*(I/2)**n, n) == 0
    assert limit_seq(n**7*(I/3)**n, n) == 0
    assert limit_seq(n/(n + 1) + (I/2)**n, n) == 1


def test_accum_bounds() -> None:
    assert limit_seq((-1)**n, n) == accum_bounds(-1, 1)
    assert limit_seq(cos(pi*n), n) == accum_bounds(-1, 1)
    assert limit_seq(sin(pi*n/2)**2, n) == accum_bounds(0, 1)
    assert limit_seq(2*(-3)**n/(n + 3**n), n) == accum_bounds(-2, 2)
    assert limit_seq(3*n/(n + 1) + 2*(-1)**n, n) == accum_bounds(1, 5)
    # (-2)**n stayed (-2)**n for an odd n (the core's power did not take
    # the parity of the exponent out of a negative base, SymPy's does):
    # the limit came out zoo instead of the bounds
    assert limit_seq((-2)**n, n) == accum_bounds(-oo, oo)


def test_factorials_and_gamma() -> None:
    assert limit_seq(subfactorial(n)/factorial(n), n) == exp(-1)
    assert limit_seq(5**n/factorial(n), n) == 0
    # the core's limit of exp(x)/gamma(x) was oo without SymPy's rewrite of
    # gamma (the tractable form exp(loggamma(x)) was SymPy's only)
    assert limit_seq(2**n/factorial(n), n) == 0
    assert limit_seq(1/gamma(n + Rational(1, 2)), n) == 0
    assert limit_seq(binomial(2*n, n)/4**n, n) == 0


def test_fibonacci() -> None:
    found = limit_seq(fibonacci(n + 1)/fibonacci(n), n)
    assert found is not None and cancel(found - GoldenRatio) == 0


def test_sums() -> None:
    x, y = Symbol('x'), Symbol('y')
    assert limit_seq(sum_(1/x, (x, 1, y)) - log(y), y) == EulerGamma
    assert limit_seq(sum_(1/x, (x, 1, y)) - 1/y, y) == oo


def test_exponential() -> None:
    assert limit_seq((1 + 1/n)**n, n) == E
