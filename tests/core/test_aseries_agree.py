"""The Rust port of the asymptotic series of the special functions
(``_eval_aseries`` of ``loggamma``, ``polygamma``, ``digamma``,
``trigamma``, ``lowergamma``, ``besseli``, ``besselk``) and of the limits
which need them, against the Python one (``limit(besseli(0, x)*exp(-x)*sqrt(x),
x, oo)`` is left out: the Python one does not return)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.limits import limit
from sympy_extras.functions import (besseli, besselk, binomial, digamma, exp, factorial, gamma, harmonic, log, loggamma,
                                    lowergamma, polygamma, rf, sqrt, subfactorial, trigamma)

from ..conftest import load

rc = load("sympy_extras_rs.core")

x = Symbol('x')
n = Symbol('n', positive=True, integer=True)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


#: (expression, the point) of the limits
LIMITS: list[tuple[Expr, Expr]] = [
    (loggamma(x) / (x * log(x)), oo),
    (gamma(x + 1) / (sqrt(2 * 3 * x) * x**x * exp(-x)), oo),
    (digamma(x) - log(x), oo),
    (trigamma(x) * x, oo),
    (polygamma(2, x) * x**2, oo),
    (polygamma(0, x + 1) - log(x), oo),
    (5**x / factorial(x), oo),
    (1 / gamma(x + Rational(1, 2)), oo),
    (binomial(2 * x, x) / 4**x, oo),
    (subfactorial(x) / factorial(x), oo),
    (harmonic(x) - log(x), oo),
    (harmonic(x, 2), oo),
    (rf(x, 3) / x**3, oo),
    (gamma(x + 1) / gamma(x), oo),
    (besselk(1, x) * exp(x) * sqrt(x), oo),
    (lowergamma(x, 2) * gamma(x + 1) / 2**x, oo),
    (loggamma(x) - (x - Rational(1, 2)) * log(x) + x, oo),
    (gamma(x) * x**Rational(1, 2) / x**x, oo),
    (factorial(x)**(1 / x) / x, oo),
]

#: (expression, number of terms) of the series at oo in x
SERIES: list[tuple[Expr, int]] = [
    (loggamma(x), 4), (digamma(x), 5), (trigamma(x), 4), (polygamma(2, x), 4), (polygamma(0, x), 1),
    (besseli(0, x) * exp(-x), 3), (besselk(Rational(1, 2), x) * exp(x), 3), (lowergamma(x, 1), 3),
]


@pytest.mark.parametrize('i', range(len(LIMITS)))
def test_limit(i: int) -> None:
    e, p = LIMITS[i]
    py = outcome(lambda: limit(e, x, p))
    rs = outcome(lambda: rc.logic_call('series', [Symbol('limit'), e, x, p, Symbol('+')]))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs


@pytest.mark.parametrize('i', range(len(SERIES)))
def test_series(i: int) -> None:
    e, k = SERIES[i]
    py = outcome(lambda: e.series(x, oo, k))
    args: list[Basic] = [Symbol('series'), e, x, oo, Integer(k), Symbol('+')]
    rs = outcome(lambda: rc.logic_call('series', args))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
