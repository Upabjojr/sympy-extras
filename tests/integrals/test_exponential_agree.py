"""The Rust port of ``integrals/exponential.py`` against the Python one (the
integrands its tests use, route by route), and the values of the
confluent hypergeometric function at large arguments (mpmath's asymptotic
series of ``_hyp1f1``, with the convergent series as its fallback)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import I, Integer, Rational, Symbol, Tuple, pi, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.functions import exp, hyper, log
import sympy_extras.integrals.exponential as EX

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, a, b, c, v, p, r = symbols('x a b c v p r')


def outcome(f: Callable[[], object]) -> str:
    try:
        res = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if res is None or str(res) == 'NONE' else str(res)


P = [x > 0]
CASES: list[tuple[str, Basic, list[Basic]]] = [
    ('composite_gaussian', (1 - 1/x**2)*exp(-(x + 1/x)**2), []), ('composite_gaussian', exp((x + 1/x)**2)/x**2, []),
    ('composite_gaussian', (x + 2*log(x))*exp(-(x + log(x)**2)**2)/x, []),
    ('exponential_antiderivative', exp(2*x)/(x + 1)**2, []), ('exponential_antiderivative', exp(2/x)/x**3, P),
    ('exponential_antiderivative', exp(sqrt(x)), P), ('exponential_antiderivative', (exp(x) + 2)**Rational(3, 2)/(exp(x) + 1), []),
    ('exponential_antiderivative', exp(x**2)/x**3, []), ('exponential_antiderivative', exp(-x**3)/x**2, P),
    ('exponential_antiderivative', exp(x**3), P), ('exponential_antiderivative', exp(-x)/x**Rational(2, 3), P),
    ('exponential_antiderivative', sqrt(x)*exp(-2*x**2), P), ('exponential_antiderivative', sqrt(x)*exp(-x), P),
    ('exponential_antiderivative', (x + 1)*exp(-x**3 - 3*x**2 - 3*x), []),
    ('exponential_antiderivative', x**2*exp(-x**2 + 3*x), []), ('exponential_antiderivative', x**2*exp(x**2), []),
    ('exponential_antiderivative', x**3*exp(x**2), []), ('exponential_antiderivative', x**5*exp(-x**3), []),
    ('exponential_antiderivative', x*exp(x**2 + 2*x), []), ('exponential_antiderivative', x**(v - 1)*exp(a*x**2), []),
    ('exponential_antiderivative', x**(v - 1)*exp(a*x**2), P), ('exponential_antiderivative', x**(v - 1)*exp(-x), P),
    ('nested_power_exponential', exp(b*sqrt(x**2)), []), ('nested_power_exponential', exp(b*(x**r)**p), []),
    ('power_exponential', exp(2*x)/x, []), ('power_exponential', exp(b*x**2), []), ('power_exponential', exp(-x**2), []),
    ('power_exponential', exp(x**2), []), ('power_exponential', x**(v - 1)*exp(p*x**2), P),
    ('rational_exponential', exp(c*x)/(x**2 - 1), []), ('rational_exponential', exp(x)/(x**2 - pi**2), []),
    ('rational_exponential', (x**3 + 1)*exp(2*x)/(x - 1)**2, []),
    ('reciprocal_square_exponential', exp(-x**2 - 1/x**2), []), ('reciprocal_square_exponential', exp(-x**2), []),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_route(i: int) -> None:
    route, f, assumptions = CASES[i]
    got = outcome(lambda: rc.logic_call('exponential', [Symbol(route), f, x, Tuple(*assumptions)]))
    if got != 'SKIP':
        assert got == outcome(lambda: getattr(EX, route)(f, x, assumptions or None)), (route, f)


@pytest.mark.parametrize('a_, b_, z', [
    (Rational(5, 3), Rational(8, 3), Integer(-107)), (Rational(2, 3), Rational(5, 3), Integer(-200)),
    (Rational(1, 2), Rational(3, 2), Integer(100)), (Rational(1, 3), Rational(1, 2), -70 + 30*I),
    (Rational(5, 3), Rational(8, 3), Integer(-64)), (Rational(1, 2), Rational(3, 2), Integer(-1000)),
])
def test_hyp1f1_large_arguments(a_: Basic, b_: Basic, z: Basic) -> None:
    e = hyper((a_,), (b_,), z)
    re, im = rc.logic_call('reliable', [Symbol('value'), e, Integer(30), Tuple()]).args
    expected = complex(e.evalf(30))
    got = complex(float(re), float(im))
    assert abs(got - expected) <= 1e-14*abs(expected)
