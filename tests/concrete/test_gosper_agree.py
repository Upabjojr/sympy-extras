"""The Rust Gosper's algorithm (``concrete/gosper.py``) against the Python
one: the terms, the indefinite and definite sums, the consecutive ratios,
compared as strings."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.concrete import gosper as gs
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.functions import binomial, factorial, rf, sin

from ..conftest import load

rc = load("sympy_extras_rs.core")
i, n, m = Symbol('i', integer=True), Symbol('n', integer=True), Symbol('m', integer=True)
a, b, x = Symbol('a'), Symbol('b'), Symbol('x')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
        return 'NONE' if r is None else str(r)
    except NotImplementedError:
        return 'SKIP'
    except Exception as e:
        return type(e).__name__


def call(op: str, *args: Basic) -> Basic:
    return rc.logic_call('gosper', [Symbol(op), *args])


TERMS: list[Basic] = [
    i*factorial(i), 1/(i*(i + 1)), binomial(m, i)*(-1)**i, i, i**2, 2**i, x**i, i*2**i, factorial(i), 1/i,
    binomial(n, i), i/((i + 1)*(i + 2)), factorial(2*i)/factorial(i)**2, rf(a, i)/factorial(i), 1/(i**2 + 3*i + 2),
    (4*i + 1)*factorial(i)/factorial(2*i + 1), i**3, binomial(i, 2), x**i*i**2,
    rf(a, i)*rf(b, i)/(factorial(i)*rf(a + b, i)), (i**2 - 1)/(i**2 + 1), 1/((2*i - 1)*(2*i + 1)),
    factorial(i + 1)/factorial(i - 1), a**i/factorial(i), 1/(i*(i + 2)),
]


@pytest.mark.parametrize('k', range(len(TERMS)))
def test_gosper(k: int) -> None:
    t = TERMS[k]
    pairs = [
        (lambda: call('term', t, i), lambda: gs.gosper_term(t, i)),
        (lambda: call('sum', t, i), lambda: gs.gosper_sum(t, i)),
        (lambda: call('sum', t, Tuple(i, Integer(0), n)), lambda: gs.gosper_sum(t, (i, Integer(0), n))),
        (lambda: call('sum', t, Tuple(i, Integer(1), n)), lambda: gs.gosper_sum(t, (i, Integer(1), n))),
        (lambda: call('ratio', t, i), lambda: gs.consecutive_ratio(t, i)),
    ]
    for rust, python in pairs:
        r = outcome(rust)
        if r != 'SKIP':
            assert r == outcome(python), t


def test_ratio_of_non_hypergeometric() -> None:
    assert outcome(lambda: call('ratio', sin(i), i)) == outcome(lambda: gs.consecutive_ratio(sin(i), i))
