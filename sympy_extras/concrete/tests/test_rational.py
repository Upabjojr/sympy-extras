from __future__ import annotations

import random

from sympy import symbols, cancel, Sum, polygamma, simplify, S
from sympy.testing.pytest import raises

from sympy_extras.concrete.rational import abramov_decomposition, rational_sum
from sympy_extras._typing import as_expr

k, n = symbols('k n')


def _consistent(f: object) -> None:
    d = abramov_decomposition(as_expr(f), k)
    assert cancel(d.g.subs(k, k + 1) - d.g + d.h - as_expr(f)) == 0


def test_abramov_decomposition() -> None:
    d = abramov_decomposition(1/(k*(k + 1)), k)
    assert d.summable and cancel(d.g + 1/k) == 0
    d = abramov_decomposition(1/k, k)
    assert not d.summable and d.h == 1/k and d.g == 0
    d = abramov_decomposition((2*k + 1)/(k**2*(k + 1)**2), k)
    assert d.summable
    d = abramov_decomposition(1/(k*(k + 2)), k)
    assert d.summable and cancel(d.g + (2*k + 1)/(2*k*(k + 1))) == 0
    d = abramov_decomposition(1/(k*(2*k + 1)), k)
    assert not d.summable        # the shift between k and k + 1/2 is not an integer
    d = abramov_decomposition(k**2 + 1/(k**2 + 1) - 1/((k + 1)**2 + 1), k)
    assert d.summable
    d = abramov_decomposition(1/(k**2 + 1), k)
    assert not d.summable
    for f in (1/(k*(k + 1)*(k + 2)), (k + 3)/(k*(k + 1)*(k + 3)), 1/((k**2 + 1)*((k + 1)**2 + 1)),
              k**3/(k + 1), 1/(k*(k + 1)) + 1/k):
        _consistent(f)
    raises(ValueError, lambda: abramov_decomposition(2**k, k))


def test_random_decompositions() -> None:
    rng = random.Random(3)
    for _ in range(12):
        factors = [k + rng.randint(-2, 3) for _ in range(rng.randint(1, 3))]
        numerator = sum((rng.randint(-3, 3)*k**i for i in range(rng.randint(1, 3))), S.Zero) + 1
        f = numerator/__import__('sympy').Mul(*factors)
        _consistent(f)


def test_rational_sum() -> None:
    assert rational_sum(1/(k*(k + 1)), (k, 1, n)) == n/(n + 1)
    total = rational_sum(1/(k*(k + 2)), (k, 1, n))
    for value in (1, 2, 5):
        assert simplify(total.subs(n, value) - Sum(1/(k*(k + 2)), (k, 1, value)).doit()) == 0
    total = rational_sum(1/k, (k, 1, n))
    assert total.has(polygamma) or total.has(Sum) or total.has(__import__('sympy').harmonic)
