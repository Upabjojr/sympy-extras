"""The Rust ``factor`` of polynomials with Gaussian coefficients (over
``ZZ_I`` and ``QQ_I``, ``polytools._gaussian_factor_list``) against the
Python one, compared as strings."""
from __future__ import annotations

import random

import pytest

from sympy_extras.core import I, Integer, Rational, Symbol, expand
from sympy_extras.core.basic import Basic
from sympy_extras.polys.polytools import factor

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = Symbol('x'), Symbol('y')


def _random_cases(seed: int, count: int) -> list[Basic]:
    rng = random.Random(seed)

    def coeff() -> Basic:
        if rng.random() < 0.7:
            return Integer(rng.randint(-3, 3)) + I*Integer(rng.randint(-3, 3))
        return Rational(rng.randint(-3, 3), rng.randint(1, 3)) + I*Rational(rng.randint(-2, 2), 2)

    def poly(variables: list[Basic], deg: int) -> Basic:
        total: Basic = Integer(0)
        for i in range(deg + 1):
            for j in range(deg + 1 - i if len(variables) > 1 else 1):
                total = total + coeff()*variables[0]**i*(variables[1]**j if len(variables) > 1 else 1)
        return total

    cases: list[Basic] = []
    for _ in range(count):
        variables = [x] if rng.random() < 0.6 else [x, y]
        cases.append(expand(poly(variables, rng.randint(1, 2))*poly(variables, rng.randint(1, 2))))
    return cases


CASES: list[Basic] = [x**2 + 1, x**4 - 1, x**2 + 2*I*x - 1, I*x**2 - I, x*(x**2 + 1)**2, (x + I)*(x - 2*I),
                      x**2*y + y*I, x**2 + y**2, (x + I*y)**2*(x - y), x**3*(I*x + 1), Rational(1, 2)*x**2 + I/2,
                      (2 + 2*I)*x**2 - 4*I] + _random_cases(1, 40)


@pytest.mark.parametrize('i', range(len(CASES)))
def test_gaussian_factor(i: int) -> None:
    e = CASES[i]
    assert str(rc.polytools_call('factor', [e], [])) == str(factor(e))
