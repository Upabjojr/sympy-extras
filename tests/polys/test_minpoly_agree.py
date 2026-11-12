"""``minimal_polynomial`` of the Rust port against
``sympy_extras.polys.numberfields.minpoly``: random sums, products and
rational powers of rationals, square and cube roots, ``I``, the golden
ratio, and ``cos``, ``sin``, ``tan``, ``exp(I*pi*c)`` at rational
multiples of ``pi``, and the non-algebraic and symbolic inputs, compared
through ``repr`` and the class of the error."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_polytools_functions_agree import compare


def leaves(rng: random.Random) -> list[Any]:
    from sympy_extras.core import GoldenRatio, I, Integer, Rational, pi, sqrt
    from sympy_extras.core.power import pow
    from sympy_extras.functions import cos, exp, sin, tan
    angle = Rational(rng.randint(1, 11), rng.choice([3, 4, 5, 6, 8, 12]))*pi
    return [Integer(rng.randint(-5, 5)), Rational(rng.randint(-4, 4), rng.randint(1, 4)), sqrt(Integer(rng.randint(2, 7))),
            pow(Integer(rng.randint(2, 5)), Rational(1, 3)), I, GoldenRatio, cos(angle), sin(angle), tan(angle),
            exp(I*angle)]


def random_algebraic(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    if depth == 0:
        return rng.choice(leaves(rng))
    c = rng.random()
    if c < 0.45:
        return add(random_algebraic(rng, depth - 1), random_algebraic(rng, depth - 1))
    if c < 0.85:
        return mul(random_algebraic(rng, depth - 1), random_algebraic(rng, depth - 1))
    return pow(random_algebraic(rng, depth - 1), rng.choice([Integer(2), Integer(-1), Rational(1, 2), Rational(2, 3)]))


@pytest.mark.parametrize("seed", range(40))
def test_minimal_polynomial(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.polys.numberfields.minpoly import minimal_polynomial
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("minpoly", seed))
    x = Symbol('x')
    compared = 0
    for _ in range(3):
        e = random_algebraic(rng, rng.choice([1, 2, 2]))
        compared += compare(rc, "minimal_polynomial", lambda: minimal_polynomial(e, x), [e], [x])
    assert compared >= 1


def test_edge_cases(rust: Any) -> None:
    from sympy_extras.core import E, I, Integer, Rational, Symbol, TribonacciConstant, pi, sqrt
    from sympy_extras.functions import cos, exp, log
    from sympy_extras.polys.numberfields.minpoly import minimal_polynomial
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    for e in [Integer(0), Integer(3), Rational(-2, 3), I, TribonacciConstant, sqrt(2) + sqrt(3), (1 + sqrt(5))/2,
              cos(pi/7), exp(2*I*pi/5), pi, E, log(2), cos(1), sqrt(2)**Rational(1, 2), 1/(sqrt(2) - 1), y + 1,
              Integer(0)**-1, (sqrt(2) + sqrt(3))**Rational(-2, 3)]:
        compare(rc, "minimal_polynomial", lambda: minimal_polynomial(e, x), [e], [x])
