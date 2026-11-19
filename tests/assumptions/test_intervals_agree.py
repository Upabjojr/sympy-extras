"""The Rust interval arithmetic (``assumptions/ivarith.rs``, mpmath's
``iv`` at 53 bits) and the enclosures of ``assumptions/intervals.py``
against the Python ones: the exact endpoints."""
from __future__ import annotations

import random

import mpmath
import pytest

from sympy_extras.assumptions.intervals import evaluate
from sympy_extras.core import E, Integer, Rational, Symbol, Tuple, oo, pi
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import mpf_to_fraction
from sympy_extras.core.numbers import number_from_fraction
from sympy_extras.functions import Abs, atan, cos, cosh, exp, log, sin, sinh, sqrt, tan, tanh

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = Symbol('x'), Symbol('y')


def _endpoint(m: object) -> str:
    v = mpmath.mpf(m)
    if v == mpmath.inf:
        return 'oo'
    if v == -mpmath.inf:
        return '-oo'
    return str(number_from_fraction(mpf_to_fraction(v)))


def _random_cases(seed: int, count: int) -> list[tuple[Basic, tuple[Basic, Basic]]]:
    rng = random.Random(seed)
    functions = [exp, log, sin, cos, tan, atan, sinh, cosh, tanh, Abs, sqrt]

    def leaf() -> Basic:
        return rng.choice([x, y, x, Integer(rng.randint(-5, 5)), Rational(rng.randint(-9, 9), rng.randint(1, 7)), pi, E])

    def gen(d: int) -> Basic:
        if d == 0:
            return leaf()
        k = rng.random()
        if k < 0.25:
            return gen(d - 1) + gen(d - 1)
        if k < 0.5:
            return gen(d - 1)*gen(d - 1)
        if k < 0.65:
            return gen(d - 1)**rng.choice([Integer(2), Integer(3), Integer(-1), Integer(-2), Rational(1, 3), Rational(3, 2)])
        if k < 0.9:
            return rng.choice(functions)(gen(d - 1))
        return gen(d - 1) - gen(d - 1)

    cases = []
    for _ in range(count):
        lo = Rational(rng.randint(-20, 20), rng.randint(1, 5))
        cases.append((gen(rng.randint(1, 2)), (lo, lo + Rational(rng.randint(0, 20), rng.randint(1, 5)))))
    return cases


CASES: list[tuple[Basic, tuple[Basic, Basic]]] = [
    (exp(x), (Integer(0), Integer(1))), (sin(x), (Integer(0), Integer(4))), (cos(x), (Integer(-1), Integer(1))),
    (tan(x), (Integer(1), Integer(2))), (atan(x), (-oo, oo)), (log(x), (Integer(1), Integer(3))),
    (sqrt(x), (Integer(0), Integer(2))), (sinh(x)*cosh(x), (Integer(-1), Integer(2))), (tanh(x), (Integer(-3), Integer(3))),
    (Abs(x - 1), (Integer(-2), Integer(2))), (pi*x + E, (Integer(0), Integer(1))), (x**3 - 2*x, (Integer(-1), Integer(1))),
    (1/x, (Integer(1), Integer(2))), (1/x, (Integer(-1), Integer(1))), (exp(1/x), (Integer(1), oo)),
    (sin(x), (Integer(100), Integer(101))), (cos(x), (Rational(1, 3), Rational(1, 2))), (atan(E)*x, (Integer(-3), Integer(4))),
] + _random_cases(7, 300)


@pytest.mark.parametrize('i', range(len(CASES)))
def test_evaluate(i: int) -> None:
    e, (lo, hi) = CASES[i]
    box = {x: (lo, hi), y: (Integer(1), Integer(2))}
    v = evaluate(e, box)
    expected = 'NONE' if v is None else str((_endpoint(v.a), _endpoint(v.b)))
    r = rc.logic_call('intervals', [Symbol('evaluate'), e, Tuple(*[Tuple(s, a, b) for s, (a, b) in box.items()])])
    got = 'NONE' if str(r) == 'NONE' else str((str(r.args[0]), str(r.args[1])))
    assert got == expected, e
