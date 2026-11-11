"""``together`` of the Rust port against
``sympy_extras.polys.rationaltools``, with its options, on random sums of
quotients, powers of them and functions of them (results by structure,
classes and reprs)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from ..core.test_arithmetic_agree import shape
from .test_polyutils_agree import random_expression


def random_rational(rng: random.Random) -> Any:
    from sympy_extras.core import Integer
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.functions.elementary.exponential import exp
    terms = []
    for _ in range(rng.randint(2, 4)):
        numerator = random_expression(rng, 2)
        denominator = random_expression(rng, 2)
        if denominator == 0:
            denominator = Integer(1)
        terms.append(mul(numerator, pow(denominator, Integer(-1))))
    e = add(*terms)
    kind = rng.random()
    if kind < 0.2:
        return pow(e, Integer(rng.randint(-2, 3)))
    if kind < 0.35:
        return mul(random_expression(rng, 1), exp(e))
    return e


@pytest.mark.parametrize("seed", range(20))
def test_together_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.polys.rationaltools import together
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("together", seed))
    compared = 0
    for _ in range(10):
        try:
            e = random_rational(rng)
        except ZeroDivisionError:
            continue
        for options in ({}, {"deep": True}, {"fraction": False}):
            expected = together(e, **options)
            try:
                got = rc.together(e, **options)
            except NotImplementedError:
                continue
            assert shape(got) == shape(expected) and repr(got) == repr(expected), (e, options, got, expected)
            compared += 1
    assert compared >= 12
