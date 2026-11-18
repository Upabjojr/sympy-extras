"""``expand(trig=True)`` of the Rust core against
``sympy_extras.core.function.expand``: the trigonometric, hyperbolic and
reciprocal functions of sums and of integer multiples (nested, with
other hints), compared by structure and printing. ``complex=True`` only
on the functions which are not nested: the real and imaginary parts of
nested ones swell beyond any time limit in Python and in Rust."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import shape

HINTS: list[dict[str, bool]] = [{}, {"deep": False}, {"mul": False, "multinomial": False, "log": False},
                                {"power_exp": False}, {"complex": True}]


def argument(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol, pi
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z', real=True)
    leaf = rng.choice([x, y, z, 2*x, 3*y, -x, 4*x, x + y, x + y + z, 2*x + y, pi/2 - x, pi + x, Rational(1, 2)*x,
                       5*z, Integer(2)])
    if depth == 0 or rng.random() < 0.5:
        return leaf
    return leaf + function(rng, depth - 1)


def function(rng: random.Random, depth: int) -> Any:
    from sympy_extras.functions import cos, cosh, cot, coth, csc, csch, sec, sech, sin, sinh, tan, tanh
    f = rng.choice([sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch])
    return f(argument(rng, depth))


@pytest.mark.parametrize("seed", range(40))
def test_expand_trig(rust: Any, seed: int) -> None:
    from sympy_extras.core.function import expand
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("expand trig", seed))
    compared = 0
    for _ in range(5):
        depth = rng.randint(0, 1)
        e = function(rng, depth)
        product = rng.random() < 0.3
        if product:
            e = e * function(rng, 0) + 1
        for hints in HINTS:
            if hints.get("complex") and (depth or product):
                continue  # the real and imaginary parts of nested functions swell (in both)
            try:
                expected = expand(e, trig=True, **hints)
            except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, NotImplementedError):
                continue
            try:
                got = rc.expand(e, trig=True, **hints)
            except NotImplementedError:
                continue
            assert shape(got) == shape(expected) and rc.sstr(got) == str(expected), (e, hints)
            compared += 1
    assert compared >= 10
