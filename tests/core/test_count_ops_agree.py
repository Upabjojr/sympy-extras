"""``count_ops`` (plain and visual) of the core against its Rust port, on
random expressions with sums, products, quotients, powers, functions,
relations, Booleans, ``Mod``, undefined functions and derivatives."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import E, Function, Integer, Rational, Symbol, pi
    from sympy_extras.core.function import Derivative
    from sympy_extras.core.mod import Mod
    from sympy_extras.functions import cos, exp, log, sin
    x, y = Symbol('x'), Symbol('y')
    if depth == 0:
        return rng.choice([x, y, Integer(2), Integer(-3), Rational(1, 2), Rational(-2, 3), pi, E])
    a, b = expression(rng, depth - 1), expression(rng, depth - 1)
    f = Function('f')
    return rng.choice([a + b, a - b, a*b, a/b, -a, a**b, a**-1, sin(a), cos(a) - b, exp(a), log(a) + 1,
                       Mod(a, b), f(a), Derivative(f(x), x), -a - b, 2*a/3])


@pytest.mark.parametrize("seed", range(40))
def test_count_ops(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.function import count_ops
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("count_ops", seed))
    x = Symbol('x')
    for _ in range(5):
        e = expression(rng, rng.randint(1, 3))
        if rng.random() < 0.2:
            try:
                e = (e > 1) & (x < 2) if rng.random() < 0.5 else ~(e <= x)
            except TypeError:  # a comparison of a non-real number
                pass
        assert outcome(lambda: rc.logic_call('count_ops', [e])) == outcome(lambda: count_ops(e)), e
        assert outcome(lambda: rc.logic_call('count_ops_visual', [e])) == \
            outcome(lambda: count_ops(e, visual=True)), e
