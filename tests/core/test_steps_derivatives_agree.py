"""``sympy_extras.steps.derivatives`` against its Rust port (``steps/core.rs``
and ``steps/derivatives.rs``): the trees of random derivatives printed by
``format_steps`` (at several depths, with and without the minor steps,
with and without ``branch``), their results and their numbers of steps."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Function, Integer, Rational, Symbol, pi
    from sympy_extras.functions import (acos, acosh, asin, asinh, atan, atanh, cos, cosh, cot, csc, exp, log, sec, sin,
                                        sinh, tan, tanh)
    x, y, u = Symbol('x'), Symbol('y'), Symbol('u')
    if depth == 0:
        return rng.choice([x, x, y, Integer(3), Rational(1, 2), pi, 2*x, x + 1, u*x])
    a, b = expression(rng, depth - 1), expression(rng, depth - 1)
    fs = [sin, cos, tan, cot, sec, csc, exp, log, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh]
    builders = [lambda: rng.choice(fs)(a), lambda: a + b, lambda: a*b, lambda: a/b, lambda: a**3, lambda: a**b,
                lambda: 2**a, lambda: x**Rational(1, 2), lambda: 5*a, lambda: Function('f')(a), lambda: a**y]
    while True:
        try:
            return rng.choice(builders)()
        except (ValueError, TypeError, ZeroDivisionError):
            continue


@pytest.mark.parametrize("seed", range(60))
def test_derivative_steps(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.steps.core import count_steps, format_steps
    from sympy_extras.steps.derivatives import derivative_steps
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("steps derivatives", seed))
    x = Symbol('x')
    for _ in range(3):
        e = expression(rng, rng.randint(1, 3))
        branch = rng.random() < 0.3
        depth = rng.choice([None, None, 0, 1, 2])
        minor = rng.random() < 0.3

        def expected() -> Any:
            steps = derivative_steps(e, x, branch=branch)
            return (format_steps(steps, depth=depth, minor=minor), str(steps.result()), count_steps(steps))

        def got() -> Any:
            r = rc.logic_call('derivative_steps', [e, x, Integer(int(branch)), Integer(-1 if depth is None else depth),
                                                   Integer(int(minor))])
            return (r.args[0].name, rc.sstr(r.args[1]), int(r.args[2]))
        assert outcome(got) == outcome(expected), (e, branch, depth, minor)
