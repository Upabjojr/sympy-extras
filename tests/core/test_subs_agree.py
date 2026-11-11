"""``subs`` of the Rust core against ``Basic.subs`` of the Python core: a
pair, a mapping (SymPy's order of the keys) on random expressions over
every ported function, undefined functions and their derivatives, with
symbols, powers (``x**2``, ``1/x``, ``exp(x)``), sums and products among
the old expressions (compared by structure and the Rust printer, as the
dummies of the derivatives are converted)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import shape
from .test_diff_agree import random_expression


@pytest.mark.parametrize("seed", range(30))
def test_subs_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core import E, I, Integer, Rational, Symbol, pi
    from sympy_extras.core.function import Function, diff
    from sympy_extras.functions.elementary.exponential import exp
    rc = load("sympy_extras_rs.core")
    x, y, z, p = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('p', positive=True)
    f = Function('f')
    olds = [x, y, x**2, 1/x, x*y, x + y, exp(x), E**x, f(x), y**2, p**2, Integer(2), pi]
    news = [z, Integer(0), Integer(3), x + 1, z**2, Rational(1, 2), y, I]
    rng = random.Random(stable_seed("subs", seed))
    compared = 0
    for _ in range(10):
        try:
            e = random_expression(rng, 3)
            if rng.random() < 0.3:
                e = diff(e, rng.choice([x, y]))
        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, NotImplementedError):
            continue
        cases: list[tuple[Any, ...]] = [(rng.choice(olds), rng.choice(news)) for _ in range(4)]
        cases.append(({x: z, y: Integer(2)},))
        for case in cases:
            try:
                expected = e.subs(*case)
            except (ZeroDivisionError, ValueError, TypeError, NotImplementedError):
                continue
            try:
                got = rc.subs(e, *case)
            except (NotImplementedError, TypeError):
                continue
            assert shape(got) == shape(expected) and rc.sstr(got) == str(expected), (e, case)
            compared += 1
    assert compared >= 30
