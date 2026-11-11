"""``expand`` of the Rust core with all its hints (``mul``,
``multinomial``, ``power_base``, ``power_exp``, ``log``, ``force``,
``deep``) against ``sympy_extras.core.function.expand``, on random
expressions with powers of products and of sums, logarithms, exponentials
and symbols with signs. ``force`` is not combined with logarithms:
``expand(log(-sin(sin(12))) + I*pi, force=True)`` does not terminate in
Python (``log(-x)`` is split into ``log(-1) + log(x)`` while ``log(x)`` of
the negative ``x`` evaluates back to ``I*pi + log(-x)``), nor in the
port."""

from __future__ import annotations

import importlib
import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import shape

HINTS: list[dict[str, bool]] = [{}, {"log": False}, {"mul": False}, {"power_base": False, "power_exp": False},
                                {"force": True}, {"deep": False}, {"multinomial": False, "mul": False, "log": False}]


def random_expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    ex = importlib.import_module("sympy_extras.functions.elementary.exponential")
    tr = importlib.import_module("sympy_extras.functions.elementary.trigonometric")
    x, y = Symbol('x'), Symbol('y')
    leaves = [x, y, Symbol('p', positive=True), Symbol('q', positive=True), Symbol('n', negative=True),
              Symbol('r', real=True), Symbol('k', integer=True), Integer(rng.randint(-3, 4)), Rational(1, 2), I,
              Integer(12), Integer(8)]
    if depth == 0:
        return rng.choice(leaves)
    c = rng.random()
    if c < 0.25:
        return add(random_expression(rng, depth - 1), random_expression(rng, depth - 1))
    if c < 0.5:
        return mul(random_expression(rng, depth - 1), random_expression(rng, depth - 1))
    if c < 0.75:
        exponent = rng.choice([Integer(2), Integer(3), Integer(-1), Rational(1, 2), Rational(3, 2), y,
                               add(x, Integer(1)), Symbol('k', integer=True)])
        return pow(random_expression(rng, depth - 1), exponent)
    if c < 0.88:
        return ex.log(random_expression(rng, depth - 1))
    if c < 0.95:
        return ex.exp(random_expression(rng, depth - 1))
    return tr.sin(random_expression(rng, depth - 1))


@pytest.mark.parametrize("seed", range(30))
def test_expand_hints_agree(rust: Any, seed: int) -> None:
    from sympy_extras.core.function import expand
    from sympy_extras.functions.elementary.exponential import Log
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("expand hints", seed))
    compared = 0
    for _ in range(8):
        try:
            e = random_expression(rng, 3)
        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, NotImplementedError):
            continue
        for hints in HINTS:
            if hints.get("force") and e.atoms(Log):
                continue
            try:
                expected = expand(e, **hints)
            except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, NotImplementedError):
                continue
            try:
                got = rc.expand(e, **hints)
            except (NotImplementedError, TypeError):
                continue
            assert shape(got) == shape(expected) and rc.sstr(got) == str(expected), (e, hints)
            compared += 1
    assert compared >= 20
