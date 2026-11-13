"""``exp`` and ``log`` of the Rust core against
``sympy_extras.functions.elementary.exponential``, mixed with the
evaluating arithmetic (so that the ``E**x`` and ``exp(x)*exp(y)`` rules
of ``pow`` and ``mul`` are exercised), on random canonical operands: the
results are compared by structure, classes and reprs, and their facts
(``facts_of``) too. The cases which need what the Rust core does not have
yet raise ``NotImplementedError`` and are counted, not compared."""

from __future__ import annotations

import importlib
import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, convertible, python_function, shape


def functions() -> dict[str, tuple[Callable[..., Any], int]]:
    ex = importlib.import_module("sympy_extras.functions.elementary.exponential")
    table: dict[str, tuple[Callable[..., Any], int]] = {
        name: (python_function(name), 1 if name == "neg" else 2)
        for name in ("add", "sub", "mul", "div", "neg", "pow")}
    table["exp"] = (ex.exp, 1)
    table["log"] = (ex.log, 1)
    table["log_base"] = (ex.log, 2)
    return table


def angles() -> list[Any]:
    from sympy_extras.core.numbers import I, Rational, pi
    return [pi * I, Rational(1, 3) * pi * I, -pi * I / 2, Rational(7, 4) * pi * I]


@pytest.mark.parametrize("seed", range(40))
def test_exponential_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core.assumptions import facts_of
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("exponential", seed))
    pool = atoms() + angles()
    table = functions()
    compared = skipped = 0
    for _ in range(30):
        name = rng.choice(sorted(table))
        function, arity = table[name]
        args = [rng.choice(pool) for _ in range(arity)]
        try:
            expected = function(*args)
        except (ZeroDivisionError, ValueError, TypeError, ArithmeticError, NotImplementedError):
            continue
        try:
            got = rc.log(*args) if name == "log_base" else getattr(rc, name)(*args)
        except NotImplementedError:
            skipped += 1
            continue
        compared += 1
        assert shape(got) == shape(expected) and repr(got) == repr(expected), (name, args, got, expected)
        if convertible(expected):
            assert rc.facts(expected) == dict(facts_of(expected)), expected
            if expected.count_ops() < 40:
                pool.append(expected)
    assert compared >= 15, (compared, skipped)
