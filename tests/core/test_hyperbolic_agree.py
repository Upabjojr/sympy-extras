"""``sinh``, ``cosh``, ``tanh``, ``coth``, ``sech`` and ``csch`` of the
Rust core against ``sympy_extras.functions.elementary.hyperbolic``: the
numbers, the imaginary arguments in both directions (``sin(I*x)`` is
``I*sinh(x)``, ``cosh(I*pi/3)`` is ``1/2``), the multiples of ``I*pi``
peeled off sums, and random operations mixing them with the
trigonometric, complex and exponential functions and the arithmetic
(results by structure, classes and reprs, and their facts)."""

from __future__ import annotations

import importlib
import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, convertible, shape
from .test_exponential_agree import angles
from .test_trigonometric_agree import multiples_of_pi
from .test_trigonometric_agree import table as trigonometric_table

NAMES = ("sinh", "cosh", "tanh", "coth", "sech", "csch")


def hyperbolic() -> Any:
    return importlib.import_module("sympy_extras.functions.elementary.hyperbolic")


def table() -> dict[str, tuple[Callable[..., Any], int]]:
    result = trigonometric_table()
    for name in NAMES:
        result[name] = (getattr(hyperbolic(), name), 1)
    return result


def test_special_arguments(rust: Any) -> None:
    from sympy_extras.core.numbers import I, Rational, oo, pi, zoo
    from sympy_extras.core.symbol import Symbol
    rc = load("sympy_extras_rs.core")
    tr = importlib.import_module("sympy_extras.functions.elementary.trigonometric")
    x = Symbol('x')
    arguments = [Rational(0), Rational(-3, 2), oo, -oo, zoo, I * x, -I * x, x + I * pi, x - I * pi / 2,
                 x + 3 * I * pi / 2] + [Rational(k, q) * I * pi for q in (1, 2, 3, 4, 6) for k in (-1, 1, 5)]
    for a in arguments:
        for module, names in ((hyperbolic(), NAMES), (tr, ("sin", "cos", "tan", "cot", "sec", "csc"))):
            for name in names:
                try:
                    expected = getattr(module, name)(a)
                except (TypeError, ValueError):
                    continue
                try:
                    got = getattr(rc, name)(a)
                except NotImplementedError:
                    continue
                assert shape(got) == shape(expected) and repr(got) == repr(expected), (name, a)


@pytest.mark.parametrize("seed", range(40))
def test_hyperbolic_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core.assumptions import facts_of
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("hyperbolic", seed))
    pool = atoms() + angles() + multiples_of_pi()
    operations = table()
    compared = skipped = 0
    for _ in range(30):
        name = rng.choice(sorted(operations))
        function, arity = operations[name]
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
        expected_parts = expected if isinstance(expected, tuple) else (expected,)
        got_parts = got if isinstance(got, tuple) else (got,)
        for g, x in zip(got_parts, expected_parts):
            assert shape(g) == shape(x) and repr(g) == repr(x), (name, args, got, expected)
        for x in expected_parts:
            if convertible(x):
                assert rc.facts(x) == dict(facts_of(x)), x
                if x.count_ops() < 40:
                    pool.append(x)
    assert compared >= 15, (compared, skipped)
