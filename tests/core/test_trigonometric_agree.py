"""``sin``, ``cos``, ``tan``, ``cot``, ``sec`` and ``csc`` of the Rust
core against ``sympy_extras.functions.elementary.trigonometric``: the
exact values at rational multiples of ``pi`` (the reductions, the product
formulas, the Chebyshev and half-angle values, the table of tenths), the
multiples of ``pi`` peeled off sums, and random operations mixing them
with the complex functions, ``exp``, ``log`` and the arithmetic (results
by structure, classes and reprs, and their facts). The cases which need
the hyperbolic functions or the accumulation bounds raise
``NotImplementedError`` in Rust and are counted, not compared."""

from __future__ import annotations

import importlib
import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, convertible, shape
from .test_complexes_agree import table as complexes_table
from .test_exponential_agree import angles

NAMES = ("sin", "cos", "tan", "cot", "sec", "csc")


def trigonometric() -> Any:
    return importlib.import_module("sympy_extras.functions.elementary.trigonometric")


def table() -> dict[str, tuple[Callable[..., Any], int]]:
    result = complexes_table()
    for name in NAMES:
        result[name] = (getattr(trigonometric(), name), 1)
    return result


def multiples_of_pi() -> list[Any]:
    from sympy_extras.core.numbers import Rational, pi
    return [Rational(k, q) * pi for q in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 60) for k in (1, 5, 7)]


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 30, 40, 60])
def test_exact_values(rust: Any, q: int) -> None:
    from sympy_extras.core.numbers import Rational, pi
    rc = load("sympy_extras_rs.core")
    # every numerator in two periods for the small denominators, a sample
    # for the large ones (each value takes a product formula)
    numerators = range(-q - 1, 2 * q + 2) if q <= 24 else (-1, 1, 7, q - 1, q + 7, 2 * q - 1)
    for k in numerators:
        a = Rational(k, q) * pi
        for name in NAMES:
            expected = getattr(trigonometric(), name)(a)
            got = getattr(rc, name)(a)
            assert shape(got) == shape(expected) and repr(got) == repr(expected), (name, a)


def test_peeled_and_symbolic(rust: Any) -> None:
    from sympy_extras.core.numbers import Rational, pi
    from sympy_extras.core.symbol import Symbol
    rc = load("sympy_extras_rs.core")
    x, n = Symbol('x'), Symbol('n', integer=True)
    for a in [x + Rational(k, 4) * pi for k in range(-5, 9)] + [n * pi, 2 * n * pi, n * pi / 2, (2 * n + 1) * pi / 2,
                                                              x * pi, 3 * x * pi / 7, -x]:
        for name in NAMES:
            expected = getattr(trigonometric(), name)(a)
            got = getattr(rc, name)(a)
            assert shape(got) == shape(expected) and repr(got) == repr(expected), (name, a)


@pytest.mark.parametrize("seed", range(40))
def test_trigonometric_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core.assumptions import facts_of
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("trigonometric", seed))
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
