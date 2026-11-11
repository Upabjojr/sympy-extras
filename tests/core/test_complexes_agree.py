"""``re``, ``im``, ``conjugate``, ``sign``, ``Abs`` and ``as_real_imag``
of the Rust core against ``sympy_extras.functions.elementary.complexes``,
mixed with ``exp``, ``log`` and the arithmetic (whose ``Abs``, ``re`` and
``im`` branches in ``pow`` and ``exp`` they unblock), on random canonical
operands; results by structure, classes and reprs, and their facts. The
cases which need the trigonometric functions, ``arg`` or the full
``expand`` raise ``NotImplementedError`` and are counted, not compared."""

from __future__ import annotations

import importlib
import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, convertible, shape
from .test_exponential_agree import angles, functions


def table() -> dict[str, tuple[Callable[..., Any], int]]:
    cx = importlib.import_module("sympy_extras.functions.elementary.complexes")
    result = functions()
    for name in ("re", "im", "conjugate", "sign", "abs_"):
        result[name] = (getattr(cx, name), 1)
    result["as_real_imag"] = (lambda e: e.as_real_imag(), 1)
    return result


@pytest.mark.parametrize("seed", range(40))
def test_complexes_agree(rust: Any, seed: int) -> None:
    from sympy_extras.core.assumptions import facts_of
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("complexes", seed))
    pool = atoms() + angles()
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
