"""``expand(complex=True)`` of the Rust core (every node as ``re +
I*im`` before the other hints, the hints of the call passed on to the
real and imaginary parts of the functions) against
``sympy_extras.core.function.expand``, on the random expressions of the
hints test with five sets of hints. Expressions with two logarithms are
left out: ``expand(log(log(x**(3/2))), complex=True, log=False)`` does
not finish in Python nor in the port."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import shape
from .test_expand_hints_agree import random_expression

HINTS: list[dict[str, bool]] = [{"complex": True}, {"complex": True, "log": False},
                                {"complex": True, "mul": False, "multinomial": False}, {"complex": True, "deep": False},
                                {"complex": True, "power_base": False, "power_exp": False}]


@pytest.mark.parametrize("seed", range(25))
def test_expand_complex_agree(rust: Any, seed: int) -> None:
    from sympy_extras.core.function import expand
    from sympy_extras.functions.elementary.exponential import Log
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("expand complex", seed))
    compared = 0
    for _ in range(8):
        try:
            e = random_expression(rng, 3)
        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, NotImplementedError):
            continue
        if len(e.atoms(Log)) >= 2:
            continue
        for hints in HINTS:
            try:
                expected = expand(e, **hints)
            except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, NotImplementedError):
                continue
            try:
                got = rc.expand(e, **hints)
            except NotImplementedError:
                continue
            assert shape(got) == shape(expected) and repr(got) == repr(expected), (e, hints)
            compared += 1
    assert compared >= 10
