"""``as_numer_denom``, ``fraction``, ``expand_mul`` and
``expand_multinomial`` of the Rust core against the Python core, on the
random evaluated expressions of the exponential comparison and on hand
picked products of sums, rational powers of sums and fractions."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, convertible, shape
from .test_exponential_agree import angles, functions


def checks() -> dict[str, Callable[[Any], Any]]:
    from sympy_extras.core.function import expand_mul, expand_multinomial
    from sympy_extras.simplify.radsimp import fraction
    return {"as_numer_denom": lambda e: e.as_numer_denom(), "fraction": fraction,
            "expand_mul": expand_mul, "expand_multinomial": expand_multinomial}


def agree(got: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        return all(agree(g, x) for g, x in zip(got, expected))
    return bool(shape(got) == shape(expected) and repr(got) == repr(expected))


@pytest.mark.parametrize("seed", range(30))
def test_expansion_agrees(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("expand", seed))
    pool = atoms() + angles()
    table = functions()
    compared = 0
    for _ in range(30):
        function, arity = table[rng.choice(sorted(table))]
        try:
            e = function(*[rng.choice(pool) for _ in range(arity)])
        except (ZeroDivisionError, ValueError, TypeError, ArithmeticError, NotImplementedError):
            continue
        if not convertible(e):
            continue
        if e.count_ops() < 30:
            pool.append(e)
        for name, check in checks().items():
            expected = check(e)
            try:
                got = getattr(rc, name)(e)
            except NotImplementedError:
                continue
            assert agree(got, expected), (name, e, got, expected)
            compared += 1
    assert compared >= 40


def test_hand_picked(rust: Any) -> None:
    from sympy_extras.core import I, Rational, Symbol, sqrt
    from sympy_extras.functions.elementary.exponential import exp, log
    rc = load("sympy_extras_rs.core")
    x, y, z = (Symbol(n) for n in 'xyz')
    for e in [x*(y + 1), (x + 1)**2*(y + z), (x + y)**Rational(5, 2), (x + y)**Rational(-7, 3),
              (x + 1)/(x*(y + 2)), exp(x*(y + 1)), (x + I)**3, 2*(x + y)*(x - y)/(z*(x + 3)), log(x*(1 + y)),
              (sqrt(2) + x)**4/(y + 1)**2, 1 + (x + 1)/(x - 1), x/2 + y/3]:
        for name, check in checks().items():
            assert agree(getattr(rc, name)(e), check(e)), (name, e)
