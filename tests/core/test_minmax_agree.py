"""``Max`` and ``Min`` of the Rust core against
``sympy_extras.functions.elementary.miscellaneous``: random arguments
(numbers, infinities, comparable constants, symbols with signs, sums,
nested ``Max`` and ``Min``), compared by structure and repr, with the
facts, the numerical values and the derivatives."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import convertible
from .test_bridge_agree import close


def outcome(f: Callable[[], Any]) -> str:
    try:
        return repr(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def leaves() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, sqrt
    x, p, n, r = Symbol('x'), Symbol('p', positive=True), Symbol('n', negative=True), Symbol('r', real=True)
    k = Symbol('k', integer=True)
    return [x, p, n, r, k, r + 1, p + 1, -p, n - 1, 2*r, r**2, Integer(0), Integer(3), Integer(-2), Rational(1, 2),
            oo, -oo, pi, sqrt(2), I]


def random_argument(rng: random.Random, depth: int) -> Any:
    from sympy_extras.functions.elementary.miscellaneous import max_, min_
    if depth == 0 or rng.random() < 0.6:
        return rng.choice(leaves())
    function = rng.choice([max_, min_])
    try:
        return function(*[random_argument(rng, depth - 1) for _ in range(rng.randint(2, 3))])
    except ValueError:
        return rng.choice(leaves())


@pytest.mark.parametrize("seed", range(40))
def test_minmax(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary import miscellaneous as m
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("minmax", seed))
    x = Symbol('x')
    for _ in range(10):
        name = rng.choice(["max_", "min_"])
        args = [random_argument(rng, 2) for _ in range(rng.randint(1, 4))]
        got = outcome(lambda: rc.logic_call(name, args))
        expected = outcome(lambda: getattr(m, name)(*args))
        assert got == expected, (name, args)
        if expected.startswith('ERROR'):
            continue
        e = getattr(m, name)(*args)
        if convertible(e):
            r = rc.logic_call(name, args)
            assert rc.facts(r) == dict(facts_of(e)), e
            assert close(rc.numeric_value(r, 53), numeric_value(e, 53), 53), e
            d = outcome(lambda: rc.diff(r, (x, 1)))
            if d != 'SKIP':
                assert d == outcome(lambda: diff(e, x)), e
