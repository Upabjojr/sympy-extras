"""Pattern matching (``sympy_extras.core.matching.match``) against its
Rust port: random patterns of wilds (some excluding ``x``) in sums,
products, powers and functions, matched against their instances (the
wilds replaced by random values) and against random expressions, with
and without ``exact`` and given bindings."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def leaf(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol
    x, y = Symbol('x'), Symbol('y')
    return rng.choice([x, y, Integer(2), Integer(-1), Integer(3), Rational(1, 2), x + 1, 2*x, x*y, x**2])


def pattern(rng: random.Random, wilds: list[Any], depth: int) -> Any:
    from sympy_extras.core import Integer
    from sympy_extras.functions import cos, exp, sin
    if depth == 0:
        return rng.choice(wilds + [leaf(rng)])
    a, b = pattern(rng, wilds, depth - 1), pattern(rng, wilds, depth - 1)
    return rng.choice([a + b, a*b, a**b, a**Integer(2), sin(a), exp(a)*b, cos(a) + b, a*b + rng.choice(wilds),
                       Integer(2)*a, a - b])


def wild_set() -> list[Any]:
    from sympy_extras.core import Symbol
    from sympy_extras.core.symbol import Wild
    x = Symbol('x')
    return [Wild('a'), Wild('b'), Wild('c', exclude=[x]), Wild('d', exclude=[x])]


@pytest.mark.parametrize("seed", range(60))
def test_match(rust: Any, seed: int) -> None:
    from sympy_extras.core.containers import Tuple
    from sympy_extras.core import Integer
    from sympy_extras.core.matching import match
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("matching", seed))
    wilds = wild_set()
    for _ in range(5):
        try:
            p = pattern(rng, wilds, rng.randint(1, 2))
            if rng.random() < 0.6:
                e = p.xreplace({w: leaf(rng) for w in wilds})
            else:
                e = pattern(rng, [leaf(rng) for _ in range(3)], rng.randint(1, 2))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        exact = rng.random() < 0.3
        given = {wilds[0]: leaf(rng)} if rng.random() < 0.2 else {}

        def expected() -> Any:
            found = match(e, p, given, exact)
            return False if found is None else sorted((str(k), str(v)) for k, v in found.items())

        def got() -> Any:
            found = rc.logic_call('match', [e, p, Integer(int(exact)), *[Tuple(k, v) for k, v in given.items()]])
            return False if found is False or str(found) == 'False' else sorted((str(t.args[0]), str(t.args[1])) for t in found.args)
        assert outcome(got) == outcome(expected), (e, p, exact, given)
