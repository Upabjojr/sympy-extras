"""``KroneckerDelta`` and ``LeviCivita`` of the Rust core against
``sympy_extras.functions.special.tensor_functions``: random indices
(integers, rationals, symbols with assumptions, sums), compared by
structure, facts, powers and derivatives."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def indices() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol
    i, j, k = Symbol('i', integer=True), Symbol('j', integer=True), Symbol('k')
    p = Symbol('p', positive=True)
    return [Integer(0), Integer(1), Integer(2), Integer(-1), Integer(3), Rational(1, 2), i, j, k, p, i + 1, j - 2,
            2*i, i + j, I, -p]


@pytest.mark.parametrize("seed", range(40))
def test_tensor_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational, Symbol, pow
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.function import diff
    from sympy_extras.functions.special import tensor_functions as tf
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("tensor", seed))
    x = Symbol('k')
    for _ in range(12):
        if rng.random() < 0.5:
            name, args = 'KroneckerDelta', [rng.choice(indices()) for _ in range(2)]
        else:
            name, args = 'LeviCivita', [rng.choice(indices()) for _ in range(rng.randint(1, 4))]
        got = outcome(lambda: rc.logic_call(name, args))
        expected = outcome(lambda: getattr(tf, name)(*args))
        assert got == expected, (name, args)
        if isinstance(expected, str):
            continue
        e = getattr(tf, name)(*args)
        r = rc.logic_call(name, args)
        assert repr(r) == repr(e)
        assert rc.facts(r) == dict(facts_of(e)), e
        for n in [Integer(2), Integer(-1), Integer(-2), Rational(1, 2), Symbol('m', positive=True)]:
            assert outcome(lambda: rc.pow(r, n)) == outcome(lambda: pow(e, n)), (e, n)
        g = outcome(lambda: rc.diff(r, (x, 1)))
        if g != 'SKIP':
            assert g == outcome(lambda: diff(e, x)), e
