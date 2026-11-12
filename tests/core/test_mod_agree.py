"""``Mod`` of the Rust core against ``sympy_extras.core.mod``: random
dividends (numbers, symbols with assumptions, sums, products, nested
moduli) and moduli (integers, rationals, symbols, zero, infinities),
compared by structure (or the error), facts, printing (in products and
powers too) and derivatives."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def atoms() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol, pi, sqrt
    n, m = Symbol('n', integer=True), Symbol('m', integer=True, positive=True)
    x, p = Symbol('x', real=True), Symbol('p', positive=True)
    return [Integer(7), Integer(-5), Integer(0), Integer(6), Rational(7, 2), Rational(-1, 3), n, m, x, p,
            Symbol('z'), sqrt(2), pi]


def random_dividend(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core.mod import mod
    choice = rng.random()
    if depth == 0 or choice < 0.4:
        return rng.choice(atoms())
    if choice < 0.65:
        return random_dividend(rng, depth - 1) + random_dividend(rng, depth - 1)
    if choice < 0.9:
        return rng.choice([2, 3, -2, 6]) * random_dividend(rng, depth - 1) * rng.choice([1, rng.choice(atoms())])
    try:
        return mod(random_dividend(rng, depth - 1), rng.choice(moduli()))
    except Exception:
        return rng.choice(atoms())


def moduli() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol, oo
    return [Integer(2), Integer(3), Integer(-3), Integer(1), Integer(6), Rational(1, 2), Rational(3, 2),
            Symbol('n', integer=True), Symbol('q', positive=True), Symbol('x', real=True), Integer(0), oo]


@pytest.mark.parametrize("seed", range(40))
def test_mod(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol, mul, pow
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.function import diff
    from sympy_extras.core.mod import mod
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("mod", seed))
    x = Symbol('x', real=True)
    for _ in range(12):
        p, q = random_dividend(rng, 2), rng.choice(moduli())
        got = outcome(lambda: rc.logic_call('mod', [p, q]))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: mod(p, q)), (p, q)
        try:
            e = mod(p, q)
        except Exception:
            continue
        r = rc.logic_call('mod', [p, q])
        assert repr(r) == repr(e)
        assert rc.facts(r) == dict(facts_of(e)), e
        for k in [Integer(2), Integer(-1), Integer(-3)]:
            assert repr(rc.mul(k, r)) == repr(mul(k, e)), (k, e)
        assert repr(rc.pow(r, Integer(2))) == repr(pow(e, Integer(2))), e
        g = outcome(lambda: rc.diff(r, (x, 1)))
        if g != 'SKIP':
            assert g == outcome(lambda: diff(e, x)), e
