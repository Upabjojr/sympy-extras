"""The B-splines of the Rust core against
``sympy_extras.functions.special.bsplines``: ``bspline_basis`` and
``bspline_basis_set`` of random degrees on random knot sequences
(rational, repeated, symbolic), compared by structure (or the error) and
printing."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def _knots(rng: random.Random) -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    kind = rng.random()
    count = rng.randint(2, 7)
    if kind < 0.15:
        a = Symbol('a', positive=True)
        return [Integer(0)] + [k*a for k in range(1, count)]
    pool: list[Any] = [Integer(0), Integer(1), Integer(2), Integer(3), Rational(1, 2), Integer(-1), Rational(5, 2),
                       Integer(4)]
    values: list[Any] = sorted((rng.choice(pool) for _ in range(count)), key=lambda v: (v.p / v.q))
    if kind < 0.4:
        values = [values[0]] * rng.randint(1, 3) + values + [values[-1]] * rng.randint(0, 2)
    return values


@pytest.mark.parametrize("seed", range(40))
def test_bsplines(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol, Tuple
    from sympy_extras.functions.special.bsplines import bspline_basis, bspline_basis_set
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("bsplines", seed))
    x = Symbol(rng.choice(['x', 'y']), real=rng.random() < 0.3)
    for _ in range(4):
        knots = _knots(rng)
        d = rng.randint(0, 3)
        n = rng.randint(0, max(0, len(knots) - d - 2))
        args = [Integer(d), Tuple(*knots), Integer(n), x]
        got = outcome(lambda: rc.logic_call('bspline_basis', args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: bspline_basis(d, knots, n, x)), (d, knots, n)
        try:
            e = bspline_basis(d, knots, n, x)
        except Exception:
            continue
        assert rc.sstr(rc.logic_call('bspline_basis', args)) == str(e), (d, knots, n)
    knots = _knots(rng)
    d = rng.randint(0, 2)
    got = outcome(lambda: rc.logic_call('bspline_basis_set', [Integer(d), Tuple(*knots), x]))
    if got != 'SKIP':
        assert got == outcome(lambda: Tuple(*bspline_basis_set(d, knots, x))), (d, knots)
