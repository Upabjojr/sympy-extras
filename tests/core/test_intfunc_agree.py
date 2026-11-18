"""``sympy_extras.core.intfunc`` against its Rust port (``core/intfunc.rs``):
gcd, lcm, the extended Euclidean algorithm, the modular inverse (negative
and unit moduli), integer roots and logarithms, digit counts and trailing
zero bits, on random small and large integers, the errors included."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def number(rng: random.Random) -> int:
    kind = rng.random()
    if kind < 0.3:
        return rng.randint(-12, 12)
    if kind < 0.6:
        return rng.randint(-10**6, 10**6)
    return rng.choice([-1, 1]) * rng.randint(0, 10**40)


@pytest.mark.parametrize("seed", range(40))
def test_intfunc(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer
    from sympy_extras.core import intfunc as f
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("intfunc", seed))

    def rust_call(name: str, *args: int) -> Any:
        r = rc.logic_call(name, [Integer(a) for a in args])
        if r.args:
            return tuple(bool(a) if str(a) in ('True', 'False') else int(a) for a in r.args)
        return int(r)

    for _ in range(10):
        a, b, c = number(rng), number(rng), number(rng)
        y, n = abs(a), rng.choice([1, 2, 3, 5, 7, 0, -1, 40])
        base = rng.choice([2, 3, 10, 16])
        cube = rng.randint(0, 10**12)**rng.choice([2, 3, 5])
        cases = [('igcd', (a, b, c)), ('ilcm', (a, b)), ('igcdex', (a, b)), ('mod_inverse', (a, b)),
                 ('mod_inverse', (a, rng.choice([1, -1, 7, -7, 12]))), ('integer_nthroot', (y, n)),
                 ('integer_nthroot', (cube, rng.choice([2, 3, 5]))), ('integer_nthroot', (a, 2)),
                 ('integer_log', (y, rng.choice([2, 3, 10, 1]))), ('integer_log', (base**rng.randint(0, 30), base)),
                 ('num_digits', (a, base)), ('trailing', (a,)), ('trailing', (a << rng.randint(0, 70),))]
        for name, args in cases:
            assert outcome(lambda: rust_call(name, *args)) == outcome(lambda: getattr(f, name)(*args)), (name, args)
