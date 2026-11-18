"""``sympy_extras_core::random::PyRandom`` (the Mersenne Twister with
CPython's seeding) against Python's ``random.Random``: string seeds (the
SHA-512 digest), integer seeds (zero, negative, beyond 32 and 64 bits)
and ``randint`` over ranges of various widths."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


SEEDS: list[Any] = ['', 'a', 'x**2 + sqrt(x + 1)', '1/(sqrt(x + 1) + sqrt(x - 1))', 'ü unicode', 0, 1, -5, 42,
                    2**32 + 7, 2**64 + 1, 3**100]


@pytest.mark.parametrize("seed", SEEDS)
def test_randint(rust: Any, seed: Any) -> None:
    from sympy_extras.core import Integer, Symbol
    rc = load("sympy_extras_rs.core")
    for lo, hi in [(1, 7), (0, 1), (-3, 3), (1, 2**31), (0, 2**40), (5, 5)]:
        expected = random.Random(seed)
        values = [expected.randint(lo, hi) for _ in range(40)]
        node = Integer(seed) if isinstance(seed, int) else Symbol(seed) if seed else None
        if node is None:
            continue  # an empty name is not a symbol
        got = rc.logic_call('pyrandom_randint', [node, Integer(lo), Integer(hi), Integer(40)])
        assert [int(v) for v in got.args] == values, (seed, lo, hi)
