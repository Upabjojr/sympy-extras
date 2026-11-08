"""The typed sqrtdenest and radsimp against SymPy (the oracle) on random
nested radicals and numerical denominators with square roots."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.simplify.radsimp import radsimp
from sympy_extras.simplify.sqrtdenest import sqrtdenest

SURDS: list[sympy.Expr] = [sympy.sqrt(2), sympy.sqrt(3), sympy.sqrt(5), sympy.sqrt(6), sympy.sqrt(10), sympy.sqrt(15)]


@pytest.mark.parametrize('seed', [1, 2])
def test_sqrtdenest_radsimp_match_sympy(seed: int) -> None:
    rng = random.Random(seed)
    different: list[tuple[str, str, str]] = []
    for _ in range(60):
        b = rng.randint(1, 4) * rng.choice(SURDS)
        if rng.random() < 0.3:
            b = b + rng.choice(SURDS + [sympy.Integer(1), sympy.Integer(2)])
        nested = sympy.sqrt(rng.randint(1, 9) + b)
        denominator = sympy.Add(*[rng.randint(-3, 3) * rng.choice(SURDS + [sympy.Integer(1), sympy.Integer(2)])
                                  for _ in range(rng.randint(2, 4))])
        for e, denesting in [(nested, True), (1 / denominator, False)]:
            if e.has(sympy.zoo, sympy.nan):
                continue
            ours_in = expr_from_sympy(e)
            if str(to_sympy(ours_in, evaluate=False)) != str(e):
                continue
            result = str(to_sympy(sqrtdenest(ours_in) if denesting else radsimp(ours_in), evaluate=False))
            expected = str(sympy.sqrtdenest(e) if denesting else sympy.radsimp(e))
            if result != expected:
                different.append((str(e), result, expected))
    assert not different, different[:3]
