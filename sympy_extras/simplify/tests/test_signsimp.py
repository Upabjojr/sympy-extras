"""The typed signsimp against SymPy (the oracle) on random rational
expressions, compared through the printed forms."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.interop import to_sympy
from sympy_extras.interop import expr_from_sympy
from sympy_extras.simplify.signsimp import signsimp

sx, sy, sa = sympy.symbols('x y a')
ATOMS: list[sympy.Expr] = [sx, sy, sa, sympy.sin(sx), sympy.sqrt(2), sympy.exp(sx), sympy.Integer(1),
                           sympy.Rational(1, 2), sympy.I]


@pytest.mark.parametrize('seed', [1, 2])
def test_signsimp_matches_sympy(seed: int) -> None:
    rng = random.Random(seed)

    def term() -> sympy.Expr:
        return rng.randint(-3, 3) * sympy.Mul(*[rng.choice(ATOMS) for _ in range(rng.randint(1, 2))])

    def total() -> sympy.Expr:
        return sympy.Add(*[term() for _ in range(rng.randint(1, 3))])

    different: list[tuple[str, str, str]] = []
    for _ in range(300):
        shape = rng.randint(0, 4)
        if shape == 0:
            e = total() / total()
        elif shape == 1:
            e = total() * total() / total()
        elif shape == 2:
            e = sympy.sin(total()) + 1 / total()
        elif shape == 3:
            e = total()**rng.choice([2, -1, -2, sympy.Rational(1, 2)])
        else:
            e = (total() / total())**-1 + total()
        if e.has(sympy.zoo, sympy.nan):
            continue
        ours = str(to_sympy(signsimp(expr_from_sympy(e)), evaluate=False))
        if ours != str(sympy.signsimp(e)):
            different.append((str(e), ours, str(sympy.signsimp(e))))
    assert not different, different[:3]
