"""The typed together against SymPy (the oracle) on random rational
expressions, compared through the printed forms."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Symbol, to_sympy
from sympy_extras.core.exprtools import decompose_power
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.functions import exp
from sympy_extras.polys.rationaltools import together

sx, sy, sa = sympy.symbols('x y a')
ATOMS: list[sympy.Expr] = [sx, sy, sa, sympy.sin(sx), sympy.sqrt(2), sympy.exp(sx), sympy.sqrt(sx), sympy.Rational(1, 2),
                           sympy.I, 1 / sx, 1 / (sx + 1), 1 / (sy - sa)**2, sympy.exp(1 / sx + 1 / sy)]


@pytest.mark.parametrize('seed', [1, 2])
def test_together_matches_sympy(seed: int) -> None:
    rng = random.Random(seed)

    def term() -> sympy.Expr:
        return rng.randint(-3, 3) * sympy.Mul(*[rng.choice(ATOMS) for _ in range(rng.randint(1, 3))])

    def total() -> sympy.Expr:
        return sympy.Add(*[term() for _ in range(rng.randint(1, 4))])

    different: list[tuple[str, str, str, str]] = []
    for _ in range(150):
        shape = rng.randint(0, 3)
        e = [total(), total() / total(), total() * total(), sympy.sin(total()) + total()][shape]
        if e.has(sympy.zoo, sympy.nan):
            continue
        ours_in = expr_from_sympy(e)
        if str(to_sympy(ours_in, evaluate=False)) != str(e):
            continue
        for deep, fraction in [(False, True), (True, True), (False, False)]:
            ours = str(to_sympy(together(ours_in, deep=deep, fraction=fraction), evaluate=False))
            theirs = str(sympy.together(e, deep=deep, fraction=fraction))
            if ours != theirs:
                different.append((str((deep, fraction)), str(e), ours, theirs))
    assert not different, different[:3]


def test_decompose_power_keeps_canonical_bases() -> None:
    # the base of exp(2*u) must be the node exp(u) itself (a Rational(1, 1)
    # coefficient left in the exponent made them different keys)
    x, y = Symbol('x'), Symbol('y')
    u = together(1 / x + 1 / y)
    base, n = decompose_power(exp(2 * u))
    assert (base, n) == (exp(u), 2) and hash(base) == hash(exp(u))
