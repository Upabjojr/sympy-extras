"""The typed TR rules of fu against SymPy's (the oracle) on random
trigonometric expressions, compared through the printed forms."""
from __future__ import annotations

import importlib
import random

import pytest
import sympy

from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy
import sympy_extras.simplify.fu as F

SF = importlib.import_module('sympy.simplify.fu')
sx, sy = sympy.symbols('x y')
RULES = ['TR0', 'TR1', 'TR2', 'TR2i', 'TR3', 'TR4', 'TR5', 'TR6', 'TR7', 'TR8', 'TR9', 'TR10', 'TR10i', 'TR11', '_TR11',
         'TR12', 'TR12i', 'TR13', 'TRmorrie', 'TR14', 'TR15', 'TR16', 'TR111', 'TR22', 'TRpower']


@pytest.mark.parametrize('seed', [1, 2])
def test_tr_rules_match_sympy(seed: int) -> None:
    rng = random.Random(seed)
    args: list[sympy.Expr] = [sx, sy, 2 * sx, sx + sy, sx - sy, -sx, sympy.pi / 3 - sx, sympy.pi / 2 + sx]
    funcs = [sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc]

    def atom() -> sympy.Expr:
        return rng.choice(funcs)(rng.choice(args))**rng.choice([1, 1, 2, 3, 4, -1])

    different: list[tuple[str, str, str, str]] = []
    for _ in range(25):
        e = sympy.Add(*[rng.randint(-2, 2) * sympy.Mul(*[atom() for _ in range(rng.randint(1, 2))])
                        for _ in range(rng.randint(1, 3))])
        if rng.random() < 0.3:
            e = e / sympy.Add(atom(), rng.randint(1, 2))
        if e.has(sympy.zoo, sympy.nan):
            continue
        ours_in = expr_from_sympy(e)
        if str(to_sympy(ours_in, evaluate=False)) != str(e):
            continue
        for name in RULES:
            ours = str(to_sympy(getattr(F, name)(ours_in), evaluate=False))
            theirs = str(getattr(SF, name)(e))
            if ours != theirs:
                different.append((name, str(e), ours, theirs))
    assert not different, different[:3]


def test_fu_matches_sympy_on_identities() -> None:
    cases: list[sympy.Expr] = [
        sympy.sin(sx)**2 + sympy.cos(sx)**2, sympy.sin(sx) / sympy.cos(sx), 1 - 2 * sympy.sin(sx)**2,
        sympy.sin(sx) * sympy.cos(sx), sympy.cos(sx)**4 - sympy.sin(sx)**4, sympy.tan(sx) * sympy.cot(sx),
        sympy.sin(sx + sy) - sympy.sin(sx) * sympy.cos(sy), (1 - sympy.cos(sx)) * (1 + sympy.cos(sx)),
        sympy.sec(sx)**2 - sympy.tan(sx)**2, sympy.cos(sx) * sympy.cos(2 * sx) * sympy.cos(4 * sx),
        2 * sympy.sin(sx) * sympy.cos(sx) / (sympy.cos(sx)**2 - sympy.sin(sx)**2), sympy.sin(3 * sx) / sympy.sin(sx),
        sympy.cos(sx) - sympy.cos(sy), sympy.sqrt(3) * sympy.cos(sx) / 2 + sympy.sin(sx) / 2]
    for e in cases:
        assert str(to_sympy(F.fu(expr_from_sympy(e)), evaluate=False)) == str(SF.fu(e)), e
