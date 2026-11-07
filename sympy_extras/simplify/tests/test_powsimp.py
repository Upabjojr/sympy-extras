"""The typed powsimp against SymPy (the oracle) on random products of
powers, and SymPy's evaluation of powers of powers."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.core.power import pow
from sympy_extras.simplify.powsimp import powsimp

sx, sy, sz = sympy.symbols('x y z')
sp = sympy.Symbol('p', positive=True)
sn = sympy.Symbol('n', integer=True)
sr = sympy.Symbol('r', real=True)


@pytest.mark.parametrize('seed', [1, 2])
def test_powsimp_matches_sympy(seed: int) -> None:
    rng = random.Random(seed)
    atoms: list[sympy.Expr] = [sx, sy, sz, sp, sympy.Integer(2), sympy.Integer(3), sympy.Rational(1, 2), sympy.exp(sx),
                               sympy.sqrt(sx), sympy.sqrt(2), sx + 1]
    exponents: list[sympy.Expr] = [sympy.Integer(2), sympy.Integer(-1), sympy.Rational(1, 2), sympy.Rational(-3, 2), sy,
                                   sn, sx + sy, 2 * sn, sympy.Rational(1, 3)]
    different: list[tuple[str, str, str, str]] = []
    for _ in range(80):
        e = sympy.Mul(*[rng.choice(atoms)**rng.choice(exponents) for _ in range(rng.randint(2, 4))])
        if rng.random() < 0.25:
            e = e + sympy.Mul(*[rng.choice(atoms)**rng.choice(exponents) for _ in range(2)])
        if e.has(sympy.zoo, sympy.nan):
            continue
        ours_in = expr_from_sympy(e)
        if str(to_sympy(ours_in, evaluate=False)) != str(e):
            continue
        for deep, combine, force in [(False, 'all', False), (False, 'exp', False), (False, 'base', False),
                                     (False, 'all', True), (True, 'all', False)]:
            ours = str(to_sympy(powsimp(ours_in, deep=deep, combine=combine, force=force), evaluate=False))
            theirs = str(sympy.powsimp(e, deep=deep, combine=combine, force=force))
            if ours != theirs:
                different.append((str((deep, combine, force)), str(e), ours, theirs))
    assert not different, different[:3]


def test_power_of_power_matches_sympy() -> None:
    cases: list[tuple[sympy.Expr, sympy.Expr]] = [
        (sympy.exp(2), sx), (sympy.exp(sx), sy), (sympy.exp(sr), sy), (sp**2, sx), (sp**sr, sy), (sx**2, sy),
        (sympy.exp(2 * sympy.I), sx), (sr**2, sympy.Rational(1, 2)), (sx**2, sympy.Rational(1, 2)),
        (sx**sympy.Rational(1, 3), sy), (sr**4, sympy.Rational(1, 2)), (sx**-1, sympy.Rational(1, 2)), (sx**sn, 2)]
    for b, e in cases:
        ours = str(to_sympy(pow(expr_from_sympy(b), expr_from_sympy(e)), evaluate=False))
        assert ours == str(sympy.Pow(b, e)), (b, e)
