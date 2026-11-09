"""The typed powsimp against SymPy (the oracle) on random products of
powers, and SymPy's evaluation of powers of powers."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.interop import to_sympy
from sympy_extras.interop import expr_from_sympy
from sympy_extras.core.power import pow
from sympy_extras.simplify.powsimp import Combine, powsimp

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
        options: list[tuple[bool, Combine, bool]] = [(False, 'all', False), (False, 'exp', False),
                                                     (False, 'base', False), (False, 'all', True), (True, 'all', False)]
        for deep, combine, force in options:
            ours = str(to_sympy(powsimp(ours_in, deep=deep, combine=combine, force=force), evaluate=False))
            theirs = str(sympy.powsimp(e, deep=deep, combine=combine, force=force))
            if ours != theirs:
                different.append((str((deep, combine, force)), str(e), ours, theirs))
    assert not different, different[:3]


def test_power_of_power_matches_sympy() -> None:
    cases: list[tuple[sympy.Expr, sympy.Expr]] = [
        (sympy.exp(2), sx), (sympy.exp(sx), sy), (sympy.exp(sr), sy), (sp**2, sx), (sp**sr, sy), (sx**2, sy),
        (sympy.exp(2 * sympy.I), sx), (sr**2, sympy.Rational(1, 2)), (sx**2, sympy.Rational(1, 2)),
        (sx**sympy.Rational(1, 3), sy), (sr**4, sympy.Rational(1, 2)), (sx**-1, sympy.Rational(1, 2)),
        (sx**sn, sympy.Integer(2))]
    for b, e in cases:
        ours = str(to_sympy(pow(expr_from_sympy(b), expr_from_sympy(e)), evaluate=False))
        assert ours == str(sympy.Pow(b, e)), (b, e)


@pytest.mark.parametrize('seed', [1, 2])
def test_powdenest_logcombine_expand_log_match_sympy(seed: int) -> None:
    from sympy_extras.core.function import expand_log
    from sympy_extras.simplify.powsimp import powdenest
    from sympy_extras.simplify.simplify import logcombine
    rng = random.Random(seed)
    sq = sympy.Symbol('q', positive=True)
    atoms: list[sympy.Expr] = [sx, sy, sp, sq, sympy.Integer(2), sympy.Integer(8), sympy.exp(sx), sympy.exp(sp), sx**2,
                               sp**3, sympy.sqrt(sx), sx * sy, sp * sq**2]
    exponents: list[sympy.Expr] = [sympy.Integer(2), sympy.Rational(1, 2), sympy.Rational(-1, 3), sy, sn, 2 * sn, sr,
                                   sp, sympy.Rational(3, 2)]
    log_atoms: list[sympy.Expr] = [sympy.log(sp), sympy.log(sq), sympy.log(sx), sympy.log(2), sympy.log(sp * sq), sr, sn,
                                   sx, sympy.Integer(2), sympy.Rational(-1, 2)]
    different: list[tuple[str, str, str, str]] = []
    for _ in range(60):
        e = sympy.Mul(*[(rng.choice(atoms)**rng.choice(exponents))**rng.choice(exponents) for _ in range(rng.randint(1, 3))])
        logs = sympy.Add(*[rng.choice([1, 2, -1, sr, sx]) * sympy.Mul(*[rng.choice(log_atoms) for _ in range(rng.randint(1, 2))])
                           for _ in range(rng.randint(1, 4))])
        for expression in (e, logs):
            if expression.has(sympy.zoo, sympy.nan):
                continue
            ours_in = expr_from_sympy(expression)
            if str(to_sympy(ours_in, evaluate=False)) != str(expression):
                continue
            checks = [('powdenest', str(to_sympy(powdenest(ours_in), evaluate=False)), str(sympy.powdenest(expression))),
                      ('force', str(to_sympy(powdenest(ours_in, force=True), evaluate=False)),
                       str(sympy.powdenest(expression, force=True))),
                      ('logcombine', str(to_sympy(logcombine(ours_in), evaluate=False)), str(sympy.logcombine(expression))),
                      ('logcombine_force', str(to_sympy(logcombine(ours_in, force=True), evaluate=False)),
                       str(sympy.logcombine(expression, force=True))),
                      ('expand_log', str(to_sympy(expand_log(ours_in), evaluate=False)), str(sympy.expand_log(expression))),
                      ('expand_log_force', str(to_sympy(expand_log(ours_in, force=True), evaluate=False)),
                       str(sympy.expand_log(expression, force=True)))]
            different += [(name, str(expression), o, t) for name, o, t in checks if o != t]
    assert not different, different[:3]
