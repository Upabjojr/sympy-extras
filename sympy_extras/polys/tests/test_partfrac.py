"""The typed apart against SymPy (the oracle) on random rational
functions of x, with and without a parameter, compared through the
printed forms."""
from __future__ import annotations

import random

import sympy

from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.polys.partfrac import apart

sx, sa = sympy.symbols('x a')


def test_apart_matches_sympy() -> None:
    rng = random.Random(1)

    def factor_of(params: bool) -> sympy.Expr:
        choices: list[sympy.Expr] = [sx - rng.randint(-3, 3), 2 * sx + rng.choice([1, 3]), sx**2 + rng.randint(1, 3),
                                     sx**2 + sx + 1]
        if params:
            choices += [sx + sa, sx - sa, sx**2 + sa]
        return rng.choice(choices)

    def numerator(params: bool) -> sympy.Expr:
        return sympy.Add(*[rng.randint(-3, 3) * sympy.Rational(1, rng.choice([1, 2])) * sx**k
                           * (sa**rng.randint(0, 1) if params else 1) for k in range(rng.randint(1, 4))])

    different: list[tuple[str, str, str]] = []
    for _ in range(60):
        params = rng.random() < 0.4
        e = numerator(params) / sympy.Mul(*[factor_of(params)**rng.choice([1, 1, 2]) for _ in range(rng.randint(1, 3))])
        if e.has(sympy.zoo, sympy.nan) or e == 0:
            continue
        ours_in = expr_from_sympy(e)
        if str(to_sympy(ours_in, evaluate=False)) != str(e):
            continue
        ours = str(to_sympy(apart(ours_in, expr_from_sympy(sx)), evaluate=False))
        if ours != str(sympy.apart(e, sx)):
            different.append((str(e), ours, str(sympy.apart(e, sx))))
    assert not different, different[:2]
