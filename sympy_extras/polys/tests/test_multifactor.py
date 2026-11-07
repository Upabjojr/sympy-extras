"""Multivariate factorisation over ZZ (Wang's algorithm), Poly.factor_list
over ZZ, QQ, ZZ[a], QQ[a] and the expression-level factor against SymPy
(the oracle), compared through the printed forms (the order of the
factors included)."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.multifactor import zz_factor
from sympy_extras.polys.polytools import Poly, factor, factor_list_fraction
from sympy_extras.polys.rings import PolyRing

NAMES = ['x', 'y', 'z', 'w']
SYMBOLS = sympy.symbols(NAMES)
sx, sy, sz, sa, sb = sympy.symbols('x y z a b')


@pytest.mark.parametrize('seed', [1, 2])
def test_zz_factor_matches_sympy(seed: int) -> None:
    rng = random.Random(seed)

    def random_polynomial(n: int) -> sympy.Expr:
        return sympy.Add(*[rng.randint(-5, 5) * sympy.Mul(*[v**rng.randint(0, 2) for v in SYMBOLS[:n]])
                           for _ in range(rng.randint(2, 3))])

    different: list[tuple[str, str, str]] = []
    for _ in range(40):
        n = rng.randint(2, 4)
        e = sympy.expand(sympy.Mul(*[random_polynomial(n)**rng.choice([1, 1, 2]) for _ in range(rng.randint(1, 3))])
                         * rng.choice([1, 2, -3, 6]))
        if e == 0 or e.is_number:
            continue
        P = sympy.Poly(e, *SYMBOLS[:n])
        c, factors = zz_factor(PolyRing(ZZ, NAMES[:n]).from_expr(expr_from_sympy(e)))
        ours = str((c, [(str(to_sympy(f.to_expr())), k) for f, k in factors]))
        C, FS = sympy.polys.factortools.dmp_zz_factor(P.rep.to_list(), n - 1, sympy.ZZ)
        theirs = str((int(C), [(str(sympy.Poly(sympy.polys.densebasic.dmp_to_dict(f, n - 1), *SYMBOLS[:n]).as_expr()), k)
                               for f, k in FS]))
        if ours != theirs:
            different.append((str(e), ours, theirs))
    assert not different, different[:2]


@pytest.mark.parametrize('seed', [3, 4])
def test_poly_factor_list_matches_sympy(seed: int) -> None:
    rng = random.Random(seed)

    def random_polynomial(variables: list[sympy.Symbol]) -> sympy.Expr:
        return sympy.Add(*[rng.randint(-4, 4) * sympy.Rational(1, rng.choice([1, 1, 2, 3]))
                           * sympy.Mul(*[v**rng.randint(0, 2) for v in variables]) for _ in range(rng.randint(1, 3))])

    different: list[tuple[str, str, str]] = []
    for _ in range(60):
        variables = rng.choice([[sx], [sx, sy], [sx, sa], [sx, sy, sa], [sx, sy, sz], [sx, sa, sb]])
        gens = rng.choice([tuple(variables), (sx,), (sx, variables[-1]) if len(variables) > 1 else (sx,)])
        e = sympy.expand(sympy.Mul(*[random_polynomial(variables)**rng.choice([1, 1, 2]) for _ in range(rng.randint(1, 3))])
                         * rng.choice([1, 2, -3, sympy.Rational(1, 2)]))
        P = sympy.Poly(e, *gens)
        if P.is_zero:
            continue
        c, factors = Poly(expr_from_sympy(e), *[expr_from_sympy(g) for g in gens]).factor_list()
        ours = str((to_sympy(c), factors))
        if ours != str(P.factor_list()):
            different.append((str(P), ours, str(P.factor_list())))
    assert not different, different[:2]


def test_factor_matches_sympy() -> None:
    rng = random.Random(5)
    atoms: list[sympy.Expr] = [sx, sy, sz, sa, sympy.sin(sx), sympy.exp(sx), sympy.sqrt(2), sympy.sqrt(sx)]

    def random_polynomial(k: int) -> sympy.Expr:
        return sympy.Add(*[rng.randint(-4, 4) * sympy.Rational(1, rng.choice([1, 1, 2, 3]))
                           * sympy.Mul(*[rng.choice(atoms[:k])**rng.randint(0, 2) for _ in range(2)])
                           for _ in range(rng.randint(1, 3))])

    different: list[tuple[str, str, str]] = []
    for _ in range(60):
        k = rng.choice([2, 3, 4, len(atoms)])
        product = sympy.Mul(*[random_polynomial(k)**rng.choice([1, 1, 2]) for _ in range(rng.randint(1, 3))])
        e = rng.choice([sympy.expand(product), sympy.expand(product) / sympy.expand(random_polynomial(k)), product])
        if e.has(sympy.zoo, sympy.nan) or e == 0:
            continue
        ours_in = expr_from_sympy(e)
        if str(to_sympy(ours_in, evaluate=False)) != str(e):
            continue
        ours = str(to_sympy(factor(ours_in), evaluate=False))
        if ours != str(sympy.factor(e)):
            different.append((str(e), ours, str(sympy.factor(e))))
        c, fp, fq = factor_list_fraction(ours_in)
        listed = str((to_sympy(c), [(to_sympy(f), to_sympy(m)) for f, m in fp], [(to_sympy(f), to_sympy(m)) for f, m in fq]))
        if listed != str(sympy.factor_list(e, frac=True)):
            different.append((str(e), listed, str(sympy.factor_list(e, frac=True))))
    assert not different, different[:2]
