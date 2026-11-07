"""The typed fraction, collect and rcollect against SymPy (the oracle),
and SymPy's substitution rules for powers, exponentials and symbols."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Symbol, to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.functions import exp
from sympy_extras.simplify.radsimp import collect, collect_dict, fraction, rcollect

sx, sy, sa, sb, sn = sympy.symbols('x y a b n')


def printed(e: Expr) -> str:
    return str(to_sympy(e, evaluate=False))


@pytest.mark.parametrize('seed', [1, 2])
def test_collect_fraction_match_sympy(seed: int) -> None:
    rng = random.Random(seed)
    atoms: list[sympy.Expr] = [sx, sy, sa, sb, sx**2, sx**sn, sympy.exp(sx), sympy.exp(2 * sx), sympy.sin(sx),
                               sympy.sqrt(sx), 1 / sx, sx * sy, (sx + 1)**2, sympy.Rational(1, 2), sympy.sqrt(2)]
    patterns: list[list[sympy.Expr]] = [[sx], [sx, sy], [sympy.exp(sx)], [sx**2], [sa, sb], [sympy.sin(sx)],
                                        [sx, sympy.exp(sx)]]
    different: list[tuple[str, str, str, str]] = []
    for _ in range(80):
        e = sympy.Add(*[rng.randint(-3, 3) * sympy.Mul(*[rng.choice(atoms) for _ in range(rng.randint(1, 3))])
                        for _ in range(rng.randint(1, 5))])
        if e.has(sympy.zoo, sympy.nan):
            continue
        ours_in = expr_from_sympy(e)
        if printed(ours_in) != str(e):
            continue
        syms = rng.choice(patterns)
        ours_syms = [expr_from_sympy(v) for v in syms]
        checks = [('collect', printed(collect(ours_in, ours_syms)), str(sympy.collect(e, syms))),
                  ('dict', str({to_sympy(k, evaluate=False): to_sympy(v, evaluate=False)
                                for k, v in collect_dict(ours_in, ours_syms).items()}),
                   str(sympy.collect(e, syms, evaluate=False))),
                  ('rcollect', printed(rcollect(ours_in, *ours_syms)), str(sympy.rcollect(e, *syms))),
                  ('fraction', str(tuple(to_sympy(v, evaluate=False) for v in fraction(ours_in))), str(sympy.fraction(e))),
                  ('exact', str(tuple(to_sympy(v, evaluate=False) for v in fraction(ours_in, exact=True))),
                   str(sympy.fraction(e, exact=True)))]
        different += [(name, str(e), o, t) for name, o, t in checks if o != t]
    assert not different, different[:3]


def test_power_and_exponential_substitution() -> None:
    x, d = Symbol('x'), Symbol('d')
    cases: list[tuple[Expr, Expr, str]] = [(exp(2 * x), exp(x), 'd**2'), (exp(3 * x + 1), exp(x), 'E*d**3'),
                                           (exp(-x), exp(x), '1/d'), (x**4, x**2, 'd**2'), (x**3, x**2, 'x**3'),
                                           ((x + 1)**4, (x + 1)**2, 'd**2'), (x, 1 / x, '1/d'), (x + 1, 1 / x, '1 + 1/d')]
    for e, old, expected in cases:
        result = e.subs(old, d)
        assert isinstance(result, Expr) and printed(result) == expected, (e, old)
