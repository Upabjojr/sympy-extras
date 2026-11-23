"""The typed TR rules of fu against SymPy's (the oracle) on random
trigonometric expressions, compared through the printed forms."""
from __future__ import annotations

import importlib
import random

import pytest
import sympy

from sympy_extras.interop import to_sympy
from sympy_extras.interop import expr_from_sympy
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


def test_TR12i_keeps_the_value() -> None:
    # sympy/sympy#31275, V228 and V229: SymPy writes the denominators left over as tan(a) + tan(b) - 1
    # (for tan(a)*tan(b) - 1), drops the rest of a power of tan(a) + tan(b), and tells two denominators
    # apart by the sum of the arguments only (these cases differ from SymPy's results, which are wrong)
    from sympy_extras.core import Float, Rational, Symbol
    from sympy_extras.core.evalf import to_complex
    from sympy_extras.functions import sqrt, tan
    a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    ta, tb, tc = tan(a), tan(b), tan(c)
    first = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    assert str(F.TR12i(first.expand())) == '-3*tan(a + b)*tan(a + c)/(2*(tan(a)*tan(b) - 1))'
    second = (((ta + tb)*(a + 1)).expand())**2/(ta*tb - 1)
    assert str(F.TR12i(second)) == '-(a + 1)**2*(tan(a) + tan(b))*tan(a + b)'
    assert F.TR12i((ta + tb)/(1 - ta*tb)) == tan(a + b)
    cases = [first, first.expand(), second, (ta + tb)**3/(1 - ta*tb)**2, (ta + tb)*(ta + tc)/(1 - ta*tb),
             (ta + tb)/((1 - ta*tb)*(1 - ta*tc)), (ta + tb)/(3 - 3*ta*tb)**2,
             (tan(2*a) + tb)/((1 - tan(2*a)*tb)*(1 - ta*tan(a + b))),
             (ta + tan(a + b))/((1 - tan(2*a)*tb)*(1 - ta*tan(a + b)))]
    for e in cases:
        result = F.TR12i(e)
        assert result != e, e
        for values in ({a: Float('0.3'), b: Float('0.7'), c: Float('-0.4')},
                       {a: Float('1.1'), b: Float('-0.2'), c: Float('0.5')}):
            left, right = to_complex(e.subs(values)), to_complex(result.subs(values))
            assert abs(left - right) < 1e-9*(1 + abs(left)), (e, result)
    # nothing to combine: a sum of tangents without its denominator (a KeyError in SymPy), a
    # denominator under a square root whose cofactor is not known to be positive
    for e in ((ta + tc)/(1 - ta*tb), (ta + tb)/sqrt(c*(1 - ta*tb)), (ta + tb)**Rational(1, 2)/(1 - ta*tb)):
        assert F.TR12i(e) == e
