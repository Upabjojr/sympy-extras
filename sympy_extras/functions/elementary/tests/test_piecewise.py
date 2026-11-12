"""The typed Piecewise against SymPy (the oracle), compared without
letting SymPy re-evaluate the results."""
from __future__ import annotations

import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.interop import to_sympy
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, and_, false, not_, or_, true
from sympy_extras.core.relational import eq, ge, gt, le, lt, ne
from sympy_extras.functions import Piecewise, conjugate, exp, piecewise, sin

x, y = Symbol('x'), Symbol('y')
p = Symbol('p', positive=True)
r = Symbol('r', real=True)

PairList = list[tuple[Expr, Boolean]]


def cases() -> list[PairList]:
    return [
        [(Integer(1), lt(-x, -1)), (Integer(4), lt(-x, -1)), (Integer(3), false), (Integer(2), true),
         (Integer(5), lt(x, 1))],
        [(x**2, lt(x, 0)), (x, true)],
        [(x, true)],
        [(Integer(1), lt(x, 0)), (Integer(1), gt(x, 2)), (Integer(0), true)],
        [(Integer(1), lt(-2*x + 1, 0)), (Integer(0), true)],
        [(x, lt(3*x, 6)), (y, ge(-x, -5)), (Integer(0), true)],
        [(x, lt(x, 0)), (y, lt(x, 0)), (Integer(1), true)],
        [(x, lt(x, 0)), (y, and_(ge(x, 0), lt(x, 1))), (Integer(1), true)],
        [(x, le(x, 0)), (y, and_(gt(x, 0), lt(y, 1)))],
        [(x, lt(x, 1)), (y, and_(lt(x, 1), lt(y, 2))), (Integer(2), true)],
        [(Integer(1), false)],
        [(Integer(1), gt(p, 0)), (Integer(2), true)],
        [(sin(x), eq(x, 0)), (exp(x), ne(x, 0))],
        [(Integer(1), or_(lt(x, 0), and_(gt(x, 1), lt(y, 0)))), (Integer(1), gt(y, 3)), (Integer(0), true)],
        [(Integer(3), lt(Rational(1, 2)*x + 1, 0)), (x, not_(lt(x, -2)))],
        [(Integer(1), lt(x, 0)), (Integer(2), ge(x, 0))],
        [(Integer(1), lt(x, 0)), (Integer(2), and_(ge(x, 0), le(x, 0))), (Integer(3), true)],
    ]


def test_construction_matches_sympy() -> None:
    different: list[tuple[str, str, str]] = []
    for pairs in cases():
        ours = str(to_sympy(piecewise(*pairs), evaluate=False))
        theirs = str(sympy.Piecewise(*[(to_sympy(e), to_sympy(c)) for e, c in pairs]))
        if ours != theirs:
            different.append((str(pairs), ours, theirs))
    assert not different, different


def test_nested_and_substitution_match_sympy() -> None:
    inner = piecewise((x, lt(x, 0)), (Integer(2), true))
    nested = piecewise((inner, lt(x, 0)), (Integer(3), true))
    s_inner = sympy.Piecewise((sympy.Symbol('x'), sympy.Symbol('x') < 0), (2, True))
    s_nested = sympy.Piecewise((s_inner, sympy.Symbol('x') < 0), (3, True))
    assert str(to_sympy(nested, evaluate=False)) == str(s_nested)
    f = piecewise((x**2, lt(x, 0)), (x + 1, lt(x, 3)), (Integer(7), true))
    for value in [Integer(-2), Integer(0), Integer(2), Integer(3), Integer(9)]:
        ours = f.subs(x, value)
        theirs = sympy_expr(to_sympy(f)).subs(sympy.Symbol('x'), to_sympy(value))
        assert str(to_sympy(ours, evaluate=False)) == str(theirs), value
    g = f.subs(x, y + 1)
    assert str(to_sympy(g, evaluate=False)) == str(sympy_expr(to_sympy(f)).subs(sympy.Symbol('x'),
                                                                                  sympy.Symbol('y') + 1))


def test_derivative_conjugate_and_facts() -> None:
    f = piecewise((x**2, lt(r, 0)), (sin(x), true))
    sf = sympy_expr(to_sympy(f))
    assert str(to_sympy(f.diff(x), evaluate=False)) == str(sf.diff(sympy.Symbol('x')))
    g = piecewise((x + I, lt(r, 0)), (r, true))
    assert str(to_sympy(conjugate(g), evaluate=False)) == str(sympy.conjugate(to_sympy(g)))
    h = piecewise((p, lt(r, 0)), (p + 1, true))
    sh = sympy_expr(to_sympy(h))
    for name in ['positive', 'real', 'finite', 'integer', 'zero', 'negative']:
        got = h._ask(name)
        if got is not None:
            assert got == getattr(sh, 'is_' + name), name
    assert h.is_positive is True


def test_round_trip_and_node() -> None:
    node = Piecewise((x, true))
    assert node.args[0].expr == x and node.doit() == x and piecewise((x, true)) == x
    f = piecewise((x**2, lt(x, 0)), (x, true))
    back = Basic.__eq__(f, f)
    assert back is True
    from sympy_extras.interop import from_sympy
    assert from_sympy(to_sympy(f)) == f


def test_piecewise_simplify_matches_sympy() -> None:
    # random Piecewise of linear conditions in x (the interval path), some
    # nonlinear (the relational solved through SymPy) or in y (the
    # conditions simplified one by one)
    import random
    import sympy
    from sympy.functions.elementary.piecewise import piecewise_simplify as sympy_piecewise_simplify
    from sympy_extras.interop import to_sympy
    from sympy_extras.interop import expr_from_sympy
    from sympy_extras.functions.elementary.piecewise import piecewise_simplify
    sx, sy = sympy.symbols('x y')
    rng = random.Random(1)
    consts = [-2, -1, 0, sympy.Rational(1, 2), 1, 2, 3]
    exprs = [sx, sx**2, 1, 0, 2*sx + 1, sympy.sin(sx), sx*sy, sy]

    def cond() -> sympy.Basic:
        rel = rng.choice([sympy.Lt, sympy.Le, sympy.Gt, sympy.Ge, sympy.Ne])
        lhs = rng.choice([sx, 2*sx, -sx, sx + 1, 3 - sx] + ([sx**2] if rng.random() < 0.1 else []))
        r = rel(lhs, rng.choice(consts))
        k = rng.random()
        if k < 0.25:
            return sympy.And(r, rng.choice([sympy.Lt, sympy.Gt])(sx, rng.choice(consts)))
        if k < 0.35:
            return sympy.Or(r, sympy.Gt(sx, rng.choice(consts)))
        return r
    checked = 0
    for _ in range(80):
        pairs = [(rng.choice(exprs), cond()) for _ in range(rng.randint(1, 3))] + [(rng.choice(exprs), True)]
        e = sympy.Piecewise(*pairs)
        if not isinstance(e, sympy.Piecewise):
            continue
        # SymPy's Or does not see complementary relations inside a nested Or
        # (the DNF of a condition with Ne): ours does, and finds intervals
        if any(isinstance(c, sympy.Ne) for c in e.atoms(sympy.Ne)) and e.atoms(sympy.Or):
            continue
        got = piecewise_simplify(expr_from_sympy(e))
        assert str(to_sympy(got, evaluate=False)) == str(sympy_piecewise_simplify(e)), e
        checked += 1
    assert checked > 50


def test_piecewise_simplify_of_xor_conditions_matches_sympy() -> None:
    # conditions with Xor: the algebraic normal form (several variables)
    # and the conjunctive normal form of Piecewise._intervals (one)
    import random
    import sympy
    from sympy.core.cache import clear_cache
    from sympy.functions.elementary.piecewise import piecewise_simplify as sympy_piecewise_simplify
    from sympy_extras.interop import expr_from_sympy
    from sympy_extras.functions.elementary.piecewise import piecewise_simplify
    sx, sy = sympy.symbols('x y')
    rng = random.Random(2)
    consts = [-2, -1, 0, sympy.Rational(1, 2), 1, 2]

    def rel() -> sympy.logic.boolalg.Boolean:
        r = rng.choice([sympy.Lt, sympy.Le, sympy.Gt, sympy.Ge, sympy.Ne])
        return r(rng.choice([sx, 2*sx, -sx, sy]), rng.choice(consts))

    def cond() -> sympy.Basic:
        k = rng.random()
        c = sympy.Xor(rel(), rel())
        if k < 0.3:
            return sympy.Not(c)
        if k < 0.5:
            return sympy.And(c, rel())
        return c
    checked = 0
    for _ in range(40):
        e = sympy.Piecewise((sx, cond()), (sy, True))
        if not isinstance(e, sympy.Piecewise) or not e.has(sympy.Xor):
            continue
        # SymPy decides the endpoints of the intervals of one variable on
        # the relation of the previous interval (see logic.boolsimp)
        if len(e.free_symbols) == 1 or e.args[0].args[1].free_symbols == {sx}:
            continue
        clear_cache()
        expected = str(sympy_piecewise_simplify(e))
        assert str(to_sympy(piecewise_simplify(expr_from_sympy(e)), evaluate=False)) == expected, e
        checked += 1
    assert checked > 15


def test_relations_with_unknown_symbols_match_sympy() -> None:
    # SymPy decides a relation by the sign of the difference only between
    # extended reals: x + 1 > x is not decided for an x not known real
    from sympy_extras.core import Symbol
    from sympy_extras.core.logic import true
    x, r = Symbol('x'), Symbol('r', real=True)
    assert str(x + 1 > x) == 'x + 1 > x' and (r + 1 > r) is true


def test_piecewise_fold_keeps_the_limits_of_a_sum() -> None:
    from sympy_extras.concrete.expr_with_limits import Sum
    from sympy_extras.core import Integer, Symbol, gt, oo, true
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise, piecewise_fold
    k, a = Symbol('k'), Symbol('a')
    # the bug: the limits, not expressions, were dropped and rebuilding the
    # Sum raised "Sum requires limits" (series_integral(sin(x) + cos(x), x,
    # 0, oo) raised it); SymPy gives the pieces of the sum
    s = Sum(piecewise((a**k, gt(a, 0)), (k, true)), (k, Integer(1), oo))
    assert piecewise_fold(s) == Piecewise((Sum(a**k, (k, Integer(1), oo)), gt(a, 0)),
                                          (Sum(k, (k, Integer(1), oo)), true))
