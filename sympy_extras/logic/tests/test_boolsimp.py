"""The simplification of Boolean formulas of sympy_extras.logic.boolsimp,
checked by the truth of the formulas at sample points (an independent
oracle) and against SymPy's ``simplify`` (the formula ported)."""
from __future__ import annotations

import random

import sympy
from sympy.core.cache import clear_cache

from sympy_extras.core import Eq, Integer, Ne, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, Not, Xor, and_, not_, or_, true
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.logic.boolsimp import simplify_boolean, simplify_univariate
from sympy.testing.pytest import raises

x, y = Symbol('x', real=True), Symbol('y', real=True)
_POINTS = [(Rational(a, 2), Rational(b, 2)) for a in range(-7, 8) for b in range(-7, 8)]


def _same_truth(f: Boolean, g: Boolean) -> bool:
    """Whether ``f`` and ``g`` agree at the sample points of the plane."""
    return all(f.subs({x: p, y: q}) == g.subs({x: p, y: q}) for p, q in _POINTS)


def _sympy_simplify(f: Boolean) -> Boolean:
    clear_cache()
    found = from_sympy(sympy.simplify(to_sympy(f)))
    assert isinstance(found, Boolean)
    return found


def test_patterns_of_conjunctions_and_disjunctions() -> None:
    cases: list[tuple[Boolean, str]] = [
        (and_(x < y, Ne(x, y)), 'x < y'),
        (and_(x <= 1, x <= 2), 'x <= 1'),
        (or_(x > 1, x > 2, x < 0), '(x > 1) | (x < 0)'),
        (and_(x <= y, y <= x), 'Eq(x, y)'),
        (or_(x < y, Eq(x, y)), 'x <= y'),
        (and_(Eq(x, 1), x + y > 2), 'Eq(x, 1) & (y + 1 > 2)'),
        (and_(x < 1, x > 2), 'False'),
    ]
    for f, expected in cases:
        s = simplify_boolean(f)
        assert str(s) == expected, (f, s)
        assert _same_truth(f, s)
        assert str(_sympy_simplify(f)) == expected


def test_endpoints_of_univariate_formulas() -> None:
    # SymPy's simplify_univariate tests the endpoints of each interval on
    # the relation built for the previous one: simplify(~((x > 0) & (x < 1)))
    # is (x >= 1) | (x < 0) there, false at x = 0 where the formula holds
    f = not_(and_(x > 0, x < 1))
    s = simplify_boolean(f)
    assert s.subs(x, Integer(0)) is true and _same_truth(f, s)
    assert str(s) == '(x >= 1) | (x <= 0)'
    u = simplify_univariate(or_(x >= 1, x <= 0))
    assert u.subs(x, Integer(0)) is true and u.subs(x, Integer(1)) is true
    # SymPy's intervals end at max(1, 0) = 1 (an 'Eq condition'): left as given
    assert str(simplify_univariate(and_(x > 1, x < 0))) == '(x > 1) & (x < 0)'


def test_random_formulas_agree_with_sympy() -> None:
    rnd = random.Random(7)
    terms: list[Expr] = [x, y, x + y, 2*x, x - y]
    constants: list[Expr] = [Integer(k) for k in (-1, 0, 1, 2)]

    def atom() -> Boolean:
        lhs = rnd.choice(terms)
        rhs = rnd.choice(constants + [x, y])
        op = rnd.randrange(6)
        return [lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs, Eq(lhs, rhs), Ne(lhs, rhs)][op]

    def formula(depth: int) -> Boolean:
        if depth == 0 or rnd.random() < 0.3:
            return atom()
        kind = rnd.randrange(3)
        if kind == 0:
            return not_(formula(depth - 1))
        args = [formula(depth - 1) for _ in range(rnd.randint(2, 3))]
        return and_(*args) if kind == 1 else or_(*args)
    for _ in range(12):
        f = formula(2)
        s = simplify_boolean(f)
        assert _same_truth(f, s), (f, s)
        assert str(s) == str(_sympy_simplify(f)), f


def test_other_booleans() -> None:
    from sympy_extras.sets.fancysets import Naturals0
    from sympy_extras.sets.sets import Contains
    c = Contains(x - 1, Naturals0)
    assert simplify_boolean(c) is c
    assert simplify_boolean(true) is true
    raises(NotImplementedError, lambda: simplify_boolean(Xor(x > 1, y > 1)))
    assert isinstance(simplify_boolean(Not(and_(x > 0, y > 0))), Boolean)
