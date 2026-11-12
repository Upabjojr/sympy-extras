"""The simplification of Boolean formulas of sympy_extras.logic.boolsimp,
checked by the truth of the formulas at sample points (an independent
oracle) and against SymPy's ``simplify`` (the formula ported). The
expected values are SymPy 1.14's ``simplify`` on the same formulas, except
where noted."""
from __future__ import annotations

import random

import sympy
from sympy.core.cache import clear_cache

from sympy_extras.core import Eq, Integer, Ne, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, BooleanAtom, Not, Xor, and_, false, not_, or_, true
from sympy_extras.core.relational import Ge, Gt, Le, Lt
from sympy_extras.functions import Abs, sin
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.logic.boolsimp import simplify_boolean, simplify_univariate

x, y = Symbol('x', real=True), Symbol('y', real=True)
u, v = Symbol('u'), Symbol('v')
_POINTS = [(Rational(a, 2), Rational(b, 2)) for a in range(-7, 8) for b in range(-7, 8)]


def _same_truth(f: Boolean, g: Boolean) -> bool:
    """Whether ``f`` and ``g`` agree at the sample points of the plane."""
    return all(f.subs({x: p, y: q}) == g.subs({x: p, y: q}) for p, q in _POINTS)


def _same_truth_plain(a: Boolean, b: Boolean) -> bool:
    """The formulas in ``u``, ``v`` agree at rational sample points."""
    for i in range(-12, 13):
        for j in (-3, 0, 2):
            point = {u: Rational(i, 4), v: Rational(j)}
            va, vb = a.xreplace(point).doit(), b.xreplace(point).doit()
            if isinstance(va, BooleanAtom) and isinstance(vb, BooleanAtom) and bool(va) != bool(vb):
                return False
    return True


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


def test_known_simplifications() -> None:
    cases: list[tuple[Boolean, Boolean]] = [
        (and_(u > 1, u < 0), false),
        (and_(u > 0, Ne(u, 0)), Gt(u, 0)),
        (and_(u > 1, u < 3), and_(Gt(u, 1), Lt(u, 3))),
        (or_(u < 1, and_(u >= 1, u < 2)), Lt(u, 2)),
        # SymPy leaves (u > 0) & (u**2 > 4): Piecewise._intervals meets the
        # empty interval (0, -2) of the term (u < -2) & (u > 0), which it
        # takes for a point; the intervals are computed exactly here
        (and_(u**2 > 4, u > 0), Gt(u, 2)),
        (or_(u**2 > 4, u < 0), or_(Gt(u, 2), Lt(u, 0))),
        (and_(Eq(u, 1), v > u), and_(Eq(u, 1), Gt(v, 1))),
        (and_(Eq(u, v), Eq(u, -v)), and_(Eq(u, 0), Eq(v, 0))),
        (or_(u >= 1, Eq(u, 0)), or_(Eq(u, 0), Ge(u, 1))),
        (and_(u < v, v < u), false),
        (or_(sin(u) > 0, sin(u) <= 0), true),
        (and_(u > v, u >= v), Gt(u, v)),
        (and_(u > 1, u > 2, v > 3), and_(Gt(u, 2), Gt(v, 3))),
        (and_(u < 2, u > -2), and_(Gt(u, -2), Lt(u, 2))),
        (or_(u > 2, u < -2), Gt(Abs(u), 2)),
    ]
    for formula, expected in cases:
        found = simplify_boolean(formula)
        assert found == expected, (formula, found)
        assert _same_truth_plain(found, formula), formula


def test_endpoints_of_univariate_formulas() -> None:
    # SymPy's simplify_univariate tests the endpoints of each interval on
    # the relation built for the previous one (it reuses the name of the
    # formula): simplify(~((x > 0) & (x < 1))) is (x >= 1) | (x < 0)
    # there, false at x = 0 where the formula holds. The intervals are
    # computed from the formula here.
    f = not_(and_(x > 0, x < 1))
    s = simplify_boolean(f)
    assert s.subs(x, Integer(0)) is true and _same_truth(f, s)
    assert str(s) == '(x >= 1) | (x <= 0)'
    w = simplify_univariate(or_(x >= 1, x <= 0))
    assert w.subs(x, Integer(0)) is true and w.subs(x, Integer(1)) is true
    formula = not_(and_(u > -1, u < 1))
    assert simplify_univariate(formula) == or_(Le(u, -1), Ge(u, 1))
    assert _same_truth_plain(simplify_boolean(formula), formula)


def test_univariate_intervals() -> None:
    assert simplify_univariate(and_(u > 1, u < 3)) == and_(Lt(1, u), Lt(u, 3))
    assert simplify_univariate(or_(u > 1, u < -1)) == or_(Gt(u, 1), Lt(u, -1))
    # SymPy's intervals end at max(1, 0) = 1 (an 'Eq condition') and leave
    # the formula as given; the empty union of intervals is false here
    assert simplify_univariate(and_(x > 1, x < 0)) is false
    # a relation which is not solved, and a formula true at a single
    # point, are left as they are
    formula = and_(sin(u) > 0, u < 0)
    assert simplify_univariate(formula) == formula
    formula = and_(u <= 1, u >= 1)
    assert simplify_univariate(formula) == formula


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
    # Xor takes the univariate intervals and simplify_logic only (SymPy's
    # Xor._eval_simplify, by the algebraic normal form, is not ported)
    f = Xor(x > 1, y > 1)
    assert _same_truth(simplify_boolean(f), f)
    assert str(simplify_boolean(Xor(x > 1, x > 2))) == "(x <= 2) & (x > 1)"
    assert isinstance(simplify_boolean(Not(and_(x > 0, y > 0))), Boolean)


def test_sympy_intervals() -> None:
    # the conditions of piecewise_simplify_arguments take SymPy's intervals
    # (Piecewise._intervals, one per term), the endpoints decided on the
    # formula
    f = or_(x <= -3, x < 1)
    assert str(simplify_univariate(f)) == 'x < 1'
    assert str(simplify_univariate(f, True)) == '(x <= -3) | (x < 1)'
    f = not_(and_(x + 1 >= 2, -x <= 3))
    assert str(simplify_boolean(f)) == 'x < 1'
    assert str(simplify_boolean(f, sympy_intervals=True)) == str(_sympy_simplify(f)) == '(x <= -3) | (x < 1)'
    assert str(simplify_univariate(not_(and_(x > 0, x < 1)), True)) == '(x >= 1) | (x <= 0)'
    # true at the isolated point x = 0: SymPy's simplify gives False
    f = and_(x < 2, x**2 <= 0, -x < 2)
    assert _sympy_simplify(f) is false
    for intervals in (False, True):
        s = simplify_boolean(f, sympy_intervals=intervals)
        assert s.subs(x, Integer(0)) is true and _same_truth(f, s), s
