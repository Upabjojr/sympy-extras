"""Tests of the port of SymPy's simplification of compound Boolean
formulas (:mod:`sympy_extras.assumptions.boolsimp`). The expected values
are SymPy 1.14's ``simplify`` on the same formulas, except where noted."""
from __future__ import annotations

from sympy_extras.core import Eq, Ne, Rational, Symbol, and_, not_, or_
from sympy_extras.core.logic import Boolean, BooleanAtom, false, true
from sympy_extras.core.relational import Ge, Gt, Le, Lt
from sympy_extras.functions import Abs, sin
from sympy_extras.assumptions.boolsimp import simplify_boolean, simplify_univariate

x, y = Symbol('x'), Symbol('y')


def _same_truth(a: Boolean, b: Boolean) -> bool:
    """The formulas agree at rational sample points of x and y."""
    for i in range(-12, 13):
        for j in (-3, 0, 2):
            point = {x: Rational(i, 4), y: Rational(j)}
            va, vb = a.xreplace(point).doit(), b.xreplace(point).doit()
            if isinstance(va, BooleanAtom) and isinstance(vb, BooleanAtom) and bool(va) != bool(vb):
                return False
    return True


def test_known_simplifications() -> None:
    cases: list[tuple[Boolean, Boolean]] = [
        (and_(x > 1, x < 0), false),
        (and_(x > 0, Ne(x, 0)), Gt(x, 0)),
        (and_(x > 1, x < 3), and_(Gt(x, 1), Lt(x, 3))),
        (or_(x < 1, and_(x >= 1, x < 2)), Lt(x, 2)),
        # SymPy leaves (x > 0) & (x**2 > 4): Piecewise._intervals meets the
        # empty interval (0, -2) of the term (x < -2) & (x > 0), which it
        # takes for a point; the intervals are computed exactly here
        (and_(x**2 > 4, x > 0), Gt(x, 2)),
        (or_(x**2 > 4, x < 0), or_(Gt(x, 2), Lt(x, 0))),
        (and_(Eq(x, 1), y > x), and_(Eq(x, 1), Gt(y, 1))),
        (and_(Eq(x, y), Eq(x, -y)), and_(Eq(x, 0), Eq(y, 0))),
        (or_(x >= 1, Eq(x, 0)), or_(Eq(x, 0), Ge(x, 1))),
        (and_(x < y, y < x), false),
        (or_(sin(x) > 0, sin(x) <= 0), true),
        (and_(x > y, x >= y), Gt(x, y)),
        (and_(x > 1, x > 2, y > 3), and_(Gt(x, 2), Gt(y, 3))),
        (and_(x < 2, x > -2), and_(Gt(x, -2), Lt(x, 2))),
        (or_(x > 2, x < -2), Gt(Abs(x), 2)),
    ]
    for formula, expected in cases:
        found = simplify_boolean(formula)
        assert found == expected, (formula, found)
        assert _same_truth(found, formula), formula


def test_univariate_intervals() -> None:
    assert simplify_univariate(and_(x > 1, x < 3)) == and_(Lt(1, x), Lt(x, 3))
    assert simplify_univariate(or_(x > 1, x < -1)) == or_(Gt(x, 1), Lt(x, -1))
    # a relation which is not solved, and a formula true at a single
    # point, are left as they are
    formula = and_(sin(x) > 0, x < 0)
    assert simplify_univariate(formula) == formula
    formula = and_(x <= 1, x >= 1)
    assert simplify_univariate(formula) == formula


def test_endpoint_of_a_negated_interval() -> None:
    # SymPy 1.14 simplifies ~((x > -1) & (x < 1)) to (x >= 1) | (x < -1),
    # which is false at x = -1: simplify_univariate reuses the name of the
    # formula for the interval it builds and decides the next endpoint on
    # it. The intervals are computed from the formula here.
    formula = not_(and_(x > -1, x < 1))
    found = simplify_univariate(formula)
    assert found == or_(Le(x, -1), Ge(x, 1))
    assert _same_truth(simplify_boolean(formula), formula)
