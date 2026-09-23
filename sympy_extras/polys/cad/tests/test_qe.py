from __future__ import annotations

import random
from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent, Xor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.sets import Interval, FiniteSet, Union
from sympy_extras._typing import QuantifierSpec, as_boolean
from sympy_extras.polys.cad.qe import (quantifier_elimination, decide,
    sample_points, solution_set, _truth_values, CellTruth)
from sympy_extras.polys.cad.lifting import CAD
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, d, e, x, y, z

qe = quantifier_elimination


def _equivalent(f: Boolean, g: Boolean, gens: list[Symbol]) -> bool:
    """Whether two quantifier-free formulas agree on every cell of a
    decomposition sign-invariant for the polynomials of both."""
    _, _, _, cells = _truth_values(Equivalent(f, g), gens, [], None)
    return all(value for _, value in cells)


def test_decide() -> None:
    assert decide(x >= 22, [('forall', x)]) is False
    assert decide(x >= 22, [('exists', x)]) is True
    assert decide(x >= 22, [('exists', [x, y])]) is True
    assert decide(Eq(x, 2*y), [('forall', x), ('exists', y)]) is True
    assert decide(Eq(x, y**2), [('forall', x), ('exists', y)]) is False
    assert decide(Eq(x, y**2), [('exists', x), ('forall', y)]) is False
    assert decide(Eq(y, x**2), [('forall', x), ('exists', y)]) is True
    assert decide(Ne(x, x + 1), [('forall', x)]) is True
    assert decide(Ne(x, 2*x), [('forall', x)]) is False
    assert decide(Eq(x, 2*x), [('exists', x)]) is True
    assert decide(And(x + 1 < y, Eq(x, y)), [('exists', [x, y])]) is False
    assert decide(Or(x >= 22, x < 22), [('forall', x)]) is True
    assert decide(And(x >= 5, x < 1), [('forall', x)]) is False
    assert decide(Implies(x > 0, x >= 0), [('forall', x)]) is True
    assert decide(Implies(x > 0, x < 0), [('forall', x)]) is False
    assert decide(Equivalent(x >= 0, x**3 >= 0), [('forall', x)]) is True
    assert decide(Xor(x > 0, x <= 0), [('forall', x)]) is True
    assert decide(Not(x**2 < 0), [('forall', x)]) is True
    assert decide(Eq(y, z*x), [('forall', [x, y]), ('exists', z)]) is False
    assert decide(Eq(y, z*x), [('forall', y), ('exists', [x, z])]) is True
    assert decide(x**2 + y**2 < 0, [('exists', [x, y])]) is False
    assert decide(x**2 + y**2 < 1, [('forall', x), ('exists', y)]) is False
    assert decide(x**2 + y**2 < 1, [('exists', x), ('exists', y)]) is True
    assert decide(x**2 + y**2 - 2*x*y >= 0, [('forall', [x, y])]) is True
    assert decide(S.true, [('forall', x)]) is True
    assert decide(S.false, [('exists', x)]) is False
    # the quantifier order matters
    assert decide(x < y, [('forall', x), ('exists', y)]) is True
    assert decide(x < y, [('exists', y), ('forall', x)]) is False
    # a real root of every odd degree polynomial, none for x**2 + 1
    assert decide(Eq(x**3 + a*x + b, 0), [('forall', [a, b]), ('exists', x)]) is True
    assert decide(Eq(x**2 + 1, 0), [('exists', x)]) is False
    # positive definite quadratic form
    assert decide(x**2 + x*y + y**2 > 0, [('forall', [x, y])]) is False
    assert decide(Or(x**2 + x*y + y**2 > 0, And(Eq(x, 0), Eq(y, 0))), [('forall', [x, y])]) is True
    assert decide(x**2 - x*y + y**2 >= 0, [('forall', [x, y])]) is True
    assert decide(x**2 - 3*x*y + y**2 >= 0, [('forall', [x, y])]) is False


def test_quantifier_elimination_one_variable() -> None:
    # solvability of the monic quadratic
    assert qe(Eq(x**2 + a*x + b, 0), [('exists', x)]) == (a**2 - 4*b >= 0)
    # positivity of the monic quadratic
    assert qe(x**2 + b*x + c > 0, [('forall', x)]) == (b**2 - 4*c < 0)
    assert qe(x**2 + b*x + c >= 0, [('forall', x)]) == (b**2 - 4*c <= 0)
    # projections of an ellipse (the answers 8*x**2 - 8*x - 29 <= 0 and
    # 8*y**2 + 16*y - 85 <= 0 are due to QEPCAD)
    ellipse = Eq(3*x**2 + 2*x*y + y**2 - x + y - 7, 0)
    r0, r1 = CRootOf(8*x**2 - 8*x - 29, 0), CRootOf(8*x**2 - 8*x - 29, 1)
    assert qe(ellipse, [('exists', y)]) == And(x >= r0, x <= r1)
    assert solution_set(ellipse, x, [('exists', y)]) == Interval(r0, r1)
    assert solution_set(8*x**2 - 8*x - 29 <= 0, x) == Interval(r0, r1)
    s0, s1 = CRootOf(8*x**2 + 16*x - 85, 0), CRootOf(8*x**2 + 16*x - 85, 1)
    assert solution_set(ellipse, y, [('exists', x)]) == Interval(s0, s1)
    assert solution_set(8*y**2 + 16*y - 85 <= 0, y) == Interval(s0, s1)
    # a circle and its inside
    assert qe(Eq(x**2 + y**2, 1), [('exists', y)]) == And(x >= -1, x <= 1)
    assert qe(x**2 + y**2 < 1, [('exists', y)]) == And(x > -1, x < 1)
    assert qe(x**2 + y**2 < 1, [('forall', y)]) == S.false
    assert qe(x**2 + y**2 > -1, [('forall', y)]) == S.true
    assert qe(Or(x**2 + y**2 < 1, x**2 + y**2 > 4), [('forall', y)]) == Or(x < -2, x > 2)
    # unions of intervals and points
    assert solution_set(x**2 > 2, x) == Union(
        Interval.open(-S.Infinity, CRootOf(x**2 - 2, 0)),
        Interval.open(CRootOf(x**2 - 2, 1), S.Infinity))
    assert solution_set(x**2 >= 2, x) == Union(
        Interval(-S.Infinity, CRootOf(x**2 - 2, 0)),
        Interval(CRootOf(x**2 - 2, 1), S.Infinity))
    assert solution_set(Eq(x**2, 2), x) == FiniteSet(CRootOf(x**2 - 2, 0), CRootOf(x**2 - 2, 1))
    assert solution_set(Or(Eq(x, 1), And(x > 2, x <= 3)), x) == Union(FiniteSet(1), Interval.Lopen(2, 3))
    assert solution_set(x**2 < 0, x) == S.EmptySet
    assert solution_set(x**2 >= 0, x) == S.Reals
    assert solution_set(Ne(x, 1), x) == Union(Interval.open(-S.Infinity, 1), Interval.open(1, S.Infinity))
    assert solution_set(And(x >= 1, x <= 1), x) == FiniteSet(1)
    assert solution_set(x**3 - x > 0, x) == Union(Interval.open(-1, 0), Interval.open(1, S.Infinity))
    assert qe(x**3 - x > 0, []) == Or(And(x > -1, x < 0), x > 1)
    # for which x is there a y with y > x on the unit circle
    assert solution_set(And(Eq(x**2 + y**2, 1), y > x), x, [('exists', y)]) == \
        Interval.Ropen(-1, CRootOf(2*x**2 - 1, 1))
    # the quantified variables need not appear
    assert qe(x > 1, [('forall', y)]) == (x > 1)
    assert solution_set(x > 1, x, [('exists', [y, z])]) == Interval.open(1, S.Infinity)
    # rational coefficients
    assert solution_set(x/2 > Rational(1, 3), x) == Interval.open(Rational(2, 3), S.Infinity)


def test_quantifier_elimination_several_variables() -> None:
    # a linear polynomial has a positive value: a /= 0 \/ b > 0
    r = qe(a*x + b > 0, [('exists', x)])
    assert _equivalent(r, Or(Ne(a, 0), b > 0), [a, b])
    # solvability of the general quadratic, QEPCAD gives
    # 4 a c - b^2 <= 0 /\ [ c = 0 \/ a /= 0 \/ 4 a c - b^2 < 0 ]
    r = qe(Eq(a*x**2 + b*x + c, 0), [('exists', x)])
    ref = And(4*a*c - b**2 <= 0, Or(Eq(c, 0), Ne(a, 0), 4*a*c - b**2 < 0))
    assert _equivalent(r, ref, [a, b, c])
    # a minimum cover of the true cells by prime implicants (the greedy
    # merge gave four terms, (a > 0 & D = 0) and (a < 0 & D <= 0) apart)
    assert r == Or(Eq(c, 0), 4*a*c - b**2 < 0, And(Ne(b, 0), 4*a*c - b**2 <= 0))
    # no real root, QEPCAD: 4 a c - b^2 >= 0 /\ c /= 0 /\ [ b = 0 \/ 4 a c - b^2 > 0 ]
    r = qe(Ne(a*x**2 + b*x + c, 0), [('forall', x)])
    ref = And(4*a*c - b**2 >= 0, Ne(c, 0), Or(Eq(b, 0), 4*a*c - b**2 > 0))
    assert _equivalent(r, ref, [a, b, c])
    assert r == Or(4*a*c - b**2 > 0, And(Eq(b, 0), Ne(c, 0), 4*a*c - b**2 >= 0))
    # a positive value of the general quadratic; Redlog answers
    # a > 0 or (b < 0 and a = 0) or (a = 0 and (b > 0 or (c > 0 and b = 0)))
    # or (a < 0 and 4*a*c - b^2 < 0)
    r = qe(a*x**2 + b*x + c > 0, [('exists', x)])
    assert r == Or(a > 0, c > 0, 4*a*c - b**2 < 0)
    ref = Or(a > 0, And(b < 0, Eq(a, 0)), And(Eq(a, 0), Or(b > 0, And(c > 0, Eq(b, 0)))),
             And(a < 0, 4*a*c - b**2 < 0))
    assert _equivalent(r, ref, [a, b, c])
    # with a /= 0 assumed the condition is the discriminant
    r = qe(And(Ne(a, 0), Eq(a*x**2 + b*x + c, 0)), [('exists', x)])
    assert _equivalent(r, And(Ne(a, 0), 4*a*c - b**2 <= 0), [a, b, c])
    # a quantifier-free formula is described by its own polynomials
    assert qe((x**2 + y**2 < 1) & (x > y), []) == And(x - y > 0, x**2 + y**2 - 1 < 0)
    assert qe(x**2 + y**2 < 0, []) == S.false
    assert qe(x**2 + y**2 >= 0, []) == S.true
    assert qe(Or(x > y, x <= y), [], free=[x, y]) == S.true
    # existence of a point of the circle over a
    r = qe(And(Eq(x**2 + y**2, 1), Eq(a, x)), [('exists', [x, y])])
    assert r == And(a >= -1, a <= 1)
    # x**2 + a*x + 1 = 0 has a real root iff |a| >= 2
    assert qe(Eq(x**2 + a*x + 1, 0), [('exists', x)]) == Or(a >= 2, a <= -2)
    # x**4 + d*x + e = 0 has a real root iff 256 e^3 <= 27 d^4
    r = qe(Eq(x**4 + d*x + e, 0), [('exists', x)])
    assert _equivalent(r, 256*e**3 - 27*d**4 <= 0, [d, e])
    # the depressed cubic has a real root for every p, q
    p, q = symbols('p q')
    assert qe(Eq(x**3 + p*x + q, 0), [('exists', x)], free=[p, q]) == S.true


def test_sample_points() -> None:
    assert sample_points(x**2 + y**2 < 0, [x, y]) == []
    assert sample_points((x**2 + y**2 < 1) & (x > y), [x, y]) == [
        {x: 0, y: -Rational(1, 2)}, {x: CRootOf(2*x**2 - 1, 1), y: 0}, {x: Rational(3, 4), y: 0}]
    pts = sample_points(Eq(x**2 + y**2, 1), [x, y])
    assert len(pts) == 4 and all((p[x]**2 + p[y]**2 - 1).equals(0) for p in pts)
    assert sample_points(x > 2, [x]) == [{x: 3}]
    assert sample_points(S.true, [x, y]) == [{x: 0, y: 0}]
    assert sample_points((x > 0) & (y > 0) & (x + y < 1), [x, y]) == [{x: Rational(1, 2), y: Rational(1, 3)}]
    f = And(x**2 + y**2 - 7 > 0, x + y - 2 > 0, x**2 + y - 10 > 0)
    pts = sample_points(f, [x, y])
    assert len(pts) == 13 and all(f.subs(list(pt.items())) for pt in pts)
    assert sample_points(And(x**2 + y**2 - 7 > 0, x + y - 2 > 0, x**2 + y - 10 > 0,
                             x**2 + y**2 - 7 < 0), [x, y]) == []


def test_errors() -> None:
    raises(ValueError, lambda: qe(x > 0, [('some', x)]))
    raises(ValueError, lambda: qe(x > 0, [('exists', x)], free=[x]))
    raises(ValueError, lambda: qe(x + y > 0, [('exists', x)], free=[]))
    raises(ValueError, lambda: qe(x + y > 0, [('exists', x)], free=[z]))
    raises(PolynomialError, lambda: qe(sqrt(x) > 0, [('exists', x)]))
    raises(ValueError, lambda: qe(x, [('exists', x)]))
    from sympy.logic.boolalg import ITE
    raises(ValueError, lambda: qe(ITE(x > 0, y > 0, y < 0), [('exists', [x, y])]))


def test_sign_formula_minimization() -> None:
    # the disc, described by one factor
    assert qe(x**2 + y**2 < 1, [], free=[x, y]) == (x**2 + y**2 - 1 < 0)
    # a half plane with a redundant polynomial in the formula
    assert qe(Or(x - y > 0, And(x - y > 0, x > 0)), [], free=[x, y]) == (x - y > 0)
    # complement of a line
    assert qe(Not(Eq(x, y)), [], free=[x, y]) == Ne(x - y, 0)
    # a formula that needs the sign of two factors
    assert qe(x*y > 0, [], free=[x, y]) == Or(And(x < 0, y < 0), And(x > 0, y > 0))
    r = qe(x*y >= 0, [], free=[x, y])
    # the greedy merge gave three terms, Eq(y, 0) | (x >= 0 & y > 0) |
    # (x <= 0 & y <= 0); the prime implicants cover the true cells with two
    assert r == Or(And(x >= 0, y >= 0), And(x <= 0, y <= 0))
    assert _equivalent(r, x*y >= 0, [x, y])
    # merged signs
    r = qe(Ne(x*y, 0), [], free=[x, y])
    assert _equivalent(r, And(Ne(x, 0), Ne(y, 0)), [x, y])
    assert r.count(Ne) == 2 or r == And(Ne(x, 0), Ne(y, 0))


def test_trial_evaluation() -> None:
    # the three-valued evaluation of a compiled formula on partial sign
    # vectors (None for a sign which is not known yet)
    from sympy_extras.polys.cad.qe import _compile
    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    gens = [x, y]
    compiled = [_compile(f, gens, polys, index) for f in [
        And(x > 0, y > 0), Or(x > 0, y > 0), Implies(x > 0, y > 0), Equivalent(x > 0, y > 0),
        Xor(x > 0, y > 0), Not(And(x > 0, y > 0))]]
    assert [c.trial([1, None]) for c in compiled] == [None, True, None, None, None, None]
    assert [c.trial([-1, None]) for c in compiled] == [False, None, True, None, None, True]
    assert [c.trial([None, 1]) for c in compiled] == [None, True, True, None, None, None]
    assert [c.trial([None, -1]) for c in compiled] == [False, None, None, None, None, True]
    assert [c.trial([1, -1]) for c in compiled] == [False, True, False, False, True, True]
    assert [c.trial([1, 1]) for c in compiled] == [True, True, True, True, False, False]
    assert [c.trial([-1, -1]) for c in compiled] == [False, False, True, True, False, True]
    assert [c.trial([None, None]) for c in compiled] == [None] * 6
    assert compiled[0].trial([0, 1]) is False and compiled[1].trial([0, 0]) is False


def test_partial_decomposition_agrees_with_the_full_one() -> None:
    # the partial decomposition (Collins-Hong) lifts the stacks which the
    # truth value depends on; the full one, the reference, is kept
    # reachable with partial=False and gives the same answers
    for formula, prefix in [
            (x**2 + y**2 < 1, [('forall', x), ('exists', y)]),
            (Eq(y, x**2), [('forall', x), ('exists', y)]),
            (Or(x**2 + x*y + y**2 > 0, And(Eq(x, 0), Eq(y, 0))), [('forall', [x, y])]),
            (And(x*y*z > 1, x + y + z < 3, x > 0, y > 0, z > 0), [('exists', [x, y, z])]),
            (Implies(x**2 + y**2 < 1, And(z**2 > x, z < y)), [('forall', [x, y]), ('exists', z)]),
            (Eq(x**3 + a*x + b, 0), [('forall', [a, b]), ('exists', x)]),
            (Ne(y*z + 2*z, 0) | (x*z - 2*y**2 - 3*y + 1 <= 0), [('exists', x), ('exists', y), ('forall', z)])]:
        assert decide(formula, prefix) is decide(formula, prefix, partial=False), formula
    for formula, prefix, free in [
            (Eq(a*x**2 + b*x + c, 0), [('exists', x)], [a, b, c]),
            (And(a*x**2 + b*x + c > 0, x > 0, x < 1), [('exists', x)], [a, b, c]),
            (Implies(Eq(y, x**2), y >= a*x + b), [('forall', x), ('forall', y)], [a, b]),
            (And(Eq(z**2, x), z > y), [('exists', z)], [x, y]),
            (And(x**2 + y**2 + z**2 < 1, z > x + y), [('exists', z)], [x, y]),
            (Eq(x**3 + b*x + c, 0) & (x > 0), [('exists', x)], [b, c])]:
        assert qe(formula, prefix, free=free) == qe(formula, prefix, free=free, partial=False), formula
    for formula, prefix in [
            (And(Eq(x**2 + y**2, 1), y > x), [('exists', y)]),
            (Implies(y**2 + z**2 < 1, x + y*z > 0), [('forall', [y, z])]),
            (Or(Eq(x*z - y, 0), Ne(x**2 + y*z + 1, 0)), [('forall', y), ('forall', z)])]:
        assert solution_set(formula, x, prefix) == solution_set(formula, x, prefix, partial=False), formula
    for formula, gens in [((x**2 + y**2 < 1) & (x > y), [x, y]), (Eq(x**2 + y**2, 1), [x, y]),
                          ((x**2 + y**2 + z**2 < 4) & (x*y > 1) & (z > x), [x, y, z])]:
        assert sample_points(formula, gens) == sample_points(formula, gens, partial=False), formula


def test_partial_decomposition_lifts_fewer_cells() -> None:
    from sympy_extras.polys.cad.qe import _truth_values
    formula = And(x**2 + y**2 + z**2 < 4, z > x*y, y > x)
    prefix: list[tuple[str, Symbol]] = [('exists', y), ('exists', z)]
    # the full decomposition of this one takes minutes (the old code, which
    # built it, did not answer in a minute); the partial one lifts the
    # stacks it needs. The end of the interval is checked on both sides
    end = CRootOf(x**4 + 2*x**2 - 4, 1)
    assert solution_set(formula, x, prefix) == Interval.open(-2, end)
    assert 1.11 < end.evalf() < 1.12
    assert decide(formula.subs(x, Rational(11, 10)), prefix) is True
    assert decide(formula.subs(x, Rational(112, 100)), prefix) is False
    assert decide(formula.subs(x, Rational(-199, 100)), prefix) is True
    assert decide(formula.subs(x, -2), prefix) is False
    formula = Implies(x**2 + y**2 < 1, And(z**2 > x, z < y))
    blocks: QuantifierSpec = [('forall', [x, y]), ('exists', z)]
    partial, _, value, _ = _truth_values(formula, [], blocks, None)
    full, _, value_, _ = _truth_values(formula, [], blocks, None, partial=False)
    assert value is value_ is True
    assert len(full.cells_at(3)) > 4 * len(partial.cells_at(3)) and len(partial.cells_at(3)) > 0
    # over a cell where the signs of the factors of the lower levels decide
    # the formula, nothing is lifted for the sample points either
    assert sample_points(And(x > 0, x**2 + y**2 < 1, y > x), [x, y]) == [{x: Rational(1, 2), y: Rational(2, 3)}]
    cad, _, _, cells = _truth_values(And(x > 0, x**2 + y**2 < 1), [x, y], [], None, True, 'false')
    assert [c.point for c in cad.cells_at(1)] == [(-2,), (-1,), (-Rational(1, 2),), (0,), (Rational(1, 2),), (1,), (2,)]
    # x > 0 fails on the first four cells of the line: nothing is lifted
    # over them; over the others the sign of the circle is not known
    assert [(c.index, v) for c, v in cells] == [((1,), False), ((2,), False), ((3,), False), ((4,), False),
        ((5, 1), False), ((5, 2), False), ((5, 3), True), ((5, 4), False), ((5, 5), False),
        ((6, 1), False), ((6, 2), False), ((6, 3), False), ((7, 1), False)]


def test_equational_constraint_in_the_formula() -> None:
    from sympy_extras.polys.cad.qe import _equational_constraint, _truth_values
    assert _equational_constraint(Eq(x*y, 1), [x, y]) == Poly(x*y - 1, x, y)
    assert _equational_constraint(And(x > 0, Eq(y, x**2), Eq(x, 2)), [x, y]) == Poly(y - x**2, x, y)
    # not implied by the formula, or not of the last level
    assert _equational_constraint(Or(Eq(x*y, 1), x > 0), [x, y]) is None
    assert _equational_constraint(And(Eq(x, 2), y > 0), [x, y]) is None
    # the reduced projection is used when the last variable is quantified;
    # the answer is that of the full decomposition
    formula = And(Eq(x**2 + y**2, 1), Eq(y, a*x + b))
    prefix: QuantifierSpec = [('exists', [x, y])]
    reduced, _, _, cells = _truth_values(formula, [a, b], prefix, None)
    full, _, _, cells_ = _truth_values(formula, [a, b], prefix, None, partial=False)
    assert len(reduced.projection[2]) < len(full.projection[2]) and len(reduced.projection[0]) < len(full.projection[0])
    # the space of the free variables is decomposed more coarsely (fewer
    # polynomials are projected); the truth values are right on every cell
    assert 0 < len(cells) < len(cells_)
    for cell, value in cells:
        if all(coordinate.is_Rational for coordinate in cell.point):
            assert decide(formula.subs(dict(zip([a, b], cell.point))), prefix) is value
    assert qe(formula, prefix, free=[a, b]) == qe(formula, prefix, free=[a, b], partial=False) == (a**2 - b**2 + 1 >= 0)
    # with the last variable free the cells must be sign-invariant for
    # every polynomial: no reduction
    plain, _, _, _ = _truth_values(formula, [a, b, x, y], [], None)
    assert plain.projection == full.projection


def _agrees_at_random_points(formula: Boolean, prefix: QuantifierSpec, answer: Boolean,
                             free: list[Symbol], points: int, seed: int) -> None:
    """The answer has the truth value of the quantified formula at random
    rational points of the space of the free variables (decide answers
    the formula with the values put in, which has no free variable)."""
    rng = random.Random(seed)
    for _ in range(points):
        values: dict[Basic | complex, Basic | complex] = {v: Rational(rng.randint(-20, 20), rng.randint(1, 5)) for v in free}
        got = as_boolean(answer.subs(values))
        assert got in (S.true, S.false), (answer, values)
        assert bool(got) is decide(as_boolean(formula.subs(values)), prefix), (formula, values)


def test_solution_formula_by_the_augmented_projection() -> None:
    # sympy-extras#9: the projection factors of two free variables may not
    # tell the true cells from the false ones. For x >= 0 the condition is
    # y < sqrt(x): the factor y**2 - x is positive on both sides of the
    # parabola and no factor vanishes on y = 0. NotImplementedError was
    # raised, then the answer was written with the root function sqrt(x);
    # Hong's augmented projection adds the derivative y of the factor and
    # the answer is a formula in polynomial sign conditions
    from sympy_extras.polys.cad.qe import _separating_refinement, _sign_formula
    formula: Boolean = Eq(z**2, x) & (z > y)
    prefix: QuantifierSpec = [('exists', z)]
    answer = qe(formula, prefix, free=[x, y])
    assert answer == Or(And(x >= 0, y < 0), x - y**2 > 0)
    assert qe(formula, prefix, free=[x, y], partial=False) == answer
    assert qe(formula, prefix, free=[x, y], method='hong') == answer
    assert _equivalent(answer, And(x >= 0, Or(y < 0, y**2 < x)), [x, y])
    _agrees_at_random_points(formula, prefix, answer, [x, y], 300, 9)
    # the refinement: every cell of the refined decomposition takes the
    # truth value of the cell of the first one which contains it; the
    # refined projection has the derivative and its cells are more
    cad, _, _, cells = _truth_values(formula, [x, y], prefix, None)
    assert _sign_formula(cad, cells, 2) is None
    refined = _separating_refinement(cad, cells, 2)
    assert refined is not None
    cad_, cells_ = refined
    assert [f.as_expr() for f in cad_.projection[1]] == [x - y**2, y] and cad_.projection[0] == cad.projection[0]
    assert len(cells_) > len(cells)
    for cell, value in cells_:
        if all(coordinate.is_Rational for coordinate in cell.point):
            assert decide(formula.subs(dict(zip([x, y], cell.point))), prefix) is value
    # the order of the free variables: the same set
    answer_ = qe(formula, prefix, free=[y, x])
    assert _equivalent(answer_, answer, [x, y])
    # roots of higher degree: the derivatives of x - y**4 and of the cubic
    # (the second derivative is needed for y - 1 below)
    formula = Eq(z**4, x) & (z > y)
    answer = qe(formula, prefix, free=[x, y])
    assert answer == Or(And(x >= 0, y < 0), x - y**4 > 0)
    _agrees_at_random_points(formula, prefix, answer, [x, y], 200, 10)
    formula = Eq(z**3 - 3*z, x) & (z > y)
    answer = qe(formula, prefix, free=[x, y])
    assert answer == Or(And(x + 2 >= 0, y - 1 < 0), x - y**3 + 3*y > 0)
    _agrees_at_random_points(formula, prefix, answer, [x, y], 200, 11)
    # a bound between two root functions
    formula = (x**2 + y**2 + z**2 < 1) & (z > x + y)
    answer = qe(formula, prefix, free=[x, y])
    assert _equivalent(answer, Or(2*x**2 + 2*x*y + 2*y**2 < 1, And(x + y < 0, x**2 + y**2 < 1)), [x, y])
    _agrees_at_random_points(formula, prefix, answer, [x, y], 200, 12)
    # the examples of the docs are described by the projection factors
    # themselves: the refinement is not needed and the answers do not change
    assert qe(Eq(a*x**2 + b*x + c, 0), [('exists', x)]) == Or(Eq(c, 0), 4*a*c - b**2 < 0,
                                                              And(Ne(b, 0), 4*a*c - b**2 <= 0))
    assert qe(x**2 + y**2 < 1, [], free=[x, y]) == (x**2 + y**2 - 1 < 0)


def test_inherited_truth_of_the_refined_cells() -> None:
    # the cells of the refinement are mapped to the cells of the coarser
    # decomposition by counting the sections of the old factors: a factor
    # which vanishes identically over the cell below has no section (the
    # refined stack over x = 0 has the double root y = 0 of x - y**2 and
    # the root of y, one section)
    from sympy_extras.polys.cad.lifting import Lifting
    from sympy_extras.polys.cad.projection import augmented_projection_sets, projection_sets
    from sympy_extras.polys.cad.qe import _inherited_truth
    projection = projection_sets([x - y**2, x*y], [x, y])
    lifting = Lifting(projection, [x, y], 'mccallum')
    levels = lifting.lift_all()
    truth = {cell.index: bool(cell.point[1] < 0) for cell in levels[1]}
    refined = augmented_projection_sets(projection, [x, y], [y - 1, x - 1])
    refined_lifting = Lifting(refined, [x, y], 'mccallum')
    cells = _inherited_truth(refined_lifting, [len(level) for level in projection], truth, 2)
    assert len(cells) > len(levels[1])
    assert all(value is bool(cell.point[1] < 0) for cell, value in cells)
    assert [cell.point for cell, _ in cells if cell.point[0] == 0] == [(0, -1), (0, 0), (0, S.Half), (0, 1), (0, 2)]


def test_solution_formulas_at_random_points() -> None:
    # random formulas with two free variables and one bound one, of
    # degree two: an equation of degree two in the bound variable with a
    # bound on it (the kind whose solution set the projection factors do
    # not describe), or a Boolean combination of sparse relations; the
    # answers are formulas in polynomial sign conditions, compared with
    # decide at random rational points, and some take the augmented
    # projection
    from sympy_extras.polys.cad import qe as qe_module
    from sympy_extras.polys.cad.cylindrical import IndexedRoot
    rng = random.Random(2024)
    gens = [x, y, z]

    def polynomial(variables: list[Symbol], degree: int, terms: int) -> Expr:
        monomials = [S.One] + list(variables)
        if degree >= 2:
            monomials += [g*h for i, g in enumerate(variables) for h in variables[i:]]
        chosen = rng.sample(monomials, rng.randint(1, min(terms, len(monomials))))
        return sum([rng.choice([-3, -2, -1, 1, 2, 3])*m for m in chosen], S.Zero)

    def relation(p: Expr) -> Boolean:
        return rng.choice([p < 0, p > 0, Eq(p, 0), p <= 0, Ne(p, 0)])

    def structured() -> Boolean:
        p = z**2 + polynomial([x, y], 2, 2)
        if rng.random() < 0.5:
            p += polynomial([x, y], 1, 2)*z
        bound = z - polynomial([x, y], rng.choice([1, 2]), 2)
        return And(Eq(p, 0), rng.choice([bound < 0, bound > 0, bound <= 0, bound >= 0]))

    def combination() -> Boolean:
        atoms = [relation(polynomial(gens, 2, 3)) for _ in range(rng.choice([1, 2, 2]))]
        formula: Boolean = atoms[0]
        for other in atoms[1:]:
            formula = rng.choice([And, Or])(formula, other)
        return formula

    refinements = 0
    original = qe_module._separating_refinement

    def counted(cad: CAD, cells: Sequence[CellTruth], k: int) -> Optional[tuple[CAD, list[CellTruth]]]:
        nonlocal refinements
        refinements += 1
        return original(cad, cells, k)

    qe_module._separating_refinement = counted
    try:
        for i in range(10):
            formula = structured() if i % 2 == 0 else combination()
            prefix: QuantifierSpec = [(rng.choice(['exists', 'forall']), z)]
            answer = qe(formula, prefix, free=[x, y])
            assert not answer.has(IndexedRoot) and not answer.has(sqrt(x)), answer
            _agrees_at_random_points(formula, prefix, answer, [x, y], 30, i)
    finally:
        qe_module._separating_refinement = original
    assert refinements > 0
