"""``PowerSet`` and ``SetOfSets`` (:mod:`sympy_extras.sets.powerset`), and
the rewrites ``Piecewise`` <-> ``ITE`` and ``PowerSet`` <-> ``FiniteSet``
(:mod:`sympy_extras.functions._rewrites_misc`), checked against SymPy 1.14
through the interop (the same printed result) on a batch of cases, against
the subsets counted by hand, and by the truth values of the piecewise
functions at sample points (an oracle independent of SymPy)."""
from __future__ import annotations

from itertools import combinations

import sympy as s
from sympy.logic.boolalg import ITE as SymPyITE

from sympy_extras.core import Basic, Expr, Rational, Symbol
from sympy_extras.core.logic import ITE, false, true
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.sets import (Contains, EmptySet, FiniteSet, PowerSet, Reals, Set, SetOfSets, UniversalSet,
    finite_set, interval, set_of_sets)
from sympy.testing.pytest import raises

x, y, z, a, b, p = (Symbol(n) for n in 'xyzabp')
sx, sy, sz, sa, sb = s.symbols('x y z a b')
SE = s.S.EmptySet


def _set(e: s.Basic) -> Set:
    n = from_sympy(e)
    assert isinstance(n, Set), e
    return n


def test_powerset_matches_sympy() -> None:
    bases = [s.FiniteSet(1, 2, 3), SE, s.FiniteSet(1), s.FiniteSet(sx, sy), s.FiniteSet(s.FiniteSet(1), s.FiniteSet(2)),
             s.Interval(0, 1), s.FiniteSet((1, 2), (3, 4)), s.S.Reals, s.PowerSet(SE)]
    elements = [SE, s.FiniteSet(1), s.FiniteSet(1, 2), s.FiniteSet(4), s.FiniteSet(sx), s.Interval(0, s.Rational(1, 2)),
                s.FiniteSet(s.FiniteSet(1)), s.Integer(1)]
    for base in bases:
        theirs = s.PowerSet(base)
        ours = from_sympy(theirs)
        assert isinstance(ours, PowerSet) and to_sympy(ours) == theirs and str(ours) == str(theirs)
        assert str(to_sympy(Basic.rewrite(ours, FiniteSet))) == str(theirs.rewrite(s.FiniteSet)), base
        assert str(to_sympy(_set(base).powerset())) == str(base.powerset()), base
        if base.is_FiniteSet:
            assert len(ours) == len(theirs) and [to_sympy(e) for e in ours] == list(theirs)
        for el in elements:
            if base.has(s.Tuple) and el == s.FiniteSet(sx):
                # the PointSet of the core: a symbol is not a point (SymPy
                # leaves Eq(x, (1, 2)) unevaluated), so {x} is no subset
                continue
            native = from_sympy(el)
            assert isinstance(native, (Expr, Set))
            assert str(to_sympy(ours.contains(native))) == str(theirs.contains(el)), (base, el)
        others = [s.PowerSet(s.FiniteSet(1, 2, 3, 4)), s.PowerSet(s.FiniteSet(1)), s.PowerSet(s.Interval(0, 2)),
                  s.PowerSet(SE)] + ([theirs.rewrite(s.FiniteSet)] if base.is_FiniteSet else [])
        for other in others:
            assert ours.is_subset(_set(other)) == theirs.is_subset(other), (theirs, other)
            assert _set(other).is_subset(ours) == other.is_subset(theirs), (other, theirs)


def test_finite_sets_of_sets_match_sympy() -> None:
    cases = [s.FiniteSet(SE, s.FiniteSet(1)), s.FiniteSet(SE, s.FiniteSet(1), s.FiniteSet(2), s.FiniteSet(1, 2)),
             s.FiniteSet(s.FiniteSet(1), s.FiniteSet(2)), s.FiniteSet(SE),
             s.FiniteSet(s.FiniteSet(1, 2, 3), s.FiniteSet(1), SE, s.FiniteSet(2)), s.FiniteSet(1, 2),
             s.FiniteSet(SE, s.Interval(0, 1)),
             s.FiniteSet(s.FiniteSet(1), s.FiniteSet(2), s.FiniteSet(1, 2), s.FiniteSet(3))]
    for e in cases:
        ours = _set(e)
        assert str(ours) == str(e) and to_sympy(ours) == e
        assert str(to_sympy(Basic.rewrite(ours, PowerSet))) == str(e.rewrite(s.PowerSet)), e
    assert isinstance(_set(cases[0]), SetOfSets)


def test_powerset_counted_by_hand() -> None:
    for n in range(5):
        base = finite_set(*range(n))
        subsets = base.powerset()
        assert len(PowerSet(base)) == 2**n
        expected = [finite_set(*c) for k in range(n + 1) for c in combinations(range(n), k)]
        assert sorted(map(str, PowerSet(base))) == sorted(map(str, expected))
        if n:
            assert isinstance(subsets, SetOfSets) and len(subsets) == 2**n
            assert all(subsets.contains(e) == true for e in expected)
            assert Basic.rewrite(subsets, PowerSet) == PowerSet(base)
    assert EmptySet.powerset() == set_of_sets(EmptySet)
    assert interval(0, 1).powerset() == PowerSet(interval(0, 1))


def test_sympy_bug_fixed_finite_set_of_sets_is_not_a_power_set() -> None:
    # SymPy 1.14 rewrites {EmptySet, {1}, {2}, {3}} as PowerSet({1}): four
    # elements, but PowerSet({1}) has two
    e = set_of_sets(EmptySet, finite_set(1), finite_set(2), finite_set(3))
    assert Basic.rewrite(e, PowerSet) == e


def test_membership_of_sets() -> None:
    q = PowerSet(finite_set(1, 2))
    assert q.contains(finite_set(1)) == true and q.contains(finite_set(3)) == false
    assert q.contains(x) == Contains(x, q) and q.contains(finite_set(x)) == Contains(finite_set(x), q)
    assert interval(0, 1).contains(finite_set(1)) == false and Reals.contains(EmptySet) == false
    assert UniversalSet.contains(finite_set(1)) == true and EmptySet.contains(EmptySet) == false
    s_ = set_of_sets(EmptySet, finite_set(1))
    assert s_.contains(EmptySet) == true and s_.contains(finite_set(2)) == false and s_.contains(x) == false
    assert (s_ | finite_set(5)).contains(EmptySet) == true
    assert str(Contains(finite_set(x), q).subs(x, 1)) == 'True'
    raises(ValueError, lambda: q._evaluated((x,)))


def test_piecewise_and_ite_rewrites_match_sympy() -> None:
    pieces = [s.Piecewise((sx, sx > 0), (sy, True)), s.Piecewise((sa, sx > 0), (sb, sx <= 0)),
              s.Piecewise((sa, sx > 0), (sb, sx < 0), (sz, True)), s.Piecewise((sa, sx > 0), (sb, sx >= 0), (sy, True)),
              s.Piecewise((sa, s.Eq(sx, 0)), (sb, True)), s.Piecewise((sa, (sx < 0) | (sx > 1)), (sb, (sx >= 0) & (sx <= 1))),
              s.Piecewise((sa, sx > sy), (sb, True)), s.Piecewise((sa, sx**2 > 1), (sb, sx**2 <= 1)),
              s.Piecewise((sa, sx > 1), (sb, sx < 2)), s.Piecewise((sa, (sx > 1) & (sy > 0)), (sb, True)),
              s.Piecewise((sa, sx < -1), (sb, (sx >= -1) & (sx <= 1)), (sz, sx > 1)), s.Piecewise((sa, 2*sx > 4), (sb, True))]
    for e in pieces:
        theirs = e.rewrite(SymPyITE)
        ours = Basic.rewrite(from_sympy(e), ITE)
        assert str(to_sympy(ours)) == str(theirs), (e, theirs, ours)
        assert str(to_sympy(Basic.rewrite(ours, Piecewise))) == str(theirs.rewrite(s.Piecewise)), e
    ites = [SymPyITE(sa, sb, sz), SymPyITE(sx > 0, sb, sz), SymPyITE(sx, SymPyITE(sy, sa, sb), sz),
            SymPyITE(sx > 0, SymPyITE(sy > 1, sa, sb), SymPyITE(sx < -1, sz, sa)), SymPyITE(s.Eq(sx, 1), sa, sb)]
    for i in ites:
        assert str(to_sympy(from_sympy(i).rewrite(Piecewise))) == str(i.rewrite(s.Piecewise)), i


def test_piecewise_rewrites_where_sympy_raises_or_errs() -> None:
    # values which are not truth values: SymPy raises TypeError, the rule
    # does not apply; conditions which do not cover the reals: SymPy raises
    # ValueError
    pw = piecewise((1, x > 0), (0, True))
    assert Basic.rewrite(pw, ITE) == pw
    gap = piecewise((a, x > 0), (b, x < -1))
    assert Basic.rewrite(gap, ITE) == gap
    # Expr.rewrite gives expressions: Basic.rewrite gives the ITE
    raises(TypeError, lambda: piecewise((a, x > 0), (b, True)).rewrite(ITE))
    # SymPy 1.14 rewrites Piecewise((a, p), (b, True)) as a (p.as_set() is
    # the universal set): the bug is not ported
    assert Basic.rewrite(piecewise((a, p), (b, True)), ITE) == ITE(p, a, b)
    # branches which are not expressions: no Piecewise of truth values
    assert ITE(x > 0, y > 0, z).rewrite(Piecewise) == ITE(x > 0, y > 0, z)
    assert ITE(x, ITE(y, a, b), z > 0).rewrite(Piecewise) == ITE(x, ITE(y, a, b), z > 0)


def test_piecewise_and_ite_agree_at_sample_points() -> None:
    # the branch chosen at each point: the same symbol
    cases = [piecewise((a, x > 0), (b, x < -1), (z, True)), piecewise((a, (x < 0) | (x > 1)), (b, True)),
             piecewise((a, x > 1), (b, x < 2)), piecewise((a, x**2 > 1), (b, True))]
    points = [Rational(n, 2) for n in range(-5, 6)]
    for pw in cases:
        it = Basic.rewrite(pw, ITE)
        assert isinstance(it, ITE), (pw, it)
        for pt in points:
            assert pw.subs(x, pt) == it.subs(x, pt), (pw, it, pt)
