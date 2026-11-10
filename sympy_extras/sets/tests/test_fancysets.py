"""The named sets of numbers against SymPy's (the oracle): the memberships
of numbers and of symbols with assumptions, and the conversion both ways."""
from __future__ import annotations

import sympy
from sympy import Rational as R

from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy, from_sympy, to_sympy
from sympy_extras.sets import Complexes, Integers, Naturals, Naturals0, Rationals, Reals, Set, contains, interval

_ELEMENTS = [0, 1, 2, -1, R(1, 2), sympy.sqrt(2), sympy.pi, sympy.I, 1 + sympy.I, sympy.oo,
             sympy.Symbol('x'), sympy.Symbol('n', integer=True), sympy.Symbol('m', integer=True, positive=True),
             sympy.Symbol('k', integer=True, nonnegative=True), sympy.Symbol('q', rational=True),
             sympy.Symbol('r', real=True), sympy.Symbol('p', positive=True), sympy.Symbol('c', complex=True)]
_SETS: list[tuple[Set, sympy.Set]] = [(Naturals, sympy.S.Naturals), (Naturals0, sympy.S.Naturals0),
                                      (Integers, sympy.S.Integers), (Rationals, sympy.S.Rationals),
                                      (Complexes, sympy.S.Complexes)]


def _ours(v: object) -> Expr:
    return expr_from_sympy(sympy.sympify(v))


def test_memberships_agree_with_sympy() -> None:
    for ours, theirs in _SETS:
        for e in _ELEMENTS:
            expected = sympy.Contains(e, theirs)
            assert str(contains(_ours(e), ours)) == str(expected), (ours, e)


def test_conversion() -> None:
    for ours, theirs in _SETS:
        assert to_sympy(ours) is theirs
        assert from_sympy(theirs) is ours
        assert str(ours) == str(theirs)
    assert from_sympy(sympy.S.Reals) is Reals


def test_named_sets_in_operations() -> None:
    # the handlers of union and intersection have no rule for them: they
    # stay unevaluated, and the empty set is absorbed
    u = Integers | interval(0, 1)
    assert str(to_sympy(u)) == str(sympy.Union(sympy.S.Integers, sympy.Interval(0, 1)))
    assert Integers.is_empty is False and Integers.is_finite_set is False
    assert str(Naturals.inf) == '1' and str(Naturals0.inf) == '0' and str(Integers.sup) == 'oo'
