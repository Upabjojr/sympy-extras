"""The native satisfiable, tautology and find_instance on objects of the
core, checked against hand computations (the SymPy face is tested in
test_sat.py)."""
from __future__ import annotations

from sympy_extras.core import Eq, Integer, Ne, Q, Rational, Symbol, oo
from sympy_extras.core.logic import BooleanTrue, Implies, and_, or_
from sympy_extras.functions import sin
from sympy_extras.sets import FiniteSet, Integers, Interval, Naturals, Reals, Union
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.quantifiers import Exists, ForAll
from sympy_extras.assumptions.sat import _compare, _floor, _integer_in, _without_real_solution, find_instance, \
    satisfiable, tautology

x, y, z, p, q = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('p'), Symbol('q')


def test_native_satisfiable() -> None:
    assert satisfiable(p & ~q) == {p: True, q: False}
    assert satisfiable(and_(p, ~p)) is False and satisfiable(True) == {} and satisfiable(False) is False
    f = and_(x**2 + y**2 < 1, x + y > 1)
    m = satisfiable(f)
    assert isinstance(m, dict) and set(m) == {x, y}
    assert isinstance(f.subs(list((k, v) for k, v in m.items() if not isinstance(v, bool))), BooleanTrue)
    assert satisfiable(and_(x**2 + y**2 < 1, x + y > 2)) is False
    # the theory rejects the propositional model with p true
    assert satisfiable(and_(or_(p, x > 1), or_(~p, x < 0), x > 0)) == {p: False, x: Integer(2)}
    # an equation does not make its variables real
    assert satisfiable(Eq(x**2, -1)) is None and satisfiable(Eq(x**2, -1), domain=Reals) is False
    assert satisfiable(and_(Q.positive(x), Q.negative(x))) is False
    assert satisfiable(sin(x) > 0) is None
    models = satisfiable(or_(x > 0, x < 0), all_models=True)
    assert models is not None and {m[x] for m in models} == {Integer(-1), Integer(1)}


def test_native_satisfiable_integers() -> None:
    assert satisfiable(and_(x > 1, x < 3), domain=Integers) == {x: Integer(2)}
    assert satisfiable(Eq(x**2, 2), domain=Integers) is False
    # 2 < x**2 < 5: the integers -2 and 2, between roots of x**2 - 2 and x**2 - 5
    m = satisfiable(and_(x**2 > 2, x**2 < 5), domain=Integers)
    assert isinstance(m, dict) and m[x] in (Integer(-2), Integer(2))
    m = satisfiable(and_(Eq(x**2, y), y > 3, y < 5), and_(element(x, Integers), element(y, Reals)))
    assert isinstance(m, dict) and m[x] in (Integer(-2), Integer(2)) and m[y] == 4
    assert satisfiable(and_(Eq(x**2, y), y > 2, y < 3), and_(element(x, Integers), element(y, Reals))) is False


def test_native_quantified_and_tautology() -> None:
    assert satisfiable(ForAll(x, x**2 >= 0)) == {} and satisfiable(ForAll(x, x**2 > 0)) is False
    assert satisfiable(and_(Exists(y, Eq(y**2, x)), x < 0)) is False
    assert tautology(p | ~p) is True and tautology(p | q) is False
    assert tautology(x**2 >= 0) is True and tautology(Ne(x**2, -1)) is None
    assert tautology(Implies(element(x, Naturals), x > 0)) is True


def test_native_find_instance() -> None:
    assert find_instance(and_(x**2 + y**2 < 1, x > y), [x, y]) == [{x: Integer(0), y: Rational(-1, 2)}]
    assert find_instance(Eq(x**2, 4), [x], count=5) == [{x: Integer(-2)}, {x: Integer(2)}]
    assert find_instance(Eq(x**2, 4), [x], Integers, count=5) == [{x: Integer(-2)}]
    assert find_instance(and_(x**2 > 5, x < 0), [x], Integers) == [{x: Integer(-3)}]
    assert find_instance(True, [x, y]) == [{x: Integer(0), y: Integer(0)}]
    assert find_instance(sin(x) > 0, [x]) is None
    for witness in find_instance(and_(Ne(x, 1), Ne(x, 2), x > 0, x < 3), [x], count=6) or []:
        value = witness[x]
        assert isinstance(value, Rational) and 0 < value < 3 and value not in (1, 2)


def test_native_integer_in() -> None:
    assert _integer_in(Interval(Rational(1, 3), Rational(2, 3))) is None
    assert _integer_in(Interval(-oo, 3, False, True)) == 2
    assert _integer_in(Interval(-oo, Rational(5, 2))) == 2
    assert _integer_in(Interval(Rational(3, 2), oo)) == 2
    assert _integer_in(Interval(2, 3, True, True)) is None
    assert _integer_in(Interval(-oo, oo)) == 0
    assert _integer_in(Union(FiniteSet(Rational(1, 2)), Interval(Rational(7, 2), 4))) == 4


def test_native_without_real_solution() -> None:
    # x, y, z are the roots of (t - 1)**3, all 1, so x > y has no solution
    system = and_(Eq(x**2 + y**2 + z**2, 3), Eq(x*y + y*z + z*x, 3), Eq(x*y*z, 1))
    assert _without_real_solution(and_(system, x - y > 0), [x, y, z])
    assert not _without_real_solution(and_(Eq(x**2 + y**2, -1), x > 0), [x, y])


def test_floor_and_compare_of_real_algebraic_numbers() -> None:
    from sympy_extras.core.mul import Mul
    from sympy_extras.polys.cad.indexedroot import IndexedRoot
    from sympy.testing.pytest import raises
    r2, r3 = IndexedRoot(x**2 - 2, x, 1), IndexedRoot(x**3 - 2, x, 0)
    # 3*sqrt(2) = 4.24..., -sqrt(2)*2**(1/3) = -1.78..., (2/7)*sqrt(2) = 0.40...
    assert _floor(3*r2) == 4 and _floor(-r2*r3) == -2 and _floor(Rational(2, 7)*r2) == 0
    assert _floor(Rational(-7, 2)) == -4 and _floor(Mul(Integer(2), Rational(1, 2))) == 1
    assert _compare(r2*r3, Integer(2)) == -1 and _compare(Integer(2), r2*r3) == 1
    assert _compare(2*r2, r2*r2*r2) == 0 and _compare(Rational(3, 2), r2) == 1
    assert _compare(-3*r2, Rational(-17, 4)) == 1      # -4.2426... > -4.25
    # not a real algebraic number of these forms
    raises(ValueError, lambda: _floor(sin(Integer(1))))
    raises(ValueError, lambda: _compare(IndexedRoot(x**2 + 1, x, 0), Integer(0)))
