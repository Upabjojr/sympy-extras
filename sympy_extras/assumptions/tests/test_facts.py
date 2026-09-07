from __future__ import annotations

from sympy import (S, Q, Eq, Ne, And, Or, Not, Implies, Xor, ITE, Interval,
    FiniteSet, Union, Intersection, Complement, Contains, Symbol, Rational, true, false)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy.abc import x, y, z

from sympy_extras.assumptions import element, Facts
from sympy_extras.assumptions.facts import (normalize, conjuncts, to_predicates,
    to_polynomial, predicates_consistent)


def test_element() -> None:
    assert element(x, S.Integers) == Contains(x, S.Integers)
    assert element(2, S.Integers) is true
    assert element(Rational(1, 2), S.Integers) is false
    assert element(x, Interval(0, 1)) == Contains(x, Interval(0, 1))
    raises(TypeError, lambda: untyped(element)(x, 3))


def test_normalize() -> None:
    assert normalize(True) is true
    assert normalize(False) is false
    assert normalize(x > 0) == (x > 0)
    assert normalize(element(x, S.Reals)) == Contains(x, S.Reals)
    assert normalize(element(x, S.Naturals)) == Contains(x, S.Naturals)
    assert normalize(element(x, Interval(0, 1))) == And(x >= 0, x <= 1)
    assert normalize(element(x, Interval.open(0, 1))) == And(x > 0, x < 1)
    assert normalize(element(x, Interval(0, S.Infinity))) == (x >= 0)
    assert normalize(element(x, FiniteSet(1, 2))) == Or(Eq(x, 1), Eq(x, 2))
    assert normalize(element(x, Union(FiniteSet(1), Interval.open(2, 3)))) == \
        Or(Eq(x, 1), And(x > 2, x < 3))
    assert normalize(element(x, Intersection(Interval(0, 2), Interval(1, 3), evaluate=False))) == \
        And(x >= 0, x <= 2, x >= 1, x <= 3)
    assert normalize(element(x, Complement(S.Reals, FiniteSet(0)))) == Or(x < 0, x > 0)
    assert normalize(element(x, Complement(S.Integers, FiniteSet(0)))) == \
        And(Contains(x, S.Integers), Ne(x, 0))
    assert normalize(element(x, S.EmptySet)) is false
    assert normalize(element(x, S.UniversalSet)) is true
    assert normalize(Not(element(x, S.Integers))) == Not(Contains(x, S.Integers))
    assert normalize(ITE(x > 0, y > 0, z > 0)) == Or(And(x > 0, y > 0), And(x <= 0, z > 0))
    assert normalize(Implies(x > 0, element(y, Interval(0, 1)))) == \
        Implies(x > 0, And(y >= 0, y <= 1))
    assert normalize(Q.positive(x)) == Q.positive(x)


def test_conjuncts() -> None:
    assert conjuncts(true) == []
    assert conjuncts(x > 0) == [x > 0]
    assert set(conjuncts(And(x > 0, y > 0))) == {x > 0, y > 0}
    assert conjuncts(Or(x > 0, y > 0)) == [Or(x > 0, y > 0)]


def test_to_predicates() -> None:
    assert to_predicates(x > 0) == Q.positive(x)
    assert to_predicates(x < 0) == Q.negative(x)
    assert to_predicates(x >= 0) == Q.nonnegative(x)
    assert to_predicates(x <= 0) == Q.nonpositive(x)
    assert to_predicates(Eq(x, 0)) == Q.zero(x)
    assert to_predicates(Ne(x, 0)) == Not(Q.zero(x))
    assert to_predicates(x > y) == Q.positive(x - y)
    assert to_predicates(2 < x) == Q.positive(x - 2)
    assert to_predicates(Contains(x, S.Integers)) == Q.integer(x)
    assert to_predicates(Contains(x, S.Naturals)) == And(Q.integer(x), Q.positive(x))
    assert to_predicates(Contains(x, S.Naturals0)) == And(Q.integer(x), Q.nonnegative(x))
    assert to_predicates(Contains(x, S.Rationals)) == Q.rational(x)
    assert to_predicates(Contains(x, S.Reals)) == Q.real(x)
    assert to_predicates(Contains(x, S.Complexes)) == Q.complex(x)
    assert to_predicates(Q.prime(x)) == Q.prime(x)
    assert to_predicates(Or(x > 0, Contains(y, S.Integers))) == Or(Q.positive(x), Q.integer(y))
    assert to_predicates(Not(Contains(x, S.Integers))) == Not(Q.integer(x))
    assert to_predicates(true) is true
    # atoms without an equivalent predicate
    assert to_predicates(Contains(x, S.Reals**2)) is None
    assert to_predicates(Or(x > 0, Symbol('p'))) is None


def test_to_polynomial() -> None:
    assert to_polynomial(x > 0, {x}) == (x > 0)
    assert to_polynomial(x > 0, set()) is None
    assert to_polynomial(x*y > 1, {x, y}) == (x*y > 1)
    assert to_polynomial(x*y > 1, {x}) is None
    assert to_polynomial(x/2 > Rational(1, 3), {x}) == (x/2 > Rational(1, 3))
    assert to_polynomial(x**Rational(1, 2) > 1, {x}) is None
    assert to_polynomial(Contains(x, S.Reals), {x}) is true
    assert to_polynomial(Contains(x, S.Integers), {x}) is None
    assert to_polynomial(Q.positive(x), {x}) == (x > 0)
    assert to_polynomial(Q.negative(x - y), {x, y}) == (x - y < 0)
    assert to_polynomial(Q.nonnegative(x), {x}) == (x >= 0)
    assert to_polynomial(Q.nonpositive(x), {x}) == (x <= 0)
    assert to_polynomial(Q.zero(x), {x}) == Eq(x, 0)
    assert to_polynomial(Q.nonzero(x), {x}) == Ne(x, 0)
    assert to_polynomial(Q.real(x), {x}) is true
    assert to_polynomial(Q.integer(x), {x}) is None
    assert to_polynomial(Q.positive(x), set()) is None
    assert to_polynomial(Or(x > 0, Q.negative(y)), {x, y}) == Or(x > 0, y < 0)
    assert to_polynomial(Xor(x > 0, y > 0), {x, y}) == Xor(x > 0, y > 0)
    assert to_polynomial(false, set()) is false


def test_predicates_consistent() -> None:
    assert predicates_consistent(Q.positive(x) & Q.integer(x)) is True
    assert predicates_consistent(Q.positive(x) & Q.negative(x)) is False
    assert predicates_consistent(Q.positive(x) & Q.zero(x)) is False
    assert predicates_consistent(Q.integer(x) & Not(Q.rational(x))) is False
    assert predicates_consistent(Q.positive(x) & Q.negative(y)) is True
    assert predicates_consistent(true) is True


def test_facts() -> None:
    f = Facts()
    assert f.formula is true and f.predicates is true and f.polynomial is true
    assert f.real == set() and f.integer == set()

    f = Facts(x > 0)
    assert f.formula == (x > 0)
    assert f.real == {x}
    assert f.polynomial == (x > 0)
    assert Q.positive(x) in f.predicates.args and Q.real(x) in f.predicates.args

    f = Facts([x > 2, element(y, S.Integers), z**3 > x*y])
    assert f.real == {x, y, z} and f.integer == {y}
    assert f.polynomial == And(x > 2, z**3 > x*y)
    assert Q.positive(x) in f.predicates.args
    assert Q.integer(y) in f.predicates.args
    assert f.is_polynomial(x*y*z > 1)
    assert not f.is_polynomial(x*Symbol('w') > 1)

    # sign consequences of comparisons with numbers
    for assumption, predicate in [(x > 3, Q.positive(x)), (x >= 3, Q.positive(x)),
                                  (x >= 0, Q.nonnegative(x)), (x < -1, Q.negative(x)),
                                  (x <= -1, Q.negative(x)), (x <= 0, Q.nonpositive(x)),
                                  (3 < x, Q.positive(x)), (Eq(x, 3), Q.positive(x)),
                                  (Eq(x, 3), Q.integer(x)), (Eq(x, -Rational(1, 2)), Q.rational(x)),
                                  (Eq(x, -Rational(1, 2)), Q.negative(x)), (Eq(x, 0), Q.zero(x)),
                                  (x >= -1, Q.real(x)), (x <= 1, Q.real(x))]:
        assert predicate in Facts(assumption).predicates.args, (assumption, predicate)
    assert Q.positive(x) not in Facts(x > -1).predicates.args
    assert Q.negative(x) not in Facts(x < 1).predicates.args

    # sets and domains
    f = Facts(element(x, Interval(0, 1)))
    assert f.real == {x} and f.polynomial == And(x >= 0, x <= 1)
    f = Facts(element(x, S.Naturals))
    assert f.real == {x} and f.integer == {x}
    f = Facts([x*y > 0], domain=S.Reals)
    assert f.real == {x, y}
    f = Facts([], domain=S.Integers, symbols=[x, y])
    assert f.real == {x, y} and f.integer == {x, y}
    f = Facts([], domain=S.Complexes, symbols=[x])
    assert f.real == set()
    raises(TypeError, lambda: untyped(Facts)([], domain=3, symbols=[x]))

    # symbols with old style assumptions are known
    p = Symbol('p', positive=True)
    n = Symbol('n', integer=True)
    f = Facts([p*n > 1])
    assert f.real == {p, n} and f.integer == {n}
    f = Facts([], symbols=[p, n])
    assert f.real == {p, n} and f.integer == {n}

    # predicates are known through sympy's ask
    f = Facts(Q.positive(x))
    assert f.real == {x} and f.polynomial == (x > 0)
    f = Facts(Q.prime(x))
    assert f.real == {x} and f.integer == {x}

    # conjuncts which cannot be translated are dropped from the parts
    f = Facts([x > 0, Or(y > 0, Symbol('p'))])
    assert f.real == {x}
    assert f.polynomial == (x > 0)
    assert f.predicates == And(Q.positive(x), Q.real(x))

    # contradictory assumptions
    raises(ValueError, lambda: Facts(false))
    raises(ValueError, lambda: Facts(And(x > 0, false)))
    raises(TypeError, lambda: Facts(x + 1))

    assert repr(Facts(x > 0)) == "Facts(x > 0)"
    g = Facts(x > 0).with_reals([x, y])
    assert g.real == {x, y}
    assert Facts(x > 0).with_reals([x]).real == {x}
