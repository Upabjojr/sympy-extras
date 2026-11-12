"""The native consistency of predicates with their known facts
(sympy_extras.assumptions.satask), checked against hand reasoning and
against SymPy's satask on random conjunctions (the oracle the module was
ported from)."""
from __future__ import annotations

import random

from sympy_extras.core import I, Integer, Q, Rational, Symbol, pi, sqrt
from sympy_extras.core.assume import Predicate
from sympy_extras.core.logic import Boolean, and_, not_, or_
from sympy_extras.functions import Abs
from sympy_extras.assumptions.satask import predicates_consistent

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def test_number_facts() -> None:
    assert predicates_consistent(and_(Q.positive(x), Q.integer(x)))
    assert not predicates_consistent(and_(Q.positive(x), Q.negative(x)))
    assert not predicates_consistent(and_(Q.even(x), Q.odd(x)))
    # an even prime is 2: not composite
    assert not predicates_consistent(and_(Q.even(x), Q.prime(x), Q.composite(x)))
    # even, positive and not prime: composite
    assert not predicates_consistent(and_(Q.even(x), Q.positive(x), not_(Q.prime(x)), not_(Q.composite(x))))
    assert not predicates_consistent(and_(Q.rational(x), Q.transcendental(x)))
    assert not predicates_consistent(and_(Q.real(x), not_(Q.rational(x)), not_(Q.irrational(x))))
    assert predicates_consistent(or_(Q.positive(x), Q.negative(x)))
    assert not predicates_consistent(and_(Q.nonzero(x), Q.zero(x)))
    assert not predicates_consistent(and_(Q.imaginary(x), Q.positive(x)))
    assert not predicates_consistent(and_(Q.infinite(x), Q.integer(x)))


def test_class_facts() -> None:
    # a sum of positive terms is positive, of negative terms negative
    assert not predicates_consistent(and_(Q.positive(x), Q.positive(y), Q.negative(x + y)))
    assert predicates_consistent(and_(Q.positive(x), Q.negative(y), Q.negative(x + y)))
    # a product is zero if and only if a factor is
    assert not predicates_consistent(and_(Q.zero(x*y), Q.nonzero(x), Q.nonzero(y)))
    assert not predicates_consistent(and_(Q.zero(x), Q.positive(x*y)))
    assert not predicates_consistent(and_(Q.integer(x), Q.integer(y), Q.even(x), Q.odd(x*y)))
    # a power of zero with a positive exponent is zero
    assert not predicates_consistent(and_(Q.zero(x**2), Q.positive(x)))
    assert not predicates_consistent(and_(Q.real(x), Q.negative(x**2)))
    assert not predicates_consistent(Q.negative(Abs(x)))
    assert not predicates_consistent(and_(Q.zero(Abs(x)), Q.nonzero(x)))
    # the numbers by their own assumptions
    assert not predicates_consistent(Q.negative(Integer(2)))
    assert not predicates_consistent(Q.rational(pi)) and not predicates_consistent(Q.real(I))
    assert predicates_consistent(and_(Q.positive(Rational(1, 2)), not_(Q.integer(Rational(1, 2)))))
    # a sum of an integer and a non-integer is not an integer
    assert not predicates_consistent(and_(Q.integer(x), Q.integer(x + Rational(1, 2))))


def test_relations_and_predicates_of_the_user() -> None:
    # relations are the binary predicates, opaque to the number facts
    assert predicates_consistent(and_(x > 1, Q.negative(x)))
    assert not predicates_consistent(and_(x > 1, ~(x > 1)))
    P = Predicate('mypred')
    assert predicates_consistent(and_(P(x), Q.positive(x)))
    assert not predicates_consistent(and_(P(x), not_(P(x))))


def test_agrees_with_sympy_satask() -> None:
    from sympy_extras.assumptions.ask_sympy import sympy_boolean
    from sympy_extras.assumptions.facts_sympy import predicates_consistent as sympy_consistent
    exprs = [x, y, z, x + y, x*y, x**2, x - 2, Integer(2), Integer(0), Rational(1, 2), pi, I, Abs(x), x*y*z,
             x + y + 1, y**3, 2*x, Integer(-3), sqrt(x), x**y]
    names = ['positive', 'negative', 'zero', 'real', 'integer', 'even', 'odd', 'rational', 'irrational', 'nonzero',
             'nonnegative', 'nonpositive', 'prime', 'composite', 'imaginary', 'complex', 'finite', 'infinite',
             'algebraic', 'transcendental', 'extended_real']
    rng = random.Random(5)
    inconsistent = 0
    for _ in range(150):
        literals: list[Boolean] = []
        for _ in range(rng.randint(1, 5)):
            literal: Boolean = Predicate._known(rng.choice(names))(rng.choice(exprs))
            if rng.random() < 0.3:
                literal = not_(literal)
            if rng.random() < 0.2:
                literal = or_(literal, Predicate._known(rng.choice(names))(rng.choice(exprs)))
            literals.append(literal)
        f = and_(*literals)
        native = predicates_consistent(f)
        assert native == sympy_consistent(sympy_boolean(f)), f
        inconsistent += not native
    assert inconsistent > 20
