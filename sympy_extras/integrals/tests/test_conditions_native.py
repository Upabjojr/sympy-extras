"""Tests of the native conditions (:mod:`sympy_extras.integrals.conditions`)
on the objects of the core; the integrators test the SymPy face."""
from __future__ import annotations

import random

from sympy_extras.assumptions.facts import element
from sympy_extras.core import I, Integer, Rational, Symbol, pi, sqrt
from sympy_extras.core.evalf import evalf
from sympy_extras.core.logic import false, or_, true
from sympy_extras.core.relational import Gt, Lt
from sympy_extras.functions import Abs, Log, arg, cos, exp, gamma, log, piecewise, re, sin
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.integrals.conditions import (ConditionalValue, _provably_different, _rational_relation, canonical,
                                               combine, decide, numerical_verdict, numerically_equal, plain_symbols,
                                               positive, real_form, sample_values, symbol_facts, with_symbol_facts)
from sympy_extras.sets import Integers, Reals

a, b, k, s, x, y = Symbol('a'), Symbol('b'), Symbol('k'), Symbol('s'), Symbol('x'), Symbol('y')


def test_conditional_values() -> None:
    v = ConditionalValue(1 / a, a > 0)
    assert v.scaled(Integer(2)) == ConditionalValue(2 / a, a > 0)
    assert v.add(ConditionalValue(a, a < 1)).condition == (a > 0) & (a < 1)
    assert v.with_condition(a < 3).condition == (a > 0) & (a < 3)
    assert ConditionalValue(a).as_piecewise(Integer(0)) == a
    assert v.as_piecewise(Integer(0)) == piecewise((1 / a, a > 0), (0, true))
    cases = ConditionalValue(piecewise((a, a > 1), (-a, a < 0)), or_(a > 1, a < 0))
    assert cases.cases() == [(a, a > 1), (-a, a < 0)]
    assert ConditionalValue(piecewise((a, a > 1), (-a, true))).cases() is None
    mapped = cases.mapped(lambda value, condition: 2 * value)
    assert isinstance(mapped.value, Piecewise) and mapped.cases() == [(2 * a, a > 1), (-2 * a, a < 0)]
    assert combine([v, ConditionalValue(Integer(1), a < 2)]) == ConditionalValue(1 / a + 1, (a > 0) & (a < 2))
    assert len({v, ConditionalValue(1 / a, a > 0)}) == 1


def test_decide_and_the_real_forms() -> None:
    assert decide((a > 0) & (b > 1), a > 1) == (b > 1)
    assert decide(a < 0, a > 1) is None
    assert decide(true, None) == true and decide(false, None) is None
    # |arg a| < pi/2 for a real a is a > 0
    assert real_form(Abs(arg(a)) < pi / 2, element(a, Reals)) == (a > 0)
    assert real_form(Abs(arg(a)) < 2 * pi, None) == true
    assert real_form(Abs(arg(a)) <= pi, None) == true
    # the real part in real parameters decides |arg(s/k + I)| < pi/2
    assert decide(Abs(arg(s / k + I)) < pi / 2, [s > 0, k > 0]) == true
    assert decide(Abs(arg((a + I * b)**2)) < pi, [a > 0, element(b, Reals)]) == true
    # an alternative which is true settles a disjunction
    assert decide(or_(Abs(arg(a)) <= 2 * pi, Abs(arg(a)) < 2 * pi), None) == true
    assert decide(k + 1 > 0, k > 0) == true
    assert canonical(k + 1 < 1) == (k < 0) and canonical(k + pi < 2 * k) == (k < 2 * k - pi)
    assert canonical(Lt(k, 1)) == (k < 1)


def test_positive() -> None:
    assert positive(a * Abs(b) * x**2, [a > 0, element(x, Reals)])
    assert positive(1 - a, a < 0) and not positive(a, None)
    # factor by factor: the factor -1 of -a is not positive
    assert not positive(-a, a < 0)
    p, n = Symbol('p', positive=True), Symbol('n', integer=True, positive=True)
    assert positive(p * n, None)


def test_symbol_flags_become_statements() -> None:
    n = Symbol('n', integer=True, positive=True)
    plain = plain_symbols([n, x])
    assert set(plain) == {n, x} and plain[n].name == 'n' and not plain[n].is_positive
    assert symbol_facts([n, x], plain) == [plain[n] > 0, element(plain[n], Integers)]
    assert with_symbol_facts(x > n, [n], plain) == [plain[n] > 0, element(plain[n], Integers), plain[x] > plain[n]]


def test_sample_values() -> None:
    for seed in range(3):
        point = sample_values([a, b], [a > 1, b < a], random.Random(seed))
        assert point is not None and point[a] > 1 and point[b] < point[a]
    assert sample_values([a], [a**2 < 0], random.Random(1)) is None
    assert sample_values([], None, random.Random(1)) == {}
    # flags are honoured
    p, n = Symbol('p', positive=True), Symbol('n', integer=True, positive=True)
    point = sample_values([p, n], None, random.Random(2))
    assert point is not None and point[p] > 0 and point[n] > 0
    # an irrational bound is approximated by a rational on the safe side
    point = sample_values([x], [x > -sqrt(2) / 2, x < 0], random.Random(3))
    assert point is not None and -sqrt(2) / 2 < point[x] < 0
    # a witness on the upper boundary is moved down (the branch for y <= 0
    # was checked at y = 0 only)
    point = sample_values([y], [y <= 0], random.Random(4))
    assert point is not None and point[y] < 0


def test_rational_relation() -> None:
    relation = _rational_relation(x > -sqrt(2) / 2)
    assert isinstance(relation, Gt) and str(evalf(relation.rhs, 8)) == '-0.70710678'
    assert relation.rhs < -sqrt(2) / 2
    assert _rational_relation(x < 1) == (x < 1)
    shifted = _rational_relation(x + y < pi)
    assert isinstance(shifted, Lt) and shifted.rhs > pi


def test_numerical_comparisons() -> None:
    assert numerically_equal(sin(x)**2 + cos(x)**2, Integer(1))
    assert numerically_equal(sqrt(x**2), x, x > 0) and not numerically_equal(sqrt(x**2), x, x < 0)
    assert numerically_equal(gamma(x + 1), x * gamma(x), x > 0)
    # agreement at the samples, and the structure theorem's second opinion
    assert not numerically_equal(exp(-10**6 * x**2), Integer(0), x > 1)
    assert _provably_different(exp(-10**6 * x**2), Integer(0), None)
    assert not _provably_different(exp(2 * x), exp(x)**2, None)
    # a value on a cut chosen by a rounding error is no evidence
    z = -6 + (-2 + sqrt(2))**2 + 4 * sqrt(2)
    assert numerically_equal(Log(-2 * sqrt(2) - 2 * sqrt(z)), log(2 * sqrt(2)) + I * pi)
    assert not numerically_equal(Log(-2 * sqrt(2) - 2 * sqrt(z)), log(2 * sqrt(2)) - I * pi)
    assert numerical_verdict(sqrt(a**2), a, a > 0) is True
    assert numerical_verdict(sqrt(a**2), a, a < 0) is False
    assert numerical_verdict(log(-a), log(a) + I * pi, None, a > 0) is True
    assert numerical_verdict(gamma(Integer(0)), gamma(Integer(0)) + 1) is None
    # a condition beyond the solver: values satisfying the assumptions at
    # which it evaluates to true
    assert numerical_verdict(sqrt(x) * sqrt(y), sqrt(x * y), [x > 0], re(y) > 0) is True
    assert numerical_verdict(Abs(a), a, None, Rational(1, 2) < a) is True
