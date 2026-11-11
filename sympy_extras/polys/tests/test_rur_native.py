"""Tests of the native rational univariate representation
(:mod:`sympy_extras.polys.rur` on core expressions and the ideals of
:mod:`sympy_extras.polys.ideals`), checked by substitution, against
Hermite's counts and against systems with known solutions."""
from __future__ import annotations

import random
from fractions import Fraction
from itertools import product

from sympy.testing.pytest import raises

from sympy_extras.core import Integer
from sympy_extras.core.expr import Expr
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions import sqrt
from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.hermite import count_complex_solutions, count_real_solutions
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.polyclasses import DensePoly
from sympy_extras.polys.rur import RationalUnivariateRepresentation, rur

t, x, y, z = [Symbol(name) for name in 'txyz']


def _random_polynomial(rng: random.Random, gens: list[Symbol], degree: int) -> Expr:
    total: Expr = Integer(0)
    for exponents in product(range(degree + 1), repeat=len(gens)):
        if sum(exponents) <= degree:
            c = rng.randint(-3, 3)
            if c:
                term: Expr = Integer(c)
                for g, e in zip(gens, exponents):
                    term = term*g**e
                total = total + term
    return total


def _vanishes(R: RationalUnivariateRepresentation, polys: list[Expr]) -> bool:
    """The coordinate polynomials substituted into the generators vanish
    modulo the polynomial of the representation."""
    coordinates = {s: g.to_expr() for s, g in zip(R.symbols, R.polynomials())}
    for p in polys:
        value = DensePoly.from_expr(p.subs(coordinates).expand(), R.t, QQ)
        if not value.rem(R.polynomial).is_zero:
            return False
    return True


def test_random_systems() -> None:
    rng = random.Random(5)
    checked = 0
    while checked < 15:
        gens = [x, y] if rng.random() < 0.6 else [x, y, z]
        polys = [_random_polynomial(rng, gens, 2) for _ in gens]
        I = Ideal(polys, *gens, domain=QQ)
        if I.is_whole_ring() or not I.is_zero_dimensional():
            continue
        R = rur(I, t=t)
        assert _vanishes(R, polys), polys
        # the degree is the number of distinct zeros, the multiplicities add
        # up to the dimension of the quotient algebra
        assert R.polynomial.degree == count_complex_solutions(I)
        assert R.polynomial.count_real_roots() == count_real_solutions(I)
        assert R.characteristic.degree == I.vector_space_dimension()
        assert R.denominator.gcd(R.polynomial).degree == 0
        for point in R.numerical_solutions(20):
            assert all(abs(complex(p.subs(point).evalf(20))) < 1e-10 for p in polys), (polys, point)
        checked += 1


def test_separating_form_and_solutions() -> None:
    # x and x + y each take one value at two of the four zeros
    R = rur([x**2 + y**2 - 1, x*y], x, y, t=t)
    assert R.form == x + 2*y and R.polynomial.to_expr() == t**4 - 5*t**2 + 4
    assert sorted(str(s) for s in R.solutions()) == ['{x: -1, y: 0}', '{x: 0, y: -1}', '{x: 0, y: 1}', '{x: 1, y: 0}']
    assert str(rur([x**2 + y**2 - 1, x - y], x, y, t=t).solutions()) == \
        '[{x: -sqrt(2)/2, y: -sqrt(2)/2}, {x: sqrt(2)/2, y: sqrt(2)/2}]'
    assert rur([x**2 + 1, y - x], x, y, t=t).solutions(real=True) == []


def test_multiplicities_and_coordinates() -> None:
    R = rur([(x - 1)**2*(x + 2), y - x**2], x, y, t=t)
    assert R.multiplicity(1) == 2 and R.multiplicity(-2) == 1 and R.multiplicity(5) == 0
    R = rur([x**2 + 4*y**2 - 4, x*y - 1], x, y, t=t)
    # two tangency points, (sqrt(2), sqrt(2)/2) and its opposite, of
    # multiplicity two each
    assert R.polynomial.degree == 2 and R.characteristic.degree == 4
    assert R.form == x and R.polynomial.to_expr() == t**2 - 2
    assert R.multiplicity(sqrt(Integer(2))) == 2 and R.multiplicity(-sqrt(Integer(2))) == 2
    assert R.coordinates(sqrt(Integer(2))) == [sqrt(Integer(2)), sqrt(Integer(2))/2]
    assert rur([x**2 - 2, y - x - 1], x, y, t=t).coordinates(sqrt(Integer(2))) == [sqrt(Integer(2)),
                                                                                  1 + sqrt(Integer(2))]


def test_a_dummy_variable_of_the_representation() -> None:
    # the polynomials were written in Symbol('t'), not in the Dummy: the
    # coordinates at a root were those of the polynomials themselves
    d = Dummy('t')
    R = rur([x**2 - 2, y - x**3], x, y, t=d)
    assert R.coordinates(sqrt(Integer(2))) == [sqrt(Integer(2)), 2*sqrt(Integer(2))]
    assert all(d in g.to_expr().free_symbols for g in R.polynomials())


def test_edge_cases() -> None:
    R = rur([x - 1, x - 2], x, t=t)
    assert R.polynomial.to_expr() == 1 and R.solutions() == [] and [g.rep for g in R.polynomials()] == [[]]
    R = Ideal([x**2 - 3], x, domain=QQ).rur(t)
    assert R.polynomial.rep == [Fraction(1), Fraction(0), Fraction(-3)]
    raises(NotImplementedError, lambda: rur([x*y], x, y))
    raises(NotImplementedError, lambda: rur([x**2 - Symbol('a')], x))
    raises(NotImplementedError, lambda: Ideal([x**2 - 2], x, domain=GF(7)).rur())
