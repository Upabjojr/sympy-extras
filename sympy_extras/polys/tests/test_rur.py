from __future__ import annotations

import random

from sympy import QQ, Expr, N, Poly, Symbol, symbols
from sympy.polys.polyerrors import NotInvertible
from sympy.testing.pytest import raises

from sympy_extras.polys.hermite_sympy import count_complex_solutions, count_real_solutions
from sympy_extras.polys.ideals_sympy import Ideal
from sympy_extras.polys.rur import rur, RationalUnivariateRepresentation

x, y, z, t = symbols('x y z t')


def _random_polynomial(rng: random.Random, gens: list[Symbol], degree: int) -> Expr:
    from itertools import product
    total: Expr = Poly(0, *gens).as_expr()
    for exponents in product(range(degree + 1), repeat=len(gens)):
        if sum(exponents) <= degree:
            c = rng.randint(-3, 3)
            if c:
                term: Expr = Poly(c, *gens).as_expr()
                for g, e in zip(gens, exponents):
                    term *= g**e
                total += term
    return total


def _substitution_check(R: RationalUnivariateRepresentation, polys: list[Expr]) -> bool:
    """The coordinates, as polynomials in t, substituted into the
    generators vanish modulo the polynomial of the representation."""
    coordinates: dict[Symbol, Expr] = dict(zip(R.symbols, [g.as_expr() for g in R.polynomials()]))
    return all(Poly(p.xreplace(coordinates), R.t).rem(R.polynomial).is_zero for p in polys)


def test_random_systems() -> None:
    rng = random.Random(5)
    checked = 0
    while checked < 25:
        gens = [x, y] if rng.random() < 0.6 else [x, y, z]
        polys = [_random_polynomial(rng, gens, 2) for _ in gens]
        I = Ideal(polys, *gens)
        if I.is_whole_ring() or not I.is_zero_dimensional():
            continue
        R = rur(I, t=t)
        assert _substitution_check(R, polys), polys
        # the degree is the number of distinct complex zeros, the number of
        # real roots the number of real zeros (Hermite's form)
        assert R.polynomial.degree() == count_complex_solutions(I)
        assert len(R.polynomial.real_roots()) == count_real_solutions(I)
        # the multiplicities add up to the dimension of the quotient algebra
        assert R.characteristic.degree() == I.vector_space_dimension()
        # the numerators and the denominator are coprime to the polynomial
        assert R.denominator.gcd(R.polynomial).degree() == 0
        # the zeros, numerically
        solutions = R.numerical_solutions(30)
        assert len(solutions) == R.polynomial.degree()
        for solution in solutions:
            size = max(1, *[abs(N(v)) for v in solution.values()])
            assert all(abs(N(p.xreplace(solution), 30)) < 1e-20*size**2 for p in polys), (polys, solution)
        checked += 1


def test_separating_form_is_needed() -> None:
    # the zeros (0, 1), (0, -1), (1, 0), (-1, 0): x and x + y both take
    # equal values at two of them
    R = rur([x**2 + y**2 - 1, x*y], x, y, t=t)
    assert R.form == x + 2*y
    assert R.polynomial == Poly(t**4 - 5*t**2 + 4, t, domain=QQ)
    assert sorted(str(s) for s in R.solutions()) == ['{x: -1, y: 0}', '{x: 0, y: -1}', '{x: 0, y: 1}', '{x: 1, y: 0}']
    assert R.solutions(real=True) == R.solutions()


def test_multiplicities() -> None:
    R = rur([x**2, x*y, y**2], x, y, t=t)
    assert R.polynomial == Poly(t, t, domain=QQ) and R.characteristic == Poly(t**3, t, domain=QQ)
    assert R.multiplicity(0) == 3 and R.solutions() == [{x: 0, y: 0}]
    R = rur([(x - 1)**2*(x + 2), y - x**2], x, y, t=t)
    assert R.multiplicity(1) == 2 and R.multiplicity(-2) == 1 and R.multiplicity(5) == 0
    assert R.solutions() == [{x: -2, y: 4}, {x: 1, y: 1}]


def test_edge_cases() -> None:
    # the whole ring: no zeros
    R = rur([x - 1, x - 2], x, t=t)
    assert R.polynomial == Poly(1, t, domain=QQ) and R.solutions() == [] and R.polynomials() == [Poly(0, t, domain=QQ)]
    # the variable of the representation may be one of the generators'
    # names only if it is not a variable of the system
    R = rur([x**2 - 3], x, t=t)
    assert R.polynomial == Poly(t**2 - 3, t, domain=QQ) and R.polynomials() == [Poly(t, t, domain=QQ)]
    # positive dimension and non-rational coefficients are refused
    raises(NotImplementedError, lambda: rur([x*y], x, y))
    raises(NotImplementedError, lambda: rur([x**2 - Symbol('a')], x))
    # an Ideal, and the method
    I = Ideal([x**2 - 2, y - x], x, y)
    assert rur(I).polynomial == I.rur().polynomial == Poly(Symbol('t')**2 - 2, Symbol('t'), domain=QQ)


def test_polynomials_are_coordinates() -> None:
    R = rur([x**2 - 2, y**2 - 3], x, y, t=t)
    px, py = R.polynomials()
    assert (px**2 - 2).rem(R.polynomial).is_zero and (py**2 - 3).rem(R.polynomial).is_zero
    assert px.degree() < 4 and py.degree() < 4
    # the fractions agree with the polynomials modulo f
    assert (R.numerators[0] - px*R.denominator).rem(R.polynomial).is_zero
    raises(NotInvertible, lambda: (Poly(0, t)).invert(R.polynomial))
