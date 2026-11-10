"""Integral bases and field discriminants: classical values (facts of the
literature) and SymPy's Round 2 (the oracle) on random polynomials."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import Poly, symbols
from sympy.polys.numberfields.basis import round_two
from sympy.testing.pytest import raises

from sympy_extras.polys.numberfields.basis import field_discriminant, integral_basis, polynomial_discriminant

x = symbols('x')


def test_quadratic_fields() -> None:
    # d = 1 mod 4: disc d, basis 1, (1 + sqrt d)/2; otherwise disc 4 d, basis 1, sqrt d
    for d in (-31, -23, -15, -11, -7, -6, -5, -3, -2, -1, 2, 3, 5, 6, 7, 10, 13, 17, 21, 29, 33):
        expected = d if d % 4 == 1 else 4*d
        assert field_discriminant([1, 0, -d]) == expected
        basis = integral_basis([1, 0, -d])
        if d % 4 == 1:
            assert basis == [[Fraction(1)], [Fraction(1, 2), Fraction(1, 2)]]
        else:
            assert basis == [[Fraction(1)], [Fraction(1), Fraction(0)]]
    # a square factor leaves the field: sqrt(12) = 2 sqrt(3), sqrt(45) = 3 sqrt(5)
    assert field_discriminant([1, 0, -12]) == 12 and field_discriminant([1, 0, -45]) == 5
    assert integral_basis([1, 0, -45]) == [[Fraction(1)], [Fraction(1, 6), Fraction(1, 2)]]


def test_known_fields() -> None:
    # pure cubic fields, Dedekind's field (2 divides the index of every ZZ[alpha]), x**3 - x - 1
    assert field_discriminant([1, 0, 0, -2]) == -108
    assert field_discriminant([1, -1, -2, -8]) == -503
    assert integral_basis([1, -1, -2, -8])[2] == [Fraction(1, 2), Fraction(1, 2), Fraction(0)]
    assert field_discriminant([1, 0, -1, -1]) == -23
    # x**3 - 10: 3 divides the index ((1 + theta + theta**2)/3 is integral)
    assert field_discriminant([1, 0, 0, -10]) == -300
    assert integral_basis([1, 0, 0, -10])[2] == [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)]
    # cyclotomic fields: disc of the p-th is (-1)**((p - 1)/2) p**(p - 2); the 8th and 12th: 256, 144
    assert field_discriminant([1, 1, 1, 1, 1]) == 125
    assert field_discriminant([1, 1, 1, 1, 1, 1, 1]) == -7**5
    assert field_discriminant([1, 0, 0, 0, 1]) == 256
    assert field_discriminant([1, 0, -1, 0, 1]) == 144
    # the maximal real subfield of the 7th: x**3 + x**2 - 2 x - 1, disc 49
    assert field_discriminant([1, 1, -2, -1]) == 49
    # QQ(sqrt 2, sqrt 3) by x**4 - 10 x**2 + 1: disc 2304
    assert field_discriminant([1, 0, -10, 0, 1]) == 2304
    # not monic, negative leading coefficient
    assert field_discriminant([4, 0, -5]) == 5 and field_discriminant([-2, 0, 1]) == 8
    assert integral_basis([4, 0, -5]) == [[Fraction(1)], [Fraction(1), Fraction(1, 2)]]
    assert polynomial_discriminant([1, 0, -5]) == 20 and polynomial_discriminant([1, 1, 1, 1, 1]) == 125
    raises(ValueError, lambda: integral_basis([1, -2, 1]))
    raises(ValueError, lambda: integral_basis([3]))


def test_against_sympy() -> None:
    rng = random.Random(97)
    checked = 0
    while checked < 30:
        n = rng.randint(2, 5)
        f = [1] + [rng.randint(-6, 6) for _ in range(n)]
        polynomial = Poly(f, x)
        if not polynomial.is_irreducible:
            continue
        ZK, dK = round_two(polynomial)
        assert field_discriminant(f) == int(dK)
        # the same lattice: the index is the product of the denominators of our echelon basis
        basis = integral_basis(f)
        index = Fraction(1)
        for element in basis:
            index *= 1/element[0]
        assert polynomial_discriminant(f) == int(dK)*index**2
        assert [len(element) for element in basis] == list(range(1, n + 1))
        checked += 1


def test_basis_elements_are_integral() -> None:
    from sympy_extras.polys.numberfields.qqbar import algebraic_roots
    # each basis element, evaluated at a root, is an algebraic integer: a monic minimal polynomial
    for f in ([1, 0, -5], [1, -1, -2, -8], [1, 0, 0, -10], [1, 0, -10, 0, 1], [1, 0, 3, 0, 9]):
        root = algebraic_roots(f)[0]
        for element in integral_basis(f):
            assert root.polynomial_value(element).minpoly[0] == 1
