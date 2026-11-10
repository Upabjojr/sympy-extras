"""Recognition of algebraic numbers from decimal approximations."""
from __future__ import annotations

import random
from fractions import Fraction

import mpmath
from sympy.testing.pytest import raises

from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots
from sympy_extras.polys.numberfields.recognize import recognize_algebraic


def _decimal(value: mpmath.mpf, digits: int) -> Fraction:
    return Fraction(int(mpmath.nint(value*10**digits)), 10**digits)


def test_known_numbers() -> None:
    with mpmath.workdps(60):
        sqrt2 = recognize_algebraic(_decimal(mpmath.sqrt(2), 15))
        assert sqrt2 is not None and sqrt2.minpoly == [1, 0, -2] and sqrt2 > 0
        golden = recognize_algebraic(_decimal(mpmath.phi, 15))
        assert golden is not None and golden.minpoly == [1, -1, -1]
        # sqrt(2) + 3**(1/3): degree 6
        mixed = recognize_algebraic(_decimal(mpmath.sqrt(2) + mpmath.cbrt(3), 40), digits=40)
        assert mixed is not None and mixed.minpoly == [1, 0, -6, -6, 12, -36, 1]
        # 2 cos(2 pi/7): x**3 + x**2 - 2 x - 1
        heptagon = recognize_algebraic(_decimal(2*mpmath.cos(2*mpmath.pi/7), 30), digits=30)
        assert heptagon is not None and heptagon.minpoly == [1, 1, -2, -1]
        # a primitive fifth root of unity
        z = mpmath.exp(2j*mpmath.pi/5)
        fifth = recognize_algebraic(_decimal(z.real, 30), _decimal(z.imag, 30), digits=30)
        assert fifth is not None and fifth.minpoly == [1, 1, 1, 1, 1] and fifth**5 == 1
        # transcendental numbers are not recognised
        for value in (mpmath.pi, mpmath.e, mpmath.log(2), mpmath.zeta(3), mpmath.euler):
            assert recognize_algebraic(_decimal(value, 30), digits=30) is None
    assert recognize_algebraic(Fraction(3, 7)) == Fraction(3, 7)
    assert recognize_algebraic(Fraction(0), Fraction(1)) == Algebraic.imaginary_unit()
    raises(ValueError, lambda: recognize_algebraic(Fraction(1), digits=0))
    raises(ValueError, lambda: recognize_algebraic(Fraction(1), max_degree=0))


def test_roots_of_random_polynomials_are_recognised() -> None:
    rng = random.Random(71)
    recognised = 0
    total = 0
    for _ in range(25):
        f = [1] + [rng.randint(-5, 5) for _ in range(rng.randint(2, 4))]
        for root in algebraic_roots(f):
            box = root.enclosure(Fraction(1, 10**45))
            re = Fraction(round((box[0] + box[1])/2*10**40), 10**40)
            im = Fraction(round((box[2] + box[3])/2*10**40), 10**40)
            total += 1
            found = recognize_algebraic(re, im, digits=40, max_degree=4)
            assert found is not None
            assert found == root
            recognised += 1
    assert recognised == total > 50


def test_random_numbers_are_not_recognised() -> None:
    rng = random.Random(72)
    for _ in range(60):
        value = Fraction(rng.randrange(10**30), 10**30) + rng.randint(-3, 3)
        assert recognize_algebraic(value, digits=30, max_degree=5) is None
