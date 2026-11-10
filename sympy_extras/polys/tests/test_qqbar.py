"""Exact arithmetic on algebraic numbers: identities which hold exactly,
and numerical values (the oracle) on random expressions."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots


def _positive_root(f: list[int]) -> Algebraic:
    (root,) = [r for r in algebraic_roots(f) if r.is_real and r > 0]
    return root


def test_square_roots() -> None:
    sqrt2 = _positive_root([1, 0, -2])
    sqrt3 = _positive_root([1, 0, -3])
    sqrt6 = _positive_root([1, 0, -6])
    assert sqrt2*sqrt2 == 2 and (sqrt2*sqrt2).is_rational
    assert sqrt2*sqrt3 == sqrt6
    s = sqrt2 + sqrt3
    assert s.minpoly == [1, 0, -10, 0, 1] and s.degree == 4 and s.is_real
    assert s*s == 5 + 2*sqrt6
    assert 1/(sqrt3 - sqrt2) == s
    assert (sqrt2 - sqrt2).is_zero and sqrt2 - sqrt2 == 0
    assert sqrt2/sqrt2 == 1
    # sqrt(3 + 2 sqrt 2) = 1 + sqrt 2 (a denesting)
    assert (1 + sqrt2)**2 == 3 + 2*sqrt2
    # (sqrt 2 + sqrt 3 + sqrt 6)**2 = 11 + 2 sqrt 12 + 2 sqrt 18 + 2 sqrt 6 ... checked through its minimal polynomial
    t = sqrt2 + sqrt3 + sqrt6
    assert t.degree == 4
    value = Algebraic.from_rational(0)
    for c in t.minpoly:
        value = value*t + c
    assert value == 0


def test_order_and_signs() -> None:
    sqrt2 = _positive_root([1, 0, -2])
    sqrt3 = _positive_root([1, 0, -3])
    assert sqrt2 < sqrt3 and sqrt3 > sqrt2 and sqrt2 <= sqrt2 and sqrt2 >= sqrt2
    assert Fraction(1414, 1000) < sqrt2 < Fraction(1415, 1000)
    assert (sqrt2 - sqrt3).sign() == -1 and (sqrt3 - sqrt2).sign() == 1 and (sqrt2 - sqrt2).sign() == 0
    # two numbers 10**-9 apart: sqrt(2) and sqrt(2) + 10**-9
    close = sqrt2 + Fraction(1, 10**9)
    assert sqrt2 < close and close != sqrt2 and (close - sqrt2) == Fraction(1, 10**9)
    # 140/99 < sqrt 2 < 99/70 (convergents: 1.41414..., 1.41428...)
    assert (sqrt2 - Fraction(99, 70)).sign() == -1 and (sqrt2 - Fraction(140, 99)).sign() == 1
    roots = algebraic_roots([1, 0, -5, 0, 5, 0, -1])     # real roots only, sorted
    assert all(r.is_real for r in roots)
    assert all(a < b for a, b in zip(roots, roots[1:]))
    i = Algebraic.imaginary_unit()
    raises(TypeError, lambda: i < sqrt2)
    raises(TypeError, lambda: i.sign())
    raises(TypeError, lambda: float(i))
    assert abs(float(sqrt2) - 2**0.5) < 1e-15


def test_complex_numbers() -> None:
    i = Algebraic.imaginary_unit()
    assert i*i == -1 and i**4 == 1 and i.conjugate() == -i and 1/i == -i
    assert not i.is_real and (i*i).is_real
    assert (1 + i)**2 == 2*i and (1 + i)**8 == 16
    z = Fraction(3, 2) - 2*i
    assert z.real_part() == Fraction(3, 2) and z.imaginary_part() == -2
    assert z*z.conjugate() == Fraction(25, 4)
    # the cube roots of unity
    one, w_low, w_high = algebraic_roots([1, 0, 0, -1])
    assert one == 1 and w_low.conjugate() == w_high and w_low != w_high
    assert w_low*w_high == 1 and w_low + w_high == -1 and w_low**3 == 1 and w_high**2 == w_low
    assert w_high.imaginary_part() > 0 > w_low.imaginary_part()
    sqrt3 = _positive_root([1, 0, -3])
    assert w_high == (-1 + sqrt3*i)/2
    # cos(2 pi/5) = (sqrt 5 - 1)/4 from the fifth roots of unity
    fifth = [r for r in algebraic_roots([1, 1, 1, 1, 1])]
    sqrt5 = _positive_root([1, 0, -5])
    assert max((r.real_part() for r in fifth), key=float) == (sqrt5 - 1)/4


def test_roots_and_vieta() -> None:
    rng = random.Random(61)
    for _ in range(12):
        n = rng.randint(2, 4)
        f = [1] + [rng.randint(-4, 4) for _ in range(n)]
        roots = algebraic_roots(f)
        # distinct roots; with a square-free f, Vieta's formulas
        assert len(set(roots)) == len(roots)
        for r in roots:
            value: Algebraic = Algebraic.from_rational(0)
            for c in f:
                value = value*r + c
            assert value == 0
        if len(roots) == n:
            total: Algebraic = Algebraic.from_rational(0)
            product: Algebraic = Algebraic.from_rational(1)
            for r in roots:
                total = total + r
                product = product*r
            assert total == -f[1]
            assert product == (-1)**n*f[n]
    # real roots first, increasing; then by real part, the lower one first
    roots = algebraic_roots([1, 0, 0, -2])
    assert roots[0].is_real and not roots[1].is_real and roots[1].conjugate() == roots[2]
    assert roots[1].imaginary_part() < 0
    assert algebraic_roots([2, -1]) == [Algebraic.from_rational(Fraction(1, 2))]
    assert algebraic_roots([5]) == []
    raises(ValueError, lambda: algebraic_roots([0, 0]))


def test_random_expressions_against_floats() -> None:
    rng = random.Random(62)
    pool: list[Algebraic] = []
    for f in ([1, 0, -2], [1, 0, -3], [1, -1, -1], [1, 0, 0, -2], [1, 0, 1], [1, 1, 1], [1, 0, -1, -1]):
        pool.extend(algebraic_roots(f))
    for _ in range(60):
        a, b = rng.choice(pool), rng.choice(pool)
        operation = rng.randrange(4)
        if operation == 0:
            c, expected = a + b, complex(a) + complex(b)
        elif operation == 1:
            c, expected = a - b, complex(a) - complex(b)
        elif operation == 2:
            c, expected = a*b, complex(a)*complex(b)
        else:
            c, expected = a/b, complex(a)/complex(b)
        assert abs(complex(c) - expected) < 1e-9
        # the minimal polynomial vanishes there and the region isolates it
        value = 0j
        for coefficient in c.minpoly:
            value = value*complex(c) + coefficient
        assert abs(value) < 1e-6*max(1, max(abs(k) for k in c.minpoly))
        box = c.enclosure(Fraction(1, 10**12))
        assert box[1] - box[0] < Fraction(1, 10**12) and box[3] - box[2] < Fraction(1, 10**12)
        assert c.is_real == (abs(expected.imag) < 1e-9)
        # consistency of the field operations
        if operation == 0:
            assert c - b == a
        elif operation == 2:
            assert c/b == a


def test_rationals_and_errors() -> None:
    half = Algebraic.from_rational(Fraction(1, 2))
    assert half.minpoly == [2, -1] and half.rational_value() == Fraction(1, 2) and half == Fraction(1, 2)
    assert half + half == 1 and half*4 == 2 and 1 - half == half and 3/half == 6 and half**-2 == 4
    assert -half == Fraction(-1, 2) and +half == half and half**0 == 1
    assert repr(half) == 'Algebraic([2, -1], 1/2)'
    assert hash(half) == hash(Algebraic.from_rational(Fraction(2, 4)))
    sqrt2 = _positive_root([1, 0, -2])
    assert repr(sqrt2) == 'Algebraic([1, 0, -2], 1.414213562)'
    assert repr(-sqrt2) == 'Algebraic([1, 0, -2], -1.414213562)'
    assert half != sqrt2 and sqrt2 != 'sqrt(2)'
    raises(ZeroDivisionError, lambda: sqrt2/(sqrt2 - sqrt2))
    raises(ValueError, lambda: sqrt2.rational_value())
    raises(ValueError, lambda: sqrt2.enclosure(Fraction(0)))
    raises(ValueError, lambda: Algebraic([1, 0, -2], None, None))
