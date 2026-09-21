"""Tests of the radical, the equidimensional parts and the minimal primes of
ideals of positive dimension. The ideals are built from primes which are
known to be prime (irreducible hypersurfaces, linear varieties, kernels of
parametrizations, maximal ideals), so that the answers are known without
the algorithm under test."""
from __future__ import annotations

from sympy import symbols
from sympy.abc import t, x, y, z
from sympy.testing.pytest import raises

from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.idealdecomposition import equidimensional_parts, minimal_primes, radical


def _intersection(ideals: list[Ideal]) -> Ideal:
    result = ideals[0]
    for ideal in ideals[1:]:
        result = result.intersect(ideal)
    return result.reduced()


def _same(found: list[Ideal], expected: list[Ideal]) -> bool:
    return len(found) == len(expected) and all(any(f == e for e in expected) for f in found)


def _twisted_cubic() -> Ideal:
    return Ideal([x - t, y - t**2, z - t**3], t, x, y, z).eliminate([t])


def test_radical_in_positive_dimension() -> None:
    assert radical(Ideal([x**2*y, x*y**3], x, y)) == Ideal([x*y], x, y)
    assert Ideal([x**2, x*y], x, y).radical() == Ideal([x], x, y)         # an embedded point
    cone = x**2 + y**2 - z**2
    assert Ideal([cone**2, (x - y)**3*cone], x, y, z).radical() == Ideal([cone], x, y, z)
    cubic, line, point = _twisted_cubic(), Ideal([x - 1, y + z], x, y, z), Ideal([x, y - 2, z**2 - 2], x, y, z)
    assert radical(cubic**2 * line * point**2) == _intersection([cubic, line, point])
    # the z axis meets the cubic at the origin, where the product is not reduced
    axis = Ideal([x, y], x, y, z)
    assert (cubic * axis).is_radical() is False and cubic.intersect(axis).is_radical() is True
    # the trivial ideals, and the dimension zero through the chains too
    assert radical(Ideal([], x, y)).is_zero() and radical(Ideal([x, x + 1], x, y)).is_whole_ring()
    J = Ideal([x**2, y**2 - 2*y + 1], x, y)
    assert radical(J) == J.radical() == Ideal([x, y - 1], x, y)
    # every generator of a radical belongs to it by Rabinowitsch's test,
    # which does not know about chains
    I = Ideal([x**2*z - y**2*z, x**3*y - x*y**3, z**2*(x - y)], x, y, z)
    R = I.radical()
    assert I.subset(R) and all(I.radical_contains(g) for g in R.exprs) and R.radical() == R


def test_equidimensional_parts() -> None:
    assert equidimensional_parts(Ideal([x*z, x*y], x, y, z)) == [Ideal([x], x, y, z), Ideal([y, z], x, y, z)]
    plane, line, point = Ideal([x], x, y, z), Ideal([y - 1, z - x], x, y, z), Ideal([x - 1, y, z**2 + 1], x, y, z)
    assert (plane**2 * line * point).equidimensional_parts() == [plane, line, point]
    # a line inside the plane is not a component: the part of dimension
    # one is cleared of what lies in the plane
    inside = Ideal([x, y], x, y, z)
    assert (plane * inside * line).equidimensional_parts() == [plane, line]
    assert Ideal([x, x + 1], x, y).equidimensional_parts() == []
    assert [p.dimension() for p in Ideal([], x, y).equidimensional_parts()] == [2]
    assert Ideal([x*z, x*y], x, y, z).height() == 1 and point.height() == 3
    raises(ValueError, lambda: Ideal([x, x + 1], x, y).height())


def test_minimal_primes() -> None:
    # three lines through the origin, two of them conjugate, and the z axis
    I = Ideal([x*z - y**2, x**2 - y*z], x, y, z)
    expected = [Ideal([x, y], x, y, z), Ideal([x - z, y - z], x, y, z), Ideal([x + y + z, y**2 + y*z + z**2], x, y, z)]
    assert _same(minimal_primes(I), expected)
    # irreducible over the rationals, not over the reals
    assert minimal_primes(Ideal([x**2 - 2*y**2], x, y)) == [Ideal([x**2 - 2*y**2], x, y)]
    assert _same(Ideal([x**4 - y**4, z**2 - x**2], x, y, z).minimal_primes(),
                 [Ideal([x - s*z, y - r*z], x, y, z) for s in (1, -1) for r in (1, -1)]
                 + [Ideal([x - s*z, y**2 + z**2], x, y, z) for s in (1, -1)])
    # the first variable alone does not separate the zeros over Q(z): the
    # form x + y does not either (it is 0 twice), x + 2*y does
    four = Ideal([x**2 - z, y**2 - z], x, y, z)
    assert _same(four.minimal_primes(), [Ideal([x - y, y**2 - z], x, y, z), Ideal([x + y, y**2 - z], x, y, z)])
    cubic, line, point = _twisted_cubic(), Ideal([x - 1, y + z], x, y, z), Ideal([x, y - 2, z**2 - 2], x, y, z)
    assert _same(minimal_primes(cubic**2 * line * point**2), [cubic, line, point])
    # an embedded component is not a minimal prime
    assert Ideal([x**2, x*y], x, y).minimal_primes() == [Ideal([x], x, y)]
    assert Ideal([x, x + 1], x, y).minimal_primes() == []


def test_is_prime_in_positive_dimension() -> None:
    assert _twisted_cubic().is_prime() is True
    assert Ideal([x**2 - y**3], x, y).is_prime() is True
    assert Ideal([x**2 - y**2], x, y).is_prime() is False
    assert Ideal([x**2], x, y).is_prime() is False                        # not radical
    assert Ideal([x*z - y**2, x**2 - y*z], x, y, z).is_prime() is False
    assert Ideal([], x, y).is_prime() is True


def test_maximal_ideals_and_separating_forms() -> None:
    # the bug: is_maximal tried five fixed linear forms, x + y, x + 2*y,
    # x + 3*y, x - 2*y, and raised NotImplementedError when none separated
    # the zeros: each of them takes the same value at the origin and at
    # one of the other four points
    points = [(0, 0), (1, -1), (2, -1), (3, -1), (2, 1)]
    five = _intersection([Ideal([x - p, y - q], x, y) for p, q in points])
    assert five.vector_space_dimension() == 5 and five.is_maximal() is False
    assert _same(five.minimal_primes(), [Ideal([x - p, y - q], x, y) for p, q in points])
    a, b, c = symbols('a b c')
    I = Ideal([a**2 - 2, b**2 - 2, c**2 - 3], a, b, c)
    assert I.is_maximal() is False and len(I.minimal_primes()) == 2
    assert Ideal([a**2 - 2, b**2 - 3, c**2 - 5], a, b, c).is_maximal() is True
    # an ideal of positive dimension is not maximal (it raised)
    assert Ideal([x*y], x, y).is_maximal() is False


def test_other_coefficients_are_refused() -> None:
    from sympy import sqrt
    raises(NotImplementedError, lambda: Ideal([(x - sqrt(2)*y)**2], x, y).radical())
