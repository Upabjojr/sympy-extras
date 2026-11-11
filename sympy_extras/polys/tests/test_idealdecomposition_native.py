"""Tests of the native decompositions (:mod:`sympy_extras.polys.idealdecomposition`
on the ideals of :mod:`sympy_extras.polys.ideals`, core expressions). The
ideals are built from primes known to be prime (linear varieties,
irreducible hypersurfaces, kernels of parametrizations, maximal ideals),
so that the answers are known without the algorithm under test."""
from __future__ import annotations

from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras.core.symbol import Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.idealdecomposition import (associated_primes, equidimensional_parts, independent_set,
                                                   is_primary, minimal_primes, primary_decomposition, radical)
from sympy_extras.polys.ideals import Ideal

t, x, y, z = [Symbol(name) for name in 'txyz']
V = (x, y, z)


def _q(gens: list[Expr]) -> Ideal[Fraction]:
    return Ideal(gens, *V, domain=QQ)


def _intersection(ideals: list[Ideal[Fraction]]) -> Ideal[Fraction]:
    result = ideals[0]
    for ideal in ideals[1:]:
        result = result.intersect(ideal)
    return result.reduced()


def _same(found: list[Ideal[Fraction]], expected: list[Ideal[Fraction]]) -> bool:
    return len(found) == len(expected) and all(any(f == e for e in expected) for f in found)


def _twisted_cubic() -> Ideal[Fraction]:
    return Ideal([x - t, y - t**2, z - t**3], t, x, y, z, domain=QQ).eliminate([t])


def test_radical_and_minimal_primes_of_known_primes() -> None:
    cubic, line, point = _twisted_cubic(), _q([x - 1, y + z]), _q([x, y - 2, z**2 - 2])
    I = cubic**2 * line * point**2
    assert radical(I) == _intersection([cubic, line, point])
    assert _same(minimal_primes(I), [cubic, line, point])
    assert [P.dimension() for P in minimal_primes(I)] == [1, 1, 0]
    cone = x**2 + y**2 - z**2
    assert radical(_q([cone**2, (x - y)**3*cone])) == _q([cone])
    # the trivial ideals
    assert radical(_q([])).is_zero() and radical(_q([x, x + 1])).is_whole_ring()
    assert minimal_primes(_q([x, x + 1])) == [] and equidimensional_parts(_q([x, x + 1])) == []
    # the methods of the native Ideal call the native functions
    assert _q([x**2*y, x*y**3]).radical() == _q([x*y]) and not _q([x**2*y]).is_radical()
    assert _same(_q([x**2 - y**2, x*z - y]).minimal_primes(),
                 [_q([x + y, z + 1]), _q([x - y, z - 1]), _q([x, y])])
    assert _q([x**2 - y**3]).is_prime() and not _q([x**2 - y**2]).is_prime()


def test_irreducible_over_the_rationals_only() -> None:
    # x**2 - 2 y**2 is prime over QQ, a pair of planes over QQ(sqrt(2))
    assert minimal_primes(_q([x**2 - 2*y**2])) == [_q([x**2 - 2*y**2])]
    # three lines through the origin and the z axis
    assert _same(minimal_primes(_q([x*z - y**2, x**2 - y*z])),
                 [_q([x - z, y - z]), _q([x, y]), _q([y**2 + y*z + z**2, x + y + z])])


def test_equidimensional_parts() -> None:
    plane, line, point = _q([x]), _q([y - 1, z - x]), _q([x - 1, y, z**2 + 1])
    assert (plane**2 * line * point).equidimensional_parts() == [plane, line, point]
    # a line inside the plane is not a component
    assert equidimensional_parts(plane * _q([x, y]) * line) == [plane, line]


def test_primary_decomposition_of_known_components() -> None:
    # the y axis with an embedded point at the origin
    found = primary_decomposition(_q([x**2, x*y]))
    assert [(str(Q), str(P)) for Q, P in found] == [('Ideal([x], x, y, z)', 'Ideal([x], x, y, z)'),
                                                    ('Ideal([x**2, y], x, y, z)', 'Ideal([x, y], x, y, z)')]
    assert associated_primes(_q([x*y, x*z, y*z])) == [_q([x, y]), _q([x, z]), _q([y, z])]
    # the intersection of the components is the ideal
    I = _q([x*z, y**2*z, (x - 1)**2*y])
    components = I.primary_decomposition()
    assert _intersection([Q for Q, _ in components]) == I.reduced()
    assert [P for _, P in components] == [_q([x - 1, z]), _q([x, y]), _q([y, z])]
    assert all(is_primary(Q) for Q, _ in components) and not is_primary(I)


def test_irrational_points_in_dimension_zero() -> None:
    # a double pair of conjugate points and a rational point; the factors of
    # the minimal polynomial were written in Symbol('y') instead of their
    # Dummy, which the substitution of the form missed: the component of
    # the pair came out as the whole ring and was lost
    f = x**2 - 2
    pair, point = Ideal([f, y], x, y, domain=QQ), Ideal([x - 1, y - 1], x, y, domain=QQ)
    I = (pair**2).intersect(point)
    found = primary_decomposition(I)
    assert [P for _, P in found] == [point, pair] or [P for _, P in found] == [pair, point]
    assert any(Q == pair**2 for Q, _ in found)
    # an embedded prime without rational points
    J = Ideal([f**2, f*y], x, y, domain=QQ)
    assert J.associated_primes() == [Ideal([f], x, y, domain=QQ), pair]


def test_is_primary_and_independent_sets() -> None:
    assert is_primary(Ideal([x**2, y**3], x, y, domain=QQ)) and not is_primary(Ideal([x**2, x*y], x, y, domain=QQ))
    assert Ideal([x**2, y], x, y, z, domain=QQ).is_primary() and not Ideal([x*y], x, y, domain=QQ).is_primary()
    assert independent_set(_q([x*y, x*z])) == [y, z] and independent_set(_q([x**2 - y, z])) == [y]
    assert independent_set(_q([x - 1, y, z])) == []
    raises(ValueError, lambda: independent_set(_q([x, x + 1])))


def test_other_fields_are_refused() -> None:
    I = Ideal([x*y], x, y, domain=GF(5))
    raises(NotImplementedError, lambda: radical(I))
    raises(NotImplementedError, lambda: minimal_primes(I))
    raises(NotImplementedError, lambda: primary_decomposition(I))
    raises(NotImplementedError, lambda: is_primary(I))
    # dimension zero needs no chain
    assert Ideal([x**2, y - 1], x, y, domain=GF(5)).radical() == Ideal([x, y - 1], x, y, domain=GF(5))
