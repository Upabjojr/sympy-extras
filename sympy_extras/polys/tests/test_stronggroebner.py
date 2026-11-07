"""Tests of the strong Gröbner bases over the integers. The oracles are
the definition (every S- and G-polynomial of the basis reduces to zero,
which is the criterion of Kandri-Rody–Kapur and Becker–Weispfenning),
the membership of the generators and of random combinations of them,
the equality of the ideals over the rationals (F4 bases), the uniqueness
of the reduced basis, and examples worked by hand."""
from __future__ import annotations

import random

from sympy import QQ, ZZ
from sympy.polys.rings import PolyElement, PolyRing, ring
from sympy.testing.pytest import raises

from sympy_extras.polys.domains.finitefield import FiniteField
from sympy_extras.polys.rings import PolyElement as TypedElement, PolyRing as TypedRing, lex as typed_lex

from typing import Optional, Sequence, Union

from sympy_extras.polys import stronggroebner as typed
from sympy_extras.polys.interop import zz_from_sympy, zz_to_sympy
from sympy_extras.polys.modulargroebner import direct_groebner

# the tests are written in SymPy's ring syntax; the typed functions are
# called through the conversions of sympy_extras.polys.interop


def _typed(polys: Sequence[PolyElement], R: PolyRing) -> list[TypedElement[int]]:
    return zz_from_sympy(polys, R)[1]


def strong_groebner(polys: Sequence[Union[PolyElement, int]], R: PolyRing) -> list[PolyElement]:
    elements = [p if isinstance(p, PolyElement) else R(p) for p in polys]
    T, converted = zz_from_sympy(elements, R)
    return zz_to_sympy(typed.strong_groebner(converted, T), R)


def strong_reduce(f: PolyElement, basis: Sequence[PolyElement]) -> PolyElement:
    R = f.ring
    T, converted = zz_from_sympy([f, *basis], R)
    return zz_to_sympy([typed.strong_reduce(converted[0], converted[1:])], R)[0]


def g_polynomial_zz(f: PolyElement, g: PolyElement) -> Optional[PolyElement]:
    T, (a, b) = zz_from_sympy([f, g], f.ring)
    found = typed.g_polynomial_zz(a, b)
    return None if found is None else zz_to_sympy([found], f.ring)[0]


def s_polynomial_zz(f: PolyElement, g: PolyElement) -> PolyElement:
    T, (a, b) = zz_from_sympy([f, g], f.ring)
    return zz_to_sympy([typed.s_polynomial_zz(a, b)], f.ring)[0]


def is_strong_groebner(basis: Sequence[PolyElement]) -> bool:
    if not basis:
        return True
    return typed.is_strong_groebner(_typed(basis, basis[0].ring))


def f4(polys: Sequence[PolyElement], R: PolyRing) -> list[PolyElement]:
    return direct_groebner(polys, R)


def random_polynomial(R: PolyRing, rng: random.Random, degree: int, terms: int, cmax: int) -> PolyElement:
    p = R.zero
    for _ in range(terms):
        m = R.one
        for _ in range(rng.randint(0, degree)):
            m *= rng.choice(R.gens)
        p += R.domain(rng.randint(-cmax, cmax))*m
    return p


def test_examples_by_hand() -> None:
    R, x, y = ring('x,y', ZZ, 'lex')
    # <2x, 3y> contains xy = 3 xy - 2 xy, the G-polynomial of the pair
    assert strong_groebner([2*x, 3*y], R) == [x*y, 2*x, 3*y]
    assert g_polynomial_zz(2*x, 3*y) == x*y and s_polynomial_zz(2*x, 3*y) == 0
    # <2x, 3x> = <x>; <4, 2x + 1> is the whole ring (2x - 1 = x*4 - (2x + 1), then 2, then 1)
    assert strong_groebner([2*x, 3*x], R) == [x]
    assert strong_groebner([4, 2*x + 1], R) == [R.one]
    # <6, 4x> = <2x, 6>: the coefficients of x are the multiples of 2
    assert strong_groebner([6, 4*x], R) == [2*x, R(6)]
    # <2x**2 + y, 3xy + 1>, checked term by term in the docstring
    G = strong_groebner([2*x**2 + y, 3*x*y + 1], R)
    assert G == [x*y + 3*y**3 + 1, 2*x - 3*y**2, 9*y**3 + 2]
    assert strong_reduce(2*x**2 + y, G) == 0 and strong_reduce(3*x*y + 1, G) == 0
    assert strong_reduce(x, G) == x and strong_reduce(y**3, G) == y**3
    # the zero ideal and the unit ideal
    assert strong_groebner([], R) == [] and strong_groebner([R.zero], R) == []
    assert strong_groebner([R(1)], R) == [R.one] and strong_groebner([R(-5), R(3)], R) == [R.one]
    assert strong_groebner([R(-4)], R) == [R(4)]  # leading coefficients are made positive


def test_reduction_and_the_g_polynomial() -> None:
    R, x, y = ring('x,y', ZZ, 'grevlex')
    basis = [2*x, 3*y, x*y]
    assert strong_reduce(7*x*y + 5*x, basis) == x
    assert strong_reduce(-7*x, basis) == x  # the remainders are nonnegative
    assert strong_reduce(6*x + 3*y, basis) == 0
    assert g_polynomial_zz(2*x, 4*y) is None
    assert g_polynomial_zz(4*x, 6*y) in (2*x*y, -2*x*y)
    assert s_polynomial_zz(2*x**2 + y, 3*x*y + 1) == 3*y**2 - 2*x
    # a QQ ring does not type-check; an integer-valued domain other than ZZ is refused at run time
    raises(ValueError, lambda: typed.strong_groebner([], TypedRing(FiniteField(5), ['u'], typed_lex)))


def test_random_ideals_satisfy_the_definition() -> None:
    rng = random.Random(11)
    for _ in range(40):
        nvars = rng.randint(1, 3)
        order = rng.choice(['lex', 'grevlex'])
        R = ring(','.join('x%d' % k for k in range(nvars)), ZZ, order)[0]
        generators = [p for p in (random_polynomial(R, rng, 2, rng.randint(1, 3), 6)
                                  for _ in range(rng.randint(1, 3))) if p]
        G = strong_groebner(generators, R)
        assert is_strong_groebner(G)
        for g in generators:
            assert strong_reduce(g, G) == 0
        # random elements of the ideal
        for _ in range(3):
            element = R.zero
            for g in generators:
                element += random_polynomial(R, rng, 1, 2, 5)*g
            assert strong_reduce(element, G) == 0
        # the basis generates the same ideal over QQ
        if generators and G:
            Rq = ring(','.join('x%d' % k for k in range(nvars)), QQ, order)[0]
            over_q = [Rq.from_dict({m: QQ(int(c)) for m, c in p.terms()}) for p in generators]
            basis_q = [Rq.from_dict({m: QQ(int(c)) for m, c in p.terms()}) for p in G]
            assert f4(over_q, Rq) == f4(basis_q, Rq)
        # the reduced basis is unique: the basis of the basis is the basis
        assert strong_groebner(G, R) == G
