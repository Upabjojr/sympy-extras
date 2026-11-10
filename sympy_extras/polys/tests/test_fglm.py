"""FGLM against the direct computation of the basis for the new order
(the reduced basis is unique)."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.f4 import f4
from sympy_extras.polys.fglm import fglm, groebner_by_fglm, is_zero_dimensional
from sympy_extras.polys.rings import MonomialOrder, PolyElement, PolyRing, grevlex, grlex, lex


def _katsura(n: int, ring: PolyRing[Fraction]) -> list[PolyElement[Fraction]]:
    """The Katsura system in ``n + 1`` unknowns ``u_0, ..., u_n``."""
    u = ring.gens

    def at(i: int) -> PolyElement[Fraction]:
        i = abs(i)
        return u[i] if i <= n else ring.zero

    system = [u[0] + sum((u[i] for i in range(1, n + 1)), ring.zero).mul_ground(Fraction(2)) - ring.one]
    for m in range(n):
        total = ring.zero
        for i in range(-n, n + 1):
            total = total + at(i)*at(m - i)
        system.append(total - u[m])
    return system


def _random_system(rng: random.Random, ring: PolyRing[int], p: int) -> list[PolyElement[int]]:
    """A zero-dimensional system: x_i**d_i + lower terms."""
    n = len(ring.names)
    system = []
    for i in range(n):
        d = rng.randint(1, 3)
        f = ring.term(tuple(d if k == i else 0 for k in range(n)), 1)
        for _ in range(rng.randint(1, 4)):
            exponents = tuple(rng.randint(0, d - 1) if k == i else rng.randint(0, 1) for k in range(n))
            if sum(exponents) < d:
                f = f + ring.term(exponents, rng.randrange(1, p))
        system.append(f)
    return system


def test_known_conversions() -> None:
    R = PolyRing(QQ, ['x', 'y'], grevlex)
    x, y = R.gens
    G = f4([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
    converted = fglm(G, R, lex)
    assert [str(g) for g in converted] == ['x + 1/2*y**3 - 5/2*y', 'y**4 - 5*y**2 + 4']
    assert all(g.ring.order == lex for g in converted)
    # back to the graded order
    assert fglm(converted, R.with_order(lex), grevlex) == G
    # the whole ring and a point
    assert [str(g) for g in fglm([R.one], R, lex)] == ['1']
    assert [str(g) for g in fglm([x - R.from_int(3), y + R.from_int(1)], R, lex)] == ['x - 3', 'y + 1']
    # positive dimension is refused
    raises(ValueError, lambda: fglm([x*y], R, lex))
    raises(ValueError, lambda: fglm([], R, lex))
    assert is_zero_dimensional([R.one], R) and not is_zero_dimensional([x*x], R)


def test_against_direct_computation_modulo_primes() -> None:
    rng = random.Random(91)
    for p in (2, 7, 101):
        for _ in range(12):
            n = rng.randint(1, 3)
            names = ['x', 'y', 'z'][:n]
            F: Domain[int] = GF(p)
            graded = PolyRing(F, names, grevlex)
            system = _random_system(rng, graded, p)
            G = f4(system, graded)
            for order in (lex, grlex):
                target = graded.with_order(order)
                direct = f4([PolyElement(dict(f.terms), target) for f in system], target)
                assert fglm(G, graded, order) == direct
                assert groebner_by_fglm([PolyElement(dict(f.terms), target) for f in system], target) == direct


def test_against_direct_computation_over_the_rationals() -> None:
    orders: list[MonomialOrder] = [lex, grlex]
    for n in (2, 3):
        names = ['u%d' % i for i in range(n + 1)]
        graded = PolyRing(QQ, names, grevlex)
        G = f4(_katsura(n, graded), graded)
        for order in orders:
            target = graded.with_order(order)
            assert fglm(G, graded, order) == f4(_katsura(n, target), target)
    # the lexicographic basis of Katsura-3 is in shape position: 2**3 solutions
    target = PolyRing(QQ, ['u0', 'u1', 'u2', 'u3'], lex)
    basis = groebner_by_fglm(_katsura(3, target), target)
    assert basis[-1].degree_in(3) == 8 and len(basis) == 4
