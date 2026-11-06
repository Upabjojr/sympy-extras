"""Tests of the F4 algorithm. The oracle is SymPy's ``groebner``: the
reduced Gröbner basis is unique, so that the results must be equal."""
from __future__ import annotations

import random

from sympy import GF, QQ, sqrt
from sympy.polys.domains import Domain
from sympy.polys.groebnertools import groebner as sympy_groebner, is_groebner
from sympy.polys.rings import PolyElement, PolyRing, ring
from sympy.testing.pytest import raises

from sympy_extras.polys.f4 import f4, reduced_basis


def random_polynomial(R: PolyRing, rng: random.Random, degree: int, terms: int) -> PolyElement:
    p = R.zero
    for _ in range(terms):
        m = R.one
        for _ in range(rng.randint(0, degree)):
            m *= rng.choice(R.gens)
        p += R.domain(rng.randint(-5, 5)) * m
    return p


def katsura(n: int, domain: Domain, order: str) -> tuple[list[PolyElement], PolyRing]:
    R = ring(','.join('x%d' % k for k in range(n + 1)), domain, order)[0]
    xs = list(R.gens)
    equations = [xs[0] + 2*sum(xs[1:], R.zero) - 1]
    for m in range(n):
        equations.append(sum((xs[abs(i)]*xs[abs(m - i)] for i in range(-n, n + 1) if abs(m - i) <= n), R.zero)
                         - xs[m])
    return equations, R


def cyclic(n: int, domain: Domain, order: str) -> tuple[list[PolyElement], PolyRing]:
    R = ring(','.join('x%d' % k for k in range(n)), domain, order)[0]
    xs = list(R.gens)
    equations = []
    for k in range(1, n):
        total = R.zero
        for i in range(n):
            product = R.one
            for j in range(k):
                product *= xs[(i + j) % n]
            total += product
        equations.append(total)
    product = R.one
    for v in xs:
        product *= v
    equations.append(product - 1)
    return equations, R


def test_f4_is_sympy_groebner_on_random_ideals() -> None:
    rng = random.Random(7)
    for _ in range(120):
        n = rng.randint(2, 4)
        order = rng.choice(['lex', 'grevlex', 'grlex'])
        domain = rng.choice([QQ, QQ, GF(32003), GF(7)])
        R = ring(','.join('x%d' % k for k in range(n)), domain, order)[0]
        polys = [p for p in (random_polynomial(R, rng, rng.randint(1, 3), rng.randint(1, 4))
                             for _ in range(rng.randint(1, 4))) if p]
        G = f4(polys, R)
        assert G == list(sympy_groebner(polys, R)), (order, domain, polys)
        assert is_groebner(G, R) if G else True


def test_f4_on_the_classical_families() -> None:
    for equations, R in [katsura(3, QQ, 'grevlex'), katsura(4, QQ, 'grevlex'), katsura(4, GF(32003), 'lex'),
                         cyclic(4, QQ, 'grevlex'), cyclic(5, GF(32003), 'grevlex'), katsura(3, QQ, 'lex')]:
        assert f4(equations, R) == list(sympy_groebner(equations, R))


def test_f4_over_an_algebraic_field_and_the_edge_cases() -> None:
    R, x, y = ring('x,y', QQ.algebraic_field(sqrt(2)), 'lex')
    polys = [x**2 - 2, x*y - sqrt(2)*y + 1]
    assert f4(polys, R) == list(sympy_groebner(polys, R))
    R, x, y = ring('x,y', QQ, 'grevlex')
    assert f4([], R) == [] and f4([R.zero], R) == [] and f4([x, R(3)], R) == [R.one]
    assert f4([x**2 - 1, x - 1], R) == [x - 1] and f4([2*x + 2*y], R) == [x + y]
    assert reduced_basis([2*x + 2*y, y**2 - 1, x*y + x], R) == [y**2 - 1, x + y]  # grevlex: y**2 > x
    Z, a = ring('a', __import__('sympy').ZZ, 'lex')
    raises(ValueError, lambda: f4([a**2 - 1], Z))
