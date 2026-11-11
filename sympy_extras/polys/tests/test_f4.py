"""Tests of the F4 algorithm. The oracle is SymPy's ``groebner``: the
reduced Gröbner basis is unique, so that the results must be equal."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import GF, QQ, sqrt
from sympy.polys.domains import Domain
from sympy.polys.groebnertools import groebner as sympy_groebner, is_groebner
from sympy.polys.rings import PolyElement, PolyRing, ring
from sympy.testing.pytest import raises

from sympy.polys.orderings import ProductOrder, grevlex

from sympy_extras.polys.domains import QQ as TQQ, ZZ as TZZ
from sympy_extras.polys.domains.finitefield import FiniteField
from sympy_extras.core import Integer as CoreInteger, sqrt as core_sqrt
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.f4 import f4, reduced_basis, row_echelon
from sympy_extras.polys.interop import gf_from_sympy, gf_to_sympy, qq_from_sympy, qq_to_sympy, sympy_modulus
from sympy_extras.polys.modulargroebner_sympy import direct_groebner
from sympy_extras.polys.rings import PolyRing as TypedRing, grevlex as typed_grevlex, lex as typed_lex
from sympy_extras.polys.interop import ring_to_sympy


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


def typed_f4(polys: list[PolyElement], R: PolyRing) -> list[PolyElement]:
    """The typed F4 on a SymPy ring, through the conversions."""
    p = sympy_modulus(R)
    if p is not None:
        T_gf, typed_gf = gf_from_sympy(polys, R, p)
        return gf_to_sympy(f4(typed_gf, T_gf), R)
    T_qq, typed_qq = qq_from_sympy(polys, R)
    return qq_to_sympy(f4(typed_qq, T_qq), R)


def test_f4_is_sympy_groebner_on_random_ideals() -> None:
    rng = random.Random(7)
    for _ in range(120):
        n = rng.randint(2, 4)
        order = rng.choice(['lex', 'grevlex', 'grlex'])
        domain = rng.choice([QQ, QQ, GF(32003), GF(7)])
        R = ring(','.join('x%d' % k for k in range(n)), domain, order)[0]
        polys = [p for p in (random_polynomial(R, rng, rng.randint(1, 3), rng.randint(1, 4))
                             for _ in range(rng.randint(1, 4))) if p]
        G = typed_f4(polys, R)
        assert G == list(sympy_groebner(polys, R)), (order, domain, polys)
        assert is_groebner(G, R) if G else True


def test_f4_on_the_classical_families() -> None:
    for equations, R in [katsura(3, QQ, 'grevlex'), katsura(4, QQ, 'grevlex'), katsura(4, GF(32003), 'lex'),
                         cyclic(4, QQ, 'grevlex'), cyclic(5, GF(32003), 'grevlex'), katsura(3, QQ, 'lex')]:
        assert typed_f4(equations, R) == list(sympy_groebner(equations, R))


def test_f4_through_the_dispatcher_and_with_block_orders() -> None:
    # Ideal and the modular algorithm reach the typed F4 through direct_groebner
    R, x, y, z = ring('x,y,z', QQ, ProductOrder((grevlex, lambda m: m[:1]), (grevlex, lambda m: m[1:])))
    polys = [x**2 - y*z, y**2 - x, z**3 - x*y]
    assert direct_groebner(polys, R) == list(sympy_groebner(polys, R))


def test_f4_over_an_algebraic_field_and_the_edge_cases() -> None:
    K = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], core_sqrt(CoreInteger(2)))
    A = TypedRing(K, ['x', 'y'], typed_lex)
    X, Y = A.gens
    root = A.ground(K.alpha)
    G = f4([X*X - A.from_int(2), X*Y - root*Y + A.one], A)
    S, x, y = ring('x,y', QQ.algebraic_field(sqrt(2)), 'lex')
    expected = list(sympy_groebner([x**2 - 2, x*y - sqrt(2)*y + 1], S))
    assert [ring_to_sympy(g).expand() for g in G] == [e.as_expr().expand() for e in expected]
    T = TypedRing(TQQ, ['x', 'y'], typed_grevlex)
    U, V = T.gens
    two = TQQ.from_int(2)
    assert f4([], T) == [] and f4([T.zero], T) == [] and f4([U, T.from_int(3)], T) == [T.one]
    assert f4([U*U - T.one, U - T.one], T) == [U - T.one]
    assert f4([(U + V).mul_ground(two)], T) == [U + V]
    assert reduced_basis([(U + V).mul_ground(two), V*V - T.one, U*V + U], T) == [V*V - T.one, U + V]  # grevlex: y**2 > x
    Z = TypedRing(TZZ, ['a'], typed_lex)
    raises(ValueError, lambda: f4([Z.gens[0]*Z.gens[0] - Z.one], Z))


def test_row_echelon() -> None:
    rows = [{0: Fraction(2), 1: Fraction(4)}, {0: Fraction(1), 2: Fraction(1)}, {1: Fraction(2), 2: Fraction(-1)}]
    echelon = row_echelon(rows, TQQ)
    # the first row minus twice the second is twice the third: rank two,
    # and the reduced form clears column 1 from the first row
    assert echelon == [{0: Fraction(1), 2: Fraction(1)}, {1: Fraction(1), 2: Fraction(-1, 2)}]
    # the generic elimination (over GF(p)) and the fraction-free one agree
    F = FiniteField(7)
    assert F.sparse_echelon([{0: 2, 1: 4}, {0: 1, 2: 1}, {1: 2, 2: 6}]) == [{0: 1, 2: 1}, {1: 1, 2: 3}]
    assert row_echelon([], TQQ) == [] and row_echelon([{}], TQQ) == []
