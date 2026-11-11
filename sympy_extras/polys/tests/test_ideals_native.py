"""Tests of the native ideals (:mod:`sympy_extras.polys.ideals`) on core
expressions and typed domains; the SymPy face is tested in
``test_ideals.py``."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Sequence

from sympy_extras._testing import untyped
from sympy_extras.core import Integer, Rational, Symbol, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.polys.bigatti import hilbert_numerator_coefficients, minimal_generators
from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.f4 import f4
from sympy_extras.polys.ideals import Ideal, groebner_walk, normal_form, separating_forms
from sympy_extras.polys.rings import (BlockOrder, PolyRing, WeightOrder, as_order, elimination_order,
                                      grevlex, grlex, lex, monomial_divides)
from sympy_extras.settings import configure

x, y, z, w, t, d = [Symbol(name) for name in 'xyzwtd']


def _strs(polys: Sequence[object]) -> list[str]:
    return [str(p) for p in polys]


def _hilbert_function(I: Ideal[Fraction], degree: int) -> int:
    """Brute force count of the standard monomials of a degree."""
    lms = I.leading_monomials(grevlex)
    n = len(I.symbols)
    return sum(1 for e in product(range(degree + 1), repeat=n)
               if sum(e) == degree and not any(monomial_divides(m, e) for m in lms))


def test_orders() -> None:
    assert as_order('lex') is lex and as_order(grlex) is grlex
    W = WeightOrder((1, 3))
    assert str(W) == 'WeightOrder((1, 3), lex)' and W == WeightOrder((1, 3), lex) and W != WeightOrder((1, 2))
    # x**2 has weight 2, y weight 3
    assert W((2, 0)) < W((0, 1)) and not W.is_graded and WeightOrder((2, 2), lex).is_graded
    B = BlockOrder([(grevlex, 1), (grevlex, 2)])
    assert str(B) == 'BlockOrder((grevlex, 1), (grevlex, 2))'
    # any power of the first variable is above the monomials without it
    assert B((1, 0, 0)) > B((0, 5, 7))
    assert elimination_order(1, 3) == B and elimination_order(0, 3) is grevlex and elimination_order(3, 3) is grevlex
    try:
        WeightOrder((1, -1))
    except ValueError:
        pass
    else:
        raise AssertionError('negative weights accepted')
    try:
        as_order('nonsense')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown order accepted')


def test_bigatti() -> None:
    assert minimal_generators([(2, 1), (1, 0), (0, 3), (1, 3)]) == [(1, 0), (0, 3)]
    # (1 - t**2)**2 for <x**2, y**2>
    assert hilbert_numerator_coefficients([(2, 0), (0, 2)]) == [1, 0, -2, 0, 1]
    assert hilbert_numerator_coefficients([(1, 0), (2, 0)]) == [1, -1]
    assert hilbert_numerator_coefficients([]) == [1] and hilbert_numerator_coefficients([(0, 0)]) == []


def test_construction_and_membership() -> None:
    I = Ideal([x**2 + y**2 - 1, x - y], x, y, domain=QQ)
    assert I.symbols == (x, y) and I.domain == QQ and I.order == grevlex
    assert _strs(I.exprs) == ['x**2 + y**2 - 1', 'x - y']
    assert _strs(I.groebner_basis()) == ['y**2 - 1/2', 'x - y']
    assert _strs(I.groebner_basis('lex')) == ['x - y', 'y**2 - 1/2']
    assert I.contains(2*y**2 - 1) and (x**2 - y**2) in I and not I.contains(x)
    assert I.reduce(x**3) == y/2 and I.reduce(x**3, 'lex') == y/2
    assert str(I) == 'Ideal([x**2 + y**2 - 1, x - y], x, y)'
    # the free symbols, sorted by name, when no symbols are given
    assert Ideal([y - x], domain=QQ).symbols == (x, y)
    assert Ideal([], x, y, domain=QQ).is_zero() and Ideal([Integer(2)], x, domain=QQ).is_whole_ring()
    assert I == Ideal([y - x, 2*x**2 - 1], x, y, domain=QQ) and I != Ideal([x - y], x, y, domain=QQ)
    assert Ideal([x - y], x, y, domain=QQ).subset(I) and not I.subset(Ideal([x - y], x, y, domain=QQ))
    for bad in (lambda: Ideal([Integer(1)], domain=QQ), lambda: I.eliminate([z]),
                lambda: I + Ideal([z], z, domain=QQ), lambda: I.univariate(z)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('accepted')
    try:
        untyped(I.__add__)(3)
    except TypeError:
        pass
    else:
        raise AssertionError('accepted')
    # ring elements as generators
    R = PolyRing(QQ, ['a', 'b'])
    a, b = R.gens
    assert _strs(Ideal([a*a - b], x, y, domain=QQ).exprs) == ['x**2 - y']
    # the same name twice: the ring keeps them apart
    x2 = Symbol('x', positive=True)
    J = Ideal([x - x2], x, x2, domain=QQ)
    assert J.ring().names == ['x', 'x_1'] and J.contains(x - x2) and not J.contains(x)


def test_domains() -> None:
    # GF(7): x**3 + 2*x + 1 is irreducible (no root), and y**2 = x
    F = Ideal([x**3 + 2*x + 1, y**2 - x], x, y, domain=GF(7))
    assert F.vector_space_dimension() == 6 and F.is_radical()
    assert str(F.univariate(y).to_expr()) == 'y**6 + 2*y**2 + 1'
    # QQ(sqrt(2)): x = y/sqrt(2), y**2 = 2
    K = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], sqrt(Integer(2)))
    A = Ideal([sqrt(Integer(2))*x - y, y**2 - 2], x, y, domain=K)
    assert A.vector_space_dimension() == 2 and A.contains(x**2 - 1)
    assert A.univariate(x).to_expr() == x**2 - 1
    # over QQ(sqrt(2)) the zeros (1, sqrt 2), (-1, -sqrt 2) are rational points: not maximal
    assert A.is_radical() and not A.is_maximal()
    # a ring which is not a field is refused
    from sympy_extras.polys.domains import ZZ
    try:
        Ideal([x], x, domain=ZZ)
    except ValueError:
        pass
    else:
        raise AssertionError('an ideal over ZZ')


def test_elimination_intersection_quotient() -> None:
    I = Ideal([x - t**2, y - t**3], t, x, y, domain=QQ)
    assert str(I.eliminate([t])) == 'Ideal([x**3 - y**2], x, y)' and I.eliminate([]) is I
    # x*y generates <x> ∩ <y>
    assert Ideal([x], x, y, domain=QQ).intersect(Ideal([y], x, y, domain=QQ)) == Ideal([x*y], x, y, domain=QQ)
    assert Ideal([x**2, y], x, y, domain=QQ).intersect(Ideal([x], x, y, domain=QQ)) == \
        Ideal([x**2, x*y], x, y, domain=QQ)
    P = Ideal([x**2*y, x*y**2], x, y, domain=QQ)
    assert P.quotient(Ideal([x*y], x, y, domain=QQ)) == Ideal([x, y], x, y, domain=QQ) == P.quotient(x*y)
    assert Ideal([x], x, y, domain=QQ).quotient(Ideal([], x, y, domain=QQ)).is_whole_ring()
    assert Ideal([x**3*y, x**2*y**2], x, y, domain=QQ).saturate(x) == Ideal([y], x, y, domain=QQ)
    C = Ideal([x*z - y**2, x**2 - y*z], x, y, z, domain=QQ)
    cubic = Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z, domain=QQ)
    assert C.saturate(y) == cubic and C.saturate(Ideal([x, y], x, y, z, domain=QQ)) == cubic
    J = Ideal([x**2, y**3], x, y, domain=QQ)
    assert J.radical_contains(x*y) and not J.contains(x*y) and not J.radical_contains(x + 1)
    assert (Ideal([x**2], x, y, domain=QQ) + Ideal([y], x, y, domain=QQ)).exprs == [x**2, y]
    assert (Ideal([x**2], x, y, domain=QQ)*y).exprs == [x**2*y]
    assert (Ideal([y], x, y, domain=QQ)**3).exprs == [y**3] and (Ideal([y], x, y, domain=QQ)**0).is_whole_ring()


def test_dimension_hilbert() -> None:
    cases: list[tuple[list[Expr], int, int]] = [
        ([x*z - y**2, x**2 - y*z], 1, 4), ([x**2 - y*z, x*y - z**2, y**2 - x*z], 1, 3),
        ([x*y, x*z], 2, 1), ([x*y], 2, 2), ([], 3, 1), ([x**2 - 1, y**3 - y, z], 0, 6), ([x, y, z], 0, 1)]
    for gens, dim, deg in cases:
        I = Ideal(gens, x, y, z, domain=QQ)
        assert I.dimension() == dim and I.degree() == deg and I.is_zero_dimensional() == (dim == 0), gens
        # the Hilbert polynomial agrees with the Hilbert function from degree 10 on
        P = I.hilbert_polynomial(d)
        assert all(P.subs(d, Integer(k)) == _hilbert_function(I, k) for k in range(10, 13)), gens
    assert Ideal([x, x + 1], x, y, domain=QQ).dimension() == -1
    assert str(Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z, domain=QQ).hilbert_series(t)) == '(2*t + 1)/(1 - t)'
    assert Ideal([x, y], x, y, domain=QQ).hilbert_series(t) == 1
    assert Ideal([x*z - y**2, x**2 - y*z], x, y, z, domain=QQ, order='lex').dimension() == 1


def test_zero_dimensional() -> None:
    I = Ideal([x**2 - y, y**2 - 1], x, y, domain=QQ)
    assert I.standard_monomials() == [Integer(1), y, x, x*y] and I.vector_space_dimension() == 4
    assert I.univariate(x).to_expr() == x**4 - 1 and I.univariate(y).to_expr() == y**2 - 1
    M = I.multiplication_matrix(x)
    # x times the class of 1 is the class of x (third standard monomial)
    assert [row[0] for row in M] == [0, 0, 1, 0]
    assert I.is_radical() and I.radical() == I and not I.is_prime() and not I.is_maximal()
    J = Ideal([x**2, y**2 - 2*y + 1], x, y, domain=QQ)
    assert J.radical() == Ideal([x, y - 1], x, y, domain=QQ) and not J.is_radical()
    assert Ideal([x**2 - 2, y - x], x, y, domain=QQ).is_maximal()
    assert not Ideal([x**2 - 2, y**2 - 2], x, y, domain=QQ).is_maximal()
    assert Ideal([x**2 - 2, y**2 - 3], x, y, domain=QQ).is_maximal()
    assert not Ideal([x**2 - 1, y - 2], x, y, domain=QQ).is_prime()
    assert Ideal([x*y], x, y, domain=QQ).is_maximal() is False
    # the minimal polynomial vanishes at the form modulo the ideal
    K = Ideal([x**2 + y**2 - 1, x - y**2], x, y, domain=QQ)
    p = K.minimal_polynomial(x + 2*y, z)
    assert p.degree == 4 and K.contains(p.to_expr().subs(z, x + 2*y))
    try:
        Ideal([x*y], x, y, domain=QQ).standard_monomials()
    except NotImplementedError:
        pass
    else:
        raise AssertionError('positive dimension accepted')


def test_change_order_and_walk() -> None:
    F = [x**2*y - z**3, x*y*z - w**2, y**3 - x*w, x**3 - y*z*w]
    I = Ideal(F, x, y, z, w, domain=QQ)
    R = I.ring()
    for target in (lex, grlex, WeightOrder((1, 2, 3, 4), grevlex)):
        L = R.with_order(target)
        direct = f4([L.from_expr(f) for f in F], L)
        assert groebner_walk(I.groebner_basis(), R, target) == direct
        assert I.change_order(target) == direct
    # cached
    assert I.change_order(lex) is not I.change_order(lex) and I.change_order(lex) == I.groebner_basis(lex)
    # dimension zero: FGLM, and the lexicographic basis comes from the graded one
    J = Ideal([x**2 - 3*y - x + 1, y**2 - 2*x + y - 1], x, y, domain=QQ)
    L2 = J.ring(lex)
    assert J.change_order(lex) == f4([L2.from_expr(x**2 - 3*y - x + 1), L2.from_expr(y**2 - 2*x + y - 1)], L2)
    assert J._converted_basis(lex) is not None and Ideal([x*y], x, y, domain=QQ)._converted_basis(lex) is None
    # an order with no weight vector cannot be walked from
    B = Ideal([t - x**2, t - y], t, x, y, domain=QQ, order=BlockOrder([(grevlex, 1), (grevlex, 2)]))
    assert _strs(B.groebner_basis()) == ['t - y', 'x**2 - y']
    try:
        B.change_order(lex)
    except ValueError:
        pass
    else:
        raise AssertionError('walked from a block order')


def test_normal_form() -> None:
    R = PolyRing(QQ, ['x', 'y'], grevlex)
    X, Y = R.gens
    G = f4([X*X + Y*Y - R.one, X*Y - R.from_int(2)], R)
    f = (X + Y)**4
    r = normal_form(f, G)
    # f - r is in the ideal and r has no term divisible by a leading monomial
    assert not normal_form(f - r, G).terms
    assert all(not monomial_divides(g.LM, m) for g in G for m in r.terms)


def test_modular_fallback_and_settings() -> None:
    eqs = [x**2 + y**2 + z**2 - 1, x*y - z/3, x + y - 2*z]
    expected = _strs(Ideal(eqs, x, y, z, domain=QQ).groebner_basis('lex'))
    with configure(groebner_direct_time=0):
        # the modular algorithm (through the face) gives the same reduced basis
        assert _strs(Ideal(eqs, x, y, z, domain=QQ).groebner_basis('lex')) == expected
    with configure(groebner_algorithm='buchberger'):
        assert _strs(Ideal(eqs, x, y, z, domain=QQ).groebner_basis('lex')) == expected
    with configure(modular_groebner=False, groebner_direct_time=0):
        assert _strs(Ideal(eqs, x, y, z, domain=QQ).groebner_basis('lex')) == expected


def test_decompositions_through_the_face() -> None:
    I = Ideal([x**2 - y**2, x*z - y], x, y, z, domain=QQ)
    primes = I.minimal_primes()
    assert [str(P) for P in primes] == ['Ideal([x + y, z + 1], x, y, z)', 'Ideal([x - y, z - 1], x, y, z)',
                                        'Ideal([x, y], x, y, z)']
    assert all(P.domain == QQ for P in primes)
    assert Ideal([x**2*y, x*y**3], x, y, domain=QQ).radical() == Ideal([x*y], x, y, domain=QQ)
    assert [str(P) for P in Ideal([x*z, x*y], x, y, z, domain=QQ).equidimensional_parts()] == \
        ['Ideal([x], x, y, z)', 'Ideal([y, z], x, y, z)']
    pairs = Ideal([x**2*y, x*y**2], x, y, domain=QQ).primary_decomposition()
    assert [(str(Q), str(P)) for Q, P in pairs] == [('Ideal([x], x, y)', 'Ideal([x], x, y)'),
                                                    ('Ideal([y], x, y)', 'Ideal([y], x, y)'),
                                                    ('Ideal([x**2, y**2], x, y)', 'Ideal([x, y], x, y)')]
    assert [str(P) for P in Ideal([x**2, x*y, x*z], x, y, z, domain=QQ).associated_primes()] == \
        ['Ideal([x], x, y, z)', 'Ideal([x, y, z], x, y, z)']
    assert Ideal([x**2, y**2], x, y, domain=QQ).is_primary() and not Ideal([x**2, x*y], x, y, domain=QQ).is_primary()
    assert Ideal([x**2 - y**3], x, y, domain=QQ).is_prime() and not Ideal([x**2 - y**2], x, y, domain=QQ).is_prime()


def test_separating_forms() -> None:
    assert _strs(list(separating_forms([x, y], 3))) == ['x', 'x + y', 'x - y', 'x + 2*y', 'x - 2*y']
    assert list(separating_forms([x], 5)) == [x] and list(separating_forms([], 2)) == [Integer(0)]
    # the last form of the search reaches the Schwartz-Zippel height
    forms: list[Expr] = list(separating_forms([x, y, z], 4))
    assert len(forms) == 7**2 and str(forms[-1]) == 'x - 3*y - 3*z'
    assert Rational(1, 2)*x != forms[0]
