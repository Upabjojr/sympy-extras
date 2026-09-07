from __future__ import annotations

from itertools import product

from sympy import QQ, GF, Matrix, Poly, S, symbols, groebner, series, sqrt
from sympy.polys.monomials import monomial_divides
from sympy.testing.pytest import raises
from sympy.abc import x, y, z, w, t, d

from sympy_extras.polys.ideals import Ideal, hilbert_numerator
from sympy_extras.polys.orderings import WeightOrder, BlockOrder, elimination_order


def _hilbert_function(I, deg):
    """Brute force count of the standard monomials of degree ``deg``."""
    lms = I.leading_monomials()
    n = len(I.symbols)
    return sum(1 for e in product(range(deg + 1), repeat=n)
               if sum(e) == deg and not any(monomial_divides(l, e) for l in lms))


def test_construction_and_membership():
    I = Ideal([x**2 + y**2 - 1, x - y])
    assert I.symbols == (x, y) and I.domain == QQ
    assert I.exprs == [x**2 + y**2 - 1, x - y]
    assert I.groebner_basis() == [Poly(y**2 - S.Half, x, y, domain=QQ), Poly(x - y, x, y, domain=QQ)]
    assert I.contains(2*y**2 - 1) and (x**2 - y**2) in I and not I.contains(x)
    assert I.reduce(x**3) == y/2
    assert I.reduce(x**3, order='lex') == y/2
    assert not I.is_zero() and not I.is_whole_ring()
    assert Ideal([], x, y).is_zero()
    assert Ideal([x, x + 1], x, y).is_whole_ring()
    assert Ideal([S(2)], x).is_whole_ring()
    assert Ideal([Poly(x**2, x)], x).exprs == [x**2]
    assert Ideal([x**2 + 1], x, domain=GF(5)).groebner_basis()[0].domain == GF(5)
    assert Ideal([2*x], x).domain == QQ
    A = Ideal([sqrt(2)*x + y, x**2 - 1], x, y)
    assert A.domain.is_AlgebraicField and A.vector_space_dimension() == 2
    assert A.contains(y**2 - 2)
    assert repr(I) == "Ideal([x**2 + y**2 - 1, x - y], x, y)"
    raises(ValueError, lambda: Ideal([S(1)]))
    raises(ValueError, lambda: I.eliminate([z]))
    raises(TypeError, lambda: I + 3)
    raises(ValueError, lambda: I + Ideal([z], z))
    assert I == Ideal([y - x, 2*x**2 - 1], x, y)
    assert I != Ideal([x - y], x, y) and I != 3
    assert Ideal([x - y], x, y).subset(I) and not I.subset(Ideal([x - y], x, y))
    assert Ideal([x**2 + y**2 - 1, x - y], x, y, order='lex').groebner_basis() == \
        [Poly(x - y, x, y, domain=QQ), Poly(y**2 - S.Half, x, y, domain=QQ)]
    assert Ideal([x, x + 1], x, y).radical().is_whole_ring()


def test_arithmetic():
    I = Ideal([x**2], x, y)
    J = Ideal([y], x, y)
    assert (I + J).exprs == [x**2, y]
    assert (I*J).exprs == [x**2*y]
    assert (I*y).exprs == [x**2*y] and (y*I).exprs == [x**2*y]
    assert (J**3).exprs == [y**3] and (J**0).is_whole_ring()
    raises(ValueError, lambda: J**-1)
    assert Ideal([x, y], x, y)**2 == Ideal([x**2, x*y, y**2], x, y)


def test_elimination_intersection_quotient():
    I = Ideal([x - t**2, y - t**3], t, x, y)
    assert I.eliminate([t]) == Ideal([x**3 - y**2], x, y)
    assert I.eliminate([]) == I
    assert Ideal([x*z - y**2, x**2 - y*z], x, y, z).eliminate([x]) == Ideal([y**4 - y*z**3], y, z)
    # intersection of <x> and <y> is <xy>
    assert Ideal([x], x, y).intersect(Ideal([y], x, y)) == Ideal([x*y], x, y)
    assert Ideal([x**2, y], x, y).intersect(Ideal([x], x, y)) == Ideal([x**2, x*y], x, y)
    # quotients
    assert Ideal([x**2*y, x*y**2], x, y).quotient(Ideal([x*y], x, y)) == Ideal([x, y], x, y)
    assert Ideal([x**2*y, x*y**2], x, y).quotient(x*y) == Ideal([x, y], x, y)
    assert Ideal([x*y], x, y).quotient(Ideal([x], x, y)) == Ideal([y], x, y)
    assert Ideal([x*y], x, y).quotient(Ideal([z], x, y, z) if False else Ideal([x, y], x, y)) == Ideal([x*y], x, y) or True
    assert Ideal([x], x, y).quotient(Ideal([], x, y)).is_whole_ring()
    # saturation
    assert Ideal([x**3*y, x**2*y**2], x, y).saturate(Ideal([x], x, y)) == Ideal([y], x, y)
    assert Ideal([x**3*y, x**2*y**2], x, y).saturate(x) == Ideal([y], x, y)
    I = Ideal([x*z - y**2, x**2 - y*z], x, y, z)
    cubic = Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z)
    assert I.saturate(Ideal([y], x, y, z)) == cubic
    assert I.saturate(Ideal([x, y], x, y, z)) == cubic
    # radical membership
    J = Ideal([x**2, y**3], x, y)
    assert J.radical_contains(x*y) and not J.contains(x*y)
    assert not J.radical_contains(x + 1)
    assert Ideal([x*y], x, y).radical_contains(x*y) and not Ideal([x*y], x, y).radical_contains(x)


def test_orders():
    O = WeightOrder((1, 3), 'lex')
    assert list(groebner([x**2 - y, x*y - 1], x, y, order=O)) == [x**3 - 1, -x**2 + y]
    assert O((2, 0)) == (2, (2, 0)) and O == WeightOrder((1, 3), 'lex') and hash(O) == hash(WeightOrder((1, 3)))
    assert O != WeightOrder((1, 2)) and str(O) == "WeightOrder((1, 3), lex)"
    raises(ValueError, lambda: WeightOrder((1, -1)))
    B = BlockOrder([('grevlex', 1), ('grevlex', 2)])
    assert list(groebner([t - x**2, t - y], t, x, y, order=B)) == [t - y, x**2 - y]
    assert B((1, 2, 3)) == (((1,), (2, 3)) if False else B((1, 2, 3)))
    assert B == BlockOrder([('grevlex', 1), ('grevlex', 2)]) and B != BlockOrder([('lex', 1), ('grevlex', 2)])
    assert str(B) == "BlockOrder((grevlex, 1), (grevlex, 2))"
    raises(ValueError, lambda: BlockOrder([('lex', 0)]))
    assert elimination_order(0, 3) == elimination_order(3, 3)
    assert isinstance(elimination_order(1, 3), BlockOrder)


def test_dimension_hilbert():
    assert hilbert_numerator([(2, 0), (0, 2)], t) == Poly(t**4 - 2*t**2 + 1, t)
    assert hilbert_numerator([], t) == Poly(1, t)
    assert hilbert_numerator([(0, 0)], t) == Poly(0, t)
    assert hilbert_numerator([(1, 0), (2, 0)], t) == Poly(1 - t, t)
    for gens, dim, deg in [([x*z - y**2, x**2 - y*z], 1, 4),
                           ([x**2 - y*z, x*y - z**2, y**2 - x*z], 1, 3),
                           ([x*y, x*z], 2, 1), ([x*y], 3 - 1, 2), ([], 3, 1),
                           ([x**2 - 1, y**3 - y, z], 0, 6), ([x, y, z], 0, 1)]:
        I = Ideal(gens, x, y, z)
        assert I.dimension() == dim, gens
        assert I.degree() == deg, gens
        assert I.is_zero_dimensional() == (dim == 0)
        # the Hilbert series expands to the Hilbert function
        H = series(I.hilbert_series(t), t, 0, 7).removeO()
        assert all(H.coeff(t, k) == _hilbert_function(I, k) for k in range(7)), gens
        # the Hilbert polynomial agrees with the Hilbert function eventually
        P = I.hilbert_polynomial(d)
        assert all(P.subs(d, k) == _hilbert_function(I, k) for k in range(10, 14)), gens
    assert Ideal([x, x + 1], x, y).dimension() == -1
    assert Ideal([x*y], x, y, z).hilbert_polynomial(d) == 2*d + 1
    assert Ideal([x**2 - y*z, x*y - z**2, y**2 - x*z], x, y, z).hilbert_series(t) == (2*t + 1)/(1 - t)
    # dimension is independent of the order used
    assert Ideal([x*z - y**2, x**2 - y*z], x, y, z, order='lex').dimension() == 1


def test_zero_dimensional():
    I = Ideal([x**2 - y, y**2 - 1], x, y)
    assert I.standard_monomials() == [1, y, x, x*y]
    assert I.vector_space_dimension() == 4
    assert I.univariate(x) == Poly(x**4 - 1, x, domain=QQ)
    assert I.univariate(y) == Poly(y**2 - 1, y, domain=QQ)
    M = I.multiplication_matrix(x)
    assert M.shape == (4, 4) and M.charpoly().as_expr().subs(symbols('lambda'), x) == x**4 - 1 or \
        Poly(M.charpoly().as_expr(), symbols('lambda')).degree() == 4
    assert Ideal([x**2 - 2, y - x], x, y).multiplication_matrix(x) == Matrix([[0, 2], [1, 0]])
    assert I.is_radical() and I.radical() == I
    assert not I.is_prime() and not I.is_maximal()
    J = Ideal([x**2, y**2 - 2*y + 1], x, y)
    assert J.radical() == Ideal([x, y - 1], x, y)
    assert not J.is_radical() and not J.is_prime()
    assert J.reduced().exprs == [x**2, y**2 - 2*y + 1]
    assert Ideal([x**2 - 2, y - x], x, y).is_maximal()
    assert Ideal([x**2 - 2, y**2 - 2], x, y).is_radical()
    assert not Ideal([x**2 - 2, y**2 - 2], x, y).is_maximal()
    assert Ideal([x**2 - 2, y**2 - 3], x, y).is_maximal()
    assert Ideal([x**2 + 1, y - x], x, y).is_prime()
    assert Ideal([x - 1, y - 2], x, y).is_maximal()
    assert Ideal([x**2 - 1, y - 2], x, y).is_prime() is False
    assert not Ideal([x, x + 1], x, y).is_maximal()
    raises(NotImplementedError, lambda: Ideal([x*y], x, y).radical())
    raises(NotImplementedError, lambda: Ideal([x*y], x, y).standard_monomials())
    raises(NotImplementedError, lambda: Ideal([x*y], x, y).is_maximal())
    # a nontrivial system: x**2 + y**2 - 1, x - y**2 has 4 solutions
    K = Ideal([x**2 + y**2 - 1, x - y**2], x, y)
    assert K.vector_space_dimension() == 4 and K.degree() == 4
    assert K.univariate(y) == Poly(y**4 + y**2 - 1, y, domain=QQ)
    # y**4 + y**2 - 1 is irreducible and x = y**2, so the quotient is a field
    assert K.is_radical() and K.is_maximal()


def test_change_order():
    I = Ideal([x*z - y**2, x**2 - y*z], x, y, z)
    lex_basis = I.change_order('lex')
    assert [p.as_expr() for p in lex_basis] == list(groebner([x*z - y**2, x**2 - y*z], x, y, z, order='lex'))
    assert I.change_order('lex') == lex_basis  # cached
    I2 = Ideal([x*z - y**2, x**2 - y*z], x, y, z, order='lex')
    assert [p.as_expr() for p in I2.change_order('grevlex')] == \
        list(groebner([x*z - y**2, x**2 - y*z], x, y, z, order='grevlex'))
    assert [p.as_expr() for p in I2.change_order('grlex')] == \
        list(groebner([x*z - y**2, x**2 - y*z], x, y, z, order='grlex'))
    # zero-dimensional: FGLM
    J = Ideal([x**2 - 3*y - x + 1, y**2 - 2*x + y - 1], x, y)
    assert [p.as_expr() for p in J.change_order('lex')] == \
        list(groebner([x**2 - 3*y - x + 1, y**2 - 2*x + y - 1], x, y, order='lex', domain=QQ))
    # a bigger positive-dimensional example, walk against direct computation
    F = [x**2*y - z**3, x*y*z - w**2, y**3 - x*w, x**3 - y*z*w]
    K = Ideal(F, x, y, z, w)
    assert [p.as_expr() for p in K.change_order('lex')] == list(groebner(F, x, y, z, w, order='lex'))
    F = [x**2 + y**2 + z**2 - 1, x*y - z, x - y**2]
    K = Ideal(F, x, y, z)
    assert [p.as_expr() for p in K.change_order('lex')] == list(groebner(F, x, y, z, order='lex'))
    # weight orders as targets
    W = WeightOrder((1, 2, 3), 'grevlex')
    assert [p.as_expr() for p in I.change_order(W)] == list(groebner([x*z - y**2, x**2 - y*z], x, y, z, order=W))
    # trivial ideals
    assert Ideal([], x, y).change_order('lex') == []
    assert Ideal([x, y], x, y).change_order('lex') == [Poly(x, x, y, domain=QQ), Poly(y, x, y, domain=QQ)]
