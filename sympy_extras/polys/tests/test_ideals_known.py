"""Ideal operations against results published elsewhere.

The examples are taken (as results, not code) from Cox, Little, O'Shea,
*Ideals, Varieties, and Algorithms* (CLO), the Macaulay2 and Singular
documentation, and the standard Katsura and cyclic benchmark families whose
numbers of solutions are well known.
"""
from __future__ import annotations

from sympy import QQ, Rational, symbols, groebner, cancel
from sympy.abc import x, y, z, w, t, u, d

from sympy_extras.polys.ideals import Ideal


def katsura(m):
    """The Katsura-m system in the variables u_0, ..., u_m, with 2**m
    solutions."""
    us = symbols('u0:%d' % (m + 1))

    def U(i):
        i = abs(i)
        return us[i] if i <= m else 0

    eqs = [us[0] + 2*sum(us[1:]) - 1]
    for r in range(m):
        eqs.append(sum(U(i)*U(r - i) for i in range(-m, m + 1)) - us[r])
    return eqs, us


def cyclic(m):
    """The cyclic-m system."""
    xs = symbols('x0:%d' % m)
    eqs = []
    for r in range(1, m):
        eqs.append(sum(prod_(xs[(i + j) % m] for j in range(r)) for i in range(m)))
    p = 1
    for v in xs:
        p *= v
    eqs.append(p - 1)
    return eqs, xs


def prod_(items):
    result = 1
    for it in items:
        result *= it
    return result


def test_cox_little_oshea():
    # CLO chapter 2, section 7, example 2: the reduced grlex basis of
    # (x^3 - 2xy, x^2 y - 2y^2 + x) is {x^2, xy, y^2 - x/2}
    I = Ideal([x**3 - 2*x*y, x**2*y - 2*y**2 + x], x, y, order='grlex')
    assert [p.as_expr() for p in I.groebner_basis()] == [x**2, x*y, y**2 - x/2]
    assert [p.as_expr() for p in I.change_order('lex')] == [x - 2*y**2, y**3]
    # CLO chapter 3, section 1: the tangent surface of the twisted cubic,
    # x = t + u, y = t^2 + 2tu, z = t^3 + 3t^2 u, has the implicit equation
    # x^3 z - 3/4 x^2 y^2 - 3/2 x y z + y^3 + 1/4 z^2 = 0
    I = Ideal([x - t - u, y - t**2 - 2*t*u, z - t**3 - 3*t**2*u], t, u, x, y, z)
    E = I.eliminate([t, u])
    assert E == Ideal([x**3*z - Rational(3, 4)*x**2*y**2 - Rational(3, 2)*x*y*z + y**3 + z**2/4], x, y, z)
    # CLO chapter 4, section 4: (xz, yz) : (z) = (x, y) and
    # (x^2 y) intersect (x y^2) = (x^2 y^2)
    assert Ideal([x*z, y*z], x, y, z).quotient(z) == Ideal([x, y], x, y, z)
    assert Ideal([x**2*y], x, y).intersect(Ideal([x*y**2], x, y)) == Ideal([x**2*y**2], x, y)
    # CLO chapter 4, section 1: y belongs to the radical of (x^2 + y^2, x^2 - y^2)? no,
    # but x and y do: V = {(0, 0)} and the radical is (x, y)
    J = Ideal([x**2 + y**2, x**2 - y**2], x, y)
    assert J.radical_contains(x) and J.radical_contains(y) and not J.contains(x)
    assert J.radical() == Ideal([x, y], x, y)
    # CLO chapter 9: the three coordinate axes (xy, xz, yz) have dimension 1
    # and degree 3; the twisted cubic in P^3 has Hilbert polynomial 3d + 1
    axes = Ideal([x*y, x*z, y*z], x, y, z)
    assert axes.dimension() == 1 and axes.degree() == 3
    cubic = Ideal([x*z - y**2, y*w - z**2, x*w - y*z], x, y, z, w)
    assert cubic.dimension() == 2 and cubic.degree() == 3
    assert cubic.hilbert_polynomial(d) == 3*d + 1
    assert cubic.hilbert_series(t) == (2*t + 1)/(1 - t)**2


def test_macaulay2_singular_documented_examples():
    # Macaulay2 "radical": radical of (x^2, y^3) is (x, y)
    assert Ideal([x**2, y**3], x, y).radical() == Ideal([x, y], x, y)
    # Macaulay2 "hilbertSeries" of the twisted cubic: (1 + 2T)/(1 - T)^2
    cubic = Ideal([x*z - y**2, y*w - z**2, x*w - y*z], x, y, z, w)
    assert cancel(cubic.hilbert_series(t) - (1 + 2*t)/(1 - t)**2) == 0
    # Macaulay2 "hilbertPolynomial" of a plane conic: 2d + 1; of (x^2, xy, y^2): 0
    assert Ideal([x**2 + y**2 - z**2], x, y, z).hilbert_polynomial(d) == 2*d + 1
    assert Ideal([x**2 + y**2 - z**2], x, y, z).degree() == 2
    assert Ideal([x**2, x*y, y**2], x, y).hilbert_series(t) == 1 + 2*t
    assert Ideal([x**2, x*y, y**2], x, y).vector_space_dimension() == 3
    # Singular "sat": the saturation of (x^2 y, x y^2) by x is (y)
    assert Ideal([x**2*y, x*y**2], x, y).saturate(x) == Ideal([y], x, y)
    # Singular "quotient": (x^2 y, x y^2) : (x y) = (x, y)
    assert Ideal([x**2*y, x*y**2], x, y).quotient(x*y) == Ideal([x, y], x, y)
    # Singular "dim"/"vdim": (x^3, y^2) has vector space dimension 6
    assert Ideal([x**3, y**2], x, y).vector_space_dimension() == 6
    assert Ideal([x**3, y**2], x, y).degree() == 6
    # Singular "eliminate": eliminating y from (x^2 - y, y^2 - z) gives (x^4 - z)
    assert Ideal([x**2 - y, y**2 - z], x, y, z).eliminate([y]) == Ideal([x**4 - z], x, z)
    # maximal ideals over QQ: (x^2 + 1, y^2 + 1) is not maximal (two primes),
    # (x^2 - 2, y^2 - 3) is
    assert not Ideal([x**2 + 1, y**2 + 1], x, y).is_maximal()
    assert Ideal([x**2 - 2, y**2 - 3], x, y).is_maximal()
    assert Ideal([x**2 + x + 1, y - x], x, y).is_prime()


def test_benchmark_systems():
    # Katsura-m has 2^m solutions
    from sympy_extras.polys.groebnerwalk import groebner_walk
    for m_, count in [(2, 4), (3, 8), (4, 16)]:
        eqs, us = katsura(m_)
        I = Ideal(eqs, *us)
        assert I.is_zero_dimensional()
        assert I.vector_space_dimension() == count, m_
        assert I.degree() == count
        assert I.is_radical()
        # FGLM (change_order) and the walk agree; for Katsura-3 also with
        # the direct lex computation (too slow in SymPy for Katsura-4)
        fglm = [p.as_expr() for p in I.change_order('lex')]
        if m_ <= 3:
            walk = [g.as_expr() for g in groebner_walk(I._basis(), I.ring(), 'lex')]
            assert fglm == walk
            assert fglm == list(groebner(eqs, *us, order='lex', domain=QQ))
    # cyclic-4 is not zero-dimensional (its solution set has dimension 1)
    eqs, xs = cyclic(4)
    C4 = Ideal(eqs, *xs)
    assert C4.dimension() == 1
    assert [p.as_expr() for p in C4.change_order('lex')] == list(groebner(eqs, *xs, order='lex', domain=QQ))
    # cyclic-5 has 70 solutions
    eqs, xs = cyclic(5)
    C5 = Ideal(eqs, *xs)
    assert C5.vector_space_dimension() == 70
    assert C5.is_radical()
    assert C5.univariate(xs[0]).degree() == 15
    walk = [g.as_expr() for g in groebner_walk(C5._basis(), C5.ring(), 'lex')]
    assert walk == [p.as_expr() for p in C5.change_order('lex')]
