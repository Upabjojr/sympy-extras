"""Tests of the differential rings, the rankings and the differential
polynomials."""
from __future__ import annotations

import random

from sympy import Function, Rational, Symbol, diff, expand, prem, symbols
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.polys.differential import DifferentialPolynomial, DifferentialRing

x, y, a = symbols('x y a')
u = Function('u')(x, y)
v = Function('v')(x, y)


def test_orderly_ranking() -> None:
    ring = DifferentialRing([u, v])
    assert ring.is_orderly
    # the order first, then the function, then the first variable
    assert ring.leader(u + v.diff(x)) == v.diff(x)
    assert ring.leader(u.diff(y) + v.diff(x)) == u.diff(y)
    assert ring.leader(u.diff(y, 2) + u.diff(x, y)) == u.diff(x, y)
    assert ring.leader(x**5*u + u.diff(x)**2*v.diff(x, y)) == v.diff(x, y)


def test_elimination_ranking() -> None:
    ring = DifferentialRing([u, v], ranking=[[v], [u]])
    assert not ring.is_orderly
    assert ring.leader(u.diff(x, 3) + v) == v
    ring = DifferentialRing([u, v], variables=[y, x])
    assert ring.leader(u.diff(y, 2) + u.diff(x, y) + u.diff(x, 2)) == u.diff(y, 2)


def test_a_ranking_is_compatible_with_the_derivations() -> None:
    generator = random.Random(3)
    for ranking in (None, [[v], [u]], [[u], [v]]):
        ring = DifferentialRing([u, v], ranking=ranking)
        for _ in range(200):
            first = (generator.randint(0, 1), (generator.randint(0, 3), generator.randint(0, 3)))
            second = (generator.randint(0, 1), (generator.randint(0, 3), generator.randint(0, 3)))
            i = generator.randint(0, 1)
            assert ring.rank_key(first) < ring.rank_key(ring.differentiate(first, i))
            if ring.rank_key(first) < ring.rank_key(second):
                assert ring.rank_key(ring.differentiate(first, i)) < ring.rank_key(ring.differentiate(second, i))


def test_conversion() -> None:
    ring = DifferentialRing([u, v], parameters=[a])
    e = (a*u.diff(x, y)**2*v - u/x + 3)/(y + 1)
    assert expand(ring.from_terms(ring.terms(e)) - e) == 0
    assert ring.terms(u - u) == {}
    assert ring.jet(u.diff(x, y, y)) == (0, (1, 2))
    assert ring.to_expr((1, (2, 0))) == v.diff(x, 2)
    raises(ValueError, lambda: ring.terms(1/u))
    raises(ValueError, lambda: ring.terms(u**a))
    raises(ValueError, lambda: ring.terms(Symbol('b')*u))
    raises(ValueError, lambda: ring.terms(Function('w')(x, y)))
    raises(ValueError, lambda: ring.leader(x + 1))


def test_construction_is_checked() -> None:
    raises(ValueError, lambda: DifferentialRing([]))
    raises(TypeError, lambda: untyped(DifferentialRing)([x]))
    raises(ValueError, lambda: DifferentialRing([Function('w')(x, 2*y)]))
    raises(ValueError, lambda: DifferentialRing([Function('w')(x, x)]))
    raises(ValueError, lambda: DifferentialRing([u], ranking=[[v]]))
    raises(ValueError, lambda: DifferentialRing([u, v], ranking=[[u]]))
    raises(ValueError, lambda: DifferentialRing([u, v], ranking=[[u, v], [u]]))


def _random_polynomial(generator: random.Random) -> Expr:
    jets = [u, v, u.diff(x), u.diff(y), v.diff(x), u.diff(x, y), v.diff(y, 2)]
    result: Expr = Rational(0)
    for _ in range(generator.randint(1, 4)):
        term: Expr = generator.choice([1, 2, -3, x, y, x*y + 1])
        for _ in range(generator.randint(1, 3)):
            term = term*generator.choice(jets)
        result = result + term
    return result


def test_derivatives_agree_with_sympy() -> None:
    generator = random.Random(5)
    ring = DifferentialRing([u, v])
    for _ in range(30):
        e = _random_polynomial(generator)
        p = DifferentialPolynomial(ring, ring.integral(ring.terms(e)))
        for i, s in enumerate((x, y)):
            assert expand(p.derivative(i).to_expr() - diff(e, s)) == 0
        assert expand(p.prolonged((1, 2)).to_expr() - diff(e, x, y, y)) == 0


def test_arithmetic_agrees_with_sympy() -> None:
    generator = random.Random(9)
    ring = DifferentialRing([u, v])
    for _ in range(30):
        e, f = _random_polynomial(generator), _random_polynomial(generator)
        p = DifferentialPolynomial(ring, ring.integral(ring.terms(e)))
        q = DifferentialPolynomial(ring, ring.integral(ring.terms(f)))
        assert expand((p*q).to_expr() - e*f) == 0
        assert expand((p - q).to_expr() - (e - f)) == 0
        assert expand((p**2 + q).to_expr() - (e**2 + f)) == 0
    assert not (p - p)


def test_initial_separant_and_rank() -> None:
    ring = DifferentialRing([u, v])
    # u_x is higher than v_y: the same order, and u is the first function
    p = DifferentialPolynomial.from_expr(ring, (x*u - 1)*u.diff(x)**3 + u*u.diff(x) + v.diff(y)**5)
    assert p.leader_expr() == u.diff(x) and p.degree() == 3
    assert expand(p.initial().to_expr() - (x*u - 1)) == 0
    assert expand(p.separant().to_expr() - (3*(x*u - 1)*u.diff(x)**2 + u)) == 0
    assert p.rank() > DifferentialPolynomial.from_expr(ring, u.diff(x)**2).rank()
    assert p.rank() > DifferentialPolynomial.from_expr(ring, v.diff(x)**7).rank()
    assert p.rank() < DifferentialPolynomial.from_expr(ring, v.diff(x, 2)).rank()
    raises(ValueError, lambda: DifferentialPolynomial.from_expr(ring, x + 1).leader())


def test_primitive_part() -> None:
    ring = DifferentialRing([u, v])
    p = DifferentialPolynomial.from_expr(ring, (4*x*u.diff(x)**2 - 6*x**2*u)/(3*y))
    assert p.to_expr() == 2*u.diff(x)**2 - 3*x*u
    # the sign is the one of the highest monomial
    assert DifferentialPolynomial.from_expr(ring, -u).to_expr() == u
    assert DifferentialPolynomial.from_expr(ring, u - u.diff(x)).to_expr() == u.diff(x) - u


def test_pseudo_remainder_agrees_with_sympy() -> None:
    ring = DifferentialRing([u, v])
    U, V, W = symbols('U V W')
    names = {u.diff(x): W, u: U, v: V}
    e = x*u.diff(x)**3*v + u.diff(x)*u - v**2
    d = u*u.diff(x)**2 - y*v
    p = DifferentialPolynomial.from_expr(ring, e)
    q = DifferentialPolynomial.from_expr(ring, d)
    r = p.pseudo_remainder(q, q.leader())
    expected = prem(e.subs(names), d.subs(names), W)
    ratio = (r.to_expr().subs(names)/expected).cancel()
    assert ratio.free_symbols <= {U}
    assert r.degree(q.leader()) < 2


def test_ritt_reduction() -> None:
    ring = DifferentialRing([u, v])
    chain = [DifferentialPolynomial.from_expr(ring, u.diff(x)**2 - 4*u)]
    p = DifferentialPolynomial.from_expr(ring, u.diff(x, y)*v.diff(y) - u + 1)
    partial = p.partial_remainder(chain)
    assert partial.is_partially_reduced(chain[0])
    # 2 u_x p = 2 u_y v_y - u_x (u - 1), modulo the derivative of the chain
    # (up to the sign, which the primitive part fixes)
    assert expand(partial.to_expr() + (2*u.diff(y)*v.diff(y) - u.diff(x)*(u - 1))) == 0
    q = DifferentialPolynomial.from_expr(ring, u.diff(x, 2)*u.diff(x)**3)
    r = q.remainder(chain)
    assert r.is_reduced(chain[0])
    # the separant 2 u_x times q is u_x**3 (2 u_x u_xx) = 4 u_x**4 = 64 u**2 modulo the chain
    assert expand(r.to_expr() - u**2) == 0
    assert not chain[0].derivative(0).remainder(chain)
