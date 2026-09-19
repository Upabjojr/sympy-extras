"""Tests of RegularChain and of the operations modulo a chain."""
from __future__ import annotations

import random
from typing import Sequence

from sympy import Integer, Mul, Rational, expand, symbols
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras._typing import as_expr
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.regularchains import RegularChain, regular_gcd, triangularize

from .oracle import covers, is_squarefree_regular_chain, saturated

x, y, z, t = symbols('x y z t')


def test_description() -> None:
    T = RegularChain([(y + 1)*x - 1, z**2 - 2, y**2 - z], x, y, z)
    assert T.polys == [z**2 - 2, y**2 - z, x*y + x - 1]
    assert T.main_variables == [z, y, x] and T.free_variables == []
    assert T.initials == [1, 1, y + 1]
    assert (T.height, T.dimension, T.degree) == (3, 0, 4)
    assert T.symbols == (x, y, z)
    S = RegularChain([y**2 - t, z*x**3 - y], x, y, z, t)
    assert S.main_variables == [y, x] and S.free_variables == [z, t]
    assert (S.height, S.dimension, S.degree) == (2, 2, 6)
    empty = RegularChain([], x, y)
    assert empty.polys == [] and empty.dimension == 2 and empty.degree == 1
    assert empty.saturated_ideal().is_zero()
    # rational coefficients are cleared, the other symbols come last
    assert RegularChain([x/2 - t/3], x).polys == [3*x - 2*t]
    assert RegularChain([x/2 - t/3], x).symbols == (x, t)
    assert str(T) == 'RegularChain([z**2 - 2, y**2 - z, x*y + x - 1], x, y, z)'


def test_equality_and_hash() -> None:
    T = RegularChain([y**2 - 2, x - y], x, y)
    assert T == RegularChain([x - y, 2*y**2 - 4], x, y)
    assert hash(T) == hash(RegularChain([x - y, 2*y**2 - 4], x, y))
    assert T != RegularChain([x**2 - 2, x - y], y, x)
    assert T != RegularChain([y**2 - 2, x + y], x, y)
    assert len({T, RegularChain([-x + y, y**2 - 2], x, y)}) == 1


def test_what_is_not_a_regular_chain() -> None:
    raises(ValueError, lambda: RegularChain([x**2 - y, x*y - 1], x, y))
    raises(ValueError, lambda: RegularChain([Integer(3)], x))
    # an initial which vanishes at one of the two roots below it
    raises(ValueError, lambda: RegularChain([y**2 - z*y, (y - z)*x - 1], x, y, z))
    # an initial which vanishes identically
    raises(ValueError, lambda: RegularChain([y**2 - 2, (y**2 - 2)*x**2 + x], x, y))
    # not squarefree, alone and modulo the polynomial below
    raises(ValueError, lambda: RegularChain([(x - y)**2], x, y))
    raises(ValueError, lambda: RegularChain([y**2 - 2, x**2 - 2*y*x + 2], x, y))
    raises(ValueError, lambda: RegularChain([x - 1], x, x))
    # these are
    assert is_squarefree_regular_chain(RegularChain([y**2 - z*y, (y + z)*x - 1], x, y, z))
    assert is_squarefree_regular_chain(RegularChain([y**2 - 2, x**2 - 2*y*x + 1], x, y))


def test_reduce_and_contains() -> None:
    T = RegularChain([z**2 - 2, y**2 - z, (y + 1)*x - 1], x, y, z)
    ideal = saturated(T)
    assert T.reduce(y**3 + x) == y*z + 3
    for f in (y**4 - 2, x*z - x - y + 1, (y**2 - z)*x**5 + (z**2 - 2)*y, Integer(0)):
        assert T.contains(f) and f in T and ideal.contains(f)
    for f in (y**3 + x, y - 1, x, Integer(1), z):
        assert not T.contains(f) and not ideal.contains(f)
        # the remainder is the polynomial times a power of the initials, modulo the ideal
        r = T.reduce(f)
        assert any(ideal.contains(k*(y + 1)**e*f - r) for e in range(4) for k in (1, -1))
    # in positive dimension the initial matters: the saturated ideal is
    # larger than the ideal of the polynomials
    S = RegularChain([z*y - 1, y*x - z], x, y, z)
    assert S.contains(x - z**2) and S.contains(y*z - 1)
    assert S.saturated_ideal() == Ideal([x - z**2, y*z - 1], x, y, z)
    raises(ValueError, lambda: T.contains(t))
    raises(ValueError, lambda: T.reduce(1/x))


def _check_regularize(T: RegularChain, f: Expr) -> None:
    zero, regular = T.regularize(f)
    parts = zero + regular
    assert all(is_squarefree_regular_chain(part) for part in parts)
    for part in zero:
        assert part.contains(f) and saturated(part).contains(f)
    for part in regular:
        ideal = saturated(part)
        assert ideal.quotient(f) == ideal and part.is_regular(f)
    # the parts lie in the closure of the chain and cover its quasi-component
    assert all(saturated(part).contains(p) for part in parts for p in T.polys)
    assert covers(parts, T.polys, T.symbols, [as_expr(Mul(*T.initials))])
    assert T.is_regular(f) == (not any(part.height == T.height for part in zero))


def test_regularize() -> None:
    S = RegularChain([y**2 - 1, x**2 - y], x, y)
    zero, regular = S.regularize(y - 1)
    assert zero == [RegularChain([y - 1, x**2 - y], x, y)]
    assert regular == [RegularChain([y + 1, x**2 - y], x, y)]
    assert S.regularize(x**2 - y) == ([S], []) and S.regularize(x + 3) == ([], [S])
    assert S.regularize(Integer(0)) == ([S], []) and S.regularize(Integer(2)) == ([], [S])
    assert not S.is_regular(y - 1) and not S.is_regular(x - 1) and S.is_regular(x - 2)
    assert not S.is_regular(Integer(0))
    # x - 1 vanishes at one of the four points
    _check_regularize(S, x - 1)
    _check_regularize(S, (x - 1)*(y + 1))
    _check_regularize(S, x*y - x + y - 1)
    T = RegularChain([z**4 - 5*z**2 + 4, y**2 - z**2, x**2 - y*z], x, y, z)
    for f in (y - z, x**2 - 1, (z - 1)*x + y - 1):
        _check_regularize(T, f)
    # positive dimension: a free variable below, a leading coefficient which
    # vanishes on a part, and a polynomial in a free variable above the chain
    U = RegularChain([y**2 - z, (z - 1)*x**2 - y], x, y, z)
    for f in (y - 1, (y - 1)*x + z - 1, z - 1, x**2*(z - 1)**2 - z, y*x - 1):
        _check_regularize(U, f)
    V = RegularChain([z**2 - 1], x, y, z)
    for f in ((z - 1)*x + y, (z - 1)*x + (z + 1)*y, (z - 1)*x*y + z - 1):
        _check_regularize(V, f)


def test_intersect() -> None:
    circle = RegularChain([x**2 + y**2 - 1], x, y)
    assert circle.intersect(x - y) == [RegularChain([2*y**2 - 1, x - y], x, y)]
    assert circle.intersect(Integer(0)) == [circle] and circle.intersect(Integer(1)) == []
    assert circle.intersect(x**2 + y**2 - 1) == [circle]
    assert circle.intersect(x**2 + y**2 - 2) == []
    cases: Sequence[tuple[RegularChain, Expr]] = [
        (circle, x*y*(x - y)),
        (RegularChain([y**2 - z, (z - 1)*x**2 - y], x, y, z), x*y - 1),
        (RegularChain([y**2 - z, (z - 1)*x**2 - y], x, y, z), z*(y - 1)),
        (RegularChain([z*y - 1], x, y, z), (y - 1)*x**2 + z - 1),
        (RegularChain([z**2 - 1], x, y, z), (z - 1)*x + (z + 1)*y),
    ]
    for T, f in cases:
        parts = T.intersect(f)
        assert all(is_squarefree_regular_chain(part) for part in parts)
        # in the zeros of f in the closure, and covering those of the quasi-component
        assert all(saturated(part).contains(g) for part in parts for g in [f] + T.polys)
        assert covers(parts, T.polys + [f], T.symbols, [as_expr(Mul(*T.initials))])
    # the initial of the chain vanishes where the line meets z = 1: the point
    # is not in the quasi-component
    U = RegularChain([(z - 1)*x - y], x, y, z)
    assert all(not part.contains(z - 1) for part in U.intersect(y))


def test_regular_gcd() -> None:
    T = RegularChain([y**2 - 1], x, y)
    assert regular_gcd(x**2 - y, x**2 - 3*x + 2*y, x, T) == [
        (1, RegularChain([y + 1], x, y)), (x - 1, RegularChain([y - 1], x, y))]
    # the definition: the leading coefficient is regular, the gcd is in the
    # ideal of the two polynomials and divides them, modulo the saturated ideal
    U = RegularChain([z**3 - z, y**2 - z**2 - 1], x, y, z)
    p = expand((x - y)*(x**2 + z*x + 1) + (z - 1)*(x**2 + y))
    q = expand((x - y)*(x + z) + (z - 1)*(x + y**2))
    results = regular_gcd(p, q, x, U)
    assert covers([part for _, part in results], U.polys, U.symbols)
    degrees = {}
    for g, part in results:
        ideal = saturated(part)
        lead = g.as_poly(x).LC()
        assert ideal.quotient(lead) == ideal
        both = Ideal(ideal.exprs + [p, q], *U.symbols)
        assert both.saturate(lead).contains(g)
        with_g = Ideal(ideal.exprs + [g], *U.symbols).saturate(lead)
        assert with_g.contains(p) and with_g.contains(q)
        degrees[str(part.polys)] = g.as_poly(x).degree()
    # x - y is a common factor at z = 1, and x + 1 at the point z = 0, y = -1
    assert degrees == {'[z + 1, y**2 - z**2 - 1]': 0, '[z, y - 1]': 0, '[z - 1, y**2 - z**2 - 1]': 1, '[z, y + 1]': 1}
    # of equal degrees, and with a free variable in the chain
    V = RegularChain([y**2 - t], x, y, t)
    # a regular gcd is defined up to a regular factor: here y*(x - y)
    [(g, part)] = regular_gcd(x**2 - t, x**2 - 2*y*x + t, x, V)
    assert part == V and V.contains(g - y*(x - y)) and g.as_poly(x).degree() == 1
    raises(ValueError, lambda: regular_gcd(x**2 - y, y - 1, x, T))
    raises(ValueError, lambda: regular_gcd(y**2 - 1, y - 1, y, T))
    raises(ValueError, lambda: regular_gcd((y - 1)*x**2 - y, x - 1, x, T))
    raises(ValueError, lambda: regular_gcd(z**2 - y, z - 1, z, T))


def test_random_operations() -> None:
    generator = random.Random(7)
    for _ in range(6):
        roots = generator.sample(range(-3, 4), 3)
        bottom = expand(Mul(*[z - r for r in roots]))
        middle = expand((y - generator.choice(roots))*(y - z) + generator.randint(0, 1)*(z - roots[0]))
        chains = triangularize([bottom, middle, x**2 - y*z - generator.randint(0, 2)], x, y, z)
        for T in chains:
            f = expand((x - generator.randint(-2, 2))*(y - generator.choice(roots)) + generator.randint(0, 1)*(z - roots[1]))
            _check_regularize(T, f)


def test_rational_input() -> None:
    T = RegularChain([y**2 - Rational(1, 4), x - y], x, y)
    assert T.polys == [4*y**2 - 1, x - y]
    assert T.contains(x**2 - Rational(1, 4)) and T.reduce(x + Rational(1, 2)) == 2*y + 1


def test_numerical_solutions() -> None:
    system = [x**3 + y*z - 2, y**3 + x*z - 3, z**2 + x*y - 1]
    solutions = [s for chain in triangularize(system, x, y, z) for s in chain.numerical_solutions(12)]
    # Bezout's number, all the solutions being simple and at finite distance
    assert len(solutions) == 18
    assert all(abs(complex(f.xreplace(s))) < 1e-9 for f in system for s in solutions)
    points = [tuple(complex(s[v]) for v in (x, y, z)) for s in solutions]
    assert all(max(abs(u - v) for u, v in zip(p, q)) > 1e-6 for k, p in enumerate(points) for q in points[:k])
    # real solutions come out real
    [circles] = triangularize([x**2 + y**2 - 1, (x - 1)**2 + y**2 - 1], x, y)
    solutions = circles.numerical_solutions(10)
    assert all(abs(s[x] - Rational(1, 2)) < 1e-9 for s in solutions) and len(solutions) == 2
    assert all(s[y].is_real for s in solutions) and abs(solutions[1][y]**2 - Rational(3, 4)) < 1e-9
    raises(ValueError, lambda: RegularChain([x*y - 1], x, y).numerical_solutions())


def test_with_symbols() -> None:
    T = RegularChain([y**2 - 2, x - y], x, y)
    U = T.with_symbols(x, t, y, z)
    assert U.symbols == (x, t, y, z) and U.polys == T.polys and U.free_variables == [t, z]
    assert U.contains(t*x**2 - 2*t + z*(x - y)) and not U.contains(t)
    assert is_squarefree_regular_chain(U)
    assert T.with_symbols(x, y) == T
    raises(ValueError, lambda: T.with_symbols(y, x))
    raises(ValueError, lambda: T.with_symbols(x, z))
    raises(ValueError, lambda: T.with_symbols(x, y, y))
