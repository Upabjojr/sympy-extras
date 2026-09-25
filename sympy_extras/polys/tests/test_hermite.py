"""Tests of real root counting by Hermite's quadratic form and of sign
determination.

The hand-checked examples have solutions known in closed form; the random
systems are checked against the number of cells of a cylindrical
algebraic decomposition on which the equations and the sign conditions
hold, an independent computation (projection, real root isolation and
lifting, no quadratic form)."""
from __future__ import annotations

import random
from itertools import product

from sympy import And, Eq, Matrix, Ne, Or, Rational, sqrt
from sympy.abc import x, y, z
from sympy.core.expr import Expr
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly, real_roots
from sympy.testing.pytest import raises

from sympy_extras._typing import DomainElement, as_boolean
from sympy_extras.polys.cad import sample_points
from sympy_extras.polys.hermite import (SignCounts, count_complex_solutions, count_real_solutions,
    decide_zero_dimensional, hermite_matrix, real_sign_counts, sign_determination, tarski_query, _inertia)
from sympy_extras.polys.ideals import Ideal


def test_circle_and_line() -> None:
    # (1/sqrt(2), 1/sqrt(2)) and its opposite
    assert count_real_solutions([x**2 + y**2 - 1, x - y], x, y) == 2
    assert real_sign_counts([x**2 + y**2 - 1, x - y], x, x, y) == SignCounts(0, 1, 1)
    # the line misses the circle: two complex solutions
    assert count_real_solutions([x**2 + y**2 - 1, x + y - 2], x, y) == 0
    assert count_complex_solutions([x**2 + y**2 - 1, x + y - 2], x, y) == 2
    # tangent: one double solution
    assert count_real_solutions([x**2 + y**2 - 2, x + y - 2], x, y) == 1
    assert count_complex_solutions([x**2 + y**2 - 2, x + y - 2], x, y) == 1


def test_no_real_solution() -> None:
    assert count_real_solutions([x**2 + 1], x) == 0
    assert count_complex_solutions([x**2 + 1], x) == 2
    assert count_real_solutions([x**2 + 1, y**2 + 1], x, y) == 0
    assert sign_determination([x**2 + 1, y - 1], [x, y], x, y) == {}


def test_multiplicities_are_not_counted() -> None:
    assert count_real_solutions([(x - 1)**3*(x + 2), y**2], x, y) == 2
    assert count_complex_solutions([(x - 1)**3*(x + 2), y**2], x, y) == 2
    assert Ideal([(x - 1)**3*(x + 2), y**2], x, y).vector_space_dimension() == 8
    # a non-radical ideal whose multiple point is not a product of powers
    I = Ideal([x**2 - y**3, x*y, y**4 - y], x, y)
    assert count_real_solutions(I) == len(sample_points(And(*[Eq(g, 0) for g in I.exprs]), [x, y],
                                                        partial=False))


def test_polynomial_vanishing_at_solutions() -> None:
    # (-1, 1), (0, 0), (1, 1): x - y is -2, 0, 0
    assert real_sign_counts([x**3 - x, y - x**2], x - y, x, y) == SignCounts(2, 0, 1)
    # x*y is -1, 0, 1 and vanishes at a double solution of the ideal
    assert real_sign_counts([x**2*(x**2 - 1), y - x], x*y, x, y) == SignCounts(1, 2, 0)
    assert real_sign_counts([x**2 - 2], x**2 - 2, x) == SignCounts(2, 0, 0)
    assert real_sign_counts([x**2 - 2], x**3, x) == SignCounts(0, 1, 1)
    assert tarski_query([x**3 - x, y - x**2], 2*y - 1, x, y) == 1
    assert tarski_query([x**3 - x, y - x**2], x**2*y, x, y) == 2


def test_sign_determination() -> None:
    signs = sign_determination([x**2 - 1, y**2 - 1], [x + y, x, x - y], x, y)
    expected: dict[tuple[int, ...], int] = {}
    for a, b in product([-1, 1], repeat=2):
        key = tuple(_sign(v) for v in (a + b, a, a - b))
        expected[key] = expected.get(key, 0) + 1
    assert signs == expected
    # the eight points (±1, ±2, ±3) and the signs of x*y*z, x + y + z
    signs = sign_determination([x**2 - 1, y**2 - 4, z**2 - 9], [x*y*z, x + y + z, z], x, y, z)
    expected = {}
    for a, b, c in product([-1, 1], [-2, 2], [-3, 3]):
        key = (_sign(a*b*c), _sign(a + b + c), _sign(c))
        expected[key] = expected.get(key, 0) + 1
    assert signs == expected
    assert sign_determination([x**2 - 2, y - x], [], x, y) == {(): 2}


def test_decide_zero_dimensional() -> None:
    circle = Eq(x**2 + y**2, 2) & Eq(x, y)  # (1, 1) and (-1, -1)
    assert decide_zero_dimensional(circle & (x > 0), [x, y]) is True
    assert decide_zero_dimensional(circle & (x > 1), [x, y]) is False
    assert decide_zero_dimensional(circle & (x >= 1), [x, y]) is True
    assert decide_zero_dimensional(circle & (x*y < 1), [x, y]) is False
    assert decide_zero_dimensional(circle & (x*y <= 1), [x, y]) is True
    assert decide_zero_dimensional(circle & Ne(x**2, 1), [x, y]) is False
    assert decide_zero_dimensional(circle & (x > 0) & (y < 0), [x, y]) is False
    assert decide_zero_dimensional(Eq(x, 1) & Eq(x, 2) & Eq(y, 0), [x, y]) is False
    assert decide_zero_dimensional(Eq(x**2, 2), [x]) is True
    # not applicable: infinitely many solutions, a disjunction, an
    # irrational coefficient
    assert decide_zero_dimensional(Eq(x**2 + y**2, 2) & (x > 0), [x, y]) is None
    assert decide_zero_dimensional(Or(Eq(x, 1), Eq(x, 2)), [x]) is None
    assert decide_zero_dimensional(Eq(x**2, sqrt(2)), [x]) is None
    assert decide_zero_dimensional(Eq(x**4, 1) & Eq(y, 0), [x, y], max_dimension=3) is None


def test_hermite_matrix() -> None:
    assert hermite_matrix([x**2 - 2], 1, x) == Matrix([[2, 0], [0, 4]])
    assert hermite_matrix([x**2 - 2], x, x) == Matrix([[0, 4], [4, 0]])
    # the trace of 1 is the number of solutions with multiplicity
    H = hermite_matrix([x**2 - y, y**2 - 1], 1, x, y)
    assert H[0, 0] == 4 and H == H.T


def test_whole_ring_and_invalid_input() -> None:
    assert count_real_solutions([x, x - 1], x) == 0
    assert count_complex_solutions([x, x - 1], x) == 0
    assert real_sign_counts([x - 1, x - 2, y], y, x, y) == SignCounts(0, 0, 0)
    raises(NotImplementedError, lambda: count_real_solutions([x*y], x, y))
    raises(NotImplementedError, lambda: count_real_solutions([x**2 - sqrt(2)], x))


def test_inertia() -> None:
    q = QQ
    # zero diagonal: the 2x2 step replaces e_0 by e_0 + e_1
    assert _inertia([[q(0), q(1)], [q(1), q(0)]]) == (1, 1)
    assert _inertia([[q(0), q(1), q(0)], [q(1), q(0), q(0)], [q(0), q(0), q(0)]]) == (1, 1)
    # the diagonal becomes zero only after the first elimination
    assert _inertia([[q(1), q(1), q(1)], [q(1), q(1), q(2)], [q(1), q(2), q(1)]]) == _descartes(
        Matrix([[1, 1, 1], [1, 1, 2], [1, 2, 1]]))
    assert _inertia([]) == (0, 0)


def test_inertia_random() -> None:
    # seeded: symmetric integer matrices of low rank with many zeros, whose
    # inertia is compared with the signs of the roots of the characteristic
    # polynomial, all real, counted by Descartes' rule of signs
    rng = random.Random(1616)
    for _ in range(60):
        n = rng.randint(1, 6)
        k = rng.randint(0, n)
        rows = [[rng.choice([0, 0, 1, -1, 2]) for _ in range(k)] for _ in range(n)]
        signs = [rng.choice([1, -1]) for _ in range(k)]
        M = Matrix(n, n, lambda i, j: sum(signs[t]*rows[i][t]*rows[j][t] for t in range(k)))
        entries: list[list[DomainElement]] = [[QQ(int(M[i, j])) for j in range(n)] for i in range(n)]
        assert _inertia(entries) == _descartes(M), M


def test_univariate_against_real_roots() -> None:
    # seeded random univariate polynomials with repeated factors, against
    # the distinct real roots isolated by SymPy
    rng = random.Random(16)
    for _ in range(30):
        p = Poly(1, x)
        for _ in range(rng.randint(1, 3)):
            factor = Poly([rng.randint(-4, 4) for _ in range(rng.randint(2, 4))], x)
            if factor.degree() > 0:
                p = p*factor**rng.randint(1, 2)
        if p.degree() <= 0:
            continue
        distinct = set(real_roots(p))
        assert count_real_solutions([p.as_expr()], x) == len(distinct), p
        g = Poly([rng.randint(-3, 3) for _ in range(3)], x)
        values = [g.eval(r) for r in distinct]
        assert real_sign_counts([p.as_expr()], g.as_expr(), x) == SignCounts(
            sum(1 for v in values if _is_zero(v)),
            sum(1 for v in values if not _is_zero(v) and v.is_positive),
            sum(1 for v in values if not _is_zero(v) and v.is_negative)), (p, g)


def test_random_systems_against_cad() -> None:
    # seeded random zero-dimensional systems in two variables (dense
    # quadrics and cubics, some with a solution at the origin where the side
    # polynomial vanishes too, some with squared equations), against the
    # number of cells of the CAD on which the equations and each sign
    # condition hold
    rng = random.Random(1616)
    checked = 0
    while checked < 12:
        through_origin = rng.random() < 0.5
        f = _random_poly(rng, rng.randint(1, 3), through_origin)
        g = _random_poly(rng, 2, through_origin)
        h = _random_poly(rng, 1, through_origin)
        if rng.random() < 0.3:
            f = f**2
        I = Ideal([f, g], x, y)
        if I.is_whole_ring() or not I.is_zero_dimensional():
            continue
        checked += 1
        equations = And(Eq(f, 0), Eq(g, 0))
        assert count_real_solutions(I) == _cells(equations), (f, g)
        assert real_sign_counts(I, h) == SignCounts(
            _cells(equations & Eq(h, 0)), _cells(equations & (h > 0)), _cells(equations & (h < 0))), (f, g, h)


def test_random_sign_determination_against_cad() -> None:
    # seeded: two side polynomials, each of the nine sign conditions
    # counted by the CAD
    rng = random.Random(61)
    checked = 0
    while checked < 5:
        f = _random_poly(rng, 2, True)
        g = _random_poly(rng, 2, False) + x*y
        P = [_random_poly(rng, 1, True), _random_poly(rng, 2, False)]
        I = Ideal([f, g], x, y)
        if I.is_whole_ring() or not I.is_zero_dimensional():
            continue
        checked += 1
        equations = And(Eq(f, 0), Eq(g, 0))
        expected: dict[tuple[int, ...], int] = {}
        for s in product([0, 1, -1], repeat=2):
            n = _cells(And(equations, *[_condition(p, e) for p, e in zip(P, s)]))
            if n:
                expected[s] = n
        assert sign_determination(I, P) == expected, (f, g, P)


def test_three_variables_in_shape_position() -> None:
    # seeded random systems of two quadrics and a plane in three variables
    # whose lexicographic Gröbner basis is x - a(z), y - b(z), p(z) (shape
    # lemma): the real solutions are the distinct real roots of p, isolated
    # by SymPy, and the sign of z - c at each is exact
    rng = random.Random(3)
    checked = 0
    while checked < 8:
        F = [_random_quadric(rng), _random_quadric(rng),
             x + rng.randint(-2, 2)*y + rng.randint(-2, 2)*z + rng.randint(-1, 1)]
        I = Ideal(F, x, y, z)
        if I.is_whole_ring() or not I.is_zero_dimensional():
            continue
        G = I.groebner_basis('lex')
        if len(G) != 3 or G[0].as_expr().diff(x) != 1 or G[1].as_expr().diff(y) != 1:
            continue
        checked += 1
        roots = set(real_roots(Poly(G[2].as_expr(), z)))
        c = Rational(rng.randint(-2, 2), 2)
        assert count_real_solutions(I) == len(roots), F
        assert real_sign_counts(I, z - c) == SignCounts(
            sum(1 for r in roots if r == c), sum(1 for r in roots if r > c),
            sum(1 for r in roots if r < c)), F


def _random_poly(rng: random.Random, degree: int, through_origin: bool) -> Expr:
    p: Expr = Rational(0)
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i == j == 0 and through_origin:
                continue
            p += rng.randint(-3, 3)*x**i*y**j
    return p


def _random_quadric(rng: random.Random) -> Expr:
    p: Expr = Rational(rng.randint(-2, 2))
    for m in (x**2, y**2, z**2, x*y, y*z, x, y, z):
        p += rng.randint(-2, 2)*m
    return p


def _cells(formula: Boolean) -> int:
    return len(sample_points(formula, [x, y], partial=False))


def _condition(p: Expr, s: int) -> Boolean:
    relation: Relational = Eq(p, 0) if s == 0 else (p > 0 if s > 0 else p < 0)
    return as_boolean(relation)


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def _is_zero(v: Expr) -> bool:
    return bool(v.equals(0))


def _descartes(M: Matrix) -> tuple[int, int]:
    """The numbers of positive and negative eigenvalues of a symmetric
    matrix, from the sign variations of its characteristic polynomial."""
    p = M.charpoly()
    return _variations(p.all_coeffs()), _variations(p.compose(Poly(-p.gen, p.gen)).all_coeffs())


def _variations(coeffs: list[Expr]) -> int:
    signs = [c > 0 for c in coeffs if c != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)
