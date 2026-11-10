"""Exact solutions of zero-dimensional systems: known solution sets,
exact substitution, numerical residuals."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras.polys.domains import QQ
from sympy_extras.polys.f4 import f4
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.rings import PolyElement, PolyRing, lex
from sympy_extras.polys.zerodim import solve_zero_dimensional, univariate_in_ideal


def _exact_value(f: PolyElement[Fraction], point: list[Algebraic]) -> Algebraic:
    total = Algebraic.from_rational(0)
    for monomial, c in f.terms.items():
        term = Algebraic.from_rational(c)
        for value, e in zip(point, monomial):
            if e:
                term = term*value**e
        total = total + term
    return total


def _residual(f: PolyElement[Fraction], point: list[Algebraic]) -> float:
    total = 0j
    for monomial, c in f.terms.items():
        term = complex(float(c))
        for value, e in zip(point, monomial):
            term *= complex(value)**e
        total += term
    return abs(total)


def test_known_solution_sets() -> None:
    R = PolyRing(QQ, ['x', 'y'], lex)
    x, y = R.gens
    # a circle and a hyperbola: (+-1, +-2), (+-2, +-1) with equal signs
    solutions = solve_zero_dimensional([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
    assert sorted((s[0].rational_value(), s[1].rational_value()) for s in solutions) == [(-2, -1), (-1, -2), (1, 2), (2, 1)]
    # a circle and a parabola: y is the golden ratio minus one, or its conjugate with imaginary x
    system = [x*x + y*y - R.one, y - x*x]
    solutions = solve_zero_dimensional(system, R)
    assert len(solutions) == 4
    real = [s for s in solutions if s[0].is_real]
    assert len(real) == 2 and all(s[1].minpoly == [1, 1, -1] and s[1] > 0 for s in real)
    assert all(s[0]*s[0] == s[1] for s in solutions)
    for s in solutions:
        for f in system:
            assert _exact_value(f, s).is_zero
    # a multiple point, the product of two pairs of points (not in shape position), no solution
    assert solve_zero_dimensional([x*x, y*y], R) == [[Algebraic.from_rational(0), Algebraic.from_rational(0)]]
    grid = solve_zero_dimensional([x*x - R.one, y*y - R.one], R)
    assert sorted((s[0].rational_value(), s[1].rational_value()) for s in grid) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    assert solve_zero_dimensional([x + y, R.one], R) == []
    assert solve_zero_dimensional([x - y, x + y - R.from_int(2), x*y - R.from_int(3)], R) == []
    # infinitely many solutions
    raises(ValueError, lambda: solve_zero_dimensional([x*y], R))
    raises(ValueError, lambda: solve_zero_dimensional([], R))
    # one variable
    U = PolyRing(QQ, ['t'], lex)
    (t,) = U.gens
    roots = solve_zero_dimensional([t*t*t - U.from_int(2)], U)
    assert len(roots) == 3 and all(len(r) == 1 and r[0]**3 == 2 for r in roots)


def test_symmetric_systems() -> None:
    # x + y + z = 1, x**2 + y**2 + z**2 = 1, x**3 + y**3 + z**3 = 1: the permutations of (1, 0, 0)
    R = PolyRing(QQ, ['x', 'y', 'z'], lex)
    x, y, z = R.gens
    solutions = solve_zero_dimensional([x + y + z - R.one, x*x + y*y + z*z - R.one, x*x*x + y*y*y + z*z*z - R.one], R)
    assert sorted(tuple(c.rational_value() for c in s) for s in solutions) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # cyclic-3: x + y + z = 0, xy + yz + zx = 0, xyz = 1: six points (w**i, w**j, w**k)
    cyclic = [x + y + z, x*y + y*z + z*x, x*y*z - R.one]
    solutions = solve_zero_dimensional(cyclic, R)
    assert len(solutions) == 6
    assert len({tuple(s) for s in solutions}) == 6
    for s in solutions:
        assert all(c**3 == 1 for c in s)
        assert (s[0] + s[1] + s[2]).is_zero and s[0]*s[1]*s[2] == 1


def test_katsura_counts_and_residuals() -> None:
    from sympy_extras.polys.tests.test_fglm import _katsura
    for n in (2, 3):
        R = PolyRing(QQ, ['u%d' % i for i in range(n + 1)], lex)
        system = _katsura(n, R)
        solutions = solve_zero_dimensional(system, R)
        # 2**n solutions (Katsura), all different
        assert len(solutions) == 2**n and len({tuple(s) for s in solutions}) == 2**n
        for s in solutions:
            for f in system:
                assert _residual(f, s) < 1e-9
        # the point (1, 0, ..., 0) is one of them
        assert [Algebraic.from_rational(1)] + [Algebraic.from_rational(0)]*n in solutions


def test_planted_solutions() -> None:
    rng = random.Random(95)
    R = PolyRing(QQ, ['x', 'y'], lex)
    x, y = R.gens
    for _ in range(10):
        a, b = Fraction(rng.randint(-3, 3)), Fraction(rng.randint(-3, 3), rng.choice([1, 2]))
        # two curves through (a, b)
        f = (x - R.ground(a))*(x + y) + (y - R.ground(b))*R.from_int(rng.randint(1, 3))
        g = (y - R.ground(b))*(x - y + R.one) + (x - R.ground(a))*(x - R.ground(a))
        solutions = solve_zero_dimensional([f, g], R)
        assert [Algebraic.from_rational(a), Algebraic.from_rational(b)] in solutions
        for s in solutions:
            assert _residual(f, s) < 1e-8 and _residual(g, s) < 1e-8


def test_univariate_in_ideal() -> None:
    R = PolyRing(QQ, ['x', 'y'], lex)
    x, y = R.gens
    G = f4([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
    assert univariate_in_ideal(G, R, 0) == [1, 0, -5, 0, 4]
    assert univariate_in_ideal(G, R, 1) == [1, 0, -5, 0, 4]
    H = f4([x*x, y*y*y - y*y], R)
    assert univariate_in_ideal(H, R, 0) == [1, 0, 0] and univariate_in_ideal(H, R, 1) == [1, -1, 0, 0]
    assert univariate_in_ideal([R.one], R, 0) == [1]
