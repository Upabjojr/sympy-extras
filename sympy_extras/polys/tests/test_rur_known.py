"""Rational univariate representations of systems with known solutions:
conics, the Katsura and cyclic families, and SymPy's own solver."""
from __future__ import annotations

from sympy import Expr, N, Poly, Symbol, sqrt, symbols
from sympy.solvers.polysys import solve_poly_system

from sympy_extras._typing import as_expr
from sympy_extras.polys.rur_sympy import rur

x, y, z, t = symbols('x y z t')


def _points(solutions: list[dict[Symbol, Expr]], gens: list[Symbol]) -> list[tuple[complex, ...]]:
    return [tuple(complex(N(s[g], 20)) for g in gens) for s in solutions]


def _close(a: list[tuple[complex, ...]], b: list[tuple[complex, ...]]) -> bool:
    return len(a) == len(b) and all(any(max(abs(u - v) for u, v in zip(p, q)) < 1e-9 for q in b) for p in a)


def test_conics() -> None:
    R = rur([x**2 + y**2 - 1, x - y], x, y, t=t)
    assert R.solutions() == [{x: -sqrt(2)/2, y: -sqrt(2)/2}, {x: sqrt(2)/2, y: sqrt(2)/2}]
    # x**2 + y**2 = 5, x y = 2: (1, 2), (2, 1), (-1, -2), (-2, -1)
    R = rur([x**2 + y**2 - 5, x*y - 2], x, y, t=t)
    assert sorted((s[x], s[y]) for s in R.solutions()) == [(-2, -1), (-1, -2), (1, 2), (2, 1)]
    # the ellipse x**2 + 4 y**2 = 4 and the hyperbola x y = 1 are tangent
    # at (sqrt(2), sqrt(2)/2) and (-sqrt(2), -sqrt(2)/2): two real zeros
    # of multiplicity two
    R = rur([x**2 + 4*y**2 - 4, x*y - 1], x, y, t=t)
    assert R.polynomial.degree() == 2 and len(R.solutions(real=True)) == 2
    assert R.characteristic.degree() == 4 and R.multiplicity(sqrt(2)) == 2
    assert R.solutions() == [{x: -sqrt(2), y: -sqrt(2)/2}, {x: sqrt(2), y: sqrt(2)/2}]


def test_katsura_and_cyclic() -> None:
    # Katsura-n has 2**n distinct solutions
    for n in (2, 3):
        gens = symbols('x0:%d' % (n + 1))
        polys = [gens[0] + 2*sum(gens[1:]) - 1]
        for m in range(n):
            polys.append(sum(gens[abs(i)]*gens[abs(m - i)] for i in range(-n, n + 1)
                             if abs(i) <= n and abs(m - i) <= n) - gens[m])
        R = rur(polys, *gens, t=t)
        assert R.polynomial.degree() == 2**n == R.characteristic.degree()
        coordinates: dict[Symbol, Expr] = dict(zip(R.symbols, [g.as_expr() for g in R.polynomials()]))
        assert all(Poly(as_expr(p).xreplace(coordinates), t).rem(R.polynomial).is_zero for p in polys)
    # cyclic-3: the permutations of the cube roots of unity, six points
    R = rur([x + y + z, x*y + y*z + z*x, x*y*z - 1], x, y, z, t=t)
    assert R.polynomial.degree() == 6
    points = _points(R.numerical_solutions(20), [x, y, z])
    assert all(abs(p[0]**3 - 1) < 1e-9 and abs(p[0] + p[1] + p[2]) < 1e-9 for p in points)
    assert len(R.solutions(real=True)) == 0


def test_against_solve_poly_system() -> None:
    for polys, gens in [([x**2 + y**2 - 1, x*y], [x, y]), ([x**2 - y, y**2 - x], [x, y]),
                        ([x**2 + y**2 + z**2 - 1, x + y + z, x*y - z], [x, y, z]),
                        ([x**3 - 2*x + 1, y**2 - x], [x, y])]:
        expected = solve_poly_system(polys, *gens)
        assert expected is not None
        found = rur(polys, *gens, t=t).numerical_solutions(20)
        assert _close(_points(found, gens), [tuple(complex(N(v, 20)) for v in sol) for sol in expected]), polys
