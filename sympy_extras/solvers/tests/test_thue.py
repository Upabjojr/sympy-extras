from __future__ import annotations

from sympy import Poly
from sympy.core.expr import Expr
from sympy.abc import x, y
from sympy.testing.pytest import raises

from sympy_extras.solvers.thue_equation import (thue, units_of_order, elements_of_norm, lll,
    short_vectors, ThueEquation)


def _brute_force(F: Expr, m: int, box: int) -> list[tuple[int, int]]:
    """The solutions with ``|x|, |y| <= box``."""
    poly = Poly(F, x, y)
    found: list[tuple[int, int]] = []
    for a in range(-box, box + 1):
        for b in range(-box, box + 1):
            if poly.eval({x: a, y: b}) == m:
                found.append((a, b))
    return sorted(found)


def test_precision_is_restored() -> None:
    """The numerical parts raise mpmath's global precision and put it back:
    the results of the other modules must not depend on whether a Thue
    equation was solved before."""
    import mpmath
    from sympy_extras.solvers.thue_equation import at_precision
    before = mpmath.mp.dps
    with at_precision(before + 20):
        assert mpmath.mp.dps == before + 20
    assert mpmath.mp.dps == before
    units_of_order([1, 0, 0, -2])
    assert mpmath.mp.dps == before
    elements_of_norm([1, 0, 0, -2], 2, units_of_order([1, 0, 0, -2])[0])
    assert mpmath.mp.dps == before
    thue(x**3 - 2*y**3, 1, x, y)
    assert mpmath.mp.dps == before
    # a right-hand side beyond the enumeration budget is refused, not run forever
    raises(NotImplementedError, lambda: thue(x**3 - 2*y**3, 10**5, x, y))
    assert mpmath.mp.dps == before


def test_short_vectors() -> None:
    assert sorted(short_vectors([[2, 0], [0, 3]], 3.5)) == [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]]
    assert len(short_vectors([[1, 0], [0, 1]], 5.0)) == 81
    raises(NotImplementedError, lambda: short_vectors([[1, 0], [0, 1]], 5.0, budget=10))


def test_lll() -> None:
    assert lll([[1, 0, 3], [0, 1, 5], [0, 0, 7]]) == [[-1, -1, -1], [-1, 2, 0], [-2, 0, 1]]
    # entries far beyond the range of floating point numbers
    big = 10**60
    reduced = lll([[1, 0, big + 3], [0, 1, 2*big + 5], [0, 0, 3*big + 7]])
    assert all(abs(v) < big for row in reduced for v in row)


def test_units_and_norms() -> None:
    # Z[cbrt(2)]: the fundamental unit is cbrt(2) - 1 (or its inverse 1 + cbrt(2) + cbrt(4))
    units, torsion = units_of_order([1, 0, 0, -2])
    assert len(units) == 1 and sorted(torsion) == [(-1, 0, 0), (1, 0, 0)]
    assert units[0] in ((1, -1, 0), (1, 1, 1))
    # the simplest cubic field x**3 - 3x - 1 (totally real, rank 2)
    units, torsion = units_of_order([1, 0, -3, -1])
    assert len(units) == 2
    assert elements_of_norm([1, 0, 0, -2], 1, units_of_order([1, 0, 0, -2])[0]) == [(1, 0, 0)]
    assert elements_of_norm([1, 0, 0, -2], 2, units_of_order([1, 0, 0, -2])[0]) == [(0, 1, 0)]


def test_thue_known() -> None:
    # Nagell: x**3 - 2 y**3 = 1 has the solutions (1, 0) and (-1, -1)
    assert thue(x**3 - 2*y**3, 1, x, y) == [(-1, -1), (1, 0)]
    assert thue(x**3 - 2*y**3, -1, x, y) == [(-1, 0), (1, 1)]
    # Thomas: the simplest cubic x**3 + x**2 y - 2 x y**2 - y**3 = 1 has exactly nine solutions
    assert thue(x**3 + x**2*y - 2*x*y**2 - y**3, 1, x, y) == [(-9, 5), (-1, -1), (-1, 1), (-1, 2), (0, -1), (1, 0),
                                                              (2, -1), (4, -9), (5, 4)]
    # x**3 - 3 x y**2 - y**3 = 1: six solutions (Tzanakis-de Weger)
    assert thue(x**3 - 3*x*y**2 - y**3, 1, x, y) == [(-3, 2), (-1, 1), (0, -1), (1, -3), (1, 0), (2, 1)]


def test_thue_against_search() -> None:
    cases = [(x**3 - 2*y**3, 5), (x**3 - 2*y**3, 3), (x**3 + 2*y**3, 1), (x**3 - 3*x*y**2 - y**3, -1),
             (2*x**3 + 3*y**3, 5), (x**3 + x*y**2 + y**3, 1), (x**3 - 2*y**3, 6)]
    for F, m in cases:
        found = thue(F, m, x, y)
        assert found == _brute_force(F, m, 60), (F, m)
        assert all(Poly(F, x, y).eval({x: a, y: b}) == m for a, b in found)


def test_thue_quartic_and_complex() -> None:
    # a totally complex form: finitely many candidates without Baker's method
    assert thue(x**4 + y**4, 2, x, y) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    assert thue(x**4 + y**4, 3, x, y) == []
    assert thue(x**4 - 2*y**4, -1, x, y) == _brute_force(x**4 - 2*y**4, -1, 40)
    assert thue(x**4 - 2*y**4, 1, x, y) == [(-1, 0), (1, 0)]


def test_thue_errors() -> None:
    raises(ValueError, lambda: thue(x**2 - 2*y**2, 1, x, y))
    raises(ValueError, lambda: thue(x**3 - y**3, 1, x, y))            # reducible
    raises(ValueError, lambda: thue(x**3 + x*y, 1, x, y))             # not homogeneous
    assert thue(x**3 - 2*y**3, 0, x, y) == [(0, 0)]
    equation = ThueEquation([1, 0, 0, -2], 1)
    assert equation.evaluate(1, 0) == 1 and equation.small_solutions(2) == [(-1, -1), (1, 0)]
