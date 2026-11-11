"""The native Thue solver on forms of the core (the other tests, in
``test_thue.py``, go through the SymPy face)."""
from __future__ import annotations

from fractions import Fraction
from typing import Callable

from sympy_extras.core.symbol import Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, Rational
from sympy_extras.core import sqrt
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.functions import sin

from sympy_extras.solvers.thue_equation import ThueEquation, _Order, _solve_rational, thue

x, y, a = Symbol('x'), Symbol('y'), Symbol('a')


def raises(exception: type[BaseException], f: Callable[[], object]) -> None:
    try:
        f()
    except exception:
        return
    raise AssertionError('%s not raised' % exception.__name__)


def _evaluate(F: Expr, u: int, v: int) -> int:
    """``F(u, v)`` from the coefficients of the form."""
    total = 0
    for (i, j), c in Poly(F, x, y).terms():
        assert isinstance(c, Rational) and c.q == 1
        total += c.p*u**i*v**j
    return total


def _brute_force(F: Expr, m: int, box: int) -> list[tuple[int, int]]:
    return sorted((u, v) for u in range(-box, box + 1) for v in range(-box, box + 1) if _evaluate(F, u, v) == m)


def test_thue_native_known() -> None:
    # Nagell: x**3 - 2 y**3 = 1 has the solutions (1, 0) and (-1, -1)
    assert thue(x**3 - 2*y**3, 1, x, y) == [(-1, -1), (1, 0)]
    # Thomas: the simplest cubic has exactly nine solutions
    assert thue(x**3 + x**2*y - 2*x*y**2 - y**3, 1, x, y) == [(-9, 5), (-1, -1), (-1, 1), (-1, 2), (0, -1),
                                                              (1, 0), (2, -1), (4, -9), (5, 4)]


def test_thue_native_against_search() -> None:
    for F, m in [(2*x**3 + 3*y**3, 5), (x**3 - 2*y**3, 6), (x**4 + y**4, 2), (x**3 - x*y**2 + y**3, -3)]:
        found = thue(F, m, x, y)
        assert found == _brute_force(F, m, 40), (F, m)
        assert all(_evaluate(F, u, v) == m for u, v in found)


def test_thue_native_errors() -> None:
    raises(ValueError, lambda: thue(x**2 - 2*y**2, 1, x, y))
    raises(ValueError, lambda: thue(x**3 - y**3, 1, x, y))            # reducible
    raises(ValueError, lambda: thue(x**3 + x*y, 1, x, y))             # not homogeneous
    raises(ValueError, lambda: thue(x*y**2 + y**3, 1, x, y))          # no x**3 term
    raises(ValueError, lambda: thue(x**3/2 - y**3, 1, x, y))          # rational coefficients
    raises(ValueError, lambda: thue(a*x**3 - y**3, 1, x, y))          # a parameter
    # the native Poly has no domain for sqrt(2): CoercionFailed escaped (SymPy's ValueError expected)
    raises(ValueError, lambda: thue(sqrt(2)*x**3 - y**3, 1, x, y))
    raises(ValueError, lambda: thue(Float(1.5)*x**3 - y**3, 1, x, y))
    raises(BasePolynomialError, lambda: thue(sin(x)*y**2 + y**3, 1, x, y))
    assert thue(x**3 - 2*y**3, 0, x, y) == [(0, 0)]
    # the content does not make a form reducible (as SymPy's is_irreducible)
    assert thue(2*x**3 + 4*y**3, 2, x, y) == [(-1, 1), (1, 0)]


def test_norm_is_the_resultant() -> None:
    order = _Order([1, 0, 0, -2])
    assert order.norm((0, 1, 0)) == 2 and order.norm((1, -1, 0)) == -1 and order.norm((3, 0, 0)) == 27
    assert order.norm((0, 0, 0)) == 0


def test_small_solutions_sorted() -> None:
    # the integer roots of F(X, y) - m for each y in turn, increasing
    equation = ThueEquation([1, 1, -2, -1], 1)
    assert equation.small_solutions(1) == [(-1, -1), (0, -1), (2, -1), (1, 0), (-1, 1)]


def test_solve_rational() -> None:
    A = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(3)]]
    assert _solve_rational(A, [Fraction(3), Fraction(5)]) == [Fraction(4, 5), Fraction(7, 5)]
    assert _solve_rational([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]], [Fraction(1), Fraction(1)]) \
        is None
