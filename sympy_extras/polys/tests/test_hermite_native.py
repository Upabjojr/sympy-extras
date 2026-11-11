"""Tests of the native Hermite counting (:mod:`sympy_extras.polys.hermite`)
on the objects of the core: hand-checked systems whose solutions are
known in closed form, and sign conditions counted directly at the known
points (no SymPy)."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Callable

from sympy_extras.core import Integer, Rational, Symbol, sqrt
from sympy_extras.core.logic import Or, true
from sympy_extras.core.relational import Eq, Ne
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.hermite import (SignCounts, _inertia, count_complex_solutions, count_real_solutions,
    decide_zero_dimensional, hermite_matrix, real_sign_counts, sign_determination, tarski_query)
from sympy_extras.polys.ideals import Ideal

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def raises(exception: type[BaseException], f: Callable[[], object]) -> None:
    try:
        f()
    except exception:
        return
    raise AssertionError('%s not raised' % exception.__name__)


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def test_counts() -> None:
    # (1/sqrt(2), 1/sqrt(2)) and its opposite
    assert count_real_solutions([x**2 + y**2 - 1, x - y], x, y) == 2
    assert real_sign_counts([x**2 + y**2 - 1, x - y], x, x, y) == SignCounts(0, 1, 1)
    # the line misses the circle; tangent: one double solution
    assert count_real_solutions([x**2 + y**2 - 1, x + y - 2], x, y) == 0
    assert count_complex_solutions([x**2 + y**2 - 1, x + y - 2], x, y) == 2
    assert count_real_solutions([x**2 + y**2 - 2, x + y - 2], x, y) == 1
    # multiplicities are not counted: the roots 1 (triple) and -2
    assert count_real_solutions([(x - 1)**3*(x + 2), y**2], x, y) == 2
    # x**3 - 2 has one real and two complex roots, x**4 - 5x**2 + 6 four real
    assert count_real_solutions([x**3 - 2], x) == 1 and count_complex_solutions([x**3 - 2], x) == 3
    assert count_real_solutions([x**4 - 5*x**2 + 6, y**2 - x], x, y) == 4
    assert count_real_solutions([Rational(1, 3)*x**2 - Rational(7, 5)], x) == 2
    # the whole ring
    assert count_real_solutions([x, x - 1], x) == 0 and count_complex_solutions([x, x - 1], x) == 0


def test_ideal_argument() -> None:
    I: Ideal[Fraction] = Ideal([x**2 - 3, y**3 - x], x, y, domain=QQ)
    # x = ±sqrt(3), y the real cube root of x
    assert count_real_solutions(I) == 2 and count_complex_solutions(I) == 6
    assert sign_determination(I, [y]) == {(-1,): 1, (1,): 1}


def test_tarski_queries() -> None:
    # (-1, 1), (0, 0), (1, 1): 2y - 1 is 1, -1, 1
    assert tarski_query([x**3 - x, y - x**2], 2*y - 1, x, y) == 1
    assert tarski_query([x**3 - x, y - x**2], x**2*y, x, y) == 2
    assert tarski_query([x**2 + 1], x, x) == 0
    # the roots of x**4 - 10x**2 + 1 are ±(sqrt(3) + sqrt(2)) ≈ ±3.146 and
    # ±(sqrt(3) - sqrt(2)) ≈ ±0.318: r**3 - 9r has the sign of r at the
    # first two and the opposite one at the others
    assert tarski_query([x**4 - 10*x**2 + 1], x**3 - 9*x, x) == 0
    assert tarski_query([x**4 - 10*x**2 + 1], x**3 - 9*x + x**2 - 1, x) == 2


def test_hermite_matrix() -> None:
    assert hermite_matrix([x**2 - 2], 1, x) == ExprMatrix.from_rows([[2, 0], [0, 4]])
    assert hermite_matrix([x**2 - 2], x, x) == ExprMatrix.from_rows([[0, 4], [4, 0]])
    # the trace of 1 is the number of solutions with multiplicity
    H = hermite_matrix([x**2 - y, y**2 - 1], 1, x, y)
    assert H[0, 0] == Integer(4) and all(H[i, j] == H[j, i] for i in range(4) for j in range(4))


def test_sign_determination_at_known_points() -> None:
    # the eight points (±1, ±2, ±3) and the signs of x*y*z, x + y + z, z
    signs = sign_determination([x**2 - 1, y**2 - 4, z**2 - 9], [x*y*z, x + y + z, z], x, y, z)
    expected: dict[tuple[int, ...], int] = {}
    for a, b, c in product([-1, 1], [-2, 2], [-3, 3]):
        key = (_sign(a*b*c), _sign(a + b + c), _sign(c))
        expected[key] = expected.get(key, 0) + 1
    assert signs == expected
    assert sign_determination([x**2 - 2, y - x], [], x, y) == {(): 2}
    assert sign_determination([x**2 + 1, y], [x], x, y) == {}
    # (1, -2, ±2): x + y + z is 1 and -3, x*y*z is -4 and 4
    assert sign_determination([x - 1, y + 2, z**2 - 4], [x + y + z, x*y*z], x, y, z) == {(-1, 1): 1, (1, -1): 1}


def test_inertia() -> None:
    q = Fraction
    # zero diagonal: the 2x2 step replaces e_0 by e_0 + e_1
    assert _inertia([[q(0), q(1)], [q(1), q(0)]]) == (1, 1)
    assert _inertia([[q(2), q(0)], [q(0), q(-3)]]) == (1, 1)
    assert _inertia([[q(1), q(2)], [q(2), q(4)]]) == (1, 0)
    assert _inertia([]) == (0, 0)


def test_decide_zero_dimensional() -> None:
    circle = Eq(x**2 + y**2, 2) & Eq(x*y, 1)    # (1, 1) and (-1, -1)
    assert decide_zero_dimensional(circle & (x > 0), [x, y]) is True
    assert decide_zero_dimensional(circle & (x > 1), [x, y]) is False
    assert decide_zero_dimensional(circle & (x >= 1), [x, y]) is True
    assert decide_zero_dimensional(circle & Ne(x**2, 1), [x, y]) is False
    assert decide_zero_dimensional(circle & (x > 0) & (y < 0), [x, y]) is False
    assert decide_zero_dimensional(Eq(x, 1) & Eq(x, 2) & Eq(y, 0), [x, y]) is False
    assert decide_zero_dimensional(Eq(x**2, 2), [x]) is True
    assert decide_zero_dimensional(Eq(x**2 + 1, 0) & Eq(y, 0), [x, y]) is False
    assert decide_zero_dimensional(true, [x]) is True and decide_zero_dimensional(False, [x]) is False
    # not of the form: a curve, a disjunction, irrational or symbolic
    # coefficients, a rational function, too many solutions
    assert decide_zero_dimensional(Eq(x**2 + y**2, 2) & (x > 0), [x, y]) is None
    assert decide_zero_dimensional(Or(Eq(x, 1), Eq(x, 2)), [x]) is None
    assert decide_zero_dimensional(Eq(x**2, sqrt(Integer(2))), [x]) is None
    assert decide_zero_dimensional(Eq(x**2, z) & Eq(y, 1), [x, y]) is None
    assert decide_zero_dimensional(Eq(1/x, 2) & Eq(y, 0), [x, y]) is None
    assert decide_zero_dimensional(Eq(x**4, 1) & Eq(y, 0), [x, y], max_dimension=3) is None


def test_invalid_input() -> None:
    raises(NotImplementedError, lambda: count_real_solutions([x*y], x, y))
    # not polynomials with rational coefficients
    raises(ValueError, lambda: count_real_solutions([x**2 - sqrt(Integer(2))], x))
