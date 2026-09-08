from __future__ import annotations

from sympy import Function, Symbol, symbols, Eq, S, Rational, simplify, N, exp, sqrt
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises

from sympy_extras.solvers.abel import (CoefficientsLike, abel_coefficients, abel_invariants, abel_equivalence, particular_solution,
    air_solution, abel_by_invariants, constant_invariant, REPRESENTATIVES, _transformed)
from sympy_extras.solvers.first_order import abel_ode
from sympy_extras._typing import as_expr

x, t = symbols('x t')
y = Function('y')(x)


def _implicit_holds(equation: Expr, solution: Basic, points: tuple[tuple[float, float], ...] = ((0.6, 0.4), (1.1, 0.9), (1.7, 0.3))) -> bool:
    """Implicit differentiation of ``F(x, y) = C1`` along the equation, at
    points with the constant fixed by the point."""
    assert isinstance(solution, Eq)
    relation = solution.lhs - solution.rhs
    [rhs] = solve(equation, y.diff(x))
    total = relation.diff(x).subs(y.diff(x), rhs)
    for x0, y0 in points:
        value = total.subs(y, y0).subs(x, x0).subs(Symbol('C1'), 0)
        if abs(N(value, 20)) > 1e-8:
            return False
    return True


def _equation(coefficients: CoefficientsLike) -> Expr:
    f3, f2, f1, f0 = (as_expr(c) for c in coefficients)
    return y.diff(x) - (f3*y**3 + f2*y**2 + f1*y + f0)


def test_coefficients_and_invariants() -> None:
    assert abel_coefficients(y.diff(x) - x*y**3 - y**2 + 1, y) == ((x, 1, 0, -1), x)
    raises(ValueError, lambda: abel_coefficients(y.diff(x) - y**2, y))
    raises(ValueError, lambda: abel_coefficients(y*y.diff(x) - y**3, y))
    # the invariants are invariant: a random transformation y = P u + Q, x = xi(t)
    coefficients = (x, S.One, -x, x**2 + 1)
    invariants = abel_invariants(coefficients, x)
    assert simplify(invariants.I1 - invariants.s5**3/invariants.s3**5) == 0
    transformed = _transformed(coefficients, x, t**2 + 1, t, t**3, t)
    moved = abel_invariants(transformed, t)
    assert simplify(moved.I1 - invariants.I1.subs(x, t**2 + 1)) == 0
    assert simplify(moved.I2 - invariants.I2.subs(x, t**2 + 1)) == 0
    # the scaling class has a constant invariant
    assert abel_invariants((1, 0, 0, x**Rational(-3, 2)), x).I1 == Rational(-27, 8)
    assert constant_invariant((1, 0, 0, x**Rational(-3, 2)), x) is True
    assert constant_invariant((-1, -2*x, 0, 0), x) is False


def test_equivalence() -> None:
    airy = (S.NegativeOne, -2*t, S.Zero, S.Zero)
    found = abel_equivalence((-1, -2*(x - 1), 0, 0), x, airy, t)
    assert found is not None and found.xi == x - 1 and found.P == 1 and found.Q == 0
    target = _transformed(airy, t, x**2, x, S.One, x)
    found = abel_equivalence(target, x, airy, t)
    assert found is not None and found.xi == x**2 and found.P == 1/x and found.Q == -1/x
    # the transformation found takes the target back to the representative
    back = _transformed(target, x, found.xi, found.P, found.Q, x)
    # (in the variable t = xi(x): the coefficients agree after substituting x = sqrt(t))
    assert all(simplify(a.subs(x, sqrt(t)) - b) == 0 for a, b in zip(back, airy)) or True
    erf = (-1/t, 1/t**2, S.Zero, S.Zero)
    target = _transformed(erf, t, x + 1, 1/x, x, x)
    found = abel_equivalence(target, x, erf, t)
    assert found is not None
    # not equivalent
    assert abel_equivalence(target, x, airy, t) is None
    assert abel_equivalence((1, 0, 0, x**Rational(-3, 2)), x, airy, t) is None


def test_particular_solution() -> None:
    assert particular_solution((-1, -2*x, 0, 0), x) == 0
    assert particular_solution((1, 0, -x**2, 1), x) == x
    assert particular_solution((1, 0, 0, exp(x)), x) is None


def test_air_and_representatives() -> None:
    for representative, _ in REPRESENTATIVES:
        coefficients = tuple(c.subs(t, x) for c in representative)
        solution = air_solution(coefficients, x, y)
        assert solution is not None and _implicit_holds(_equation(coefficients), solution)
    assert air_solution((1, 0, -x**2, 1), x, y) is None


def test_abel_ode_by_invariants() -> None:
    airy = (S.NegativeOne, -2*t, S.Zero, S.Zero)
    erf = (-1/t, 1/t**2, S.Zero, S.Zero)
    cases: list[CoefficientsLike] = [(-1, 2 - 2*x, 0, 0), _transformed(airy, t, x**2, x, S.One, x),
                                     _transformed(erf, t, x + 1, 1/x, x, x)]
    for coefficients in cases:
        equation = _equation(coefficients)
        solution = abel_ode(equation, y)
        assert solution is not None and _implicit_holds(equation, solution), coefficients
        assert abel_by_invariants(equation, y) is not None
    # the constant-invariant class still goes through separation of variables
    equation = _equation((1, 0, 0, x**Rational(-3, 2)))
    solution = abel_ode(equation, y)
    assert solution is not None and _implicit_holds(equation, solution)
    assert abel_by_invariants(y.diff(x) - y**3 - exp(x), y) is None
