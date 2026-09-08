from __future__ import annotations

from sympy import Function, Symbol, Eq, N, Rational, Tuple, Basic, sqrt
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras.solvers.first_order import chini_ode, abel_ode, lagrange_ode, dsolve_first_order
from sympy_extras._typing import as_expr

x = Symbol('x')
y = Function('y')(x)


def _implicit_holds(equation: Expr, solution: Basic, points: tuple[float, ...] = (0.6, 1.1)) -> bool:
    """Differentiate the implicit solution, replace y' by its value from
    the equation, and check numerically at points (x, y) on the curve
    for the constant fixed by the point."""
    assert isinstance(solution, Eq)
    relation = as_expr(solution.lhs - solution.rhs)
    C1 = Symbol('C1')
    dy = Function('y')(x).diff(x)
    from sympy.solvers.solvers import solve
    [rhs] = solve(equation, dy)
    total = relation.diff(x).subs(dy, rhs)
    for x0 in points:
        for y0 in (0.4, 0.9):
            value = total.subs({y: y0}).subs(x, x0)
            value = value.subs(C1, 0)
            if abs(N(value, 20)) > 1e-10:
                return False
    return True


def test_chini() -> None:
    eq = y.diff(x) - y**3 - x**Rational(-3, 2)
    solution = chini_ode(eq, y)
    assert solution is not None and _implicit_holds(eq, solution)
    eq = y.diff(x) - x**2*y**3 + x**2
    solution = chini_ode(eq, y)
    assert solution is not None and _implicit_holds(eq, solution)
    # with a linear term
    eq = y.diff(x) - y**3 - y - 1
    assert chini_ode(eq, y) is not None
    # not a constant invariant
    assert chini_ode(y.diff(x) - y**3 - x, y) is None
    assert chini_ode(y.diff(x) - y**2*x - y**3, y) is None


def test_abel() -> None:
    eq = y.diff(x) - y**3 - 3*y**2 - 3*y
    solution = abel_ode(eq, y)
    assert solution is not None and _implicit_holds(eq, solution)
    eq = y.diff(x) - x**2*(y**3 + 3*y**2 + 3*y + 1) - x**2
    solution = abel_ode(eq, y)
    assert solution is not None and _implicit_holds(eq, solution)
    # second kind: y y' = 3 y**2 + 3 y + 1 becomes w' = -w**3 - 3 w**2 - 3 w for w = 1/y
    eq = y*y.diff(x) - 3*y**2 - 3*y - 1
    solution = abel_ode(eq, y)
    assert solution is not None and _implicit_holds(eq, solution)
    assert abel_ode(y.diff(x) - y**3 - x, y) is None
    raises(ValueError, lambda: abel_ode(y.diff(x, 2) - y**3, y))


def test_lagrange() -> None:
    solution = lagrange_ode(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
    assert solution is not None and len(solution) == 2
    p = Symbol('p')
    first, second = solution
    assert isinstance(first, Eq) and isinstance(second, Eq)
    X, Y = as_expr(first.rhs), as_expr(second.rhs)
    # y = 2 x p + p**2 along the curve and dy/dx = p
    assert (Y - 2*X*p - p**2).simplify() == 0
    assert (Y.diff(p)/X.diff(p) - p).simplify() == 0
    solution = lagrange_ode(Eq(y, x*y.diff(x)**2 + y.diff(x)), y)
    assert solution is not None
    assert lagrange_ode(Eq(y, x*y.diff(x) + y.diff(x)**2), y) is None    # Clairaut: SymPy's
    assert lagrange_ode(y.diff(x) - y**2, y) is None
    result = dsolve_first_order(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
    assert isinstance(result, Tuple)
    # Chini's equation with n = 1/2 (constant invariant)
    eq = y.diff(x) - sqrt(y) - x
    chini = dsolve_first_order(eq, y)
    assert isinstance(chini, Eq) and _implicit_holds(eq, chini)
    assert dsolve_first_order(y.diff(x) - sqrt(y) - x**2, y) is None
