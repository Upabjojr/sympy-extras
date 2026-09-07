from __future__ import annotations

from sympy import Function, exp, sqrt, simplify, Eq, Rational, cancel, S
from sympy.abc import x
from sympy.testing.pytest import raises

from sympy_extras.solvers.linear_ode import (LinearOperator, polynomial_solutions, rational_solutions,
    hyperexponential_solutions, reduce_order_linear, dsolve_linear)

y = Function('y')(x)


def _solves(L: LinearOperator, solution: object) -> bool:
    from sympy_extras._typing import as_expr
    return simplify(L(as_expr(solution))) == 0


def test_operator() -> None:
    L = LinearOperator.from_equation(Eq(x*y.diff(x, 2), (x + 2)*y.diff(x) - 2*y), y)
    assert L.coefficients == [2, -x - 2, x] and L.order == 2
    assert L(x**2 + 2*x + 2) == 0
    L = LinearOperator.from_equation(y.diff(x, 2) + y.diff(x)/x + y, y)
    assert L.coefficients == [x, 1, x]
    raises(ValueError, lambda: LinearOperator.from_equation(y.diff(x) - y**2, y))
    raises(ValueError, lambda: LinearOperator.from_equation(y.diff(x) - exp(x)*y, y))
    T = L.transformed(1/x)
    assert all(c.is_polynomial(x) for c in T.coefficients)


def test_polynomial_solutions() -> None:
    L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    assert polynomial_solutions(L) == [x**2 + 2*x + 2]
    # Legendre's equation with n = 3
    L = LinearOperator.from_equation((1 - x**2)*y.diff(x, 2) - 2*x*y.diff(x) + 12*y, y)
    [p] = polynomial_solutions(L)
    assert cancel(p/(5*x**3 - 3*x)).is_constant(x)
    # a two dimensional space
    L = LinearOperator.from_equation(y.diff(x, 3), y)
    assert len(polynomial_solutions(L)) == 3
    assert polynomial_solutions(LinearOperator([S(-2), S.Zero, S.One], x)) == []
    assert polynomial_solutions(LinearOperator.from_equation(y.diff(x) - y, y)) == []


def test_rational_solutions() -> None:
    L = LinearOperator.from_equation(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    assert set(rational_solutions(L)) == {1/x, 1/x**2}
    L = LinearOperator.from_equation((x**2 + 1)*y.diff(x, 2) + 2*x*y.diff(x) - 2*y, y)
    # solutions x and x*atan(x) + 1: only x is rational
    assert rational_solutions(L) == [x]
    L = LinearOperator.from_equation((x - 1)**2*y.diff(x, 2) + (x - 1)*y.diff(x) - y, y)
    found = rational_solutions(L)
    assert len(found) == 2 and all(_solves(L, s) for s in found)
    # denominators with an irreducible quadratic factor
    L = LinearOperator.from_equation((x**2 + 1)**2*y.diff(x, 2) + 4*x*(x**2 + 1)*y.diff(x) + 2*(x**2 + 1)*y + 0*y, y)
    found = rational_solutions(L)
    assert all(_solves(L, s) for s in found)


def test_hyperexponential_solutions() -> None:
    L = LinearOperator.from_equation(4*x**2*y.diff(x, 2) + 4*x*y.diff(x) - y, y)
    assert set(hyperexponential_solutions(L)) == {sqrt(x), 1/sqrt(x)}
    # Euler equation with exponents 1/3 and -1
    L = LinearOperator.from_equation(3*x**2*y.diff(x, 2) + 5*x*y.diff(x) - y, y)
    found = hyperexponential_solutions(L)
    assert len(found) == 2 and all(_solves(L, s) for s in found)
    # the hypergeometric equation with a, b, c = 1/2, 1/2, 3/2: (1-x^2) solutions
    L = LinearOperator.from_equation(x*(1 - x)*y.diff(x, 2) + (Rational(3, 2) - 3*x)*y.diff(x) - y/4, y)
    found = hyperexponential_solutions(L)
    assert all(_solves(L, s) for s in found)
    # not Fuchsian: nothing is attempted
    assert hyperexponential_solutions(LinearOperator.from_equation(y.diff(x, 2) - x*y, y)) == []


def test_reduce_order() -> None:
    L = LinearOperator.from_equation(y.diff(x, 2) - 3*y.diff(x) + 2*y, y)
    M = reduce_order_linear(L, exp(x))
    assert M.order == 1 and M.coefficients == [-1, 1]
    raises(ValueError, lambda: reduce_order_linear(L, x))


def test_dsolve_linear() -> None:
    found = dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    assert found == [x**2 + 2*x + 2, exp(x)]
    found = dsolve_linear(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    assert set(found) == {1/x, 1/x**2}
    L = LinearOperator.from_equation((x**2 + 1)*y.diff(x, 2) + 2*x*y.diff(x) - 2*y, y)
    found = dsolve_linear((x**2 + 1)*y.diff(x, 2) + 2*x*y.diff(x) - 2*y, y)
    assert len(found) == 2 and all(_solves(L, s) for s in found)
    # third order with a rational and an exponential solution: reduction of order
    L = LinearOperator.from_equation(x*y.diff(x, 3) - (x + 3)*y.diff(x, 2) + 3*y.diff(x) - 0*y, y)
    found = dsolve_linear(x*y.diff(x, 3) - (x + 3)*y.diff(x, 2) + 3*y.diff(x), y)
    assert len(found) == 3 and all(_solves(L, s) for s in found)
    # Bessel of order 1/2 through Kovacic
    L = LinearOperator.from_equation(x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - Rational(1, 4))*y, y)
    found = dsolve_linear(x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - Rational(1, 4))*y, y, use_dsolve=False)
    assert len(found) == 2 and all(_solves(L, s) for s in found)
    # Airy: no Liouvillian solution and dsolve's special functions
    found = dsolve_linear(y.diff(x, 2) - x*y, y, use_dsolve=False)
    assert found == []
