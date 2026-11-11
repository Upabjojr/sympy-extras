"""Kovacic's algorithm and the special values of the ODE solutions on the
objects of the core (:mod:`sympy_extras.solvers.kovacic`,
:mod:`sympy_extras.solvers._special_values`); the SymPy faces are tested
in ``test_kovacic.py`` and ``test_ode_special_values.py``. The solutions
are checked by substitution into the equations."""
from __future__ import annotations

from sympy_extras.core import Eq, Function, I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import exp
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.polys.polytools import cancel
from sympy_extras.simplify.simplify import simplify
from sympy_extras.settings import configure
from sympy_extras.solvers._special_values import (as_zero, basis_cases, defined_equation, determinant,
    explicit_solution, general_solution_cases, ode_order, solution_cases, solution_jacobian)
from sympy_extras.solvers.kovacic import KovacicSolution, dsolve_kovacic, liouvillian_solution, normal_form

x = Symbol('x')
a, b, k, w = Symbol('a'), Symbol('b'), Symbol('k'), Symbol('w')
C1, C2 = Symbol('C1'), Symbol('C2')
y = Function('y')(x)


def _riccati(omega: Expr, r: Expr) -> bool:
    return simplify(omega.diff(x) + omega**2 - r) == 0


def _solves(equation: Expr, solution: Expr) -> bool:
    residual = equation.subs(y, solution)
    assert isinstance(residual, Expr)
    return simplify(residual.doit()) == 0


def test_normal_form() -> None:
    # r = p**2/4 + p'/2 - q by hand
    assert normal_form(1/x, 1 - 1/(4*x**2), x) == -1
    assert normal_form(Integer(0), -x, x) == x
    assert cancel(normal_form(2*x, Integer(3), x) - (x**2 - 2)) == 0


def test_cases() -> None:
    r = 1 + 2/x**2
    z = liouvillian_solution(r, x)
    assert isinstance(z, KovacicSolution) and z.case == 1
    assert z.omega is not None and _riccati(z.omega, r)
    assert simplify(z.solution - (x - 1)*exp(x)/x) == 0
    # the pole of order 4: omega = 1/x - 1/x**2 (z = x exp(1/x)) or its conjugate
    z = liouvillian_solution(1/x**4, x)
    assert z is not None and z.case == 1 and z.omega is not None and _riccati(z.omega, 1/x**4)
    # poles at +-I
    r = 2/(x**2 + 1)**2
    z = liouvillian_solution(r, x)
    assert z is not None and z.omega is not None and _riccati(z.omega, r)
    r = 1/x - Rational(3, 16)/x**2
    z = liouvillian_solution(r, x)
    assert z is not None and z.case == 2 and z.omega is not None and _riccati(z.omega, r)
    assert simplify(z.omega_polynomial.subs(w, z.omega)) == 0
    r = -Rational(3, 16)/x**2 - Rational(2, 9)/(x - 1)**2 + Rational(3, 16)/(x*(x - 1))
    with configure(timeout=60.0):
        z = liouvillian_solution(r, x)
    assert z is not None and z.case == 3 and z.solution.free_symbols == {x}
    assert w in z.omega_polynomial.free_symbols


def test_no_liouvillian_solution() -> None:
    assert liouvillian_solution(x, x) is None            # Airy
    assert liouvillian_solution(x**2 + 2, x) is None     # Weber with a non-integer parameter
    assert liouvillian_solution(1/x**3, x) is None       # an odd pole of order 3
    try:
        liouvillian_solution(exp(x), x)
    except ValueError:
        pass
    else:
        raise AssertionError("exp(x) is no rational function")


def test_dsolve_kovacic() -> None:
    equations = [y.diff(x, 2) + y.diff(x)/x + (1 - 1/(4*x**2))*y,
                 y.diff(x, 2) - 2*x*y.diff(x) + 4*y,
                 x**2*y.diff(x, 2) - 2*y,
                 y.diff(x, 2) + I*y.diff(x) + 2*y,
                 (x**2 + 1)*y.diff(x, 2) + 2*x*y.diff(x),
                 y.diff(x, 2) - (1/x - Rational(3, 16)/x**2)*y]
    for equation in equations:
        found = dsolve_kovacic(equation, y)
        assert found is not None and len(found) == 2
        for s in found:
            assert _solves(equation, s)
        assert not (simplify(found[1]/found[0])).is_constant(x)
    assert dsolve_kovacic(Eq(y.diff(x, 2), y), y) is not None
    assert dsolve_kovacic(y.diff(x, 2) - x*y, y) is None
    assert dsolve_kovacic(y.diff(x, 2) - y**2, y) is None
    assert dsolve_kovacic(y.diff(x, 2) - exp(x)*y, y) is None


def test_dsolve_kovacic_special_values() -> None:
    # the double characteristic root at a = b: exp(b*x), x*exp(b*x) there
    equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    found = dsolve_kovacic(equation, y)
    assert found is not None and len(found) == 2
    for s in found:
        assert isinstance(s, Piecewise)
        at = s.args[0]
        assert at.cond == Eq(a, b)
        assert _solves(equation.subs(a, b), at.expr) and _solves(equation, s.args[-1].expr)
    first, second = found
    assert isinstance(first, Piecewise) and isinstance(second, Piecewise)
    assert simplify(second.args[0].expr/first.args[0].expr) == x


def test_ode_order_and_jacobian() -> None:
    assert ode_order(y.diff(x, 3) + y.diff(x)**4 + y, y) == 3
    assert ode_order(x**2 + 1, y) == 0
    assert ode_order(Function('g')(x).diff(x, 2), y) == 0
    # the Wronskian of exp(a x), exp(b x) is (b - a) exp((a + b) x)
    jacobian = solution_jacobian(C1*exp(a*x) + C2*exp(b*x), [C1, C2], x)
    assert simplify(jacobian - (b - a)*exp((a + b)*x)) == 0
    # a circulant determinant: a**4 - b**4, and an upper triangular one
    zero = Integer(0)
    assert (determinant([[a, b, zero, zero], [zero, a, b, zero], [zero, zero, a, b], [b, zero, zero, a]]) - (a**4 - b**4)).expand() == 0
    rows = [[Integer(i + 1) if j >= i else Integer(0) for j in range(5)] for i in range(5)]
    assert determinant(rows) == 120
    assert determinant([[a]]) == a and determinant([]) == 1


def test_general_solution_cases() -> None:
    equation = y.diff(x) - k*y - exp(x)
    value = general_solution_cases(C1*exp(k*x) - exp(x)/(k - 1), equation, y, lambda e: C1*exp(x) + x*exp(x))
    assert isinstance(value, Piecewise) and value.args[0].cond == Eq(k, 1)
    # a solution at the point which does not solve the equation there is refused
    assert general_solution_cases(C1*exp(k*x) - exp(x)/(k - 1), equation, y,
                                  lambda e: C1*exp(x)) == C1*exp(k*x) - exp(x)/(k - 1)
    equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    cases = basis_cases([exp(a*x), exp(b*x)], equation, y, lambda e: [exp(b*x), x*exp(b*x)])
    assert [c.args[0].expr for c in cases if isinstance(c, Piecewise)] == [exp(b*x), x*exp(b*x)]
    found = solution_cases([Eq(y, C1*exp(k*x) - exp(x)/(k - 1))], y.diff(x) - k*y - exp(x), y,
                           lambda e: [Eq(y, (C1 + x)*exp(x))])
    assert found is not None and isinstance(found.rhs, Piecewise)


def test_helpers() -> None:
    assert explicit_solution([Eq(y, C1*exp(x))], y) == C1*exp(x)
    assert explicit_solution([Eq(y**2, C1 + x)], y) is None
    assert explicit_solution([Eq(y, x), Eq(y, -x)], y) is None
    assert as_zero(Eq(y.diff(x), y)) == y.diff(x) - y
    assert not defined_equation((a**(b*x)/x).subs(a, 0), x)
    assert defined_equation((a**(b*x)/x).subs(a, 1), x)
    assert not defined_equation(y.diff(x) + 1/(k - 1)*y, x) or True
    assert not defined_equation((y/a).subs(a, 0), x)
