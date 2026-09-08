from __future__ import annotations

from sympy import Function, Symbol, symbols, exp, sqrt, simplify, Eq, checkodesol, Derivative, log, cos
from sympy.core.expr import Expr
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises

from sympy_extras.solvers.second_order import (second_order_rhs, integrating_factor_xy, integrating_factor_p,
    is_linearizable, linearize, rectify_symmetries, commuting_pair, dsolve_second_order, FirstIntegral)
from sympy_extras.solvers.ode import ode_symmetries

x, y, p = symbols('x y p')
f = Function('y')(x)


def _is_first_integral(R: FirstIntegral, Phi: Expr) -> bool:
    """``D_x R = 0`` along ``y'' = Phi``."""
    total = R.integral.diff(x) + R.integral.diff(y)*p + R.integral.diff(p)*Phi
    return simplify(total) == 0


def _from_linear(phi: Expr, psi: Expr) -> Expr:
    """``Phi`` of the equation obtained from ``u'' = 0`` by ``t = phi(x, y)``,
    ``u = psi(x, y)``."""
    T, U = phi.subs(y, f), psi.subs(y, f)
    ut = U.diff(x)/T.diff(x)
    utt = ut.diff(x)/T.diff(x)
    ypp = Symbol('ypp')
    [value] = solve(utt.subs(Derivative(f, (x, 2)), ypp), ypp)
    return simplify(value.subs(Derivative(f, x), p).subs(f, y))


def test_second_order_rhs() -> None:
    Phi, x_, y_, p_ = second_order_rhs(x*f.diff(x, 2) + f.diff(x) - f**2, f)
    assert Phi.subs({y_: y, p_: p}) == (y**2 - p)/x and x_ == x
    raises(ValueError, lambda: second_order_rhs(f.diff(x, 2)**2 - f, f))
    raises(ValueError, lambda: second_order_rhs(f.diff(x, 3) - f, f))


def test_integrating_factor_xy() -> None:
    for Phi in (-p**2/y, -p/x, -p/x - y/x**2, -2*p/x, (p - y/x)/x):
        R = integrating_factor_xy(Phi, x, y, p)
        assert R is not None and _is_first_integral(R, Phi), Phi
    R = integrating_factor_xy(-p**2/y, x, y, p)
    assert R is not None and R.factor == y and R.integral == p*y
    assert integrating_factor_xy(y**2, x, y, p) is None
    assert integrating_factor_xy(p**3, x, y, p) is None


def test_integrating_factor_p() -> None:
    for Phi in (p**3*(x + y*p), (x + y*p)/p, exp(-p)*(y + x*p)):
        R = integrating_factor_p(Phi, x, y, p)
        assert R is not None and _is_first_integral(R, Phi), Phi
    R = integrating_factor_p(p**3*(x + y*p), x, y, p)
    assert R is not None and R.factor == p**(-3)
    # A_y != B_x
    assert integrating_factor_p(p**3*(y**2 + x*p), x, y, p) is None


def test_lie_test() -> None:
    assert is_linearizable(p**2, x, y, p)
    assert is_linearizable(-3*y*p - y**3, x, y, p)
    assert is_linearizable(_from_linear(x, exp(y) + x**2*y), x, y, p)
    assert is_linearizable(_from_linear(x + y**2, y*x), x, y, p)
    assert is_linearizable(_from_linear(x*y, x + y**3), x, y, p)
    assert not is_linearizable(y**2, x, y, p)
    assert is_linearizable(-(p**2 + 1)/y, x, y, p)       # (y**2)'' = -2
    assert not is_linearizable((p**2 + 1)/y, x, y, p)
    assert not is_linearizable(p**4, x, y, p)


def test_linearize() -> None:
    L = linearize(p**2, x, y, p)
    assert L is not None and L.t == x and L.u == -exp(-y) and L.rhs == 0
    Phi = _from_linear(x, exp(y) + x**2*y)
    L = linearize(Phi, x, y, p)
    assert L is not None and L.t == x and simplify(L.u - (exp(y) + x**2*y)) == 0
    # not fibre preserving in either orientation
    assert linearize(-3*y*p - y**3, x, y, p) is None
    assert linearize(y**2, x, y, p) is None


def test_rectify_symmetries() -> None:
    L = rectify_symmetries(-3*y*p - y**3, x, y, p, f)
    assert L is not None and L.t == x - 1/y and L.u == 1/(2*y**2) and L.rhs == 1
    found = ode_symmetries(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f, degree=3)
    pair = commuting_pair(found, x, found[0].jets.u[0])
    assert pair is not None
    bracket = pair[0].commutator(pair[1])
    assert all(simplify(c) == 0 for c in bracket.xi + bracket.eta)
    assert rectify_symmetries(y**2, x, y, p, f) is None


def test_dsolve_second_order() -> None:
    cases = [f.diff(x, 2) - f.diff(x)**2, f.diff(x, 2) + 3*f*f.diff(x) + f**3, f*f.diff(x, 2) + f.diff(x)**2,
             x**2*f.diff(x, 2) + x*f.diff(x) + f, f.diff(x, 2) + f*f.diff(x)**2]
    for equation in cases:
        solutions = dsolve_second_order(equation, f)
        assert solutions is not None, equation
        for solution in solutions:
            assert isinstance(solution, Eq)
            assert checkodesol(equation, solution)[0] is True, (equation, solution)
    [solution] = dsolve_second_order(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f) or []
    assert isinstance(solution, Eq) and solution.rhs.subs({Symbol('C1'): 0, Symbol('C2'): 0}) == 2/x
    [solution] = dsolve_second_order(f.diff(x, 2) - f.diff(x)**2, f) or []
    assert isinstance(solution, Eq) and solution.rhs.has(log)
    assert dsolve_second_order(f.diff(x, 2) - f**2, f) is None
    assert sqrt(4) == 2 and cos(0) == 1
