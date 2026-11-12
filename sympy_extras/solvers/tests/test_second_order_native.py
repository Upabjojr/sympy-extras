"""Tests of the native integrating factors and linearisations of second
order equations (:mod:`sympy_extras.solvers.second_order`) on objects of
the core: a first integral is checked by its total derivative along the
equation, a linearisation by the transformed equation, a solution by
substitution."""
from __future__ import annotations

from sympy_extras.core import Eq, Function, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Derivative
from sympy_extras.functions import exp
from sympy_extras.functions.elementary.exponential import Log
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers.lie import JetSpace, Symmetry
from sympy_extras.solvers.ode import check_ode_solution, ode_symmetries
from sympy_extras.solvers.second_order import (FirstIntegral, Linearization, commuting_pair, dsolve_second_order,
    integrating_factor_p, integrating_factor_xy, is_linearizable, linearize, rectify_symmetries, second_order_rhs,
    transformed_rhs)

x, y, p, v = Symbol('x'), Symbol('y'), Symbol('p'), Symbol('v')
f = Function('y')(x)


def _is_first_integral(R: FirstIntegral, Phi: Expr) -> bool:
    """``D_x R = 0`` along ``y'' = Phi``."""
    total = R.integral.diff(x) + R.integral.diff(y)*p + R.integral.diff(p)*Phi
    return simplify(total) == 0


def _residual(equation: Expr, value: Expr) -> Expr:
    replacements: dict[Basic, Basic] = {f: value}
    for d in equation.atoms(Derivative):
        if isinstance(d, Derivative) and d.expr == f:
            replacements[d] = value.diff(*d.variables)
    substituted = equation._xreplace_evaluated(replacements)
    assert isinstance(substituted, Expr)
    return simplify(substituted)


def test_second_order_rhs() -> None:
    Phi, x_, y_, p_ = second_order_rhs(x*f.diff(x, 2) + f.diff(x) - f**2, f)
    assert Phi.subs({y_: y, p_: p}) == (y**2 - p)/x and x_ == x
    for wrong in (f.diff(x, 2)**2 - f, f.diff(x, 3) - f):
        try:
            second_order_rhs(wrong, f)
        except ValueError:
            continue
        raise AssertionError(wrong)


def test_integrating_factor_xy() -> None:
    for Phi in (-p**2/y, -p/x, -p/x - y/x**2, -2*p/x, (p - y/x)/x):
        R = integrating_factor_xy(Phi, x, y, p)
        assert R is not None and _is_first_integral(R, Phi), Phi
    R = integrating_factor_xy(-p**2/y, x, y, p)
    assert R is not None and R.factor == y and R.integral == p*y
    assert R.equation(f) == Eq(f*f.diff(x), Symbol('C1'))
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


def test_lie_test_and_linearize() -> None:
    assert is_linearizable(p**2, x, y, p) and is_linearizable(-3*y*p - y**3, x, y, p)
    assert is_linearizable(-(p**2 + 1)/y, x, y, p)       # (y**2)'' = -2
    assert not is_linearizable(y**2, x, y, p) and not is_linearizable((p**2 + 1)/y, x, y, p)
    assert not is_linearizable(p**4, x, y, p)
    L = linearize(p**2, x, y, p)
    assert isinstance(L, Linearization) and L.t == x and L.u == -exp(-y) and L.rhs == 0
    # not fibre preserving in either orientation
    assert linearize(-3*y*p - y**3, x, y, p) is None
    assert linearize(y**2, x, y, p) is None


def test_rectify_symmetries() -> None:
    L = rectify_symmetries(-3*y*p - y**3, x, y, p, f)
    assert L is not None and L.t == x - 1/y and L.u == 1/(2*y**2) and L.rhs == 1
    assert transformed_rhs(-3*y*p - y**3, L.t, L.u, x, y, p, v) == 1
    found = ode_symmetries(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f, degree=3)
    pair = commuting_pair(found, x, found[0].jets.u[0])
    assert pair is not None
    bracket = pair[0].commutator(pair[1])
    assert all(simplify(c) == 0 for c in bracket.xi + bracket.eta)
    assert rectify_symmetries(y**2, x, y, p, f) is None


def test_commuting_pair_in_a_centralizer() -> None:
    # the translations d/dx, d/dy are no basis elements nor their sums
    # here: X1 = d/dx + d/dy, X2 = d/dx - d/dy, X3 = x d/dx + y d/dy
    jets = JetSpace([f])
    u = jets.u[0]
    basis = [Symmetry(jets, [1], [1]), Symmetry(jets, [1], [-1]), Symmetry(jets, [x], [u])]
    pair = commuting_pair(basis, x, u)
    assert pair is not None
    X, Y = pair
    bracket = X.commutator(Y)
    assert all(simplify(c) == 0 for c in bracket.xi + bracket.eta)
    assert simplify(X.xi[0]*Y.eta[0] - Y.xi[0]*X.eta[0]) != 0


def test_dsolve_second_order() -> None:
    for equation in (f.diff(x, 2) - f.diff(x)**2, f.diff(x, 2) + 3*f*f.diff(x) + f**3, f*f.diff(x, 2) + f.diff(x)**2,
                     x**2*f.diff(x, 2) + x*f.diff(x) + f):
        solutions = dsolve_second_order(equation, f)
        assert solutions, equation
        for solution in solutions:
            assert isinstance(solution, Eq) and solution.lhs == f and isinstance(solution.rhs, Expr)
            assert _residual(equation, solution.rhs) == 0, (equation, solution)
    [solution] = dsolve_second_order(f.diff(x, 2) - f.diff(x)**2, f) or []
    assert isinstance(solution, Eq) and solution.has(Log)
    # an implicit solution through the quadrature of the first integral
    equation = f.diff(x, 2) + f*f.diff(x)**2
    for solution in dsolve_second_order(equation, f) or []:
        # the relation erfi(y/sqrt(2)) = C1 x + C2 (SymPy's dsolve wrote it
        # with an Integral, which its solve gives no solution of) is not
        # solved for y: SymPy's solve gives up on it
        assert check_ode_solution(equation, solution, f, solve_for_func=False), solution
    assert dsolve_second_order(f.diff(x, 2) - f**2, f) is None
