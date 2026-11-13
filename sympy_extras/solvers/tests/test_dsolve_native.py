"""The classical hints of :mod:`sympy_extras.solvers.dsolve` on the core.

Every solution is checked by substitution into the equation (an implicit
one through its derivative), and the general solutions by their number
of constants; the expected forms are those of the textbooks (Tenenbaum
and Pollard), not the module's own output.
"""
from __future__ import annotations

from sympy_extras.core import Eq, Function, Integer, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, diff
from sympy_extras.functions import cos, exp, log, sin
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers.dsolve import (dsolve, explicit_solution, linear_basis, linear_coefficients,
                                         solution_parts)

x = Symbol('x')
y = Function('y')(x)


def _substituted(equation: Expr, f: AppliedUndef, rhs: Expr) -> Expr:
    value = equation.subs(f, rhs)
    assert isinstance(value, Expr)
    done = value.doit()
    assert isinstance(done, Expr)
    return simplify(done)


def _solves(equation: Expr, solution: Eq, f: AppliedUndef) -> bool:
    lhs, rhs = solution.lhs, solution.rhs
    assert isinstance(lhs, Expr) and isinstance(rhs, Expr)
    if lhs == f:
        return _substituted(equation, f, rhs) == 0
    # an implicit F(x, y) = G(x, y): y' = -F_x/F_y for F - G, substituted
    u = Symbol('u_')
    relation = (lhs - rhs).subs(f, u)
    assert isinstance(relation, Expr)
    slope = -diff(relation, x)/diff(relation, u)
    first = equation.subs(Derivative(f, x), slope.subs(u, f))
    assert isinstance(first, Expr)
    return simplify(first) == 0


def _constants(solution: Basic) -> int:
    return len([s for s in solution.free_symbols if isinstance(s, Symbol) and s.name.startswith('C')])


def _check_general(equation: Expr, order: int) -> list[Eq]:
    found = dsolve(equation, y)
    assert found, equation
    for s in found:
        assert _solves(equation, s, y), (equation, s)
        assert _constants(s) == order, (equation, s)
    return found


def test_constant_coefficients() -> None:
    # the characteristic roots: simple, repeated, conjugate pairs
    [s] = _check_general(y.diff(x, 2) + y, 2)
    assert s.rhs.has(sin(x)) and s.rhs.has(cos(x))
    [s] = _check_general(y.diff(x, 2) - 2*y.diff(x) + y, 2)
    assert {s.rhs.coeff(c) for c in (Symbol('C1'), Symbol('C2'))} == {exp(x), x*exp(x)}
    [s] = _check_general(y.diff(x, 2) - 2*y.diff(x) + 5*y, 2)
    assert exp(x)*sin(2*x) in {s.rhs.coeff(c) for c in (Symbol('C1'), Symbol('C2'))}
    _check_general(y.diff(x, 4) - y, 4)
    _check_general(y.diff(x, 3) - 3*y.diff(x, 2) + 3*y.diff(x) - y, 3)
    a = Symbol('a')
    _check_general(y.diff(x, 2) - a**2*y, 2)
    # the characteristic polynomial m**2 + m/k: the native Poly took no
    # 1/k as a coefficient and no solution was found
    k = Symbol('k')
    _check_general(k*y.diff(x, 2) + y.diff(x), 2)


def test_euler_equations() -> None:
    [s] = _check_general(x**2*y.diff(x, 2) - 2*y, 2)
    assert s.rhs.has(x**2) and s.rhs.has(1/x)
    # a repeated indicial root: x and x log(x)
    [s] = _check_general(x**2*y.diff(x, 2) - x*y.diff(x) + y, 2)
    assert {s.rhs.coeff(c) for c in (Symbol('C1'), Symbol('C2'))} == {x, x*log(x)}
    # complex indicial roots: cos(log(x)), sin(log(x))
    [s] = _check_general(x**2*y.diff(x, 2) + x*y.diff(x) + y, 2)
    assert s.rhs.has(sin(log(x)))
    _check_general(y.diff(x) - y/(2*x), 1)


def test_inhomogeneous_equations() -> None:
    # the forms of undetermined coefficients, by variation of parameters
    _check_general(y.diff(x, 2) + 4*y - sin(2*x), 2)
    _check_general(y.diff(x, 2) - 2*y.diff(x) + y - exp(2*x), 2)
    _check_general(y.diff(x, 2) - 1, 2)
    _check_general(x**2*y.diff(x, 2) - 2*y - x, 2)
    k = Symbol('k')
    _check_general(y.diff(x) - k*y - exp(x), 1)


def test_first_order_linear_and_reducible() -> None:
    _check_general((x**2 + 1)*y.diff(x) + 2*x*y, 1)
    _check_general(y.diff(x) + y/x - 1, 1)
    # no y: reduced to the lowest derivative
    _check_general(x*y.diff(x, 2) + 2*y.diff(x), 2)
    z = Symbol('z')
    F = Function('F')(z)
    # 2 F'' + z F' = 0: F' = exp(-z**2/4), F = erf
    found = dsolve(2*F.diff(z, 2) + z*F.diff(z), F)
    assert len(found) == 1 and _substituted(2*F.diff(z, 2) + z*F.diff(z), F, found[0].rhs) == 0


def test_first_order_nonlinear() -> None:
    # separable, explicit with both branches of a square
    found = _check_general(y*y.diff(x) - x, 1)
    assert len(found) == 2
    _check_general(y.diff(x) - y**2*sin(x), 1)
    # Bernoulli
    _check_general(y.diff(x) + y - x*y**3, 1)
    # homogeneous coefficients
    _check_general(y.diff(x) - (x + y)/x, 1)
    _check_general(x*y*y.diff(x) - x**2 - y**2, 1)
    # exact: (2 x y + 1) + (x**2 + 2 y) y' = 0, implicit
    _check_general(2*x*y + 1 + (x**2 + 2*y**3)*y.diff(x), 1)


def test_factorable() -> None:
    found = dsolve((y.diff(x) - y)*(y.diff(x) + y), y)
    assert len(found) == 2
    for s in found:
        assert _solves(y.diff(x) - y, s, y) or _solves(y.diff(x) + y, s, y)


def test_not_solved() -> None:
    # a nonlinear equation of second order which no hint covers
    assert dsolve(y.diff(x, 2) - y**3 - x, y) == []
    assert dsolve(y - x, y) == []


def test_helpers() -> None:
    assert linear_coefficients(x*y.diff(x, 2) + y - x**2, y) == ([Integer(1), Integer(0), x], x**2)
    assert linear_coefficients(y.diff(x) - y**2, y) is None
    basis = linear_basis([Integer(1), Integer(2), Integer(1)], x)
    assert basis is not None and len(basis) == 2
    t = Symbol('t')
    I = Function('I')(t)
    found = explicit_solution(I.diff(t) + I/t - 1, I)
    assert found is not None
    rhs, constants = found
    assert len(constants) == 1 and _substituted(I.diff(t) + I/t - 1, I, rhs) == 0
    parts = solution_parts(y.diff(x, 2) - y, y)
    assert parts is not None and len(parts) == 2
    assert all(_substituted(y.diff(x, 2) - y, y, p) == 0 for p in parts)
    # the constants avoid the symbols of the equation
    C1, C2 = Symbol('C1'), Symbol('C2')
    [s] = dsolve(y.diff(x) - C1*y, y)
    assert C1 in s.free_symbols and _constants(s) == 2 and C2 in s.free_symbols
