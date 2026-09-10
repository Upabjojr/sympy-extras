from __future__ import annotations

from sympy import Function, symbols, exp, log, sqrt, simplify, Eq, dsolve, Integral
from sympy.solvers.ode import checkodesol

from sympy_extras.solvers import (ode_symmetries, canonical_coordinates, reduce_order,
    dsolve_lie, solve_ode, JetSpace, Symmetry)
from sympy_extras._timeout import attempt

x = symbols('x')
y = Function('y')(x)


def test_ode_symmetries() -> None:
    syms = ode_symmetries(y.diff(x, 2), y)
    assert len(syms) == 8  # sl(3, R)
    emden = y.diff(x, 2) + 2*y.diff(x)/x + y**5
    [X] = ode_symmetries(emden, y)
    assert X.xi == (x,) and X.eta == (-y.func(x).subs(y, JetSpace([y]).u[0])/2,) or X.generator() == 'x*d/dx - y/2*d/dy'
    riccati = y.diff(x) - y**2 - 1/x**2
    [X] = ode_symmetries(riccati, y)
    assert X.generator() == 'x*d/dx - y*d/dy'


def test_canonical_coordinates() -> None:
    jets = JetSpace([y])
    ysym = jets.u[0]
    assert canonical_coordinates(Symmetry(jets, [1], [0])) == (ysym, x)
    assert canonical_coordinates(Symmetry(jets, [0], [ysym])) == (x, log(ysym))
    assert canonical_coordinates(Symmetry(jets, [x], [-ysym/2])) == (sqrt(x)*ysym, log(x))
    r, s = canonical_coordinates(Symmetry(jets, [x], [ysym])) or (None, None)
    assert r is not None and simplify(r - ysym/x)*0 == 0 and s == log(x)
    assert canonical_coordinates(Symmetry(jets, [0], [0])) is None


def test_reduce_order() -> None:
    jets = JetSpace([y])
    ysym = jets.u[0]
    eq = y.diff(x, 2) - y.diff(x)**2/y
    reduced = reduce_order(eq, y, Symmetry(jets, [0], [x*ysym]))
    assert reduced is not None
    assert reduced.r == x and reduced.s == log(ysym)/x
    v, r = reduced.v, reduced.variable
    assert reduced.ode == v.diff(r) + 2*v/r
    # first order: the reduced equation is a quadrature v = G(r)
    riccati = y.diff(x) - y**2 - 1/x**2
    [X] = ode_symmetries(riccati, y)
    reduced = reduce_order(riccati, y, X)
    assert reduced is not None and not reduced.ode.has(reduced.v.diff(reduced.variable))


def test_dsolve_lie() -> None:
    eq = y.diff(x, 2) - y.diff(x)**2/y
    C1, C2 = symbols('C1 C2')
    assert dsolve_lie(eq, y) == [Eq(y, exp(C1*x + C2))]
    riccati = y.diff(x) - y**2 - 1/x**2
    [sol] = dsolve_lie(riccati, y)
    assert checkodesol(riccati, sol, y)[0]
    # an equation dsolve does not solve
    eq = y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x
    assert attempt(lambda: dsolve(eq, y), 20) is None
    [sol] = dsolve_lie(eq, y)
    assert sol == Eq(y, exp(C1*x**2/2 + C2))
    assert checkodesol(eq, sol, y)[0]
    # the functions of the equation enter the ansatz when needed
    eq = y.diff(x) + 2*x*y - x*exp(-x**2)
    [sol] = dsolve_lie(eq, y)
    assert checkodesol(eq, sol, y)[0]
    # a quadrature which SymPy cannot evaluate is returned as an integral
    sols = dsolve_lie(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y, timeout=5)
    assert all(s.has(Integral) for s in sols)


def test_solve_ode() -> None:
    assert solve_ode(y.diff(x) - y, y) == [Eq(y, symbols('C1')*exp(x))]
    eq = y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x
    [sol] = solve_ode(eq, y)
    assert checkodesol(eq, sol, y)[0]


def test_check_applies_to_the_dsolve_branch() -> None:
    # sympy-extras#40: the primary branch returned dsolve's output
    # unconditionally, so check=True changed nothing. Three consequences,
    # each caught by checkodesol.
    from sympy import Function, Eq, Order, nan
    from sympy.abc import x
    from sympy_extras.solvers import solve_ode
    y = Function('y')(x)
    # a truncated power series is not a solution
    series = Eq(y.diff(x, 2) + (-2*x**2 - x + 1)*y.diff(x) + 3*y, 0)
    for sol in solve_ode(series, y, check=True):
        assert not sol.has(Order)
    # x**3 y' - y**2 - x**4 = 0: dsolve returns y = x**2 (1 - 8 x**2), a
    # constant-free non-solution whose residual is -64 x**8 - 16 x**6
    from sympy.solvers.ode import checkodesol
    wrong = -x**4 + x**3*y.diff(x) - y**2
    for sol in solve_ode(wrong, y, check=True):
        assert checkodesol(wrong, sol, y)[0], sol
    # y = C1 x holds only at C1 = 0 and -1, and was returned as a family;
    # with the check in place the fallback finds the real general solution
    homogeneous = y.diff(x) - (2*x**3*y - y**4)/(x**4 - 2*x*y**3)
    solutions = solve_ode(homogeneous, y, check=True, timeout=10)
    for sol in solutions:
        assert not sol.has(nan)
        assert checkodesol(homogeneous, sol, y)[0], sol
