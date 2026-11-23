"""Tests of the native symmetry method for ODEs
(:mod:`sympy_extras.solvers.ode`) on objects of the core; every solution
is checked by substitution into the equation (the residual simplified by
the native ``simplify``), not against the printed form."""
from __future__ import annotations

from sympy_extras.core import Eq, Function, Integer, Rational, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative
from sympy_extras.functions import besselj, cos, exp, hyper, log, sin, sqrt
from sympy_extras.settings import configure
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers.lie import JetSpace, Symmetry
from sympy_extras.solvers.linear_ode import has_integral
from sympy_extras.solvers.ode import (ReducedODE, canonical_coordinates, check_ode_solution, dsolve_lie,
    ode_symmetries, reduce_order, solve_ode)

x, k = Symbol('x'), Symbol('k')
y = Function('y')(x)


def _residual(equation: Expr, f: AppliedUndef, value: Expr) -> Expr:
    """The equation with ``f = value`` substituted, simplified."""
    replacements: dict[Basic, Basic] = {f: value}
    for d in equation.atoms(Derivative):
        if isinstance(d, Derivative) and d.expr == f:
            replacements[d] = value.diff(*d.variables)
    substituted = equation._xreplace_evaluated(replacements)
    assert isinstance(substituted, Expr)
    return simplify(substituted)


def _solves(equation: Expr, solution: Eq) -> bool:
    assert solution.lhs == y and isinstance(solution.rhs, Expr) and not solution.rhs.has(y), solution
    return _residual(equation, y, solution.rhs) == 0


def test_ode_symmetries() -> None:
    assert len(ode_symmetries(y.diff(x, 2), y)) == 8  # sl(3, R)
    [X] = ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y)
    assert X.generator() == 'x*d/dx - y/2*d/dy'
    [X] = ode_symmetries(y.diff(x) - y**2 - 1/x**2, y)
    assert X.generator() == 'x*d/dx - y*d/dy'


def test_canonical_coordinates() -> None:
    jets = JetSpace([y])
    u = jets.u[0]
    for xi, eta in ((Integer(1), Integer(0)), (Integer(0), u), (x, -u/2), (x, u)):
        found = canonical_coordinates(Symmetry(jets, [xi], [eta]))
        assert found is not None, (xi, eta)
        r, s = found
        # X r = 0 and X s = 1
        assert simplify(xi*r.diff(x) + eta*r.diff(u)) == 0
        assert simplify(xi*s.diff(x) + eta*s.diff(u) - 1) == 0
    assert canonical_coordinates(Symmetry(jets, [x], [-u/2])) == (sqrt(x)*u, log(x))
    assert canonical_coordinates(Symmetry(jets, [0], [0])) is None


def test_reduce_order() -> None:
    jets = JetSpace([y])
    u = jets.u[0]
    reduced = reduce_order(y.diff(x, 2) - y.diff(x)**2/y, y, Symmetry(jets, [0], [x*u]))
    assert isinstance(reduced, ReducedODE)
    assert reduced.r == x and reduced.s == log(u)/x
    v, r = reduced.v, reduced.variable
    assert reduced.ode == v.diff(r) + 2*v/r
    # first order: the reduced equation is a quadrature v = G(r)
    riccati = y.diff(x) - y**2 - 1/x**2
    [X] = ode_symmetries(riccati, y)
    reduced = reduce_order(riccati, y, X)
    assert reduced is not None and not reduced.ode.has(reduced.v.diff(reduced.variable))


def test_dsolve_lie() -> None:
    for equation in (y.diff(x, 2) - y.diff(x)**2/y, y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x,
                     y.diff(x) + 2*x*y - x*exp(-x**2)):
        solutions = dsolve_lie(equation, y)
        assert solutions, equation
        for solution in solutions:
            assert _solves(equation, solution), (equation, solution)
    C1, C2 = Symbol('C1'), Symbol('C2')
    assert dsolve_lie(y.diff(x, 2) - y.diff(x)**2/y, y) == [Eq(y, exp(C1*x + C2))]
    # a quadrature which SymPy cannot evaluate is returned as an integral
    found = dsolve_lie(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y, timeout=5)
    assert all(has_integral(s) for s in found)


def test_solve_ode() -> None:
    assert solve_ode(y.diff(x) - y, y) == [Eq(y, Symbol('C1')*exp(x))]
    equation = y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x
    [solution] = solve_ode(equation, y)
    assert _solves(equation, solution)
    # the resonance at k = 1 gets its own case, checked there
    [cased] = solve_ode(y.diff(x) - k*y - exp(x), y)
    assert str(cased) == 'Eq(y(x), Piecewise(((C1 + x)*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True)))'


def test_check_ode_solution() -> None:
    C1, C2 = Symbol('C1'), Symbol('C2')
    assert check_ode_solution(y.diff(x) - y, Eq(y, C1*exp(x)), y)
    assert not check_ode_solution(y.diff(x) - y, Eq(y, C1*exp(2*x)), y)
    # the right-hand side alone, and the solution written the other way
    assert check_ode_solution(y.diff(x, 2) + y.diff(x), C1 + C2*exp(-x), y)
    assert check_ode_solution(Eq(y.diff(x), y), Eq(C1*exp(x), y), y)
    # an implicit solution, solved for y or differentiated
    assert check_ode_solution(y*y.diff(x) + x, Eq(x**2 + y**2, C1), y)
    # a relation which SymPy's solve gives up on is checked through its
    # derivative (checkodesol raises NotImplementedError; the verdict
    # was None)
    assert check_ode_solution((1 + cos(y))*y.diff(x) - 1, Eq(y + sin(y), x + C1), y) is True
    # the relation solved by LambertW was checked in that form only, which
    # simplify does not prove (None; checkodesol gives False); its
    # derivative proves it
    assert check_ode_solution(y.diff(x) - 1/(y*exp(y)), Eq(x - (y - 1)*exp(y), C1), y) is True
    assert check_ode_solution(y*y.diff(x) - x, Eq(x**2 + y**2, C1), y) is False


def test_unproved_correct_solutions_are_not_rejected() -> None:
    # issue #25: SymPy's checkodesol answers False for these correct
    # solutions (residuals in hyper and besselj which simplify does not
    # reduce to zero); the verdict is None, False only for a residual
    # shown nonzero numerically
    a, b = Symbol('a'), Symbol('b')
    weber = (x**2)**Rational(3, 4)*exp(-x**2/2)*hyper((a/4 + Rational(3, 4),), (Rational(3, 2),), x**2)/sqrt(x)
    assert check_ode_solution(y.diff(x, 2) - (x**2 + a)*y, Eq(y, weber), y) is None
    kummer = x*y.diff(x, 2) - x*y.diff(x) - a*y
    assert check_ode_solution(kummer, Eq(y, x*hyper((a + 1,), (2,), x)), y) is None
    assert check_ode_solution(kummer, Eq(y, x*hyper((a + 2,), (2,), x)), y) is False
    bessel = sqrt(x)*x**(-a/2)*besselj(a/2 - Rational(1, 2), sqrt(b)*x)
    assert check_ode_solution(x*y.diff(x, 2) + a*y.diff(x) + b*x*y, Eq(y, bessel), y) is None
    # without the numerical checks a wrong solution is not proved wrong
    assert check_ode_solution(y.diff(x) - y, Eq(y, Symbol('C1')*exp(2*x)), y) is False
    with configure(numerical_checks=False):
        assert check_ode_solution(y.diff(x) - y, Eq(y, Symbol('C1')*exp(2*x)), y) is None
    # an implicit solution which is not proved (no point of the relation
    # is sampled: it is linear in no symbol): unknown, not wrong
    cubic = Eq(x**2*y**2 + y**3 - x**3, 0)
    assert check_ode_solution(y.diff(x)*(2*x**2*y + 3*y**2) - 2*x*y**2 - 3*x**2, cubic, y,
                              solve_for_func=False) is None


def test_every_branch_of_an_implicit_solution_is_checked() -> None:
    # the old module read checkodesol(...)[0], which for a relation solved
    # into several branches is the first (verdict, residual) pair, a
    # nonempty tuple and always true: y**2 = x was a solution of
    # y' = 1/(2 sqrt(x)), whose branch y = -sqrt(x) is not
    assert not check_ode_solution(y.diff(x) - 1/(2*sqrt(x)), Eq(y**2, x), y)
    assert check_ode_solution(2*y*y.diff(x) - 1, Eq(y**2, x), y)


def test_implicit_solutions_are_checked() -> None:
    # relations which the native solve gives up on (or solves into
    # branches which simplify does not prove) were None, right or
    # wrong; they are checked through their derivatives now, a wrong one
    # shown wrong at a point of the relation
    C1, C2, a = Symbol('C1'), Symbol('C2'), Symbol('a')
    cases: list[tuple[Expr, Eq, bool]] = [
        ((1 + cos(y))*y.diff(x) + 1, Eq(y + sin(y), x + C1), False),
        ((5*y**4 + 1)*y.diff(x) - 1, Eq(y**5 + y, x + C1), True),
        ((5*y**4 + 1)*y.diff(x) - 2, Eq(y**5 + y, x + C1), False),
        ((1 + exp(y))*y.diff(x) - 2*x, Eq(y + exp(y), x**2 + C1), True),
        # a parameter of the equation in the relation
        ((1 + a*cos(y))*y.diff(x) - 1, Eq(y + a*sin(y), x + C1), True),
        ((1 + a*cos(y))*y.diff(x) - a, Eq(y + a*sin(y), x + C1), False),
        # the residual y' - y/x = C1*x/y - y/x vanishes modulo the
        # relation only
        (y.diff(x) - y/x, Eq(y**2, C1*x**2), True),
        (y.diff(x) - 2*y/x, Eq(y**2, C1*x**2), False),
        # second order: (y**2)'' = 0, and y'' = -exp(y)*y'**2/(1 + exp(y))
        (y*y.diff(x, 2) + y.diff(x)**2, Eq(y**2, C1*x + C2), True),
        (y*y.diff(x, 2) - y.diff(x)**2, Eq(y**2, C1*x + C2), False),
        ((1 + exp(y))*y.diff(x, 2) + exp(y)*y.diff(x)**2, Eq(y + exp(y), C1*x + C2), True),
        ((1 + exp(y))*y.diff(x, 2) - exp(y)*y.diff(x)**2, Eq(y + exp(y), C1*x + C2), False),
        # a relation without constants which is linear in no symbol: the
        # derivative proves it (solve gives three Cardano branches, and
        # their checks ran out of time: None)
        (y.diff(x)*(2*x**2*y + 3*y**2) + 2*x*y**2 - 3*x**2, Eq(x**2*y**2 + y**3 - x**3, 0), True),
    ]
    for ode, relation, verdict in cases:
        for solve_for_func in (True, False):
            assert check_ode_solution(ode, relation, y, solve_for_func=solve_for_func) is verdict, (ode, relation)
    # without the numerical checks a wrong relation is not shown wrong
    with configure(numerical_checks=False):
        assert check_ode_solution((1 + cos(y))*y.diff(x) + 1, Eq(y + sin(y), x + C1), y) is None
    # the branch y = -sqrt(x) of y**2 = x does not solve y' = 1/(2*sqrt(x));
    # the points sampled have y > 0, where the relation is right: unknown
    # without solving
    assert check_ode_solution(y.diff(x) - 1/(2*sqrt(x)), Eq(y**2, x), y, solve_for_func=False) is None
