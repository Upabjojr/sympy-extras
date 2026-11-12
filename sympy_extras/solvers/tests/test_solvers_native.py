"""Tests of the native port of SymPy's ``solve`` (sympy_extras.solvers.solvers
and sympy_extras.solvers.bivariate): every solution is checked by
substitution, and the expected sets are those of SymPy's ``solve`` (SymPy
1.14) or hand computations."""
from __future__ import annotations

from sympy_extras.core import Dummy, Function, I, Rational, Symbol, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.functions import asin, atan, cos, exp, log, sin
from sympy_extras.functions.elementary.exponential import LambertW
from sympy_extras.solvers.bivariate import solve_lambert
from sympy_extras.solvers.solvers import checksol, invert, solve_equation, solve_system, zero_truth

x, y, a, b, C1, C2 = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b'), Symbol('C1'), Symbol('C2')


def _residual_is_small(e: Expr, v: Symbol, value: Expr, values: dict[Symbol, Expr]) -> bool:
    residual = e.subs(v, value).subs(values)
    assert isinstance(residual, Expr)
    return abs(complex(residual.evalf(30))) < 1e-20


def _checked(e: Expr, v: Symbol, expected_count: int) -> list[Expr]:
    found = solve_equation(e, v)
    assert found is not None
    assert len(found) == expected_count, found
    values: dict[Symbol, Expr] = {s: Rational(7 + 2*k, 5) for k, s in enumerate(sorted(e.free_symbols - {v},
                                                                                      key=lambda s: s.name))}
    for value in found:
        assert _residual_is_small(e, v, value, values), (e, value)
    return found


def test_polynomials() -> None:
    assert solve_equation(x**2 - a, x) == [-sqrt(a), sqrt(a)]
    _checked(x**3 - 1, x, 3)
    _checked(x**3 - x**3*sin(a) + 2, x, 3)       # coefficients outside the domains of Poly
    _checked(x**4 - (4 + 2*sqrt(2))*x**2 + 1, x, 4)
    assert solve_equation(x - x + 1, x) == []
    assert solve_equation(1/x, x) == []


def test_rational_functions_drop_the_poles() -> None:
    # SymPy's solve: -_v*y**3/(_v*y + 1) for -p/(p*y + y**3) - v
    p, v = Symbol('p'), Symbol('v')
    assert solve_equation(-p/(p*y + y**3) - v, p) == [-v*y**3/(v*y + 1)]
    assert solve_equation((x**2 - 1)/(x - 1), x) == [-1]


def test_generators() -> None:
    assert solve_equation(log(y) - C1*x - C2, y) == [exp(C1*x + C2)]
    assert solve_equation(exp(2*x) - 3*exp(x) + 2, x) == [0, log(2)]
    _checked(x**Rational(2, 3) - 4, x, 1)
    # the arctangent inverted: SymPy's solve of a Riccati solution for y
    _checked(-C1 + log(x) - 2*sqrt(3)*atan(2*sqrt(3)*x*y/3 + sqrt(3)/3)/3, y, 1)


def test_radicals() -> None:
    assert solve_equation(sqrt(x) + 1, x) == []
    assert solve_equation(2*sqrt(x) - a, x) == [a**2/4]
    # x - sqrt(x**2 - 1) eliminates to a nonzero constant: no solution
    assert solve_equation(x - sqrt(x**2 - 1), x) == []
    # a radical of a rational base (charpit): SymPy's solve gives the three
    # cube roots, which it cannot check; the real one solves the equation
    # for positive values of the symbols
    u = Symbol('u')
    e = 2*u/(3*sqrt(1/(u*a))) - a*y - b - x
    found = solve_equation(e, u)
    assert found is not None and len(found) == 3
    values: dict[Symbol, Expr] = {a: Rational(7, 5), b: Rational(9, 5), x: Rational(11, 5), y: Rational(13, 5)}
    assert any(_residual_is_small(e, u, value, values) for value in found)
    # with the opposite sign the real root is provably extraneous (checksol
    # with positive symbols), the two complex ones are kept as by SymPy
    other = solve_equation(-2*u/(3*sqrt(1/(u*a))) - a*y - b - x, u)
    assert other is not None and len(other) == 2


def test_cardano_candidates_are_checked_at_their_exact_roots() -> None:
    # the root 0.7816... of the eliminated cubic (Cardano's radicals with
    # an imaginary part exactly zero, which reliable values do not decide)
    # was kept although 5*sqrt(x)/3 + 3 + 1/(3*(x - 1)) is 2.947 there
    e = 5*sqrt(x)/3 + 3 + 1/(3*(x - 1))
    found = solve_equation(e, x)
    assert found is not None and len(found) == 1
    assert _residual_is_small(e, x, found[0], {})


def test_the_sine_and_the_cosine_have_two_inverses() -> None:
    # SymPy's multi_inverses: [pi - asin(3/2), asin(3/2)]; only asin(3/2)
    # was returned
    assert solve_equation(sin(x) - Rational(3, 2), x) == [pi - asin(Rational(3, 2)), asin(Rational(3, 2))]
    _checked(cos(2*x) - Rational(1, 3), x, 2)


def test_lambert() -> None:
    assert solve_equation(x*exp(x) - 1, x) == [LambertW(1)]
    found = solve_equation(x - exp(x) + 2, x)
    assert found is not None and len(found) == 2
    for value in found:
        assert abs(complex((value - exp(value) + 2).evalf(30))) < 1e-20
    found_log = solve_lambert(x + log(x) - 2, x)
    assert found_log is not None and abs(complex((found_log[0] + log(found_log[0]) - 2).evalf(30))) < 1e-20
    assert solve_lambert(x**2 - 2, x) is None


def test_an_integral_with_the_unknown_in_its_limits_has_no_solution() -> None:
    # SymPy's solve gives [] for Integral(g(y), (y, f)) - c, and
    # checkodesol then goes on: no inversion is attempted for a node which
    # is not a function (a wrapped SymPy Integral)
    from sympy_extras.solvers.solvers import _by_generator
    f = Dummy('f')
    g = Function('g')
    assert _by_generator(g(f) - 1, f, 0) is None


def test_invert_and_checks() -> None:
    assert invert(2*exp(x + 1) - y, x) == (log(y/2) - 1, x)
    assert checksol(x**2 - 2, x, sqrt(2)) is True
    assert checksol(x**2 - 2, x, Rational(3, 2)) is False
    # with positive symbols the candidate fails (SymPy's checksol posifies)
    assert checksol(x + sqrt(x**2), x, -a) is None
    assert checksol(x + sqrt(x**2), x, a) is False
    assert zero_truth(sqrt(2)**2 - 2) is True and zero_truth(I) is False


def test_systems() -> None:
    assert solve_system([x + y - 2, x - y], [x, y]) == [{x: 1, y: 1}]
    assert solve_system([x*y - 1, x - y], [x, y]) == [{x: -1, y: -1}, {x: 1, y: 1}]
    assert solve_system([x + y - a], [x, y]) == [{x: a - y}]
    assert solve_system([x**2 + 1, x - 1], [x]) == []
    # an inconsistent system of an Abel equation's undetermined
    # coefficients: SymPy's solve gave [] after 3.8 s
    a0, a1, a2 = Symbol('a0'), Symbol('a1'), Symbol('a2')
    system = [-a2**3, -3*a1*a2**2, -3*a0*a2**2 - 3*a1**2*a2, -6*a0*a1*a2 - a1**3, -3*a0**2*a2 - 3*a0*a1**2,
              -3*a0**2*a1 + 2*a2 - 1, -a0**3 + a1]
    assert solve_system(system, [a0, a1, a2]) == []
    # a positive-dimensional system: every family satisfies the equations
    z = Symbol('z')
    families = solve_system([x*z - y, x - y*z], [x, y, z])
    assert families is not None and len(families) == 3
    for family in families:
        for e in (x*z - y, x - y*z):
            assert zero_truth(e.subs(family)) is True
