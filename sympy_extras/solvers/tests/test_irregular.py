from __future__ import annotations

from sympy import QQ, Expr, Function, Matrix, Poly, Rational, cancel, exp, simplify, sqrt, together
from sympy.abc import x

from sympy_extras._typing_sympy import as_expr
from sympy_extras.solvers.irregular_sympy import (Place, conjugate_sum, finite_exponential_parts, irregular_places,
                                            local_exponential_parts)
from sympy_extras.solvers.linear_ode_sympy import LinearOperator, hyperexponential_search, hyperexponential_solutions

y = Function('y')(x)


def _operator_with_solutions(solutions: list[Expr]) -> LinearOperator:
    """The monic operator (with polynomial coefficients) annihilating the
    given independent functions: the Wronskian determinant."""
    n = len(solutions)
    rows = [[as_expr(s.diff(x, k)) for k in range(n + 1)] for s in solutions]
    coefficients: list[Expr] = []
    for k in range(n + 1):
        minor = Matrix([[row[j] for j in range(n + 1) if j != k] for row in rows])
        coefficients.append(as_expr((-1)**(n - k)*minor.det()))
    # divide by the common hyperexponential factor: every coefficient is a
    # rational function times the product of the solutions
    product = as_expr(Matrix([row[:n] for row in rows]).det())
    normalised = [as_expr(cancel(together(c/product))) for c in coefficients]
    return LinearOperator(normalised, x).primitive()


def _solves(L: LinearOperator, s: Expr) -> bool:
    return simplify(as_expr(L(s))) == 0


def _same_span(found: list[Expr], expected: list[Expr]) -> bool:
    if len(found) != len(expected):
        return False
    # each expected function is a multiple of a found one (the solutions
    # of distinct exponential parts are not proportional)
    for e in expected:
        if not any(simplify(as_expr(f/e)).is_constant() for f in found):
            return False
    return True


def test_conjugate_sum() -> None:
    P = Place(Poly(x**2 - 2, x), x)
    s = conjugate_sum(P, as_expr(P.alpha + 1), 3)
    direct = (sqrt(2) + 1)/(x - sqrt(2))**3 + (1 - sqrt(2))/(x + sqrt(2))**3
    assert simplify(s - direct) == 0
    P = Place(Poly(x**3 - 2, x), x)
    s = conjugate_sum(P, as_expr(P.alpha**2), 2)
    # the sum over the three roots is rational: check at a point
    r = Poly(x**3 - 2, x).all_roots()
    direct_sum = sum(complex((a**2/(3 - a)**2).evalf(30)) for a in r)
    assert abs(float(s.subs(x, 3)) - direct_sum.real) < 1e-9 and abs(direct_sum.imag) < 1e-9


def test_rational_irregular_point() -> None:
    # exp(+-1/x): rho = -+1/x**2
    L = _operator_with_solutions([exp(1/x), exp(-1/x)])
    assert irregular_places(L) == [Poly(x, x, domain=QQ)]
    parts, complete = finite_exponential_parts(L)
    assert complete and set(parts) == {0, 1/x, -1/x}
    found, complete = hyperexponential_search(L)
    assert complete and _same_span(found, [exp(1/x), exp(-1/x)])
    # two terms in the exponential part, and a local exponent
    L = _operator_with_solutions([exp(1/x**2 + 2/x)*sqrt(x), exp(-1/x**2)*(x + 1)])
    found, complete = hyperexponential_search(L)
    assert complete and _same_span(found, [exp(1/x**2 + 2/x)*sqrt(x), exp(-1/x**2)*(x + 1)])
    # an exponential part at infinity as well, and a second irregular point
    L = _operator_with_solutions([exp(x**2 + 1/x), exp(1/(x - 1))*x])
    found, complete = hyperexponential_search(L)
    assert complete and _same_span(found, [exp(x**2 + 1/x), exp(1/(x - 1))*x])
    assert all(_solves(L, s) for s in found)


def test_conjugate_irregular_points() -> None:
    # rho = -2x/(x**2 - 2)**2: conjugate principal parts at +-sqrt(2)
    L = _operator_with_solutions([exp(1/(x**2 - 2)), x])
    assert irregular_places(L) == [Poly(x**2 - 2, x, domain=QQ)]
    found, complete = hyperexponential_search(L)
    assert complete and _same_span(found, [exp(1/(x**2 - 2)), x])
    # a part of order three at the roots of x**2 + 1
    L = _operator_with_solutions([exp(x/(x**2 + 1)**2)*(x - 1), exp(-x/(x**2 + 1)**2)])
    found, complete = hyperexponential_search(L)
    assert complete and len(found) == 2 and all(_solves(L, s) for s in found)


def test_third_order_and_first_order() -> None:
    L = LinearOperator.from_equation(x**2*y.diff(x) + y, y)
    assert _same_span(hyperexponential_solutions(L), [exp(1/x)])
    L = _operator_with_solutions([exp(1/x), exp(-1/x), exp(2/x)*x**2])
    found, complete = hyperexponential_search(L)
    assert complete and _same_span(found, [exp(1/x), exp(-1/x), exp(2/x)*x**2])


def test_no_hyperexponential_solution_at_an_irregular_point() -> None:
    # x**3 y'' - y: ramified exponential parts at 0 (slope 3/2, solutions
    # exp(+-2/sqrt(x)) times series), no hyperexponential solution; the
    # search is complete
    L = LinearOperator.from_equation(x**3*y.diff(x, 2) - y, y)
    assert irregular_places(L) == [Poly(x, x, domain=QQ)]
    assert local_exponential_parts(L, Place(Poly(x, x), x)) == ([0], True)
    found, complete = hyperexponential_search(L)
    assert found == [] and complete
    # x y'' + y'/2 - y is regular singular at 0 (exponents 0 and 1/2)
    assert irregular_places(LinearOperator.from_equation(x*y.diff(x, 2) + y.diff(x)/2 - y, y)) == []
    # a leading coefficient outside Q(alpha): r**2 = alpha at the roots
    # alpha of x**2 - 2 (rho ~ r/(x - alpha)**2): incomplete, nothing found
    L = LinearOperator.from_equation((x**2 - 2)**4*y.diff(x, 2) - x*y, y)
    found, complete = hyperexponential_search(L)
    assert found == [] and not complete


def test_exponent_part_with_rational_coefficient_half() -> None:
    L = _operator_with_solutions([exp(Rational(1, 2)/x)*x**Rational(1, 3), exp(-3/x)])
    found, complete = hyperexponential_search(L)
    assert complete and _same_span(found, [exp(Rational(1, 2)/x)*x**Rational(1, 3), exp(-3/x)])
