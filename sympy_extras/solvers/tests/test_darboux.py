from __future__ import annotations

from sympy import Function, Symbol, Eq, Rational, I, S, sqrt, exp, log, erf, cancel, factor_list, simplify
from sympy.abc import x, y, a
from sympy.core.expr import Expr
from sympy.solvers.ode import checkodesol

from sympy_extras.solvers.darboux import (DarbouxPolynomial, darboux_polynomials, exponential_factors,
    darboux_integrating_factor, darboux_first_integral, prelle_singer, vector_field)
from sympy_extras.solvers.first_order import dsolve_first_order
from sympy_extras._typing import ExprLike, as_expr


def _D(P: ExprLike, Q: ExprLike, e: Expr) -> Expr:
    """The vector field ``Q d/dx + P d/dy`` applied to ``e``."""
    return as_expr(cancel(as_expr(Q)*e.diff(x) + as_expr(P)*e.diff(y)))


def _is_first_integral(P: ExprLike, Q: ExprLike, I_: Expr) -> bool:
    return simplify(_D(P, Q, I_)) == 0 and (I_.has(x) or I_.has(y))


def _dependent(F: Expr, G: Expr) -> bool:
    """Whether two functions of ``x, y`` are functionally dependent
    (their Jacobian vanishes)."""
    return cancel(F.diff(x)*G.diff(y) - F.diff(y)*G.diff(x)) == 0


def _checked(P: ExprLike, Q: ExprLike, polynomials: list[DarbouxPolynomial]) -> None:
    for d in polynomials:
        # D f = g f exactly, f irreducible over the rationals
        assert _D(P, Q, d.polynomial) == cancel(d.cofactor*d.polynomial)
        assert len(factor_list(d.polynomial)[1]) == 1


def test_darboux_polynomials_lotka_volterra() -> None:
    # the axes are the invariant lines of x' = x (1 - y), y' = y (x - 1)
    P, Q = y*(x - 1), x*(1 - y)
    found = darboux_polynomials(P, Q, x, y, degree=3)
    _checked(P, Q, found)
    assert {(d.polynomial, d.cofactor) for d in found} == {(x, 1 - y), (y, x - 1)}


def test_conjugate_lines_give_a_rational_polynomial() -> None:
    # x + I y and x - I y are the Darboux polynomials over the complex
    # numbers; over the rationals their product is found, with the sum
    # of their cofactors
    P, Q = x + y, x - y
    found = darboux_polynomials(P, Q, x, y, degree=2)
    _checked(P, Q, found)
    assert found == [DarbouxPolynomial(x**2 + y**2, S(2))]
    # over a field containing I, the lines themselves
    over_i = darboux_polynomials(P, Q, x, y, degree=1, extension=I)
    assert {d.polynomial for d in over_i} == {x - I*y, x + I*y}


def test_darboux_polynomials_of_a_pencil() -> None:
    # the field with the rational first integral F = (x**2 + y)/(y**2 + x):
    # its level sets x**2 + y - c (y**2 + x) are Darboux polynomials with
    # the same cofactor; the reducible members (c = 1, c = -1) split
    f1, f2 = x**2 + y, y**2 + x
    P = -(f1.diff(x)*f2 - f1*f2.diff(x))
    Q = f1.diff(y)*f2 - f1*f2.diff(y)
    found = darboux_polynomials(P, Q, x, y, degree=2)
    _checked(P, Q, found)
    polynomials = {d.polynomial for d in found}
    assert {f1, f2, x - y, x + y - 1} <= polynomials
    cofactor = {d.cofactor for d in found if d.polynomial in (f1, f2)}
    assert len(cofactor) == 1


def test_dicritical_line_at_infinity() -> None:
    # x P_d - y Q_d vanishes for y' = y/x: every line through the origin
    # is invariant, and the charts of the forms of degree N are searched
    found = darboux_polynomials(y, x, x, y, degree=1)
    _checked(y, x, found)
    assert {x, y} <= {d.polynomial for d in found}
    # the Darboux polynomial x of y' = (x y - 1)/x**2 (dicritical too) was
    # lost when the elimination divided a row by a content which vanishes
    # at the cofactor (b**3 = 0 became 1 = 0)
    found = darboux_polynomials(x*y - 1, x**2, x, y, degree=1)
    assert any(d.polynomial == x for d in found)


def test_exponential_factors() -> None:
    # the linear equation y' = x y + 1 has the integrating factor
    # exp(-x**2/2): exp(x**2) with cofactor 2 x
    found = exponential_factors(x*y + 1, 1, x, y, degree=1)
    assert any(cancel(e.argument/x**2).is_number and cancel(e.cofactor/x) == 2*cancel(e.argument/x**2)
               for e in found)
    for e in found:
        assert _D(x*y + 1, 1, e.argument) == e.cofactor
    # exp(1/x) for y' = y/x**2 (denominator the Darboux polynomial x)
    found = exponential_factors(y, x**2, x, y, degree=1)
    assert any(cancel(e.argument + 1/x) == 0 for e in found)


def test_integrating_factor_known() -> None:
    # Lotka-Volterra: 1/(x y) (Volterra 1926)
    assert darboux_integrating_factor(y*(x - 1), x*(1 - y), x, y) == 1/(x*y)
    # the linear equation: exp(-int x dx)
    assert darboux_integrating_factor(x*y + 1, 1, x, y, degree=1) == exp(-x**2/2)
    # y' = (x + y)/(x - y): 1/(x**2 + y**2)
    R = darboux_integrating_factor(x + y, x - y, x, y, degree=2)
    assert R is not None and cancel(R - 1/(x**2 + y**2)) == 0


def test_first_integral_known() -> None:
    # Lotka-Volterra: x - log x + y - log y (Volterra 1926), up to sign
    I_ = darboux_first_integral(y*(x - 1), x*(1 - y), x, y)
    assert I_ is not None and _dependent(I_, x - log(x) + y - log(y))
    assert _is_first_integral(y*(x - 1), x*(1 - y), I_)
    # y' = (x + y)/(x - y): log(x**2 + y**2)/2 - atan(y/x)
    I_ = darboux_first_integral(x + y, x - y, x, y)
    assert I_ is not None and _is_first_integral(x + y, x - y, I_)
    assert not I_.has(I)
    # the rational first integral of the pencil
    f1, f2 = x**2 + y, y**2 + x
    P = -(f1.diff(x)*f2 - f1*f2.diff(x))
    Q = f1.diff(y)*f2 - f1*f2.diff(y)
    I_ = darboux_first_integral(P, Q, x, y)
    assert I_ is not None and I_.is_rational_function() and _dependent(I_, f1/f2)
    # the linear equation: a Liouvillian first integral through erf
    I_ = darboux_first_integral(x*y + 1, 1, x, y)
    assert I_ is not None and I_.has(erf) and _is_first_integral(x*y + 1, 1, I_)
    # y' = y/x**2: y exp(1/x)
    I_ = darboux_first_integral(y, x**2, x, y)
    assert I_ is not None and _dependent(I_, y*exp(1/x))


def test_real_logarithms() -> None:
    # the quadrature of the integrating factor 1/(x**2 + y**2) came out
    # with complex logarithms, (1/2 - I/2) log(x - y + ...), when x and y
    # were integrated as complex symbols
    I_ = darboux_first_integral(x + y, x - y, x, y)
    assert I_ is not None and not I_.has(I)


def test_no_first_integral_within_the_bound() -> None:
    # y' = y**2 + x (Airy's Riccati equation) has no Liouvillian first
    # integral: nothing is found, in a short time
    assert darboux_first_integral(y**2 + x, 1, x, y, degree=3) is None
    assert darboux_polynomials(y**2 + x, 1, x, y, degree=3) == []


def test_parameters_and_extensions() -> None:
    # y' = a y/x: the first integral y/x**a with the parameter a
    I_ = darboux_first_integral(a*y, x, x, y, degree=1)
    assert I_ is not None and simplify(_D(a*y, x, I_)) == 0
    # coefficients in an algebraic extension
    found = darboux_polynomials(sqrt(2)*y, x, x, y, degree=1)
    assert {d.polynomial for d in found} >= {x, y}
    assert {d.cofactor for d in found} == {1, sqrt(2)}
    # over Q(i) the lines x - I y and x + I y with conjugate cofactors
    # (the linear systems over an algebraic field got SymPy numbers among
    # their entries and failed with AttributeError)
    found = darboux_polynomials(-(2*x**2 + x*y + y**2), x*(y - x), x, y, degree=1, extension=I)
    assert {d.polynomial for d in found} == {x, x - I*y, x + I*y}


def test_prelle_singer_implicit_solutions() -> None:
    f = Function('y')(x)
    C1 = Symbol('C1')
    equations = [
        f.diff(x) - (x + f)/(x - f),
        x*(1 - f)*f.diff(x) - f*(x - 1),
        f.diff(x) - x*f - 1,
        f.diff(x) - f**3 - x*f,                       # Bernoulli
        2*x*f*f.diff(x) - (x**2 + f**2 - 1),          # Kamke 1.  (exact after the factor 1/x**2)
        f.diff(x) - (2*x*f)/(x**2 - f**2),
    ]
    for equation in equations:
        solution = prelle_singer(equation, f)
        assert isinstance(solution, Eq), equation
        # explicit (the linear equation, through erf) or Eq(I, C1)
        assert solution.lhs == f or (solution.rhs == C1 and solution.lhs.has(f))
        checked = checkodesol(equation, solution, f, solve_for_func=False)
        assert checked[0] is True, (equation, solution, checked)


def test_prelle_singer_through_the_dispatcher() -> None:
    # dsolve_first_order tries the Prelle-Singer procedure after the
    # Riccati, Abel and Chini classes
    f = Function('y')(x)
    equation = f.diff(x) - (x + f)/(x - f)
    solution = dsolve_first_order(equation, f)
    assert isinstance(solution, Eq) and checkodesol(equation, solution, f, solve_for_func=False)[0] is True


def test_vector_field_parsing() -> None:
    f = Function('y')(x)
    parsed = vector_field((x - f)*f.diff(x) - x - f, f)
    assert parsed is not None
    P, Q, x_, y_ = parsed
    assert (P, Q, x_) == (x + y_, x - y_, x) and y_.name == 'y'
    # common factors are cancelled
    parsed = vector_field(f.diff(x) - (x*f**2 - f)/(x**2*f - x), f)
    assert parsed is not None and parsed[:2] == (parsed[3], x)
    # not polynomial, or not linear in y': no field
    assert vector_field(f.diff(x) - exp(f), f) is None
    assert vector_field(f.diff(x)**2 - f, f) is None
    assert prelle_singer(f.diff(x) - sqrt(f), f) is None


def test_degree_bound_raised_one_by_one() -> None:
    # y' = (x + y)/(x - y) needs the Darboux polynomial x**2 + y**2 over
    # the rationals: with degree one the search over Q(i) (the roots of
    # x**2 + y**2) gives the complex Darboux first integral
    # (x + I y)/(x - I y)**I instead; with degree two, the real one
    from sympy_extras.solvers.darboux import _field_of, _first_integral_over
    assert _first_integral_over(_field_of(x + y, x - y, x, y), 1) is None
    complex_form = darboux_first_integral(x + y, x - y, x, y, degree=1)
    assert complex_form is not None and complex_form.has(I) and _is_first_integral(x + y, x - y, complex_form)
    real_form = darboux_first_integral(x + y, x - y, x, y, degree=2)
    assert real_form is not None and not real_form.has(I)
    # the pencil's first integral is found with its lines already: the
    # quotient of two exponential factors with linear denominators
    f1, f2 = x**2 + y, y**2 + x
    P = -(f1.diff(x)*f2 - f1*f2.diff(x))
    Q = f1.diff(y)*f2 - f1*f2.diff(y)
    I_ = darboux_first_integral(P, Q, x, y, degree=1)
    assert I_ is not None and _dependent(I_, f1/f2)


def test_rational_first_integral_from_the_cofactors() -> None:
    # the two members of the pencil have the same cofactor: the first
    # integral is their quotient, no quadrature needed
    I_ = darboux_first_integral(y**2 - x**2 + 1, 2*x*y, x, y, degree=2)
    assert I_ is not None and _dependent(I_, (x**2 + y**2 + 1)/x)
    I_ = darboux_first_integral(x*y - 1, x**2, x, y, degree=2)
    assert I_ is not None and _dependent(I_, y/x - Rational(1, 2)/x**2)


def test_explicit_when_the_relation_is_rational() -> None:
    f = Function('y')(x)
    C1 = Symbol('C1')
    # a relation rational in y with one root: solved for y
    solution = prelle_singer(x**2*f.diff(x) - x*f + 1, f)
    assert isinstance(solution, Eq) and solution.lhs == f and not solution.rhs.has(f)
    assert checkodesol(x**2*f.diff(x) - x*f + 1, solution, f)[0] is True
    # linear in y through erf: explicit too
    solution = prelle_singer(f.diff(x) - x*f - 1, f)
    assert isinstance(solution, Eq) and solution.lhs == f and solution.rhs.has(erf)
    # a transcendental relation stays implicit (solve would cut it to
    # one branch of LambertW)
    solution = prelle_singer(x*(1 - f)*f.diff(x) - f*(x - 1), f)
    assert isinstance(solution, Eq) and solution.rhs == C1 and solution.lhs.has(log)


def test_quadratic_extension_of_the_top_part() -> None:
    from sympy_extras.solvers.darboux import _field_of, _quadratic_extension
    # x P_d - y Q_d = (x**2 + y**2) * ... : the search over the rationals
    # is repeated over Q(i), where the conjugate lines appear
    P, Q = -(2*x**2 + x*y + y**2), x*(y - x)
    extension = _quadratic_extension(_field_of(P, Q, x, y))
    assert extension is not None and cancel(extension**2 + 4) == 0
    found = darboux_polynomials(P, Q, x, y, degree=1, extension=extension)
    assert {d.polynomial for d in found} == {x, x - I*y, x + I*y}
    # no quadratic factor: no extension
    assert _quadratic_extension(_field_of(y*(x - 1), x*(1 - y), x, y)) is None


def test_parameters_with_undecided_signs() -> None:
    # Kamke 1.23, y' = b - a y**2: integrated with real parameters, the
    # quadrature is a Piecewise on the sign of a b without a generic
    # branch, and the equation lost its solution; the complex symbols
    # give the logarithms
    f = Function('y')(x)
    b = Symbol('b')
    equation = f.diff(x) + a*f**2 - b
    solution = prelle_singer(equation, f)
    assert isinstance(solution, Eq)
    assert checkodesol(equation, solution, f, solve_for_func=False)[0] is True
