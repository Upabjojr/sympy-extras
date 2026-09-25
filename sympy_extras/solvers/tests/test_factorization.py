from __future__ import annotations

from random import Random
from typing import Optional

import mpmath
from mpmath import workdps
from sympy import Float, Function, Integral, Matrix, Poly, Rational, S, besselj, exp, lambdify, meijerg, sqrt, symbols
from sympy.simplify.simplify import simplify
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.solvers.factorization import (OperatorFactorization, factor_operator, left_factor, _rational_ratio,
    right_factor, solve_factored, variation_of_parameters)
from sympy_extras.solvers.kovacic import liouvillian_solution, normal_form
from sympy_extras.solvers.linear_ode import (LinearOperator, dsolve_linear, gcrd, hyperexponential_search,
    hyperexponential_solutions, lclm, log_derivative, rational_solutions, _exponential_parts, _exponents, _independent,
    _proportional)

x = symbols('x')
y = Function('y')(x)
D = LinearOperator([S.Zero, S.One], x)


def _op(*coefficients: ExprLike) -> LinearOperator:
    return LinearOperator([as_expr(c) for c in coefficients], x)


def _c(c: ExprLike) -> LinearOperator:
    return LinearOperator([as_expr(c)], x)


# ---------------------------------------------------------------------------
# arithmetic

def test_multiplication() -> None:
    # D a = a D + a'
    assert D*_c(x**2) == _op(2*x, x**2)
    A = _op(1/x, x, 1)
    B = _op(x, 0, 1/(x + 1))
    C = _op(-1, x**2)
    assert (A*B)*C == A*(B*C)
    assert A*(B + C) == A*B + A*C
    # the product applied to a function is the composition
    f = exp(x)*x**3
    assert simplify((A*B)(f) - A(B(f))) == 0
    assert A - A == _c(0) and (A - A).is_zero


def test_right_division() -> None:
    A = _op(x, 1/x, 0, 1, x**2)
    B = _op(1, x, 1 + x**2)
    Q, R = A.right_divmod(B)
    assert R.order < B.order
    assert Q*B + R == A
    raises(ZeroDivisionError, lambda: A.right_divmod(_c(0)))
    # an exact factor leaves no remainder
    assert (A*B).right_remainder(B).is_zero
    assert (A*B).right_quotient(B) == A


def test_gcrd_lclm() -> None:
    R = D - _c(1/x)
    A = (D*D + _c(1))*R
    B = (D - _c(x))*R
    assert gcrd(A, B) == R
    assert gcrd(D - _c(1), D - _c(2)) == _c(1)
    M = lclm(D - _c(1), D - _c(1/x))
    # the solutions exp(x) and x
    assert M.order == 2 and simplify(M(exp(x))) == 0 and simplify(M(x)) == 0
    assert M.right_remainder(D - _c(1)).is_zero and M.right_remainder(D - _c(1/x)).is_zero
    # a common right factor: the solution spaces of A R and B R meet in
    # that of R, so lclm(A R, B R) has order 3 + 2 - 1
    assert lclm(A, B).order == 4


def test_adjoint() -> None:
    A = _op(x, 1/x, 2, x)
    B = _op(1, x**2, 1)
    assert A.adjoint().adjoint() == A
    assert (A*B).adjoint() == B.adjoint()*A.adjoint()
    # Lagrange's identity: v L(u) - u L*(v) is a derivative; for L = D, L* = -D
    assert D.adjoint() == -D


def test_exterior_and_symmetric_powers() -> None:
    # y'' + y = 0: the products of two solutions satisfy y''' + 4 y' = 0
    L = _op(1, 0, 1)
    S2 = L.symmetric_power(2)
    assert S2 == _op(0, 4, 0, 1)
    assert L.exterior_power(1) == L
    # the Wronskian of all solutions: W' = -a_{n-1} W
    L = _op(x, 1/x, 3, 1)
    assert L.exterior_power(3) == _op(3, 1)
    # the second exterior power of a third order operator annihilates
    # the Wronskians of pairs of solutions: y''' = y has exp(x), exp(w x),
    # exp(w**2 x) with w a cube root of unity; W(exp(a x), exp(b x)) =
    # (b - a) exp((a + b) x), and a + b is minus the third root
    E = _op(-1, 0, 0, 1).exterior_power(2)
    assert E == _op(1, 0, 0, 1)
    assert simplify(E(exp(-x))) == 0
    # a symmetric square with a solution of each kind
    L = _op(-2/x**2, 0, 1)  # solutions x**2 and 1/x
    S2 = L.symmetric_power(2)
    for f in (x**4, x, 1/x**2):
        assert simplify(S2(f)) == 0
    raises(ValueError, lambda: L.exterior_power(3))
    raises(ValueError, lambda: L.symmetric_power(0))


def test_log_derivative() -> None:
    assert log_derivative(exp(x**2)*x**Rational(1, 3), x) == (6*x**2 + 1)/(3*x)
    assert log_derivative(S(3), x) == 0
    # the logarithmic derivatives of rational functions: integer residues
    f, decided = _rational_ratio(2/x - 1/(x - 1), x)
    assert decided and f is not None and simplify(f - x**2/(x - 1)) == 0
    f, decided = _rational_ratio(3*x/(x**2 + 1), x)
    assert decided and f is None
    f, decided = _rational_ratio(S.One, x)
    assert decided and f is None
    f, decided = _rational_ratio(2*x/(x**2 - 2), x)
    assert decided and f is not None and simplify(f - (x**2 - 2)) == 0


# ---------------------------------------------------------------------------
# the indicial polynomial at the roots of a factor which is not monic

def test_exponents_at_non_monic_factors() -> None:
    # (2x + 1)**2 y'' - 8 y = 0 is Euler's equation (x + 1/2)**2 y'' - 2 y = 0,
    # exponents 2 and -1 at -1/2; the factor q'(c)**order was missing from
    # the indicial polynomial, which came out as m(m - 1) - 8 with
    # irrational roots, and neither 1/(2x + 1) nor (2x + 1)**2 was found
    L = _op(-8, 0, (2*x + 1)**2)
    assert _exponents(L, Poly(2*x + 1, x)) == [-1, 2]
    found = rational_solutions(L)
    assert len(found) == 2 and all(simplify(L(f)) == 0 for f in found)
    assert any(simplify(f*(2*x + 1)).is_constant(x) for f in found + hyperexponential_solutions(L))
    # an irreducible quadratic factor: ((x**2 + 2) y)'' = 0 has the solution
    # 1/(x**2 + 2), whose exponent -1 at the roots c of x**2 + 2 needs the
    # factor 2 c in the indicial polynomial
    L = LinearOperator.from_equation((x**2 + 2)*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    assert any(simplify(f*(x**2 + 2)).is_constant(x) for f in rational_solutions(L))


# ---------------------------------------------------------------------------
# factorisation

def test_small_factorisations() -> None:
    F = factor_operator(_op(0, -1, 0, 1))
    assert F.expand() == _op(0, -1, 0, 1)
    assert sorted(f.order for f in F.factors) == [1, 1, 1] and F.is_complete
    # an irreducible operator: Airy
    F = factor_operator(_op(-x, 0, 1))
    assert len(F.factors) == 1 and F.irreducible == [True]
    # Bessel's operator of order 0 on the left of a first order factor
    A = _op(1, 1/x, 1)
    F = factor_operator(A*(D - _c(1/x)))
    assert [f.order for f in F.factors] == [2, 1] and F.is_complete
    assert F.expand() == A*(D - _c(1/x))
    raises(ValueError, lambda: factor_operator(_c(0)))
    assert isinstance(F, OperatorFactorization)


def test_second_order_right_factor() -> None:
    # Beke's method on the second exterior power: a right factor with
    # Airy functions as solutions (no hyperexponential solution)
    A = _op(-x, 0, 1)
    L = (D - _c(1/x))*(D*D - _c(1/x**2))*A
    R = right_factor(L, 2)
    assert R == A
    # left factors through the adjoint
    L = (D - _c(x))*A
    assert left_factor(L, 1) == D - _c(x)
    assert right_factor(L, 1) is None
    raises(ValueError, lambda: right_factor(L, 3))
    raises(ValueError, lambda: left_factor(L, 0))


def test_two_second_order_factors() -> None:
    # Airy times Bessel: a product of two irreducible second order factors
    A = _op(-x, 0, 1)
    B = _op(1, 1/x, 1)
    F = factor_operator(B*A)
    assert [f.order for f in F.factors] == [2, 2] and F.is_complete
    assert F.expand() == B*A
    assert F.factors[1] == A


def test_family_of_right_factors() -> None:
    # the solutions of lclm(A, B), B with the solutions x Ai and x Bi, are
    # (a + b x) Ai, (a + b x) Bi: every [a : b] gives a second order right
    # factor, the Wronskians (a + b x)**2 W(Ai, Bi) span a three
    # dimensional space of hyperexponential solutions of the associated
    # equation, and the decomposable ones are found through the
    # Grassmann-Pluecker relations
    A = _op(-x, 0, 1)
    B = A.transformed(1/x).monic()
    L = lclm(A, B)
    assert L.order == 4
    R = right_factor(L, 2)
    assert R is not None and R.order == 2 and L.right_remainder(R).is_zero
    F = factor_operator(L)
    assert [f.order for f in F.factors] == [2, 2] and F.is_complete
    assert F.expand() == L


def _rational(g: Random) -> Rational:
    return Rational(g.randint(-3, 3), g.choice([1, 1, 2]))


def _first(g: Random) -> LinearOperator:
    r: Expr = S.Zero
    for _ in range(g.randint(0, 2)):
        r = as_expr(r + _rational(g)/(x - g.randint(-2, 2)))
    if g.random() < 0.5:
        r = as_expr(r + _rational(g))
    return D - _c(r)


def _second(g: Random) -> LinearOperator:
    # regular singular at the finite points
    a: Expr = S.Zero
    b: Expr = S.Zero
    for _ in range(g.randint(0, 2)):
        c = g.randint(-2, 2)
        a = as_expr(a + _rational(g)/(x - c))
        b = as_expr(b + _rational(g)/(x - c)**2 + _rational(g)/(x - c))
    b = as_expr(b + _rational(g)*g.choice([S.Zero, S.One, x]))
    return D*D + _op(b, a)


def _random_product(seed: int) -> tuple[LinearOperator, list[LinearOperator]]:
    g = Random(seed)
    shape = g.choice([(1, 2), (2, 1), (1, 1, 1), (2, 2), (1, 1, 2), (2, 1, 1), (1, 2, 1)])
    factors = [_first(g) if s == 1 else _second(g) for s in shape]
    L = factors[0]
    for f in factors[1:]:
        L = L*f
    return L, factors


def _irreducible_second_order(F: LinearOperator) -> bool:
    """Independently of the factorisation: no hyperexponential solution
    found by an exhaustive search, or Kovacic's algorithm outside case 1."""
    solutions, complete = hyperexponential_search(F)
    if solutions:
        return False
    if complete:
        return True
    q, p = F.monic().coefficients[:2]
    z = liouvillian_solution(normal_form(p, q, x), x)
    return z is None or z.case != 1


def test_random_products() -> None:
    # seeded products of random first and second order operators with
    # rational coefficients: the product of the factors is the operator,
    # the factors claimed irreducible are, and the orders of the factors
    # agree with those of a factorisation of the factors themselves
    # (Jordan-Hoelder: the multiset of orders is an invariant)
    for seed in (0, 1, 3, 4, 7, 8, 13, 14, 21, 28, 29):
        L, factors = _random_product(seed)
        F = factor_operator(L)
        assert F.expand() == L, seed
        assert F.is_complete, seed
        for f, flag in zip(F.factors, F.irreducible):
            if f.order == 2:
                assert _irreducible_second_order(f), seed
        orders: list[int] = []
        for f in factors:
            orders.extend(g.order for g in factor_operator(f).factors)
        assert sorted(orders) == sorted(f.order for f in F.factors), seed


# ---------------------------------------------------------------------------
# solving

def _value(e: Expr, point: Rational) -> Optional[complex]:
    """The value of ``e`` at ``point``, the indefinite integrals (one
    level) taken from 1 by mpmath's quadrature at 20 digits."""
    values: dict[Integral, Expr] = {}
    for integral in e.atoms(Integral):
        assert not integral.function.has(Integral)
        f = lambdify(x, integral.function, 'mpmath')
        with workdps(20):
            values[integral] = Float(mpmath.quad(f, [1, mpmath.mpf(int(point.p))/int(point.q)]), 20)
    value = reliable_value(as_expr(e.xreplace(values).subs(x, point)), 15)
    return None if value is None else complex(value)


def _check_basis(L: LinearOperator, basis: list[Expr]) -> None:
    """Every solution by substitution (exactly, or numerically at two
    points), and the Wronskian nonzero at a rational point."""
    for s in basis:
        residual = L(s)
        if not residual.has(Integral) and residual.count_ops() < 300:
            exact = attempt(lambda: simplify(residual), 10)
            if exact == 0:
                continue
        for point in (Rational(3, 2), Rational(7, 3)):
            r = _value(residual, point)
            assert r is not None and abs(r) < 1e-8, (s, r)
    n = len(basis)
    W = Matrix([[as_expr(b.diff(x, i)) if i else b for b in basis] for i in range(n)])
    w = _value(as_expr(W.det()), Rational(5, 4))
    assert w is not None and abs(w) > 1e-8


def test_variation_of_parameters() -> None:
    L = _op(-1, 0, 1)
    solution = variation_of_parameters(L, [exp(x), exp(-x)], x)
    assert simplify(L(solution) - x) == 0
    L = _op(0, x)
    solution = variation_of_parameters(L, [S.One], exp(x))
    assert simplify(L(solution) - exp(x)) == 0
    raises(ValueError, lambda: variation_of_parameters(_op(-1, 0, 1), [exp(x)], x))
    raises(ValueError, lambda: variation_of_parameters(_op(-1, 0, 1), [exp(x), 2*exp(x)], x))


def test_solve_factored() -> None:
    # (D - 1/x) D: 1 and log(x)... the product D*(D - 1/x) has x and x log(x)
    found = solve_factored([D, D - _c(1/x)], hyperexponential_solutions)
    assert len(found) == 2
    _check_basis(D*(D - _c(1/x)), found)


def test_dsolve_third_order_airy() -> None:
    # (D - 1)(D**2 - x): the Airy functions and the solution of
    # y'' - x y = exp(x), which needs variation of parameters (before, no
    # solution was found: the operator has no hyperexponential solution
    # to reduce the order with)
    L = (D - _c(1))*_op(-x, 0, 1)
    equation = L(y)
    found = dsolve_linear(equation, y)
    assert len(found) == 3
    _check_basis(L, found)


def test_dsolve_fourth_order_bessel() -> None:
    # (D**2 + 1/x D + 1)(D**2 - 2/x**2): x**2, 1/x and two solutions with
    # Bessel functions of order 0 in integrals (found before as well, by
    # reduction of order and SymPy's dsolve)
    L = _op(1, 1/x, 1)*_op(-2/x**2, 0, 1)
    found = dsolve_linear(L(y), y)
    assert len(found) == 4
    _check_basis(L, found)


def test_dsolve_zimmermann() -> None:
    # Postel-Zimmermann: y'''' - 4 y''/x**2 + 8 y'/x**3 - 8 y/x**4 = 0
    equation = y.diff(x, 4) - 4*y.diff(x, 2)/x**2 + 8*y.diff(x)/x**3 - 8*y/x**4
    found = dsolve_linear(equation, y)
    assert len(found) == 4
    _check_basis(LinearOperator.from_equation(equation, y), found)
    # (D**2 + 1)**2
    equation = y.diff(x, 4) + 2*y.diff(x, 2) + y
    found = dsolve_linear(equation, y)
    assert len(found) == 4
    _check_basis(LinearOperator.from_equation(equation, y), found)


def test_exponential_parts_decrease() -> None:
    # the exponential parts at infinity were followed to the depth limit
    # with terms of any degree at every level (each level transforms the
    # operator again): this operator of order 4 did not finish in five minutes;
    # the terms after the first now have lower degrees
    L = (D + _c(1))*_op((-x**3 - x**2 - 5*x - 2)/(2*x**3 + 2*x**2), -1/(2*x), 1)*(D + _c((2*x + 2)/(x**2 + 2*x)))
    result = attempt(lambda: _exponential_parts(L.monic().primitive()), 120)
    assert result is not None
    parts, complete = result
    assert complete and set(parts) == {S.Zero, S.NegativeOne, sqrt(2)/2, -sqrt(2)/2}


def test_independence_with_integrals() -> None:
    # _independent compared solutions with ``is_constant``, which called
    # simplify on unevaluated integrals and could run into heurisch for
    # minutes; a ratio with an Integral is now taken as not constant
    a = exp(x)*Integral(besselj(0, x)*exp(-x)*sqrt(x), x)
    b = exp(-x)*Integral(besselj(0, x)*exp(x)/sqrt(x), x)
    kept = attempt(lambda: _independent([a, b, 2*a], x), 10)
    assert kept is not None and len(kept) == 2


def test_independence_with_meijer_g() -> None:
    # SymPy bug (sympy-extras#25): meijerg(((1/2, 1/2), (1, 3/2, 3/2, 2)),
    # ((1/2, 1, 1), (1,)), zoo).is_finite raises "AttributeError: 'NoneType'
    # object has no attribute 'has'" in meijerg._eval_evalf, and
    # is_constant substitutes x = 0 into x**2 * meijerg(..., x**-2): the
    # comparison of the solutions of a fourth order equation crashed
    # dsolve_linear; such a ratio is now taken as not constant
    m = meijerg(((S.Half, S.Half), (1, Rational(3, 2), Rational(3, 2), 2)), ((S.Half, 1, 1), (1,)), x**-2)
    assert _proportional(as_expr(x**2*m*besselj(0, x)), S.One, x) is False
