"""The native linear ODE cluster (linear_ode, factorization, irregular,
special) on the objects of the core: results checked by substitution, by
the operator arithmetic, and against hand computations."""
from __future__ import annotations

from sympy_extras.core import Function, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.power import sqrt
from sympy_extras.functions import exp
from sympy_extras.polys.polytools import cancel
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers.factorization import factor_operator, left_factor, right_factor, _rational_ratio
from sympy_extras.solvers.irregular import (Place, conjugate_sum, finite_exponential_parts, irregular_places,
                                            local_exponential_parts)
from sympy_extras.solvers.linear_ode import (LinearOperator, UPoly, gcrd, hyperexponential_search,
                                             hyperexponential_solutions, lclm, log_derivative, polynomial_solutions,
                                             rational_solutions, reduce_order_linear, _exponents, _indicial_at,
                                             _nullspace)
from sympy_extras.solvers.special import bessel_solutions, hypergeometric_solutions, whittaker_m

x = Symbol('x')
y = Function('y')(x)
ZERO, ONE = Integer(0), Integer(1)
D = LinearOperator([ZERO, ONE], x)


def _c(c: Expr) -> LinearOperator:
    return LinearOperator([c], x)


def _op(*coefficients: Expr) -> LinearOperator:
    return LinearOperator(list(coefficients), x)


def _solves(L: LinearOperator, s: Expr) -> bool:
    return cancel(L(s)) == ZERO or simplify(L(s)) == ZERO


def test_ore_arithmetic() -> None:
    # D a = a D + a'
    assert D*_c(x**2) == _op(2*x, x**2)
    A = _op(1/x, x, ONE)
    B = _op(x, ZERO, 1/(x + 1))
    C = _op(Integer(-1), x**2)
    assert (A*B)*C == A*(B*C)
    assert A*(B + C) == A*B + A*C
    f = exp(x)*x**3
    assert simplify((A*B)(f) - A(B(f))) == ZERO
    Q, R = A.right_divmod(B)
    assert R.order < B.order and Q*B + R == A
    # (AB)* = B* A*
    assert (A*B).adjoint() == B.adjoint()*A.adjoint()
    assert _op(ZERO, x, ONE).adjoint() == _op(Integer(-1), -x, ONE)


def test_gcrd_lclm() -> None:
    R = D - _c(1/x)
    assert gcrd((D*D + _c(ONE))*R, (D - _c(x))*R) == R
    M = lclm(_op(Integer(-1), ONE), _op(Integer(-1), x))
    assert M.order == 2 and _solves(M, exp(x)) and _solves(M, x)


def test_powers() -> None:
    # y''' = y: the Wronskians of two solutions solve the adjoint
    assert _op(Integer(-1), ZERO, ZERO, ONE).exterior_power(2) == _op(ONE, ZERO, ZERO, ONE)
    # the squares of the solutions of y'' + y = 0: 1, cos(2x), sin(2x)
    assert _op(ONE, ZERO, ONE).symmetric_power(2) == _op(ZERO, Integer(4), ZERO, ONE)


def test_polynomial_and_rational_solutions() -> None:
    L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    assert L.coefficients == (Integer(2), -x - 2, x)
    assert polynomial_solutions(L) == [x**2 + 2*x + 2]
    assert polynomial_solutions(_op(Integer(-2), ZERO, ONE)) == []
    L = LinearOperator.from_equation(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    found = rational_solutions(L)
    assert len(found) == 2 and all(_solves(L, s) for s in found)
    # the indicial polynomial at the root of 2*x + 1 (a non monic factor)
    L = _op(ZERO, -(2*x + 1), (2*x + 1)**2)
    assert _exponents(L, UPoly.of(2*x + 1, x)) == [ZERO, Rational(3, 2)]


def test_hyperexponential_solutions() -> None:
    L = LinearOperator.from_equation(4*x**2*y.diff(x, 2) + 4*x*y.diff(x) - y, y)
    assert hyperexponential_solutions(L) == [1/sqrt(x), sqrt(x)]
    L = LinearOperator.from_equation(y.diff(x, 2) + 4*y.diff(x) + 5*y, y)
    found = hyperexponential_solutions(L)
    assert len(found) == 2 and all(_solves(L, s) for s in found)
    assert hyperexponential_search(_op(-x, ZERO, ONE)) == ([], True)
    # x**((1 +- sqrt(5))/2): exponents which are not rational
    assert hyperexponential_search(_op(Integer(-1), ZERO, x**2)) == ([], False)


def test_irregular_points() -> None:
    L = LinearOperator.from_equation(x**4*y.diff(x, 2) + 2*x**3*y.diff(x) - y, y)
    assert [q.as_expr() for q in irregular_places(L)] == [x]
    assert finite_exponential_parts(L) == ([ZERO, 1/x, -1/x], True)
    assert hyperexponential_solutions(L) == [exp(1/x), exp(-1/x)]
    L = LinearOperator.from_equation(x**3*y.diff(x) - (2 + x)*y, y)
    assert local_exponential_parts(L, Place(UPoly.of(x, x), x)) == ([ZERO, -1/x**2, -1/x - 1/x**2], True)
    # a conjugate pair of irregular points x = +-sqrt(2): exp(1/(x**2 - 2))
    M = lclm(D - _c(-2*x/(x**2 - 2)**2), D - _c(ONE))
    found, complete = hyperexponential_search(M)
    assert complete and len(found) == 2 and all(_solves(M, s) for s in found)


def test_conjugate_sum() -> None:
    P = Place(UPoly.of(x**2 - 2, x), x)
    # sqrt(2)/(x - sqrt(2))**2 - sqrt(2)/(x + sqrt(2))**2
    assert conjugate_sum(P, P.alpha, 2) == 8*x/(x**2 - 2)**2
    r2 = sqrt(2)
    expected = (r2 + 1)/(x - r2)**3 + (1 - r2)/(x + r2)**3
    assert cancel(conjugate_sum(P, P.alpha + 1, 3) - expected) == ZERO


def test_reduction_of_order() -> None:
    L = LinearOperator.from_equation(y.diff(x, 2) - 3*y.diff(x) + 2*y, y)
    assert reduce_order_linear(L, exp(x)) == _op(Integer(-1), ONE)


def test_log_derivative() -> None:
    assert log_derivative(sqrt(x - 1)*exp(x**2)/x, x) == cancel(1/(2*(x - 1)) + 2*x - 1/x)
    assert log_derivative(Integer(3), x) == ZERO


def test_nullspace() -> None:
    a = Symbol('a')
    assert _nullspace([[ONE, a], [a, a**2]], 2) == [[-a, ONE]]
    # an entry which is not a rational function
    assert _nullspace([[exp(x), ONE]], 2) == [[-exp(-x), ONE]]


def test_factorisation() -> None:
    F = factor_operator(_op(ZERO, Integer(-1), ZERO, ONE))
    assert len(F.factors) == 3 and F.is_complete and F.expand() == _op(ZERO, Integer(-1), ZERO, ONE)
    A = D*D - _c(2/x**2)
    L = (D - _c(1/x))*A
    assert right_factor(L, 2) == A
    L = (D - _c(1/x))*(D*D + _c(ONE))
    assert left_factor(L, 1) == D - _c(1/x)
    # an irreducible Airy operator
    F = factor_operator(_op(-x, ZERO, ONE))
    assert len(F.factors) == 1 and F.irreducible == (True,)


def test_rational_ratio() -> None:
    f, decided = _rational_ratio(2/x - 1/(x - 1), x)
    assert decided and f is not None and cancel(log_derivative(f, x) - (2/x - 1/(x - 1))) == ZERO
    # log(x**2 + 1)*3/2 is not rational
    assert _rational_ratio(3*x/(x**2 + 1), x) == (None, True)


def test_special_functions() -> None:
    found = bessel_solutions(-x, x)   # Airy
    assert found is not None and len(found) == 2
    for z in found:
        assert simplify(z.diff(x, 2) + x*z) == ZERO
    t = Symbol('t')
    assert str(whittaker_m(ZERO, Rational(1, 2), t)) == 't*exp(-t/2)*hyper((1,), (2,), t)'
    found = hypergeometric_solutions(-1/(4*x**2) - 1/(4*(x - 1)**2) + 1/(x*(x - 1)), x)
    assert found is not None and len(found) == 1


def test_upoly() -> None:
    a = Symbol('a')
    p = UPoly.of(x**3 - a**2*x, x)
    q, r = p.div(UPoly.of(a*x - a**2, x))
    assert r.is_zero and cancel(q.as_expr()*(a*x - a**2) - p.as_expr()) == ZERO
    # coefficients the typed Poly refuses: sqrt(2) and I
    p = UPoly.of(x**2 - 2, x)
    assert p.rem(UPoly.of(x - sqrt(2), x)).is_zero
    assert UPoly.of(x**2 + 1, x).rem(UPoly.of(x - 1, x)).as_expr() == Integer(2)
    L = _op(Integer(-1), ZERO, x**2)
    assert _indicial_at(L, UPoly.of(x, x)).degree() == 2
