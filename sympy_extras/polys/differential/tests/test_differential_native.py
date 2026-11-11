"""Tests of the native differential algebra (on objects of the core),
checked by substitution of known solutions, by hand computations and by
the known invariants of classical systems."""
from __future__ import annotations

import random

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import Eq, Function, Integer, Rational, Symbol, diff, expand, oo
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef
from sympy_extras.functions import cos, exp, sin
from sympy_extras.polys.differential import (DifferentialPolynomial, DifferentialRing, janet_basis,
    janet_multiplicative_variables, rosenfeld_groebner)
from sympy_extras.polys.regularchains import RegularChain

x, y, z, t, a, c, g = [Symbol(name) for name in 'xyztacg']
u, v, w = Function('u'), Function('v'), Function('w')


def _vanishes(expr: Expr, solution: dict[AppliedUndef, Expr]) -> bool:
    """Whether the expression is zero on the solution, by substitution
    (and at a few rational points, for the trigonometric solutions)."""
    replaced = expr
    for f, value in solution.items():
        replaced = replaced.subs(f, value)
    replaced = expand(replaced.doit())
    if replaced == 0:
        return True
    for k in range(1, 4):
        value = replaced.subs(x, Rational(k, 7)).subs(c, Rational(2, k + 4)).subs(t, Rational(k, 3))
        if abs(float(value.evalf(30))) > 1e-20:
            return False
    return True


def test_rankings_and_leaders() -> None:
    U, V = u(x, y), v(x, y)
    ring = DifferentialRing([U, V])
    assert ring.is_orderly and ring.n == 2
    assert ring.leader(U + V.diff(x)) == V.diff(x)
    assert ring.leader(U.diff(y) + V.diff(x)) == U.diff(y)
    assert ring.leader(x**5*U + U.diff(x)**2*V.diff(x, y)) == V.diff(x, y)
    eliminating = DifferentialRing([U, V], ranking=[[V], [U]])
    assert not eliminating.is_orderly and eliminating.leader(U.diff(x, 3) + V) == V
    assert ring.jet(U.diff(x, y, y)) == (0, (1, 2)) and ring.to_expr((1, (2, 0))) == V.diff(x, 2)


def test_conversion_and_derivatives_agree_with_the_core() -> None:
    U, V = u(x, y), v(x, y)
    ring = DifferentialRing([U, V], parameters=[a])
    e = (a*U.diff(x, y)**2*V - U/x + 3)/(y + 1)
    assert expand(ring.from_terms(ring.terms(e)) - e) == 0
    generator = random.Random(5)
    jets = [U, V, U.diff(x), U.diff(y), V.diff(x), U.diff(x, y)]
    for _ in range(15):
        f: Expr = Integer(0)
        for _ in range(generator.randint(1, 3)):
            term: Expr = generator.choice([Integer(1), Integer(-3), x, x*y + 1])
            for _ in range(generator.randint(1, 3)):
                term = term*generator.choice(jets)
            f = f + term
        p = DifferentialPolynomial(ring, ring.integral(ring.terms(f)))
        # (compared through the ring: the core's diff writes u_xyx for u_xxy)
        for i, s in enumerate((x, y)):
            assert ring.terms(p.derivative(i).to_expr() - diff(f, s)) == {}
        q = DifferentialPolynomial(ring, ring.integral(ring.terms(f + U)))
        assert ring.terms((p*q).to_expr() - f*(f + U)) == {}


def test_errors() -> None:
    U = u(x, y)
    ring = DifferentialRing([U])
    raises(ValueError, lambda: ring.terms(1/U))
    raises(ValueError, lambda: ring.terms(sin(U)))
    raises(ValueError, lambda: ring.terms(Symbol('b')*U))
    raises(ValueError, lambda: ring.terms(w(x, y)))
    raises(ValueError, lambda: ring.leader(x + 1))
    raises(TypeError, lambda: untyped(DifferentialRing)([x]))
    raises(ValueError, lambda: DifferentialRing([w(x, 2*y)]))
    raises(ValueError, lambda: janet_basis([U.diff(x)**2 - U], [U]))
    raises(ValueError, lambda: rosenfeld_groebner([u(x).diff(x) - u(x)**Rational(1, 2)], [u(x)]))


def test_ritt_reduction_by_hand() -> None:
    U, V = u(x, y), v(x, y)
    ring = DifferentialRing([U, V])
    chain = [DifferentialPolynomial.from_expr(ring, U.diff(x)**2 - 4*U)]
    p = DifferentialPolynomial.from_expr(ring, U.diff(x, y)*V.diff(y) - U + 1)
    # 2 u_x p = 2 u_y v_y - u_x (u - 1) modulo the derivative of the chain
    assert expand(p.partial_remainder(chain).to_expr() + (2*U.diff(y)*V.diff(y) - U.diff(x)*(U - 1))) == 0
    # 2 u_x u_xx u_x**3 = 4 u_x**4 = 64 u**2 modulo the chain
    assert expand(DifferentialPolynomial.from_expr(ring, U.diff(x, 2)*U.diff(x)**3).remainder(chain).to_expr()
                  - U**2) == 0
    # the primitive part over ZZ[x, y], the sign of the highest monomial
    assert DifferentialPolynomial.from_expr(ring, (4*x*U.diff(x)**2 - 6*x**2*U)/(3*y)).to_expr() == \
        2*U.diff(x)**2 - 3*x*U


def test_janet_example() -> None:
    F = u(x, y, z)
    basis = janet_basis([F.diff(z, 2) + y*F.diff(x, 2), F.diff(y, 2)], [F])
    # Janet (1929): integrability conditions of the orders 3 and 4, 12 solutions
    assert len(basis.equations) == 7 and basis.dimension == 12
    assert [basis.hilbert_function(q) for q in range(6)] == [1, 3, 4, 3, 1, 0]
    assert expand(basis.hilbert_series(t) - (t**4 + 3*t**3 + 4*t**2 + 3*t + 1)) == 0
    assert basis.hilbert_polynomial(t) == 0
    assert basis.contains(F.diff(z, 4)) and not basis.contains(F.diff(z, 3))
    leaders = [jet[1] for jet in (basis.ring.jet(e) for e in basis.leaders) if jet is not None]
    assert janet_multiplicative_variables(leaders) == basis.multiplicative


def test_infinite_solution_spaces() -> None:
    F, G = u(x, y), v(x, y)
    # Cauchy-Riemann: two Taylor coefficients per order, one holomorphic function
    basis = janet_basis([F.diff(x) - G.diff(y), F.diff(y) + G.diff(x)], [F, G])
    assert basis.dimension == oo and basis.hilbert_polynomial(t) == 2
    # the wave equation: two functions of one variable
    assert [janet_basis([F.diff(x, 2) - F.diff(y, 2)], [F]).hilbert_function(q) for q in range(4)] == [1, 2, 2, 2]
    # three variables, no equation: (s + 1)(s + 2)/2 coefficients of order s
    assert expand(janet_basis([], [u(x, y, z)]).hilbert_polynomial(t) - (t + 1)*(t + 2)/2) == 0


def test_series_solutions() -> None:
    F = u(x, y)
    # the solutions are the multiples of exp(x*y): the Taylor polynomial at (0, 1)
    basis = janet_basis([F.diff(x, 2) - y**2*F, Eq(F.diff(y), x*F)], [F])
    series = basis.series_solution(4, point=[Integer(0), Integer(1)])[F].subs(Symbol('C0'), 1)
    f = exp(x*y)
    expected: Expr = Integer(0)
    for i in range(5):
        for j in range(5 - i):
            value = diff(f, x, i, y, j) if i and j else diff(f, x, i) if i else diff(f, y, j) if j else f
            expected = expected + value.subs(x, 0).subs(y, 1)*x**i*(y - 1)**j/(_factorial(i)*_factorial(j))
    assert expand(series - expected) == 0
    # the sine and the cosine
    assert janet_basis([u(x).diff(x, 2) + u(x)], [u(x)]).series_solution(3)[u(x)] == \
        Symbol('C0') - Symbol('C0')*x**2/2 + Symbol('C1')*x - Symbol('C1')*x**3/6


def _factorial(n: int) -> int:
    return 1 if n < 2 else n*_factorial(n - 1)


def test_a_singular_point_is_refused() -> None:
    F = u(x, y)
    basis = janet_basis([F.diff(x) - F/(x + y), F.diff(y) - F/(x + y)], [F])
    # the coefficient x + y of the leaders vanishes at the origin: the port
    # substituted with the core's xreplace, which leaves 0 + 0 unevaluated,
    # and returned the Taylor polynomial zoo*C0*x + zoo*C0*y + C0
    raises(ValueError, lambda: basis.series_solution(2))
    assert basis.series_solution(1, point=[Integer(1), Integer(0)])[F].subs(Symbol('C0'), 1) == x + y


def test_general_and_singular_solutions() -> None:
    U = u(x)
    ideal = rosenfeld_groebner([U.diff(x)**2 - 4*U], [U])
    assert len(ideal) == 2
    general, singular = ideal.components
    assert general.equations == [U.diff(x)**2 - 4*U] and general.inequations == [U.diff(x)]
    assert singular.equations == [U]
    assert _vanishes(general.equations[0], {U: (x + c)**2})
    assert general.contains(U.diff(x, 2) - 2) and not singular.contains(U.diff(x, 2) - 2)
    assert ideal.contains(U.diff(x)*(U.diff(x, 2) - 2)) and not ideal.contains(U.diff(x, 2) - 2)
    for component in ideal.components:
        assert component.is_consistent
        for chain in component.regular_chains():
            assert isinstance(chain, RegularChain)
    # u'' + u = 0 is implied by u'**2 + u**2 = 1 on the general component
    sines = rosenfeld_groebner([U.diff(x)**2 + U**2 - 1], [U])
    assert sines.components[0].contains(U.diff(x, 2) + U)
    assert _vanishes(sines.components[0].equations[0], {U: sin(x + c)})
    V = v(x)
    system = rosenfeld_groebner([V - U.diff(x), V.diff(x) + U], [U, V])
    for equation in system.components[0].equations:
        assert _vanishes(equation, {U: cos(x), V: -sin(x)})


def test_pendulum() -> None:
    X, Y, L = Function('X')(t), Function('Y')(t), Function('L')(t)
    system = [X.diff(t, 2) + L*X, Y.diff(t, 2) + L*Y + g, X**2 + Y**2 - 1]
    ideal = rosenfeld_groebner(system, [L, X, Y])
    assert len(ideal) == 3
    general = ideal.components[0]
    # by hand: L = X'**2 + Y'**2 - g Y, and the energy gives L' = -3 g Y'
    assert general.contains(L - X.diff(t)**2 - Y.diff(t)**2 + g*Y)
    assert general.contains(L.diff(t) + 3*g*Y.diff(t)) and not general.contains(L.diff(t))
    assert all(component.contains(L + g*Y) and component.contains(X) for component in ideal.components[1:])
    assert all(ideal.contains(e) for e in system)


def test_partial_differential_system() -> None:
    # Boulier, Lazard, Ollivier, Petitot (1995): the component contains u_y**2 - 2u
    W, V = w(x, y), v(x, y)
    system = [W.diff(x)**2 - 4*W, W.diff(x, y)*V.diff(y) - W + 1, V.diff(x, 2) - W.diff(x)]
    ideal = rosenfeld_groebner(system, [W, V])
    assert len(ideal) == 1 and ideal.contains(W.diff(y)**2 - 2*W) and not ideal.contains(W.diff(y))


def test_factors_split_the_system() -> None:
    U = u(x)
    ideal = rosenfeld_groebner([(U.diff(x) - U)*(U.diff(x) + U)*(U - x)], [U])
    assert sorted(str(component.equations) for component in ideal.components) == \
        ['[-u(x) + Derivative(u(x), x)]', '[-x + u(x)]', '[u(x) + Derivative(u(x), x)]']
    for solution in (exp(x), exp(-x), x):
        assert sum(1 for component in ideal.components
                   if _vanishes(component.equations[0], {U: solution})) == 1


def test_functions_of_fewer_variables() -> None:
    W, A = w(x, y), Function('a')(x)
    # the derivative of a(x) in y is zero in the core too: the factors are
    # converted back from the symbols of the derivatives, not from expressions
    ideal = rosenfeld_groebner([W.diff(y) - A*W, W.diff(x)], [W, A])
    assert len(ideal) == 2 and ideal.contains(A.diff(x)*W)
    assert not ideal.contains(W) and not ideal.contains(A.diff(x))
    basis = janet_basis([u(x, y).diff(y) - Function('k')(x), u(x, y).diff(x)], [u(x, y), Function('k')(x)])
    assert basis.contains(Function('k')(x).diff(x))
