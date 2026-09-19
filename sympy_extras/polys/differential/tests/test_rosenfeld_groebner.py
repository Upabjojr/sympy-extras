"""Tests of the Rosenfeld-Groebner algorithm."""
from __future__ import annotations

from sympy import Function, Rational, cos, exp, expand, simplify, sin, symbols
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras.polys.differential import (DifferentialPolynomial, RadicalDifferentialIdeal, janet_basis,
    rosenfeld_groebner)
from sympy_extras.polys.differential.rosenfeld_groebner import _characteristic_set, _delta_polynomials

x, y, t, c, g = symbols('x y t c g')
u = Function('u')(x)


def _vanishes(expr: Expr, solution: dict[Basic, Basic]) -> bool:
    return simplify(expr.xreplace(solution).doit()) == 0


def _is_regular(ideal: RadicalDifferentialIdeal) -> bool:
    """The defining properties of the components: autoreduced and coherent
    equations, with the initials and separants among the inequations up
    to factors."""
    for component in ideal.components:
        chain = component.chain
        for k, p in enumerate(chain):
            if not all(p.is_reduced(q) for j, q in enumerate(chain) if j != k):
                return False
        for delta in _delta_polynomials(chain):
            if not component.contains(delta.to_expr()):
                return False
        for p in chain:
            if component.contains(p.initial().to_expr()) or component.contains(p.separant().to_expr()):
                return False
    return True


def test_general_and_singular_solutions() -> None:
    # u'**2 = 4 u: the parabolas (x + c)**2 and the singular solution 0
    ideal = rosenfeld_groebner([u.diff(x)**2 - 4*u], [u])
    assert len(ideal) == 2 and _is_regular(ideal)
    general, singular = ideal.components
    assert general.equations == [u.diff(x)**2 - 4*u] and general.inequations == [u.diff(x)]
    assert singular.equations == [u]
    assert general.contains(u.diff(x, 2) - 2) and not singular.contains(u.diff(x, 2) - 2)
    assert ideal.contains(u.diff(x)*(u.diff(x, 2) - 2)) and not ideal.contains(u.diff(x, 2) - 2)
    for equation in general.equations:
        assert _vanishes(equation, {u: (x + c)**2})


def test_clairaut_equation() -> None:
    # u = x u' + u'**2: the lines c x + c**2 and their envelope -x**2/4
    ideal = rosenfeld_groebner([u - x*u.diff(x) - u.diff(x)**2], [u])
    assert len(ideal) == 2 and _is_regular(ideal)
    general, singular = ideal.components
    assert general.contains(u.diff(x, 2)) and not singular.contains(u.diff(x, 2))
    assert [expand(e) for e in singular.equations] == [x**2 + 4*u]
    assert _vanishes(general.equations[0], {u: c*x + c**2})
    assert _vanishes(singular.equations[0], {u: -x**2/4})


def test_inconsistent_systems() -> None:
    assert not rosenfeld_groebner([u.diff(x) - u, u.diff(x, 2) - u - 1], [u]).is_consistent
    assert not rosenfeld_groebner([u.diff(x)**2 + 1, u.diff(x, 2) - 1], [u]).is_consistent
    assert not rosenfeld_groebner([u.diff(x) - u], [u], inequations=[u.diff(x, 2) - u]).is_consistent
    assert rosenfeld_groebner([u.diff(x) - u], [u]).is_consistent
    # everything vanishes on the empty set
    assert rosenfeld_groebner([u, u - 1], [u]).contains(u.diff(x) - 7)


def test_inequations_select_the_components() -> None:
    ideal = rosenfeld_groebner([u.diff(x)**2 - 4*u], [u], inequations=[u])
    assert len(ideal) == 1 and ideal.contains(u.diff(x, 2) - 2)
    ideal = rosenfeld_groebner([u.diff(x)*(u.diff(x) - 1)], [u], inequations=[u.diff(x)])
    assert ideal.contains(u.diff(x) - 1)
    ideal = rosenfeld_groebner([u.diff(x)*(u.diff(x) - 1)], [u])
    assert len(ideal) == 2 and not ideal.contains(u.diff(x) - 1) and ideal.contains(u.diff(x, 2))


def test_pendulum() -> None:
    # the hidden constraints of the pendulum, a differential-algebraic system of index 3
    X, Y, L = Function('X')(t), Function('Y')(t), Function('L')(t)
    system = [X.diff(t, 2) + L*X, Y.diff(t, 2) + L*Y + g, X**2 + Y**2 - 1]
    ideal = rosenfeld_groebner(system, [L, X, Y])
    assert _is_regular(ideal)
    assert len(ideal) == 3
    general = ideal.components[0]
    # the velocity constraint, the multiplier, and its derivative (by hand:
    # L = X'**2 + Y'**2 - g Y, and the energy gives L' = -3 g Y')
    assert general.contains(X*X.diff(t) + Y*Y.diff(t))
    assert general.contains(L - X.diff(t)**2 - Y.diff(t)**2 + g*Y)
    assert general.contains(L.diff(t) + 3*g*Y.diff(t))
    assert not general.contains(X) and not general.contains(L.diff(t))
    # the equilibria are the other components
    equilibria = ideal.components[1:]
    assert all(component.contains(X) and component.contains(Y**2 - 1) for component in equilibria)
    # there the multiplier balances the weight: L = -g Y with Y = 1 or -1
    assert all(component.contains(L + g*Y) for component in equilibria)
    assert not general.contains(L + g*Y)
    for equation in system:
        assert ideal.contains(equation)


def test_elimination_of_the_coordinates_of_the_pendulum() -> None:
    X, Y, L = Function('X')(t), Function('Y')(t), Function('L')(t)
    system = [X.diff(t, 2) + L*X, Y.diff(t, 2) + L*Y + g, X**2 + Y**2 - 1]
    ideal = rosenfeld_groebner(system, [X, Y, L], ranking=[[X, Y], [L]])
    general = ideal.components[0]
    alone = [e for e in general.equations if not e.has(X) and not e.has(Y)]
    # one equation for the multiplier alone, of the second order
    assert len(alone) == 1
    assert general.ring.leader(alone[0]) == L.diff(t, 2)


def test_partial_differential_system() -> None:
    # the example of Boulier, Lazard, Ollivier and Petitot (1995)
    w, v = Function('w')(x, y), Function('v')(x, y)
    system = [w.diff(x)**2 - 4*w, w.diff(x, y)*v.diff(y) - w + 1, v.diff(x, 2) - w.diff(x)]
    ideal = rosenfeld_groebner(system, [w, v])
    assert len(ideal) == 1 and _is_regular(ideal)
    # the published component contains u_y**2 - 2 u
    assert ideal.contains(w.diff(y)**2 - 2*w)
    for equation in system:
        assert ideal.contains(equation)
    assert not ideal.contains(w.diff(y))


def test_linear_systems_agree_with_janet_bases() -> None:
    f = Function('f')(x, y)
    # the solutions are the multiples of exp(x*y)
    system = [f.diff(x, 2) - y**2*f, f.diff(y) - x*f]
    ideal = rosenfeld_groebner(system, [f])
    basis = janet_basis(system, [f])
    assert len(ideal) == 1 and basis.dimension == 1
    candidates = [f.diff(x) - y*f, f.diff(x, y) - (1 + x*y)*f, f.diff(x, y) - x*y*f, f.diff(y, 3) - x**3*f,
                  f.diff(y, 3) - x**2*f, f]
    for e in candidates:
        assert ideal.contains(e) == basis.contains(e)
    assert [ideal.contains(e) for e in candidates] == [True, True, False, True, False, False]


def test_solutions_satisfy_the_components() -> None:
    # u'' + u = 0 with u'**2 + u**2 = 1: the sines, and no singular solution
    # other than the ones excluded by the separant
    ideal = rosenfeld_groebner([u.diff(x)**2 + u**2 - 1], [u])
    general = ideal.components[0]
    assert general.contains(u.diff(x, 2) + u)
    assert _vanishes(general.equations[0], {u: sin(x + c)})
    singular = ideal.components[1:]
    assert len(singular) == 2
    assert {str(component.equations[0]) for component in singular} == {'u(x) - 1', 'u(x) + 1'}
    # a system with two functions: v = u', u'' = -u
    v = Function('v')(x)
    ideal = rosenfeld_groebner([v - u.diff(x), v.diff(x) + u], [u, v])
    assert len(ideal) == 1
    for equation in ideal.components[0].equations:
        assert _vanishes(equation, {u: cos(x), v: -sin(x)})
    assert ideal.contains(u.diff(x, 2) + u) and ideal.contains(v.diff(x, 2) + v)
    assert ideal.contains((u**2 + v**2).diff(x)) and not ideal.contains(u**2 + v**2 - 1)


def test_coefficients_in_the_variable_and_parameters() -> None:
    ideal = rosenfeld_groebner([x*u.diff(x) - c*u], [u])
    assert len(ideal) == 1
    assert ideal.contains(x**2*u.diff(x, 2) - c*(c - 1)*u)
    assert not ideal.contains(x**2*u.diff(x, 2) - c*u)
    ideal = rosenfeld_groebner([u.diff(x) - u/x], [u])
    assert ideal.contains(u.diff(x, 2))
    assert _vanishes(ideal.components[0].equations[0], {u: 3*x})
    assert _vanishes(rosenfeld_groebner([u.diff(x) - 2*x*u], [u]).components[0].equations[0], {u: exp(x**2)})


def test_characteristic_sets() -> None:
    ring = rosenfeld_groebner([u], [u]).ring
    polynomials = [DifferentialPolynomial.from_expr(ring, e)
                   for e in (u.diff(x, 2)*u - 1, u.diff(x)**3 - u, u.diff(x)**2 + u**5, u.diff(x, 3))]
    chain = _characteristic_set(polynomials)
    # the lowest rank first; u'' u - 1 is not reduced with respect to u'**2 + u**5
    assert [p.to_expr() for p in chain] == [u.diff(x)**2 + u**5]
    assert _delta_polynomials(chain) == []


def test_arguments_are_checked() -> None:
    raises(ValueError, lambda: rosenfeld_groebner([u.diff(x) - sin(u)], [u]))
    raises(ValueError, lambda: rosenfeld_groebner([u.diff(x) - u**Rational(1, 2)], [u]))
    raises(ValueError, lambda: rosenfeld_groebner([u.diff(x) - Function('w')(x)], [u]))
    assert not rosenfeld_groebner([u.diff(x)], [u], inequations=[u - u]).is_consistent


def test_random_autonomous_equations() -> None:
    # u'**2 = P(u): on the general solution 2 u'' = P'(u) (differentiate and
    # divide by u'), on the singular solutions P(u) = 0 and u'' = 0
    import random
    generator = random.Random(17)
    for _ in range(6):
        P = sum(generator.randint(-3, 3)*u**k for k in range(3)) + generator.choice([1, 2])*u**3
        ideal = rosenfeld_groebner([u.diff(x)**2 - P], [u])
        assert _is_regular(ideal)
        general = ideal.components[0]
        consequence = 2*u.diff(x, 2) - P.diff(u)
        assert general.contains(consequence)
        assert general.contains(2*u.diff(x, 3) - P.diff(u, 2)*u.diff(x))
        assert not general.contains(u.diff(x)) and not general.contains(P)
        assert ideal.contains(u.diff(x)*consequence)
        for singular in ideal.components[1:]:
            assert singular.contains(P) and singular.contains(u.diff(x))
        assert len(ideal) >= 2 and not ideal.contains(consequence)


def test_functions_of_fewer_variables() -> None:
    w, a = Function('w')(x, y), Function('a')(x)
    # no component at all was returned: the derivative of a(x) with respect to
    # y, which SymPy evaluates to zero, was lost when the factors were
    # converted back, and the branch was dropped as if it had no solution
    ideal = rosenfeld_groebner([w.diff(y) - a*w, w.diff(x)], [w, a])
    assert ideal.is_consistent
    # w_xy = a' w + a w_x = 0: either w = 0 or a is constant
    assert ideal.contains(a.diff(x)*w)
    assert not ideal.contains(w) and not ideal.contains(a.diff(x))
    assert len(ideal) == 2
    assert all(e != 0 for c in ideal.components for e in c.equations)
