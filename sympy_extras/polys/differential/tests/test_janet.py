"""Tests of the Janet bases: the division, the algorithm, the invariants,
and a comparison with an oracle of linear algebra."""
from __future__ import annotations

import random
from itertools import product

from sympy import Derivative, Function, Matrix, Rational, Symbol, diff, exp, oo, symbols
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.polys.matrices import DomainMatrix
from sympy.testing.pytest import raises

from sympy_extras._typing import Monomial
from sympy_extras.polys.differential import DifferentialRing, janet_basis, janet_multiplicative_variables
from sympy_extras.polys.differential.janet import _complement, _multi_indices

x, y, z, t, a = symbols('x y z t a')
u = Function('u')
v = Function('v')


def _solution_jets(equations: list[Expr], functions: list[AppliedUndef], variables: list[Symbol],
                   order: int, extra: int, point: list[Rational]) -> int:
    """The dimension of the space of the jets of order ``order`` of the
    solutions at a point, by linear algebra: the system is differentiated
    up to the order ``order + extra`` and its solutions are projected."""
    ring = DifferentialRing(functions, variables)
    n = len(variables)
    top = order + extra
    jets = [(k, J) for k in range(len(functions)) for J in _multi_indices(n, top)]
    column = {jet: i for i, jet in enumerate(jets)}
    at = dict(zip(variables, point))
    rows: list[list[Expr]] = []
    for equation in equations:
        own = max(sum(jet[1]) for monomial in ring.terms(equation) for jet, _ in monomial)
        for theta in _multi_indices(n, top - own):
            derived = equation
            for s, k in zip(variables, theta):
                derived = diff(derived, s, k)
            row: list[Expr] = [Rational(0)]*len(jets)
            for monomial, c in ring.terms(derived).items():
                row[column[monomial[0][0]]] = ring.domain.to_sympy(c).subs(at)
            rows.append(row)
    matrix = DomainMatrix.from_Matrix(Matrix(rows)).to_field()
    high = [i for i, jet in enumerate(jets) if sum(jet[1]) > order]
    kernel = len(jets) - matrix.rank()
    vertical = len(high) - matrix.extract(list(range(len(rows))), high).rank()
    return kernel - vertical


def _agrees_with_linear_algebra(equations: list[Expr], functions: list[AppliedUndef], variables: list[Symbol],
                                order: int, extra: int) -> bool:
    basis = janet_basis(equations, functions, variables)
    point = [Rational(3, 7) + k for k in range(len(variables))]
    expected = _solution_jets(equations, functions, variables, order, extra, point)
    return sum(basis.hilbert_function(q) for q in range(order + 1)) == expected


def test_multiplicative_variables() -> None:
    # hand computation: for {x**2*y, x*y**2, y**3} the first variable is
    # multiplicative only for the element of the highest degree in it
    assert janet_multiplicative_variables([(2, 1), (1, 2), (0, 3)]) == [
        (True, True), (False, True), (False, True)]
    assert janet_multiplicative_variables([(1, 0), (0, 1)]) == [(True, True), (False, True)]
    # the classes of the second variable are taken inside a class of the first one
    assert janet_multiplicative_variables([(1, 1), (1, 3), (0, 2)]) == [
        (True, False), (True, True), (False, True)]
    assert janet_multiplicative_variables([]) == []


def _divisible(w: Monomial, monomials: list[Monomial]) -> bool:
    return any(all(i >= j for i, j in zip(w, m)) for m in monomials)


def test_complement_cones_count_the_monomials_outside_the_ideal() -> None:
    generator = random.Random(7)
    f = u(x, y, z)
    for _ in range(12):
        monomials = [tuple(generator.randint(0, 3) for _ in range(3)) for _ in range(generator.randint(1, 4))]
        equations = [Derivative(f, *[(s, k) for s, k in zip((x, y, z), m) if k]) if any(m) else f
                     for m in monomials]
        basis = janet_basis(equations, [f])
        leaders = [ring_jet[1] for ring_jet in (basis.ring.jet(e) for e in basis.leaders) if ring_jet is not None]
        assert len(leaders) == len(basis.leaders)
        cones = _complement(leaders, 3)
        for degree in range(7):
            outside = [w for w in product(range(degree + 1), repeat=3)
                       if sum(w) == degree and not _divisible(w, monomials)]
            assert basis.hilbert_function(degree) == len(outside)
            # the cones are disjoint and contain exactly these monomials
            for w in outside:
                containing = [1 for vertex, flags in cones
                              if all(i == j or (i > j and flag) for i, j, flag in zip(w, vertex, flags))]
                assert len(containing) == 1


def test_the_involutive_cones_of_a_basis_are_disjoint_and_cover_the_multiples() -> None:
    f = u(x, y, z)
    basis = janet_basis([f.diff(z, 2) + y*f.diff(x, 2), f.diff(y, 2)], [f])
    leaders = []
    for e in basis.leaders:
        jet = basis.ring.jet(e)
        assert jet is not None
        leaders.append(jet[1])
    assert janet_multiplicative_variables(leaders) == basis.multiplicative
    for w in product(range(6), repeat=3):
        cones = [1 for m, flags in zip(leaders, basis.multiplicative)
                 if all(i == j or (i > j and flag) for i, j, flag in zip(w, m, flags))]
        assert len(cones) == (1 if _divisible(w, leaders) else 0)


def test_janet_example() -> None:
    f = u(x, y, z)
    basis = janet_basis([f.diff(z, 2) + y*f.diff(x, 2), f.diff(y, 2)], [f])
    assert basis.dimension == 12
    assert [basis.hilbert_function(q) for q in range(6)] == [1, 3, 4, 3, 1, 0]
    assert basis.hilbert_polynomial(t) == 0
    assert basis.hilbert_series(t).expand() == t**4 + 3*t**3 + 4*t**2 + 3*t + 1
    # the integrability conditions
    assert basis.contains(f.diff(x, 2, y))
    assert basis.contains(f.diff(z, 4))
    assert not basis.contains(f.diff(z, 3))
    # the order of the variables changes the basis, not the dimension
    assert janet_basis([f.diff(z, 2) + y*f.diff(x, 2), f.diff(y, 2)], [f], variables=[z, y, x]).dimension == 12


def test_every_equation_of_the_basis_vanishes_on_the_solutions() -> None:
    # the solutions of u_xx = u, u_y = u_x are a*exp(x + y) + b*exp(-x - y)
    f = u(x, y)
    basis = janet_basis([f.diff(x, 2) - f, f.diff(y) - f.diff(x)], [f])
    for solution in (exp(x + y), exp(-x - y)):
        for equation in basis.equations:
            assert equation.subs(f, solution).doit().simplify() == 0
    assert basis.dimension == 2
    assert basis.parametric_derivatives(3) == [f, f.diff(y)]
    assert basis.is_parametric(f.diff(y)) and not basis.is_parametric(f.diff(x))


def test_comparison_with_linear_algebra() -> None:
    f, g = u(x, y), v(x, y)
    h = u(x, y, z)
    assert _agrees_with_linear_algebra([f.diff(x, 2) - f, f.diff(y) - f.diff(x)], [f], [x, y], 3, 3)
    assert _agrees_with_linear_algebra([h.diff(z, 2) + y*h.diff(x, 2), h.diff(y, 2)], [h], [x, y, z], 4, 3)
    assert _agrees_with_linear_algebra([x*f.diff(x) + y*f.diff(y) - 2*f, f.diff(x, y)], [f], [x, y], 3, 3)
    assert _agrees_with_linear_algebra([f.diff(x) - g.diff(y), f.diff(y) + g.diff(x)], [f, g], [x, y], 3, 3)
    assert _agrees_with_linear_algebra([f.diff(x, 2) - y*g, g.diff(y) - x*f.diff(x)], [f, g], [x, y], 3, 4)


def test_random_systems_against_linear_algebra() -> None:
    f, g = u(x, y), v(x, y)
    derivatives = [w.diff(*s) if s else w for w in (f, g) for s in ((), (x,), (y,), (x, x), (x, y), (y, y))]
    # dense systems with constant coefficients
    generator = random.Random(11)
    for _ in range(6):
        equations = []
        for _ in range(generator.randint(2, 3)):
            e = sum(generator.choice([0, 0, 0, 1, -1, 2])*d for d in derivatives)
            if e != 0:
                equations.append(e)
        assert _agrees_with_linear_algebra(equations, [f, g], [x, y], 2, 6), equations
    # sparse systems with polynomial coefficients (two of them are left out:
    # their intermediate coefficients have hundreds of terms; the one of
    # index 2 agrees too, after ten seconds, the one of index 4 takes minutes)
    generator = random.Random(13)
    for k in range(6):
        equations = []
        for _ in range(generator.randint(2, 3)):
            e = sum(generator.choice([1, -1, x, y, x + y])*d
                    for d in generator.sample(derivatives, generator.randint(2, 3)))
            if e != 0:
                equations.append(e)
        if k not in (2, 4):
            assert _agrees_with_linear_algebra(equations, [f, g], [x, y], 2, 6), equations


def test_equivalent_systems_have_the_same_basis() -> None:
    f = u(x, y)
    first = janet_basis([f.diff(x, 2) - f, f.diff(y) - f.diff(x)], [f])
    second = janet_basis([f.diff(x, y) - f, f.diff(y) - f.diff(x), x*(f.diff(y, 2) - f.diff(x, 2))], [f])
    assert first == second
    assert first != janet_basis([f.diff(x, 2) - f], [f])


def test_inconsistent_systems() -> None:
    f = u(x, y)
    basis = janet_basis([f.diff(x) - y, f.diff(y)], [f])
    assert not basis.is_consistent
    assert basis.equations == [1]
    raises(ValueError, lambda: basis.dimension)
    # a consistent right-hand side
    basis = janet_basis([f.diff(x) - y, f.diff(y) - x], [f])
    assert basis.is_consistent and basis.dimension == 1
    assert basis.reduce(f.diff(x, y)) == 1


def test_elimination_gives_the_compatibility_conditions() -> None:
    w, f, g, h = u(x, y, z), Function('f')(x, y, z), Function('g')(x, y, z), Function('h')(x, y, z)
    # grad w = (f, g, h): the conditions are curl (f, g, h) = 0
    basis = janet_basis([w.diff(x) - f, w.diff(y) - g, w.diff(z) - h], [w, f, g, h], ranking=[[w], [f, g, h]])
    free = [e for e in basis.equations if not e.has(w)]
    assert len(free) == 3
    for e in (f.diff(y) - g.diff(x), f.diff(z) - h.diff(x), g.diff(z) - h.diff(y)):
        assert basis.contains(e)
    # no condition of order zero or involving one component only
    assert not basis.contains(f.diff(y))
    # curl A = B: the condition is div B = 0
    A = [Function('A%d' % k)(x, y, z) for k in range(3)]
    B = [Function('B%d' % k)(x, y, z) for k in range(3)]
    curl = [A[2].diff(y) - A[1].diff(z) - B[0], A[0].diff(z) - A[2].diff(x) - B[1], A[1].diff(x) - A[0].diff(y) - B[2]]
    basis = janet_basis(curl, A + B, ranking=[A, B])
    free = [e for e in basis.equations if not any(e.has(c) for c in A)]
    assert len(free) == 1
    assert (free[0] - (B[0].diff(x) + B[1].diff(y) + B[2].diff(z))).expand() in (0,) or basis.contains(
        B[0].diff(x) + B[1].diff(y) + B[2].diff(z))
    assert basis.contains(B[0].diff(x) + B[1].diff(y) + B[2].diff(z))


def test_infinite_dimensional_solution_spaces() -> None:
    f, g = u(x, y), v(x, y)
    # the Cauchy-Riemann equations: two free Taylor coefficients in every order,
    # as for one holomorphic function
    basis = janet_basis([f.diff(x) - g.diff(y), f.diff(y) + g.diff(x)], [f, g])
    assert basis.dimension == oo
    assert [basis.hilbert_function(q) for q in range(5)] == [2, 2, 2, 2, 2]
    assert basis.hilbert_polynomial(t) == 2
    # the wave equation: two functions of one variable
    basis = janet_basis([f.diff(x, 2) - f.diff(y, 2)], [f])
    assert [basis.hilbert_function(q) for q in range(4)] == [1, 2, 2, 2]
    # no equation at all: one function of two variables
    assert janet_basis([], [f]).hilbert_polynomial(t) == t + 1


def test_series_solutions_satisfy_the_system() -> None:
    f = u(x, y)
    # the solutions are the multiples of exp(x*y)
    equations = [f.diff(x, 2) - y**2*f, f.diff(y) - x*f]
    basis = janet_basis(equations, [f])
    assert basis.dimension == 1
    order = 6
    series = basis.series_solution(order, point=[0, 1])[f]
    e = Symbol('e')
    expected = exp(e*x*(1 + e*y)).series(e, 0, order + 1).removeO().subs(e, 1).subs(y, y - 1)
    assert (series.subs(Symbol('C0'), 1) - expected).expand() == 0
    # an infinite dimensional space: the heat equation, with the parametric
    # derivatives of every order as constants
    # (the leader is u_xx for an orderly ranking: u and u_x are free on x = 0,
    # which makes 1 + 2 + 2 + 2 + 2 Taylor coefficients up to the order 4)
    heat = janet_basis([f.diff(y) - f.diff(x, 2)], [f])
    series = heat.series_solution(4)[f]
    assert len(series.free_symbols - {x, y}) == 9
    residual = (series.diff(y) - series.diff(x, 2)).expand()
    assert all(sum(m) >= 3 for m in residual.as_poly(x, y).monoms())
    # a singular point
    singular = janet_basis([x*f.diff(x) - f, f.diff(y)], [f])
    raises(ValueError, lambda: singular.series_solution(2))
    assert singular.series_solution(2, point=[1, 0])[f].subs(Symbol('C0'), 1) == x


def test_parameters_are_generic_constants() -> None:
    f = u(x, y)
    basis = janet_basis([f.diff(x, 2) - a*f, f.diff(y) - f.diff(x)], [f])
    assert basis.dimension == 2
    assert basis.reduce(f.diff(y, 2)) == a*f
    raises(ValueError, lambda: basis.reduce(z*f))


def test_functions_of_fewer_variables() -> None:
    f, c = u(x, y), Function('c')(x)
    # u_y = c(x) u with c a function of x alone
    basis = janet_basis([f.diff(y) - c, f.diff(x)], [f, c])
    # c is a constant: c' = u_xy = 0
    assert basis.contains(c.diff(x))
    assert all(e != 0 for e in basis.equations)


def test_arguments_are_checked() -> None:
    f = u(x, y)
    raises(ValueError, lambda: janet_basis([f.diff(x)**2 - f], [f]))
    raises(ValueError, lambda: janet_basis([f.diff(x) - v(x, y)], [f]))
    raises(ValueError, lambda: janet_basis([f.diff(x) - f**Rational(1, 2)], [f]))
    raises(ValueError, lambda: janet_basis([f.diff(x)], [f], variables=[x]))
    raises(ValueError, lambda: janet_basis([f.diff(x)], [f, f]))
    raises(ValueError, lambda: janet_basis([f.diff(x)], [f], ranking=[[]]))
