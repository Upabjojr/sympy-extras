"""The native Lie point symmetries (:mod:`sympy_extras.solvers.lie`) and
similarity reductions (:mod:`sympy_extras.solvers.pde`) on objects of the
core; ``test_lie.py`` and ``test_pde.py`` test the SymPy faces."""
from __future__ import annotations

from sympy_extras.core import Derivative, Eq, Expr, Function, Integer, Rational, Symbol, diff, expand, oo
from sympy_extras.functions import exp, sin
from sympy_extras.solvers.lie import (JetSpace, Symmetry, check_symmetry, determining_system,
    infinitesimal_condition, leading_derivatives, symmetries, symmetry_janet_basis)
from sympy_extras.solvers.pde import check_pde_solution, pde_symmetries, similarity_reduction

x, t, m = Symbol('x'), Symbol('t'), Symbol('m')
u = Function('u')(x, t)
y = Function('y')(x)
heat = u.diff(t) - u.diff(x, 2)
burgers = u.diff(t) + u*u.diff(x) - u.diff(x, 2)
kdv = u.diff(t) + u*u.diff(x) + u.diff(x, 3)


def _generators(syms: list[Symmetry]) -> set[str]:
    return {s.generator() for s in syms}


def test_jet_space() -> None:
    jets = JetSpace([u])
    assert jets.p == 2 and jets.q == 1 and jets.x == (x, t) and jets.u == (Symbol('u'),)
    usym, u_x, u_xx, u_t, u_xt = [jets.jet(0, J) for J in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]]
    e = jets.to_jet(u.diff(t) - u.diff(x, 2) + u*u.diff(x, t))
    assert e == u_t - u_xx + usym*u_xt
    assert jets.from_jet(u_t) == u.diff(t)
    assert jets.order(e) == 2
    # mixed partials are canonical, whatever the order of differentiation
    assert jets.to_jet(diff(u, t, x)) == jets.to_jet(diff(u, x, t)) == u_xt
    assert jets.total_derivative(usym**2, 0) == 2*usym*u_x
    assert jets.total_derivative(x*u_x, 1) == x*u_xt
    # the jets of another instance are recognised by name
    assert JetSpace([u]).index(u_xt) == (0, (1, 1))
    assert jets.jets_in(e) == [(0, (0, 0)), (0, (0, 1)), (0, (1, 1)), (0, (2, 0))]
    # SymPy's itermonomials sorted by degree and printed form
    assert [str(mono) for mono in jets.monomials(1)] == ['1', 't', 'u', 'x']
    assert len(jets.monomials(3)) == 20


def test_prolongation_by_hand() -> None:
    jets = JetSpace([u])
    usym = jets.u[0]
    u_x, u_t, u_xx = jets.jet(0, (1, 0)), jets.jet(0, (0, 1)), jets.jet(0, (2, 0))
    # the Galilean boost of the heat equation (Olver, example 2.41), xi = 2t,
    # tau = 0, eta = -x u, prolonged by hand: phi^x = D_x eta - u_x D_x xi,
    # phi^t = D_t eta - u_x D_t xi, phi^xx = D_x phi^x - u_xx D_x xi
    galilean = Symmetry(jets, [2*t, 0], [-x*usym])
    phi = galilean.prolongation(2)
    assert expand(phi[(0, (1, 0))] - (-usym - x*u_x)) == 0
    assert expand(phi[(0, (0, 1))] - (-x*u_t - 2*u_x)) == 0
    assert expand(phi[(0, (2, 0))] - (-2*u_x - x*u_xx)) == 0
    assert galilean.characteristic() == (-x*usym - 2*t*u_x,)
    X = Symmetry(jets, [t, 0], [0])
    assert infinitesimal_condition(heat, u, X) == [-u_x]
    assert check_symmetry(heat, u, galilean) and not check_symmetry(heat, u, X)
    dx, dt = Symmetry(jets, [1, 0], [0]), Symmetry(jets, [0, 1], [0])
    assert (dx + 2*dt).xi == (Integer(1), Integer(2)) and (galilean - galilean).is_zero()
    assert dt.commutator(galilean) == 2*dx
    assert dx.commutator(galilean).eta == (-usym,)
    assert repr(dx) == "Symmetry(xi=(1, 0), eta=(0,))"
    assert galilean.generator() == "2*t*d/dx - u*x*d/du"


def test_leading_derivatives() -> None:
    jets = JetSpace([u])
    u_xx, u_t = jets.jet(0, (2, 0)), jets.jet(0, (0, 1))
    assert leading_derivatives(jets, [jets.to_jet(heat)]) == {u_xx: u_t}
    # issue #72: a quadratic is not solved, one root would be one branch
    try:
        leading_derivatives(jets, [u_xx**2 - u_t])
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError")
    # a denominator is cleared first
    assert leading_derivatives(jets, [1/u_xx - u_t]) == {u_xx: 1/u_t}


def test_known_symmetry_algebras() -> None:
    # dimensions from Olver, Applications of Lie Groups, examples 2.41-2.44
    # and Ibragimov's handbook: heat 6 (without superposition), Burgers 5,
    # KdV 4, the wave equation 7, nonlinear diffusion 4 for generic m
    cases = [(heat, 3, False, 6), (burgers, 2, True, 5), (kdv, 2, True, 4),
             (u.diff(t, 2) - u.diff(x, 2), 2, False, 7), (u.diff(t) - (u**m*u.diff(x)).diff(x), 2, True, 4)]
    for equation, degree, superposition, dimension in cases:
        found = symmetries(equation, u, degree=degree, superposition=superposition)
        assert len(found) == dimension, (equation, found)
        assert all(check_symmetry(equation, u, s) for s in found)
    assert {'d/dx', 'd/dt', 'u*d/du', 'x*d/dx + 2*t*d/dt'} <= _generators(symmetries(heat, u, 3, superposition=False))
    assert 'x*d/dx + 3*t*d/dt - 2*u*d/du' in _generators(symmetries(kdv, u))


def test_emden_fowler_and_oscillators() -> None:
    # y'' + 2 y'/x + y**5 = 0 has the scaling x d/dx - y/2 d/dy only; the
    # cubic oscillator y'' + y**3 the translation and x d/dx - y d/dy
    emden = y.diff(x, 2) + 2*y.diff(x)/x + y**5
    [X] = symmetries(emden, y)
    assert X.generator() == 'x*d/dx - y/2*d/dy' and check_symmetry(emden, y, X)
    oscillator = y.diff(x, 2) + y**3
    assert _generators(symmetries(oscillator, y)) == {'d/dx', 'x*d/dx - y*d/dy'}
    # the free particle: the eight projective symmetries
    assert len(symmetries(y.diff(x, 2), y)) == 8
    assert symmetry_janet_basis(y.diff(x, 2), y).dimension == 8


def test_basis_systems_and_irrational_coefficients() -> None:
    eq = y.diff(x) + 2*x*y - x*exp(-x**2)
    assert symmetries(eq, y) == []
    found = symmetries(eq, y, basis=[exp(-x**2)])
    assert any(s.xi == (Integer(0),) and s.eta == (exp(-x**2),) for s in found)
    assert all(check_symmetry(eq, y, s) for s in found)
    v = Function('v')(x, t)
    system = [Eq(u.diff(t), v.diff(x)), Eq(v.diff(t), u.diff(x))]
    found = symmetries(system, [u, v], degree=1)
    assert found and all(check_symmetry(system, [u, v], s) for s in found)
    # a coefficient outside QQ(parameters): sqrt(2) is kept as a constant
    found = symmetries(y.diff(x, 2) + Rational(2)**Rational(1, 2)*y.diff(x), y)
    assert found and all(check_symmetry(y.diff(x, 2) + Rational(2)**Rational(1, 2)*y.diff(x), y, s) for s in found)


def test_janet_basis_of_the_determining_system() -> None:
    assert symmetry_janet_basis(y.diff(x, 3) + y*y.diff(x, 2), y).dimension == 2
    assert symmetry_janet_basis(kdv, u).dimension == 4
    assert symmetry_janet_basis(heat, u).dimension == oo
    system, unknowns, variables = determining_system(y.diff(x, 2), y)
    assert [str(f) for f in unknowns] == ['xi0(x, y)', 'eta0(x, y)'] and len(system) == 4


def test_similarity_reductions() -> None:
    jets = JetSpace([u])
    # stationary solutions of Burgers' equation: F F' = F''
    r = similarity_reduction(burgers, u, Symmetry(jets, [0, 1], [0]))
    F, z = r.F, r.variable
    assert r.z == x and r.solution == F.function(x)
    assert expand(r.ode + F*F.diff(z) - F.diff(z, 2)) == 0 or expand(r.ode - F*F.diff(z) + F.diff(z, 2)) == 0
    # the heat equation in time: F'' = 0
    r = similarity_reduction(heat, u, Symmetry(jets, [0, 1], [0]))
    assert r.ode == r.F.diff(r.variable, 2)


def test_pde_symmetries_and_solutions() -> None:
    syms = pde_symmetries(heat, u, degree=3)
    assert [s.generator() for s in syms][:3] == ['u*d/du', 'd/dt', 'd/dx']
    assert check_pde_solution(heat, u, exp(-t)*sin(x))
    assert check_pde_solution(Eq(u.diff(t), u.diff(x, 2)), u, x**2 + 2*t)
    assert not check_pde_solution(heat, u, x**2 + t)


def test_unevaluated_derivative_of_a_product() -> None:
    # issue #71: u_t = (u**2 u_x)_x written with an unevaluated Derivative
    jets = JetSpace([u])
    usym = jets.u[0]
    u_x, u_xx, u_t = [jets.jet(0, J) for J in [(1, 0), (2, 0), (0, 1)]]
    div = u.diff(t) - Derivative(u**2*u.diff(x), x)
    assert jets.to_jet(div) == u_t - usym**2*u_xx - 2*usym*u_x**2
    assert _generators(symmetries(div, u, 2)) == _generators(symmetries(expand(div.doit()), u, 2))
    assert len(symmetries(div, u, 2)) == 4
    assert not check_symmetry([div], [u], Symmetry(jets, [0, 0], [1]))
    assert check_symmetry([div], [u], Symmetry(jets, [x, 0], [usym]))


def test_all_branches_of_an_equation_nonlinear_in_the_derivative() -> None:
    # issue #72: (y y' + x)(y' - 1) = 0, Kamke 1.471; the symmetries of the
    # branch y' = 1 (or of the circles) alone are not symmetries
    jets = JetSpace([y])
    Y = jets.u[0]
    eq = y*y.diff(x)**2 - (y - x)*y.diff(x) - x
    assert [(X.xi, X.eta) for X in symmetries(eq, y, 2)] == [((x,), (Y,))]
    branch: list[tuple[Expr, Expr]] = [(Integer(1), Integer(0)), (Y, x), (Integer(0), Integer(1)),
        (Integer(0), Y - x), (Y, -x)]
    for xi, eta in branch:
        assert not check_symmetry([eq], [y], Symmetry(jets, [xi], [eta]))
    assert check_symmetry([eq], [y], Symmetry(jets, [x], [Y]))
    # Kamke 1.527: a cubic in y', no nan components
    cubic = y.diff(x)**3 - x*y**4*y.diff(x) - y**5
    assert [(X.xi, X.eta) for X in symmetries(cubic, y, 2)] == [((x,), (-3*Y/2,))]


def test_all_branches_of_an_equation_not_polynomial_in_the_derivative() -> None:
    # issue #72: an equation which is not polynomial in y' was solved for
    # its first root, and the symmetries of that branch came out (all nine
    # quadratic ones, y -> y + c among them) and were accepted
    jets = JetSpace([y])
    Y = jets.u[0]
    p = y.diff(x)
    translation = Symmetry(jets, [0], [1])
    # a denominator: (y y' + x)(y' - 1)/y'
    rational = (y*p + x)*(p - 1)/p
    assert [(X.xi, X.eta) for X in symmetries(rational, y, 2)] == [((x,), (Y,))]
    assert not check_symmetry(rational, y, translation)
    # a polynomial in exp(y'): y' = 2 pi i k or y' = log(y) + 2 pi i k
    for eq in [(exp(p) - 1)*(exp(p) - y), exp(2*p) - (1 + y)*exp(p) + y]:
        assert [(X.xi, X.eta) for X in symmetries(eq, y, 2)] == [((Integer(1),), (Integer(0),))]
        assert not check_symmetry(eq, y, translation)
        assert check_symmetry(eq, y, Symmetry(jets, [1], [0]))
    # a polynomial in sqrt(y'): y' = 1 or y' = y**2
    radical = (p**Rational(1, 2) - 1)*(p**Rational(1, 2) - y)
    assert [(X.xi, X.eta) for X in symmetries(radical, y, 2)] == [((Integer(1),), (Integer(0),))]
    assert not check_symmetry(radical, y, translation)
    # one branch: sqrt(y') = -y is y' = y**2, a Riccati equation
    found = symmetries(p**Rational(1, 2) + y, y, 2)
    assert len(found) == 4 and all(check_symmetry(p**Rational(1, 2) + y, y, X) for X in found)
    assert check_symmetry(p**Rational(1, 2) + y, y, Symmetry(jets, [x**2], [-2*x*Y - 1]))
    # exp(y') = x: y' = log(x) + 2 pi i k
    found = symmetries(exp(p) - x, y, 2)
    assert len(found) == 2 and all(check_symmetry(exp(p) - x, y, X) for X in found)
    assert check_symmetry(exp(p) - x, y, Symmetry(jets, [x], [x + Y]))
    assert infinitesimal_condition(exp(p) - y, y, Symmetry(jets, [0], [1])) == [Integer(-1)]
    system, _, _ = determining_system(exp(p) - y, y)
    assert len(system) == 3
    # sin(y') = x: neither, refused rather than solved for one branch
    for call in (lambda: symmetries(sin(p) - x, y, 1),
                 lambda: check_symmetry(sin(p) - x, y, translation)):
        try:
            call()
        except NotImplementedError:
            pass
        else:
            raise AssertionError("expected NotImplementedError")
