from __future__ import annotations

from sympy import Function, symbols, Eq, exp, sin, Rational

from sympy_extras.solvers import JetSpace, Symmetry, symmetries, check_symmetry
from sympy_extras.solvers.lie import infinitesimal_condition

x, t, m = symbols('x t m')
u = Function('u')(x, t)


def _generators(syms: list[Symmetry]) -> set[str]:
    return {s.generator() for s in syms}


def test_jet_space() -> None:
    jets = JetSpace([u])
    assert jets.p == 2 and jets.q == 1
    assert jets.x == (x, t) and jets.u[0].name == 'u'
    e = jets.to_jet(u.diff(t) - u.diff(x, 2) + u*u.diff(x, t))
    usym, u_x, u_xx, u_t, u_xt = [jets.jet(0, J) for J in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]]
    assert e == u_t - u_xx + usym*u_xt
    assert jets.from_jet(e) == u.diff(t) - u.diff(x, 2) + u*u.diff(x, t)
    assert jets.from_jet(u_t) == u.diff(t)
    assert jets.order(e) == 2
    # mixed partials are canonical
    assert jets.to_jet(u.diff(t, x)) == jets.to_jet(u.diff(x, t)) == u_xt
    assert jets.total_derivative(usym**2, 0) == 2*usym*u_x
    assert jets.total_derivative(x*u_x, 1) == x*u_xt
    # jets of another instance are recognised by name
    other = JetSpace([u])
    assert other.index(u_xt) == (0, (1, 1))
    assert other.jets_in(e) == [(0, (0, 0)), (0, (0, 1)), (0, (1, 1)), (0, (2, 0))]


def test_prolongation() -> None:
    jets = JetSpace([u])
    usym = jets.u[0]
    u_x, u_xx = jets.jet(0, (1, 0)), jets.jet(0, (2, 0))
    X = Symmetry(jets, [t, 0], [0])
    phi = X.prolongation(2)
    assert phi[(0, (1, 0))] == 0 and phi[(0, (0, 1))] == -u_x and phi[(0, (2, 0))] == 0
    galilean = Symmetry(jets, [2*t, 0], [-x*usym])
    assert galilean.characteristic() == (-x*usym - 2*t*u_x,)
    heat = u.diff(t) - u.diff(x, 2)
    assert infinitesimal_condition(heat, u, X) == [-u_x]
    assert check_symmetry(heat, u, galilean)
    assert not check_symmetry(heat, u, X)
    # linear combinations and brackets
    dx = Symmetry(jets, [1, 0], [0])
    dt = Symmetry(jets, [0, 1], [0])
    assert (dx + 2*dt).xi == (1, 2)
    assert (galilean - galilean).is_zero()
    assert dt.commutator(galilean) == 2*dx
    assert dx.commutator(galilean).eta == (-usym,)
    assert (galilean.apply(u_xx) - phi[(0, (2, 0))]).has(u_xx)
    assert str(dx) == "Symmetry(xi=(1, 0), eta=(0,))"


def test_symmetries_heat() -> None:
    heat = u.diff(t) - u.diff(x, 2)
    syms = symmetries(heat, u, degree=3, superposition=False)
    assert len(syms) == 6
    assert all(check_symmetry(heat, u, s) for s in syms)
    assert {'d/dx', 'd/dt', 'u*d/du', 'x*d/dx + 2*t*d/dt'} <= _generators(syms)
    # the superposition symmetries: polynomial solutions of the heat equation
    with_superposition = symmetries(heat, u, degree=3, superposition=True)
    extra = [s for s in with_superposition if s.xi == (0, 0) and not s.eta[0].has(JetSpace([u]).u[0])]
    assert extra
    for s in extra:
        h = s.eta[0]
        assert (h.diff(t) - h.diff(x, 2)).expand() == 0


def test_symmetries_burgers_kdv() -> None:
    burgers = u.diff(t) + u*u.diff(x) - u.diff(x, 2)
    syms = symmetries(burgers, u)
    assert len(syms) == 5 and all(check_symmetry(burgers, u, s) for s in syms)
    assert 't*d/dx + d/du' in _generators(syms)
    kdv = u.diff(t) + u*u.diff(x) + u.diff(x, 3)
    syms = symmetries(kdv, u)
    assert len(syms) == 4 and all(check_symmetry(kdv, u, s) for s in syms)
    assert 'x*d/dx + 3*t*d/dt - 2*u*d/du' in _generators(syms)


def test_symmetries_parameters() -> None:
    # nonlinear diffusion u_t = (u^m u_x)_x has 4 symmetries for generic m
    diffusion = u.diff(t) - (u**m*u.diff(x)).diff(x)
    syms = symmetries(diffusion, u)
    assert len(syms) == 4 and all(check_symmetry(diffusion, u, s) for s in syms)
    # the wave equation: translations, boost, scaling, two conformal ones and u d/du
    wave = u.diff(t, 2) - u.diff(x, 2)
    syms = symmetries(wave, u, superposition=False)
    assert len(syms) == 7 and all(check_symmetry(wave, u, s) for s in syms)
    assert 't*d/dx + x*d/dt' in _generators(syms)
    # a nonlinear equation keeps its u-free symmetries: d/du of the potential Burgers equation
    potential = u.diff(t) - u.diff(x, 2) - u.diff(x)**2
    syms = symmetries(potential, u, degree=3, superposition=False)
    assert len(syms) == 6 and any(s.xi == (0, 0) and s.eta == (1,) for s in syms)


def test_symmetries_with_basis() -> None:
    y = Function('y')(x)
    eq = y.diff(x) + 2*x*y - x*exp(-x**2)
    assert symmetries(eq, y) == []
    syms = symmetries(eq, y, basis=[exp(-x**2)])
    assert any(s.xi == (0,) and s.eta == (exp(-x**2),) for s in syms)
    assert all(check_symmetry(eq, y, s) for s in syms)
    eq = y.diff(x, 2) + y
    syms = symmetries(eq, y, basis=[sin(x), sin(2*x)])
    assert all(check_symmetry(eq, y, s) for s in syms)
    assert len(syms) >= 3


def test_systems_and_eq() -> None:
    v = Function('v')(x, t)
    # u_t = v_x, v_t = u_x: the wave equation as a first order system
    system = [Eq(u.diff(t), v.diff(x)), Eq(v.diff(t), u.diff(x))]
    syms = symmetries(system, [u, v], degree=1)
    assert syms and all(check_symmetry(system, [u, v], s) for s in syms)
    assert any(s.xi == (1, 0) and s.eta == (0, 0) for s in syms)
    jets = JetSpace([u, v])
    assert jets.q == 2 and jets.jet(1, (0, 1)).name == 'v_t'
    # a Symmetry needs the right number of coefficients
    try:
        Symmetry(jets, [1], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_nonlinear_leading_derivative() -> None:
    # (u_xx)^2 = u_t: the leading derivative is solved with solve
    eq = u.diff(x, 2)**2 - u.diff(t)
    syms = symmetries(eq, u, degree=1)
    assert all(check_symmetry(eq, u, s) for s in syms)
    assert {'d/dx', 'd/dt'} <= _generators(syms)
    eq = u.diff(t) - u.diff(x)**Rational(1, 2)
    syms = symmetries(eq, u, degree=1)
    assert all(check_symmetry(eq, u, s) for s in syms)
