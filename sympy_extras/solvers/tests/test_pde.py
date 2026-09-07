from __future__ import annotations

from sympy import Function, symbols, exp, sqrt, erf, simplify
from sympy.solvers.pde import checkpdesol
from sympy.testing.pytest import raises

from sympy_extras.solvers import (pde_symmetries, similarity_reduction, pdsolve_lie,
    check_symmetry, JetSpace, Symmetry)

x, t = symbols('x t')
u = Function('u')(x, t)
heat = u.diff(t) - u.diff(x, 2)
burgers = u.diff(t) + u*u.diff(x) - u.diff(x, 2)


def test_pde_symmetries() -> None:
    syms = pde_symmetries(heat, u, degree=3)
    assert len(syms) == 6
    assert [s.generator() for s in syms][:3] == ['u*d/du', 'd/dt', 'd/dx']
    assert all(check_symmetry(heat, u, s) for s in syms)
    assert len(pde_symmetries(heat, u, degree=3, superposition=True)) > 6


def test_similarity_reduction_heat() -> None:
    jets = JetSpace([u])
    usym = jets.u[0]
    scaling = Symmetry(jets, [x, 2*t], [0])
    r = similarity_reduction(heat, u, scaling)
    assert r.z == x/sqrt(t)
    assert r.solution == r.F.func(x/sqrt(t))
    F, z = r.F, r.variable
    assert simplify(r.ode/(z*F.diff(z) + 2*F.diff(z, 2))).is_number
    galilean = Symmetry(jets, [2*t, 0], [-x*usym])
    r = similarity_reduction(heat, u, galilean)
    assert r.z == t
    assert simplify(r.solution/(r.F.func(t)*exp(-x**2/(4*t)))) == 1
    # translation in time: stationary solutions
    r = similarity_reduction(heat, u, Symmetry(jets, [0, 1], [0]))
    assert r.z == x and r.ode == r.F.diff(r.variable, 2)
    # no similarity variable without a motion of the independent variables
    raises(NotImplementedError, lambda: similarity_reduction(heat, u, Symmetry(jets, [0, 0], [usym])))
    raises(NotImplementedError, lambda: similarity_reduction(heat, u, Symmetry(jets, [usym, 0], [0])))


def test_pdsolve_lie_heat() -> None:
    sols = pdsolve_lie(heat, u, degree=3)
    rhs = [s.rhs for s in sols]
    C1, C2 = symbols('C1 C2')
    assert C1 + C2*erf(x/(2*sqrt(t))) in rhs
    assert C1*exp(-x**2/(4*t))/sqrt(t) in rhs
    for s in sols:
        assert checkpdesol(heat, s, u)[0]


def test_pdsolve_lie_burgers_and_traveling_waves() -> None:
    sols = pdsolve_lie(burgers, u)
    assert sols
    for s in sols:
        assert checkpdesol(burgers, s, u)[0]
    # a traveling wave from the sum of the two translations
    jets = JetSpace([u])
    c = symbols('c', positive=True)
    wave = Symmetry(jets, [c, 1], [0])
    r = similarity_reduction(burgers, u, wave)
    assert r.z == x - c*t
    assert r.ode.has(r.F.diff(r.variable, 2))
