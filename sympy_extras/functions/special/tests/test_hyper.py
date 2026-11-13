"""The typed hyper and meijerg against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

import pytest
import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.interop import from_sympy
from sympy_extras.functions import Hyper, MeijerG, hyper, meijerg

a, b, c, z, x = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('z'), Symbol('x')


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_construction_and_radius_match_sympy() -> None:
    sa, sb, sc, sz = sympy.symbols('a b c z')
    cases: list[tuple[list[Expr], list[Expr], list[sympy.Expr], list[sympy.Expr]]] = [
        ([b, a], [c], [sb, sa], [sc]), ([a, c], [c], [sa, sc], [sc]), ([a, a, c], [a, c], [sa, sa, sc], [sa, sc]),
        ([], [], [], []), ([Integer(-2), a], [c], [sympy.Integer(-2), sa], [sc]), ([Integer(-2)], [Integer(-3)], [sympy.Integer(-2)], [sympy.Integer(-3)]),
        ([Integer(-3)], [Integer(-2)], [sympy.Integer(-3)], [sympy.Integer(-2)]), ([a, b, c], [Integer(1)], [sa, sb, sc], [sympy.Integer(1)]), ([a], [b, c], [sa], [sb, sc])]
    for ap, bq, sap, sbq in cases:
        ours, theirs = hyper(ap, bq, z), sympy.hyper(sap, sbq, sz)
        assert same(ours, theirs), (ours, theirs)
        assert isinstance(ours, Hyper)
        assert str(to_sympy(ours.radius_of_convergence)) == str(theirs.radius_of_convergence), ours
    ours_g = meijerg([b, a], [c], [Integer(1)], [], z)
    assert same(ours_g, sympy.meijerg([sb, sa], [sc], [1], [], sz))
    with pytest.raises(ValueError):
        meijerg([Integer(3)], [], [Integer(1)], [], z)
    with pytest.raises(ValueError):
        meijerg([oo], [], [], [], z)


def test_derivatives_numbers_round_trip() -> None:
    sa, sb, sc, sz, sx = sympy.symbols('a b c z x')
    for ours, theirs in [(hyper([a, b], [c], x**2), sympy.hyper([sa, sb], [sc], sx**2)),
                         (hyper([a, x], [c], x), sympy.hyper([sa, sx], [sc], sx)),
                         (meijerg([a], [b], [c], [], x**2), sympy.meijerg([sa], [sb], [sc], [], sx**2)),
                         (meijerg([], [b], [c], [], x), sympy.meijerg([], [sb], [sc], [], sx)),
                         (meijerg([], [], [], [b], x), sympy.meijerg([], [], [], [sb], sx))]:
        assert same(ours.diff(x), theirs.diff(sx)), ours
        assert from_sympy(to_sympy(ours)) == ours
    for ours, theirs in [(hyper([Rational(1, 2), 1], [Rational(3, 2)], Rational(1, 4)),
                          sympy.hyper([sympy.Rational(1, 2), 1], [sympy.Rational(3, 2)], sympy.Rational(1, 4))),
                         (meijerg([], [], [0], [], Rational(1, 3)), sympy.meijerg([], [], [0], [], sympy.Rational(1, 3)))]:
        assert abs(complex(sympy_value(ours.evalf(20))) - complex(sympy.N(theirs, 20))) < 1e-14, ours
    f = hyper([a, b], [c], z)
    assert same(f.subs(a, c), sympy.hyper([sympy.Symbol('c'), sympy.Symbol('b')], [sympy.Symbol('c')], sympy.Symbol('z')))


def test_meijerg_with_half_integer_parameters_has_a_value() -> None:
    # the bug: the parameters went to mpmath's meijerg as complex numbers,
    # which it compares with integers to find the poles: TypeError, so the
    # G-function (and any reliable value of an expression with it) had none
    from sympy_extras.core.evalf import numeric_value
    h = Rational(1, 2)
    g = meijerg([], [h], [Rational(-7, 2), -h, Rational(7, 2)], [], Rational(9, 100))
    value = numeric_value(g, 120)
    half = sympy.Rational(1, 2)
    expected = sympy.meijerg(((), (half,)), ((-7 * half, -half, 7 * half), ()), sympy.Rational(9, 100)).evalf(30)
    assert value is not None and abs(complex(value) - complex(expected)) < 1e-6 * abs(complex(value))


def test_polar_arguments_are_unpolarified_where_single_valued() -> None:
    # the bug: hyper kept exp_polar(I*pi) where SymPy's hyper.eval writes
    # -1 (p <= q, or p = q + 1 and |z| <= 1), so that simplify, through a
    # hyperexpand which found no formula, returned a polar number which
    # the Marichev-Adamchik method then refused as an unresolved branch
    from sympy_extras.core import I, pi
    from sympy_extras.functions import exp_polar
    w = exp_polar(I * pi)
    assert hyper([Rational(1, 6)], [Rational(1, 2)], w) == hyper([Rational(1, 6)], [Rational(1, 2)], -1)
    assert hyper([1, 1], [2], w) == hyper([1, 1], [2], -1)
    # p = q + 2: multivalued, the polar number stays (as in SymPy)
    assert hyper([1, 1, 1], [2], w).args[-1] == w


def test_meijerg_convergence_quantities_match_sympy() -> None:
    # nu, delta and get_period, used by the Meijer G-function integrator
    h, third, zero, one = Rational(1, 2), Rational(1, 3), Integer(0), Integer(1)
    cases: list[tuple[list[Expr], list[Expr], list[Expr], list[Expr]]] = [
        ([a], [], [b, zero], [h]), ([], [], [h], [zero]), ([one], [], [h, third], []), ([], [], [zero, one], []),
        ([third], [h], [h], [third])]
    for an, aother, bm, bother in cases:
        g = meijerg(an, aother, bm, bother, z)
        assert isinstance(g, MeijerG)
        expected = sympy.meijerg([to_sympy(e) for e in an], [to_sympy(e) for e in aother],
                                 [to_sympy(e) for e in bm], [to_sympy(e) for e in bother], to_sympy(z))
        assert to_sympy(g.nu) == expected.nu
        assert to_sympy(g.delta) == expected.delta
        assert to_sympy(g.get_period()) == expected.get_period()
        assert tuple(to_sympy(e) for e in g.ap) == tuple(expected.ap)
        assert tuple(to_sympy(e) for e in g.bq) == tuple(expected.bq)
