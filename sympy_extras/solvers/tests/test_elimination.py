"""Elimination: classical implicit equations, the relations checked by
substituting the parametrisation back, resultants as an independent
computation of the eliminant of two polynomials."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Eq, I, Integer, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, cosh, exp, log, sin, sinh, sqrt, tan
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.ideals import Ideal
from sympy_extras.simplify.normalform import normal_form
from sympy_extras.solvers.elimination import eliminate, implicitize

s, t, x, y, z, a, b = (Symbol(n) for n in 'stxyzab')


def _same(result: list[Expr], expected: list[Expr], *symbols: Symbol) -> bool:
    """The two lists generate the same ideal (the printed forms and the
    signs of the generators are not compared)."""
    if not result or not expected:
        return result == expected
    return Ideal(result, *symbols, domain=QQ) == Ideal(expected, *symbols, domain=QQ)


def _vanishes(relations: list[Expr], coordinates: list[Symbol], parametrization: list[Expr]) -> None:
    """Every relation is zero on the parametrisation, and there is one."""
    assert relations and relations != [1]
    for relation in relations:
        value = relation
        for coordinate, f in zip(coordinates, parametrization):
            value = value.subs(coordinate, f)
        assert sympy.simplify(sympy_expr(to_sympy(value))) == 0


def test_classical_curves() -> None:
    # Cox, Little, O'Shea, Chapter 3: the twisted cubic, the cuspidal cubic, the circle
    assert _same(implicitize([t, t**2, t**3], [t], [x, y, z]), [y - x**2, z - x**3], x, y, z)
    assert _same(implicitize([t**2, t**3], [t], [x, y]), [x**3 - y**2], x, y)
    assert _same(implicitize([(1 - t**2)/(1 + t**2), 2*t/(1 + t**2)], [t], [x, y]), [x**2 + y**2 - 1], x, y)
    assert _same(implicitize([cos(t), sin(t)], [t], [x, y]), [x**2 + y**2 - 1], x, y)
    # the folium of Descartes x**3 + y**3 = 3 x y, the lemniscate (x**2 + y**2)**2 = x**2 - y**2
    assert _same(implicitize([3*t/(1 + t**3), 3*t**2/(1 + t**3)], [t], [x, y]), [x**3 + y**3 - 3*x*y], x, y)
    lemniscate = implicitize([cos(t)/(1 + sin(t)**2), sin(t)*cos(t)/(1 + sin(t)**2)], [t], [x, y])
    assert _same(lemniscate, [(x**2 + y**2)**2 - x**2 + y**2], x, y)
    # a Lissajous curve and the astroid x**(2/3) + y**(2/3) = 1, i.e. (1 - x**2 - y**2)**3 = 27 x**2 y**2
    assert _same(implicitize([cos(t), sin(2*t)], [t], [x, y]), [4*x**4 - 4*x**2 + y**2], x, y)
    assert _same(implicitize([cos(t)**3, sin(t)**3], [t], [x, y]), [(1 - x**2 - y**2)**3 - 27*x**2*y**2], x, y)


def test_surfaces() -> None:
    assert _same(implicitize([s*t, s*t**2, s**2], [s, t], [x, y, z]), [x**4 - y**2*z], x, y, z)
    sphere = implicitize([sin(s)*cos(t), sin(s)*sin(t), cos(s)], [s, t], [x, y, z])
    assert _same(sphere, [x**2 + y**2 + z**2 - 1], x, y, z)
    # the hyperboloid of one sheet, the torus with radii 2 and 1, a hyperbolic paraboloid
    hyperboloid = implicitize([cosh(s)*cos(t), cosh(s)*sin(t), sinh(s)], [s, t], [x, y, z])
    assert _same(hyperboloid, [x**2 + y**2 - z**2 - 1], x, y, z)
    torus = implicitize([(2 + cos(s))*cos(t), (2 + cos(s))*sin(t), sin(s)], [s, t], [x, y, z])
    assert _same(torus, [(x**2 + y**2 + z**2 + 3)**2 - 16*(x**2 + y**2)], x, y, z)
    assert _same(implicitize([s*t, s, t], [s, t], [x, y, z]), [x - y*z], x, y, z)


def test_parametrisations_vanish_on_the_result() -> None:
    cases: list[tuple[list[Expr], list[Symbol]]] = [
        ([a*cos(t), b*sin(t)], [x, y]),
        ([t + 1/t, t - 1/t], [x, y]),
        ([sqrt(t), t**2 + 1], [x, y]),
        ([t**3 - t, t**2 + a], [x, y]),
        ([tan(t), 1/cos(t)], [x, y]),
        ([cos(t) + cos(2*t), sin(t) - sin(2*t)], [x, y]),
        ([t**2 - 1, t*(t**2 - 1), t**4], [x, y, z]),
    ]
    for parametrization, coordinates in cases:
        _vanishes(implicitize(parametrization, [t], coordinates), coordinates, parametrization)


def test_against_resultants() -> None:
    rnd = random.Random(91)
    sx, sy = sympy.Symbol('x'), sympy.Symbol('y')
    for _ in range(12):
        def poly() -> Expr:
            return sum((Integer(rnd.randint(-2, 2))*x**i*y**j for i in range(3) for j in range(3 - i)), Integer(0))
        f, g = poly() + x**2, poly() + x
        result = eliminate([f, g], [x])
        resultant = sympy.resultant(sympy_expr(to_sympy(f)), sympy_expr(to_sympy(g)), sx)
        if resultant == 0:
            continue
        # the eliminant divides the resultant and has the same roots: its squarefree part divides ours
        assert len(result) == 1
        ours = sympy_expr(to_sympy(result[0]))
        assert sympy.rem(resultant, ours, sy) == 0 or ours == 1
        if ours != 1:
            radical = sympy.Poly(resultant, sy)
            squarefree = sympy.quo(radical, sympy.gcd(radical, radical.diff(sy))).as_expr()
            assert sympy.rem(sympy.Poly(ours, sy)**sympy.degree(resultant, sy), sympy.Poly(squarefree, sy)).is_zero


def test_equations_and_parameters() -> None:
    P, V, T, U, n, R, c = (Symbol(name) for name in 'PVTUnRc')
    assert _same(eliminate([Eq(P*V, n*R*T), Eq(U, c*n*T)], [T]), [P*V*c - R*U], P, V, U, n, R, c)
    # two variables at once: the intersection of two planes projected on a line
    assert eliminate([x + y + z - 1, x - y + 2*z], [x, y]) == []
    # x = 3: y = 3 + 2 z and 3 z = -5
    assert _same(eliminate([x + y + z - 1, x - y + 2*z, x - 3], [x, y]), [3*z + 5], z)
    # kernels of the parameters stay, kernels of the eliminated variable go
    [relation] = eliminate([Eq(x, sin(a)*t), Eq(y, t**2)], [t])
    assert normal_form(relation - (x**2 - y*sin(a)**2)) == 0 or normal_form(relation + (x**2 - y*sin(a)**2)) == 0
    # the core writes exp(t)**2 as exp(2*t): both are powers of one indeterminate ([] before)
    assert _same(eliminate([Eq(x, exp(t)), Eq(y, exp(t)**2 + 1)], [t]), [x**2 - y + 1], x, y)
    assert _same(eliminate([Eq(x, log(t)), Eq(y, 2*log(t) + 1)], [t]), [2*x - y + 1], x, y)
    # cosh and sinh through exponentials: x = (e**t + e**-t)/2, y = (e**t - e**-t)/2
    assert _same(eliminate([Eq(2*x, exp(t) + exp(-t)), Eq(2*y, exp(t) - exp(-t))], [t]), [x**2 - y**2 - 1], x, y)
    assert _same(eliminate([Eq(x, exp(t/2)), Eq(y, exp(t/3))], [t]), [x**2 - y**3], x, y)


def test_degenerate_cases() -> None:
    assert eliminate([], [t]) == [] and eliminate([x - x], [t]) == []
    assert _same(eliminate([x - y], [z]), [x - y], x, y)
    assert eliminate([x, x - 1], [x]) == [1] and eliminate([x**2 - 1], [x]) == []
    assert eliminate([Integer(1)], [x]) == [1]
    # a vanishing denominator gives no relation: x = t/t is x = 1 for t != 0
    assert _same(eliminate([Eq(x, t/t)], [t]), [x - 1], x)
    assert _same(eliminate([Eq(x*t, 1), Eq(y*t, 1)], [t]), [x - y], x, y)
    with pytest.raises(NotImplementedError):
        eliminate([x - I*t], [t])
    with pytest.raises(ValueError):
        implicitize([t, t**2], [t], [x])
    with pytest.raises(ValueError):
        implicitize([t, t**2], [t], [t, y])
