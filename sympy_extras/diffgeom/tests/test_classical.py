"""Curves and surfaces: the examples of do Carmo, the Frenet formulas,
and Gauss' Theorema Egregium (the Gaussian curvature against half the
scalar curvature of the first fundamental form, computed by Metric)."""
from __future__ import annotations

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.diffgeom import Metric
from sympy_extras.diffgeom.classical import Curve, Surface
from sympy_extras.functions import abs_, cos, cosh, exp, sin, sinh, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.simplify.normalform import normal_form

t, u, v = Symbol('t'), Symbol('u'), Symbol('v')
a, b, R, r = (Symbol(name, positive=True) for name in ['a', 'b', 'R', 'r'])


def _zero(e: Expr) -> bool:
    return bool(sympy.simplify(sympy_expr(to_sympy(e))) == 0)


def test_helix() -> None:
    # do Carmo, Section 1.5, exercise 1: curvature a/(a**2 + b**2), torsion b/(a**2 + b**2)
    helix = Curve([a*cos(t), a*sin(t), b*t], t)
    assert helix.speed() == sqrt(a**2 + b**2)
    assert helix.curvature() == a/(a**2 + b**2) and helix.torsion() == b/(a**2 + b**2)
    tangent, normal, binormal = helix.frenet_frame()
    assert normal == [-cos(t), -sin(t), 0]
    # an orthonormal, positively oriented frame
    for first, second, expected in [(tangent, tangent, 1), (normal, normal, 1), (binormal, binormal, 1), (tangent, normal, 0),
                                    (tangent, binormal, 0), (normal, binormal, 0)]:
        assert normal_form(sum((p*q for p, q in zip(first, second)), Integer(0)) - expected) == 0
    # the Frenet formulas: T' = kappa |r'| N, B' = -tau |r'| N
    speed, kappa, tau = helix.speed(), helix.curvature(), helix.torsion()
    for i in range(3):
        assert normal_form(tangent[i].diff(t) - kappa*speed*normal[i]) == 0
        assert normal_form(binormal[i].diff(t) + tau*speed*normal[i]) == 0


def test_plane_curves() -> None:
    circle = Curve([R*cos(t), R*sin(t)], t)
    assert circle.curvature() == 1/R and circle.signed_curvature() == 1/R and circle.torsion() == 0
    assert Curve([R*cos(t), -R*sin(t)], t).signed_curvature() == -1/R            # clockwise
    assert circle.tangent() == [-sin(t), cos(t)]
    # the parabola y = x**2: 2/(1 + 4 x**2)**(3/2); the catenary y = cosh(x): 1/cosh(x)**2
    parabola = Curve([t, t**2], t)
    assert _zero(parabola.curvature() - 2/(1 + 4*t**2)**Rational(3, 2))
    assert _zero(Curve([t, cosh(t)], t).curvature()**2 - 1/cosh(t)**4)
    # the ellipse at its vertices: a/b**2 and b/a**2
    ellipse = Curve([a*cos(t), b*sin(t)], t)
    assert _zero(ellipse.curvature().subs(t, 0) - a/b**2)
    # a straight line has no curvature, and its torsion is not defined
    line = Curve([t, 2*t, 3*t], t)
    assert line.curvature() == 0
    with pytest.raises(ValueError):
        line.torsion()
    with pytest.raises(ValueError):
        line.frenet_frame()
    with pytest.raises(ValueError):
        Curve([t], t)
    with pytest.raises(ValueError):
        line.signed_curvature()


def test_twisted_cubic_and_parameters_without_sign() -> None:
    cubic = Curve([t, t**2, t**3], t)
    # do Carmo: kappa**2 = 4 (9 t**4 + 9 t**2 + 1)/(9 t**4 + 4 t**2 + 1)**3, tau = 3/(9 t**4 + 9 t**2 + 1)
    assert _zero(cubic.curvature()**2 - 4*(9*t**4 + 9*t**2 + 1)/(9*t**4 + 4*t**2 + 1)**3)
    assert cubic.torsion() == 3/(9*t**4 + 9*t**2 + 1)
    # with a parameter of unknown sign the curvature of the helix has an absolute value
    c = Symbol('c')
    assert Curve([c*cos(t), c*sin(t), t], t).curvature() == abs_(c)/(c**2 + 1)


def _egregium(surface: Surface) -> None:
    E, F, G = surface.first_fundamental_form()
    metric = Metric([[E, F], [F, G]], [surface.u, surface.v])
    assert normal_form(surface.gaussian_curvature() - metric.ricci_scalar()/2) == 0


def test_surfaces_of_do_carmo() -> None:
    sphere = Surface([R*sin(u)*cos(v), R*sin(u)*sin(v), R*cos(u)], u, v)
    assert sphere.first_fundamental_form() == (R**2, 0, R**2*sin(u)**2)
    assert sphere.gaussian_curvature() == 1/R**2
    assert normal_form(sphere.mean_curvature()**2 - 1/R**2) == 0
    # the torus: K = cos(u)/(r (R + r cos(u)))  (do Carmo, Section 3.3, example 5)
    torus = Surface([(R + r*cos(u))*cos(v), (R + r*cos(u))*sin(v), r*sin(u)], u, v)
    assert normal_form(torus.gaussian_curvature() - cos(u)/(r*(R + r*cos(u)))) == 0
    # minimal surfaces: the catenoid and the helicoid, with K = -1/cosh(u)**4 for the catenoid
    catenoid = Surface([cosh(u)*cos(v), cosh(u)*sin(v), u], u, v)
    helicoid = Surface([u*cos(v), u*sin(v), v], u, v)
    assert catenoid.mean_curvature() == 0 and helicoid.mean_curvature() == 0
    assert normal_form(catenoid.gaussian_curvature() + 1/cosh(u)**4) == 0
    assert normal_form(helicoid.gaussian_curvature() + 1/(1 + u**2)**2) == 0
    # the pseudosphere (tractrix revolved) has K = -1; a cylinder and a cone are flat
    pseudosphere = Surface([cos(v)/cosh(u), sin(v)/cosh(u), u - sinh(u)/cosh(u)], u, v)
    assert pseudosphere.gaussian_curvature() == -1
    assert Surface([cos(u), sin(u), v], u, v).gaussian_curvature() == 0
    assert Surface([v*cos(u), v*sin(u), v], u, v).gaussian_curvature() == 0
    for surface in (sphere, torus, catenoid, helicoid, pseudosphere):
        _egregium(surface)


def test_graphs_and_fundamental_forms() -> None:
    # the graph z = f(x, y): K = (f_xx f_yy - f_xy**2)/(1 + f_x**2 + f_y**2)**2
    for f in (u*v, u**2 - v**2, u**3 + u*v**2, exp(u)*v):
        graph = Surface([u, v, f], u, v)
        fx, fy = f.diff(u), f.diff(v)
        expected = (f.diff(u, 2)*f.diff(v, 2) - f.diff(u).diff(v)**2)/(1 + fx**2 + fy**2)**2
        assert normal_form(graph.gaussian_curvature() - expected) == 0
        _egregium(graph)
        # the second fundamental form: (f_xx, f_xy, f_yy)/sqrt(1 + f_x**2 + f_y**2)
        L, M, N = graph.second_fundamental_form()
        W = 1 + fx**2 + fy**2
        assert normal_form(L**2*W - f.diff(u, 2)**2) == 0 and normal_form(N**2*W - f.diff(v, 2)**2) == 0
        normal = graph.normal()
        assert normal_form(sum((c*c for c in normal), Integer(0)) - 1) == 0
        # the principal curvatures have the product K and the half sum H
        k1, k2 = graph.principal_curvatures()
        assert _zero(k1*k2 - graph.gaussian_curvature()) and _zero((k1 + k2)/2 - graph.mean_curvature())
    assert Surface([u, v, u*v], u, v).area_element() == sqrt(u**2 + v**2 + 1)
    with pytest.raises(ValueError):
        Surface([u, v], u, v)
    with pytest.raises(ValueError):
        Surface([u, u, u], u, u)
    with pytest.raises(ValueError):
        Surface([u + v, u + v, u + v], u, v).gaussian_curvature()
