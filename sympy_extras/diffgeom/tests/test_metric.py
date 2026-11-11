"""Curvature from a metric: the textbook spacetimes and surfaces, the
symmetries of the Riemann tensor, the Bianchi identity, and SymPy's
``sympy.diffgeom`` as an independent oracle on other metrics."""
from __future__ import annotations

from typing import Sequence

import pytest
import sympy
from sympy.diffgeom import CoordSystem, Manifold, Patch, TensorProduct, metric_to_Ricci_components, metric_to_Riemann_components

from sympy_extras.core import Function, Integer, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, exp, sin, sinh, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.diffgeom import Metric
from sympy_extras.matrices import ExprMatrix
from sympy_extras.simplify.normalform import normal_form

t, r, theta, phi, a, m, x, y, z, s = (Symbol(name) for name in ['t', 'r', 'theta', 'phi', 'a', 'm', 'x', 'y', 'z', 's'])


def _schwarzschild() -> Metric:
    f = 1 - 2*m/r
    return Metric([[-f, 0, 0, 0], [0, 1/f, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]], [t, r, theta, phi])


def _oracle(rows: Sequence[Sequence[Expr]], coordinates: Sequence[Symbol]) -> tuple[sympy.Array, sympy.Array]:
    """The Riemann and Ricci components by sympy.diffgeom (R^i_jkl and
    R_jl, with the same conventions)."""
    manifold = Manifold('M', len(coordinates))
    patch = Patch('P', manifold)
    system = CoordSystem('C', patch, [sympy.Symbol(c.name, real=True) for c in coordinates])
    functions = system.base_scalars()
    forms = system.base_oneforms()
    names = [sympy.Symbol(c.name) for c in coordinates]
    rule: dict[sympy.Basic | complex, sympy.Expr | complex] = dict(zip(names, functions))
    metric = sympy.Integer(0)
    for i in range(len(names)):
        for j in range(len(names)):
            metric = metric + sympy_expr(to_sympy(rows[i][j])).subs(rule)*TensorProduct(forms[i], forms[j])
    back = dict(zip(functions, names))
    riemann = metric_to_Riemann_components(metric).subs(back)
    ricci = metric_to_Ricci_components(metric).subs(back)
    return riemann, ricci


def _against_oracle(rows: Sequence[Sequence[Expr]], coordinates: Sequence[Symbol]) -> None:
    metric = Metric(rows, coordinates)
    riemann, ricci = _oracle(rows, coordinates)
    n = len(coordinates)
    for i in range(n):
        for j in range(n):
            assert sympy.simplify(to_sympy(metric.ricci()[i, j]) - ricci[i, j]) == 0
            for k in range(n):
                for l in range(n):
                    assert sympy.simplify(to_sympy(metric.riemann(i, j, k, l)) - riemann[i, j, k, l]) == 0


def test_sphere() -> None:
    sphere = Metric([[a**2, 0], [0, a**2*sin(theta)**2]], [theta, phi])
    # do Carmo, Differential Geometry of Curves and Surfaces, Section 4.3: the Gaussian curvature 1/a**2
    assert sphere.christoffel_symbols() == {(0, 1, 1): -sin(theta)*cos(theta), (1, 0, 1): cos(theta)/sin(theta)}
    assert sphere.ricci_scalar() == 2/a**2
    assert sphere.riemann_lower(0, 1, 0, 1) == a**2*sin(theta)**2
    assert sphere.einstein() == ExprMatrix.zeros(2, 2)          # in dimension two the Einstein tensor vanishes
    assert sphere.determinant() == a**4*sin(theta)**2
    with pytest.raises(ValueError):
        sphere.weyl(0, 1, 0, 1)


def test_flat_metrics() -> None:
    polar = Metric([[1, 0], [0, r**2]], [r, phi])
    spherical = Metric([[1, 0, 0], [0, r**2, 0], [0, 0, r**2*sin(theta)**2]], [r, theta, phi])
    # Rindler coordinates and a metric which is not diagonal: dx**2 + 2 x dx dy + (1 + x**2) dy**2 = (dx + x dy)**2 + dy**2
    rindler = Metric([[-x**2, 0], [0, 1]], [t, x])
    for metric in (polar, spherical, rindler):
        assert metric.is_flat() and metric.ricci_scalar() == 0
    assert not Metric([[1, 0], [0, sin(theta)**2]], [theta, phi]).is_flat()
    # MTW, Box 8.4 / any text: the connection of spherical coordinates
    assert spherical.christoffel(0, 1, 1) == -r and spherical.christoffel(0, 2, 2) == -r*sin(theta)**2
    assert spherical.christoffel(1, 0, 1) == 1/r and spherical.christoffel(2, 1, 2) == cos(theta)/sin(theta)
    assert spherical.christoffel(1, 2, 2) == -sin(theta)*cos(theta)


def test_schwarzschild() -> None:
    metric = _schwarzschild()
    # Carroll, Spacetime and Geometry, eq. (5.12): the connection coefficients
    assert normal_form(metric.christoffel(1, 0, 0) - m*(r - 2*m)/r**3) == 0
    assert normal_form(metric.christoffel(0, 0, 1) - m/(r*(r - 2*m))) == 0
    assert normal_form(metric.christoffel(1, 1, 1) + m/(r*(r - 2*m))) == 0
    assert normal_form(metric.christoffel(1, 2, 2) + (r - 2*m)) == 0
    assert normal_form(metric.christoffel(1, 3, 3) + (r - 2*m)*sin(theta)**2) == 0
    # a vacuum solution, with the Kretschmann scalar 48 m**2/r**6 (Carroll, eq. (5.20))
    assert metric.ricci() == ExprMatrix.zeros(4, 4) and metric.ricci_scalar() == 0
    assert metric.einstein() == ExprMatrix.zeros(4, 4)
    assert metric.kretschmann() == 48*m**2/r**6
    assert not metric.is_flat()
    # in vacuum the Weyl tensor is the Riemann tensor
    assert metric.weyl(0, 1, 0, 1) == metric.riemann_lower(0, 1, 0, 1) == -2*m/r**3
    # the Killing vectors of time translations and rotations
    assert metric.is_killing([1, 0, 0, 0]) and metric.is_killing([0, 0, 0, 1])
    assert metric.is_killing([0, 0, sin(phi), cos(phi)*cos(theta)/sin(theta)])
    assert not metric.is_killing([0, 1, 0, 0])


def test_kerr() -> None:
    # Boyer-Lindquist coordinates: a vacuum solution, with the Kretschmann scalar
    # 48 m**2 (r**2 - a**2 cos**2)((r**2 + a**2 cos**2)**2 - 16 a**2 r**2 cos**2)/(r**2 + a**2 cos**2)**6
    # (Visser, The Kerr spacetime: a brief introduction, arXiv:0706.0622, eq. (43))
    rho2 = r**2 + a**2*cos(theta)**2
    delta = r**2 - 2*m*r + a**2
    cross = -2*m*r*a*sin(theta)**2/rho2
    metric = Metric([[-(1 - 2*m*r/rho2), 0, 0, cross], [0, rho2/delta, 0, 0], [0, 0, rho2, 0],
                     [cross, 0, 0, (r**2 + a**2 + 2*m*r*a**2*sin(theta)**2/rho2)*sin(theta)**2]], [t, r, theta, phi])
    assert metric.ricci() == ExprMatrix.zeros(4, 4)
    c2 = cos(theta)**2
    expected = 48*m**2*(r**2 - a**2*c2)*((r**2 + a**2*c2)**2 - 16*a**2*r**2*c2)/(r**2 + a**2*c2)**6
    assert normal_form(metric.kretschmann() - expected) == 0
    assert metric.is_killing([1, 0, 0, 0]) and metric.is_killing([0, 0, 0, 1]) and not metric.is_killing([0, 0, 1, 0])


def test_friedmann_robertson_walker() -> None:
    scale = Function('a')(t)
    k = Symbol('k')
    metric = Metric([[-1, 0, 0, 0], [0, scale**2/(1 - k*r**2), 0, 0], [0, 0, scale**2*r**2, 0],
                     [0, 0, 0, scale**2*r**2*sin(theta)**2]], [t, r, theta, phi])
    first, second = scale.diff(t), scale.diff(t, 2)
    # Carroll, eqs. (8.44)-(8.46)
    assert normal_form(metric.ricci()[0, 0] + 3*second/scale) == 0
    assert normal_form(metric.ricci()[1, 1] - (scale*second + 2*first**2 + 2*k)/(1 - k*r**2)) == 0
    assert normal_form(metric.ricci_scalar() - 6*(second/scale + first**2/scale**2 + k/scale**2)) == 0
    # the Friedmann combination: G_tt = 3 (a'**2 + k)/a**2
    assert normal_form(metric.einstein()[0, 0] - 3*(first**2 + k)/scale**2) == 0
    # conformally flat
    assert all(metric.weyl(i, j, 0, l) == 0 for i in range(4) for j in range(4) for l in range(4))


def test_spaces_of_constant_curvature() -> None:
    # the hyperbolic plane (curvature -1) in two charts
    assert Metric([[1/y**2, 0], [0, 1/y**2]], [x, y]).ricci_scalar() == -2
    assert Metric([[1, 0], [0, sinh(r)**2]], [r, phi]).ricci_scalar() == -2
    # de Sitter space in static coordinates: R_ij = 3 H**2 g_ij, R = 12 H**2
    H = Symbol('H')
    f = 1 - H**2*r**2
    de_sitter = Metric([[-f, 0, 0, 0], [0, 1/f, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]], [t, r, theta, phi])
    assert de_sitter.ricci_scalar() == 12*H**2
    assert all(normal_form(de_sitter.ricci()[i, j] - 3*H**2*de_sitter.g[i, j]) == 0 for i in range(4) for j in range(4))
    # the three-sphere: R = 6
    chi = Symbol('chi')
    assert Metric([[1, 0, 0], [0, sin(chi)**2, 0], [0, 0, sin(chi)**2*sin(theta)**2]], [chi, theta, phi]).ricci_scalar() == 6


def test_riemann_symmetries_and_bianchi() -> None:
    # a metric without symmetry, not diagonal
    rows: list[list[Expr]] = [[1 + x**2, x*y, Integer(0)], [x*y, Integer(2), y], [Integer(0), y, 1 + z**2]]
    metric = Metric(rows, [x, y, z])
    n = 3
    lowered = [[[[metric.riemann_lower(i, j, k, l) for l in range(n)] for k in range(n)] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    assert normal_form(lowered[i][j][k][l] + lowered[j][i][k][l]) == 0
                    assert normal_form(lowered[i][j][k][l] + lowered[i][j][l][k]) == 0
                    assert normal_form(lowered[i][j][k][l] - lowered[k][l][i][j]) == 0
                    # the first Bianchi identity
                    assert normal_form(lowered[i][j][k][l] + lowered[i][k][l][j] + lowered[i][l][j][k]) == 0
    assert metric.ricci() == metric.ricci().transpose()
    # in dimension three the Weyl tensor vanishes
    assert all(metric.weyl(i, j, k, l) == 0 for i in range(n) for j in range(n) for k in range(n) for l in range(n))
    # the contracted Bianchi identity: the divergence of the Einstein tensor is zero
    einstein, inverse = metric.einstein(), metric.inverse
    coordinates = metric.coordinates
    mixed = [[sum((inverse[i, k]*einstein[k, j] for k in range(n)), Integer(0)) for j in range(n)] for i in range(n)]
    for j in range(n):
        total: Expr = Integer(0)
        for i in range(n):
            total = total + mixed[i][j].diff(coordinates[i])
            for k in range(n):
                total = total + metric.christoffel(i, i, k)*mixed[k][j] - metric.christoffel(k, i, j)*mixed[i][k]
        assert normal_form(total) == 0


def test_against_sympy_diffgeom() -> None:
    _against_oracle([[1 + x**2, x*y], [x*y, 1 + y**2]], [x, y])
    _against_oracle([[exp(2*x), Integer(0)], [Integer(0), exp(2*x)*y**2]], [x, y])
    _against_oracle([[-(1 + r**2), Integer(0), Integer(0)], [Integer(0), 1/(1 + r**2), Integer(0)], [Integer(0), Integer(0), r**2]],
                    [t, r, phi])


def test_geodesics_and_derivatives() -> None:
    polar = Metric([[1, 0], [0, r**2]], [r, phi])
    R, P = Function('r')(s), Function('phi')(s)
    equations = polar.geodesic_equations(s)
    assert normal_form(equations[0] - (R.diff(s, 2) - R*P.diff(s)**2)) == 0
    assert normal_form(equations[1] - (P.diff(s, 2) + 2*R.diff(s)*P.diff(s)/R)) == 0
    # the straight line x = 1 in polar coordinates: r = 1/cos(phi), phi = atan(s), i.e. r = sqrt(1 + s**2)
    from sympy_extras.functions import atan
    line = {R: sqrt(1 + s**2), P: atan(s)}
    for equation in equations:
        value = sympy_expr(to_sympy(equation)).subs({to_sympy(k): to_sympy(v) for k, v in line.items()}).doit()
        assert sympy.simplify(value) == 0
    # the Laplacian in spherical coordinates on 1/r and on r**2
    spherical = Metric([[1, 0, 0], [0, r**2, 0], [0, 0, r**2*sin(theta)**2]], [r, theta, phi])
    assert spherical.laplacian(1/r) == 0 and spherical.laplacian(r**2) == 6
    assert spherical.laplacian(r*cos(theta)) == 0                   # the harmonic function z
    # the divergence of the radial field r d/dr is 3, the gradient of r**2/2 is that field
    assert spherical.divergence([r, 0, 0]) == 3 and spherical.gradient(r**2/2) == [r, 0, 0]
    assert spherical.covariant_derivative([1, 0, 0]) == ExprMatrix.from_rows([[0, 0, 0], [0, 1/r, 0], [0, 0, 1/r]])
    assert spherical.lower([0, 1, 0]) == [0, r**2, 0] and spherical.raise_index([0, r**2, 0]) == [0, 1, 0]
    assert spherical.christoffel_first_kind(0, 1, 1) == -r


def test_line_element_and_rejections() -> None:
    dt, dr, dphi = Symbol('dt'), Symbol('dr'), Symbol('dphi')
    metric = Metric.from_line_element(-dt**2 + dr**2 + r**2*dphi**2 + 2*a*dt*dphi, [t, r, phi], [dt, dr, dphi])
    assert metric.g == ExprMatrix.from_rows([[-1, 0, a], [0, 1, 0], [a, 0, r**2]])
    assert metric.inverse*metric.g == ExprMatrix.eye(3) or (metric.inverse*metric.g).applyfunc(normal_form) == ExprMatrix.eye(3)
    with pytest.raises(ValueError):
        Metric.from_line_element(dt**2 + dr, [t, r], [dt, dr])
    with pytest.raises(ValueError):
        Metric([[1, 2], [3, 4]], [x, y])
    with pytest.raises(ValueError):
        Metric([[1, 0], [0, 1]], [x, x])
    with pytest.raises(ValueError):
        Metric([[1, 0, 0], [0, 1, 0]], [x, y])
    with pytest.raises(ValueError):
        Metric([[1, 1], [1, 1]], [x, y]).inverse
    with pytest.raises(ValueError):
        Metric([[1, 0], [0, 0]], [x, y]).inverse
