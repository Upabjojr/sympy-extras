"""Vector calculus in orthogonal coordinates: the tables of the
textbooks, the identities curl grad = 0 and div curl = 0, the Laplacian
against the Laplace-Beltrami operator of the metric, and the operators
computed in Cartesian coordinates through the change of variables."""
from __future__ import annotations

import pytest
import sympy

from sympy_extras.core import Function, Integer, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.diffgeom import Metric
from sympy_extras.functions import cos, exp, sin
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.simplify.normalform import normal_form
from sympy_extras.vector import OrthogonalCoordinates, cartesian, cylindrical, spherical

x, y, z, r, theta, phi, rho, u, v = (Symbol(s) for s in ['x', 'y', 'z', 'r', 'theta', 'phi', 'rho', 'u', 'v'])


def test_tables() -> None:
    S = spherical(r, theta, phi)
    f = Function('f')(r, theta, phi)
    # Arfken, Weber, Harris, eqs. (3.156)-(3.158)
    assert S.gradient(f) == [f.diff(r), f.diff(theta)/r, f.diff(phi)/(r*sin(theta))]
    expected = (f.diff(r, 2) + 2*f.diff(r)/r + f.diff(theta, 2)/r**2 + cos(theta)*f.diff(theta)/(r**2*sin(theta))
                + f.diff(phi, 2)/(r**2*sin(theta)**2))
    assert normal_form(S.laplacian(f) - expected) == 0
    Fr, Ft, Fp = (Function(name)(r, theta, phi) for name in ['Fr', 'Ft', 'Fp'])
    divergence = (r**2*Fr).diff(r)/r**2 + (sin(theta)*Ft).diff(theta)/(r*sin(theta)) + Fp.diff(phi)/(r*sin(theta))
    assert normal_form(S.divergence([Fr, Ft, Fp]) - divergence) == 0
    curl = S.curl([Fr, Ft, Fp])
    assert normal_form(curl[0] - ((sin(theta)*Fp).diff(theta) - Ft.diff(phi))/(r*sin(theta))) == 0
    assert normal_form(curl[1] - (Fr.diff(phi)/sin(theta) - (r*Fp).diff(r))/r) == 0
    assert normal_form(curl[2] - ((r*Ft).diff(r) - Fr.diff(theta))/r) == 0
    C = cylindrical(rho, phi, z)
    g = Function('g')(rho, phi, z)
    assert normal_form(C.laplacian(g) - ((rho*g.diff(rho)).diff(rho)/rho + g.diff(phi, 2)/rho**2 + g.diff(z, 2))) == 0
    assert C.volume_element() == rho and S.volume_element() == r**2*sin(theta)
    assert cartesian(x, y, z).laplacian(x**2*y + z**3) == 2*y + 6*z


def test_identities() -> None:
    systems = [spherical(r, theta, phi), cylindrical(rho, phi, z),
               OrthogonalCoordinates.from_transformation([u, v, z], [(u**2 - v**2)/2, u*v, z])]
    for system in systems:
        a, b, c = system.coordinates
        scalar = a**2*sin(b) + c*a + exp(c)*b
        field: list[Expr] = [a*b, sin(b)*c + a**2, a*c**2 + b]
        assert system.curl(system.gradient(scalar)) == [0, 0, 0]
        assert system.divergence(system.curl(field)) == 0
        assert normal_form(system.laplacian(scalar) - system.divergence(system.gradient(scalar))) == 0
        # the Laplace-Beltrami operator of the metric diag(h_i**2) is an independent computation
        h = system.scale_factors
        metric = Metric([[h[i]**2 if i == j else Integer(0) for j in range(3)] for i in range(3)], [a, b, c])
        assert normal_form(system.laplacian(scalar) - metric.laplacian(scalar)) == 0
        # and the divergence is that of the contravariant components F_i/h_i
        assert normal_form(system.divergence(field) - metric.divergence([field[i]/h[i] for i in range(3)])) == 0


def test_against_cartesian_coordinates() -> None:
    # f = x**2 y + z in spherical coordinates: its Laplacian 2 y, and the norm of its gradient, are scalars
    sx, sy, sz = sympy.symbols('x y z')
    sr, st, sp = sympy.symbols('r theta phi')
    change = {sx: sr*sympy.sin(st)*sympy.cos(sp), sy: sr*sympy.sin(st)*sympy.sin(sp), sz: sr*sympy.cos(st)}
    cartesian_f = sx**2*sy + sz
    S = spherical(r, theta, phi)
    f = (r*sin(theta)*cos(phi))**2*(r*sin(theta)*sin(phi)) + r*cos(theta)
    laplacian = sympy_expr(to_sympy(S.laplacian(f)))
    assert sympy.simplify(laplacian - (2*sy).subs(change)) == 0
    norm2 = sum((sympy_expr(to_sympy(c))**2 for c in S.gradient(f)), sympy.Integer(0))
    expected = sum((sympy.diff(cartesian_f, s)**2 for s in (sx, sy, sz)), sympy.Integer(0)).subs(change)
    assert sympy.simplify(norm2 - expected) == 0
    # the field (-y, x, 0) = rho e_phi has the curl (0, 0, 2) = 2 e_z
    assert cylindrical(rho, phi, z).curl([0, rho, 0]) == [0, 0, 2]
    # the vector Laplacian of a field with harmonic Cartesian components vanishes: (x, y, z) = r e_r
    assert S.vector_laplacian([r, 0, 0]) == [0, 0, 0]


def test_transformations() -> None:
    S = OrthogonalCoordinates.from_transformation([r, theta, phi], [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)])
    assert S.scale_factors[0] == 1 and normal_form(S.scale_factors[1]**2 - r**2) == 0
    assert normal_form(S.scale_factors[2]**2 - r**2*sin(theta)**2) == 0
    # harmonic in the plane: Re and Im of (u + i v)**3 in parabolic coordinates
    parabolic = OrthogonalCoordinates.from_transformation([u, v], [(u**2 - v**2)/2, u*v])
    assert parabolic.laplacian(u**3 - 3*u*v**2) == 0
    with pytest.raises(ValueError):
        OrthogonalCoordinates.from_transformation([u, v], [u + v, v])
    with pytest.raises(ValueError):
        OrthogonalCoordinates([u, v], [1])
    with pytest.raises(ValueError):
        OrthogonalCoordinates([u, u], [1, 1])
    with pytest.raises(ValueError):
        parabolic.curl([1, 0])
    with pytest.raises(ValueError):
        parabolic.divergence([1, 0, 0])
