"""Tests of the region integrals with a rotational symmetry on the objects
of the core (:mod:`sympy_extras.integrals.axisymmetric`): the volumes and
areas against the classical formulas and mpmath's quadrature
(independent sources), the rotation tests, the aligned frame and the
agreement with the SymPy face."""
from __future__ import annotations

import mpmath

from sympy_extras.core import Integer, Rational, Symbol, eq, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import and_
from sympy_extras.functions import exp, heaviside, piecewise
from sympy_extras.integrals.axisymmetric import (_invariant, _orthogonalize, aligned_frame, axisymmetric_integral,
                                                 rotation_group, rotation_invariant)

x, y, z, w = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('w')
ball = x**2 + y**2 + z**2 < 1
sphere = eq(x**2 + y**2 + z**2, 1)


def _close(value: Expr, expected: float) -> bool:
    return abs(complex(value.evalf(20)) - expected) < 1e-10


def test_rotation_invariance() -> None:
    assert rotation_invariant(x**2 + y**2, [x, y])
    assert rotation_invariant(exp(-(x**2 + y**2)) * z, [x, y])
    # sympy-extras#57: piecewise constant integrands, invisible to the derivative test
    assert not rotation_invariant(heaviside(x), [x, y])
    assert not rotation_invariant(piecewise((1, x * y > 0), (0, True)), [x, y])
    assert not _invariant(heaviside(x), [x, y])
    assert _invariant((x**2 + y**2)**2, [x, y])


def test_rotation_group() -> None:
    assert rotation_group(Integer(1), and_(ball, z > x**2 + y**2), [x, y, z]) == (x, y)
    assert rotation_group(Integer(1), and_(ball, (x - 1)**2 + y**2 + z**2 < 1), [x, y, z]) == (y, z)
    assert rotation_group(Integer(1), ball, [x, y, z]) == (x, y, z)
    assert rotation_group(x * y, ball, [x, y, z]) is None
    assert rotation_group(Integer(1), and_(x**2 + y**2 + z**2 + w**2 < 1, w > Rational(1, 2)), [x, y, z, w]) \
        == (x, y, z)


def test_gram_schmidt() -> None:
    # SymPy's Matrix.orthogonalize(a, e1, e2, e3, normalize=True): three
    # orthonormal vectors, the first along a, the dependent one dropped
    one, zero = Integer(1), Integer(0)
    basis = _orthogonalize([[one, one, one], [one, zero, zero], [zero, one, zero], [zero, zero, one]])
    assert len(basis) == 3
    assert basis[0] == [1 / sqrt(3)] * 3
    for i in range(3):
        for j in range(3):
            product = sum((u * v for u, v in zip(basis[i], basis[j])), Integer(0))
            assert (product - (1 if i == j else 0)).expand() == 0


def test_aligned_frame() -> None:
    frame = aligned_frame(Integer(1), and_(ball, x + y + z > 1), [x, y, z])
    assert frame is not None and frame.ratio == sqrt(3)
    Y1, Y2, Y3 = frame.variables
    assert frame.condition == and_(Y1 - 1 > 0, Y1**2 / 3 + Y2**2 / 3 + Y3**2 / 3 - 1 < 0)
    assert aligned_frame(Integer(1), and_(ball, x + y + z > 0, x - y > 0), [x, y, z]) is None


def test_volumes_and_areas() -> None:
    # the lens of two unit balls a distance 1 apart, pi (4 R + d)(2 R - d)**2/12
    shifted = (x - 1)**2 + y**2 + z**2 < 1
    assert axisymmetric_integral(Integer(1), and_(ball, shifted), [x, y, z]) == 5 * pi / 12
    # a cap of height 1/2, pi h**2 (3 R - h)/3, and its area 2 pi R h
    assert axisymmetric_integral(Integer(1), and_(ball, z > Rational(1, 2)), [x, y, z]) == 5 * pi / 24
    assert axisymmetric_integral(Integer(1), and_(sphere, z > Rational(1, 2)), [x, y, z], measure='hausdorff') == pi
    # a ball cut by a tilted plane, through the aligned frame
    h = 1 - 1 / mpmath.sqrt(3)
    cap = axisymmetric_integral(Integer(1), and_(ball, x + y + z > 1), [x, y, z])
    assert cap is not None and _close(cap, float(mpmath.pi * h**2 * (3 - h) / 3))
    # no symmetry: declined
    assert axisymmetric_integral(Integer(1), and_(x**2 + y**2 < 1, y**2 + z**2 < 1), [x, y, z]) is None


def test_agreement_with_the_face() -> None:
    import sympy
    from sympy_extras.integrals.axisymmetric_sympy import axisymmetric_integral as face_integral
    from sympy_extras.interop import to_sympy
    X, Y, Z = sympy.symbols('x y z')
    face = face_integral(1, (X**2 + Y**2 + Z**2 < 1) & (X**2 + Y**2 < sympy.Rational(1, 4)), [X, Y, Z])
    native = axisymmetric_integral(Integer(1), and_(ball, x**2 + y**2 < Rational(1, 4)), [x, y, z])
    assert face is not None and native is not None and face == to_sympy(native)
