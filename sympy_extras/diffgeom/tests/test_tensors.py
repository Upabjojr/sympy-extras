"""Tensors on a metric: the identities of the Ricci calculus (metric
compatibility, the Ricci identity, both Bianchi identities, Leibniz),
and the curvature scalars against the methods of Metric."""
from __future__ import annotations

import pytest

from sympy_extras.core import Function, Integer, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.diffgeom import Metric
from sympy_extras.diffgeom.tensors import Tensor, einstein_tensor, metric_tensor, ricci_tensor, riemann_tensor
from sympy_extras.functions import sin
from sympy_extras.simplify.normalform import normal_form

t, r, theta, phi, m, x, y, z = (Symbol(n) for n in ['t', 'r', 'theta', 'phi', 'm', 'x', 'y', 'z'])


def _schwarzschild() -> Metric:
    f = 1 - 2*m/r
    return Metric([[-f, 0, 0, 0], [0, 1/f, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]], [t, r, theta, phi])


def _generic() -> Metric:
    rows: list[list[Expr]] = [[1 + x**2, x*y, Integer(0)], [x*y, Integer(2), y], [Integer(0), y, 1 + z**2]]
    return Metric(rows, [x, y, z])


def test_metric_compatibility_and_index_gymnastics() -> None:
    for metric in (_schwarzschild(), _generic()):
        g = metric_tensor(metric)
        assert g.covariant_derivative().is_zero
        assert g.with_positions('uu').covariant_derivative().is_zero
        n = metric.dimension
        delta = g.raise_index(0)
        assert delta.components == tuple(Integer(1) if i == j else Integer(0) for i in range(n) for j in range(n))
        # compared by their difference: the entries given (2*m/r - 1) are not in the form of the results
        assert g.trace() == n and (g.with_positions('uu').lower_index(0).lower_index(1) - g).is_zero
    metric = _generic()
    T = Tensor(metric, [x, y*z, Integer(1), x*y, Integer(0), z, y, x + z, z**2], 'ud')
    assert T.lower_index(0).raise_index(0) == T and T.with_positions('du').with_positions('ud') == T
    assert T.transpose(0, 1).transpose(0, 1) == T and T.transpose(0, 1).positions == 'du'
    # the trace does not depend on which index is up
    assert normal_form(T.trace() - T.with_positions('du').trace()) == 0 == normal_form(T.trace() - T.with_positions('dd').trace())


def test_curvature_tensors_agree_with_the_metric() -> None:
    metric = _schwarzschild()
    R = riemann_tensor(metric)
    assert R.contract(0, 2) == ricci_tensor(metric) and ricci_tensor(metric).is_zero
    # the Kretschmann scalar as a full contraction
    lowered, raised = R.with_positions('dddd'), R.with_positions('uuuu')
    product = lowered*raised
    for _ in range(4):
        product = product.contract(0, product.rank//2)
    assert product.scalar() == metric.kretschmann() == 48*m**2/r**6
    generic = _generic()
    Ric = ricci_tensor(generic)
    assert normal_form(Ric.trace() - generic.ricci_scalar()) == 0
    G = einstein_tensor(generic)
    assert (Ric - metric_tensor(generic).scale(generic.ricci_scalar()/2)) == G


def test_ricci_identity() -> None:
    # (nabla_c nabla_d - nabla_d nabla_c) V^a = R^a_bcd V^b fixes the sign and the order of the indices
    metric = _generic()
    V = Tensor(metric, [x*y, z + x, y**2], 'u')
    second = V.covariant_derivative().covariant_derivative()           # V^a ;d ;c with the positions 'udd' (a, d, c)
    commutator = second.transpose(1, 2) - second                       # (a, c, d): V^a;c;d... minus the other order
    expected = (riemann_tensor(metric)*V).contract(1, 4)               # R^a_bcd V^b -> (a, c, d)
    assert (commutator - expected).is_zero
    # for a covector: (nabla_c nabla_d - nabla_d nabla_c) w_b = -R^a_bcd w_a
    w = Tensor(metric, [y, x*z, Integer(1)], 'd')
    second = w.covariant_derivative().covariant_derivative()
    commutator = second.transpose(1, 2) - second
    expected = (riemann_tensor(metric)*w).contract(0, 4)
    assert (commutator + expected).is_zero


def test_bianchi_identities() -> None:
    metric = _generic()
    G = einstein_tensor(metric)
    assert G.divergence(0).is_zero and G.divergence(1).is_zero
    # the second identity: R^a_b[cd;e] = 0 (on a surface without symmetry, to keep the 2**5 components cheap)
    surface = Metric([[1 + x**2, x*y], [x*y, 2 + y**4]], [x, y])
    D = riemann_tensor(surface).covariant_derivative()                 # (a, b, c, d, e)
    cyclic = D + D.transpose(2, 3).transpose(3, 4) + D.transpose(3, 4).transpose(2, 3)
    assert cyclic.is_zero and not D.is_zero
    # with matter: the Robertson-Walker metric, whose Einstein tensor is not zero
    a = Function('a')(t)
    frw = Metric([[-1, 0, 0, 0], [0, a**2, 0, 0], [0, 0, a**2, 0], [0, 0, 0, a**2]], [t, x, y, z])
    E = einstein_tensor(frw)
    assert not E.is_zero and E.divergence(0).is_zero


def test_leibniz_and_parts() -> None:
    metric = Metric([[1, 0], [0, sin(theta)**2]], [theta, phi])
    V = Tensor(metric, [phi*theta, sin(theta)], 'u')
    w = Tensor(metric, [theta**2, phi], 'd')
    left = (V*w).covariant_derivative()                                # (a, b, c)
    right = V.covariant_derivative()*w                                 # (a, c, b)
    assert (left - right.transpose(1, 2) - V*w.covariant_derivative()).is_zero
    # a scalar: the derivative of the contraction is the contraction of the derivative
    scalar = (V*w).contract(0, 1)
    assert scalar.rank == 0 and scalar.covariant_derivative().components == (scalar.scalar().diff(theta), scalar.scalar().diff(phi))
    assert (left.contract(0, 1) - scalar.covariant_derivative()).is_zero
    S = Tensor(metric, [theta, phi, Integer(1), theta*phi], 'dd')
    assert S.symmetrize(0, 1) + S.antisymmetrize(0, 1) == S
    assert S.antisymmetrize(0, 1).transpose(0, 1) == -S.antisymmetrize(0, 1)
    # the Killing field of the rotations, and one which is not
    assert Tensor(metric, [0, 1], 'u').lower_index(0).covariant_derivative().symmetrize(0, 1).is_zero
    assert not Tensor(metric, [1, 0], 'u').lower_index(0).covariant_derivative().symmetrize(0, 1).is_zero


def test_rejections() -> None:
    metric = Metric([[1, 0], [0, 1]], [x, y])
    T = Tensor(metric, [1, 2, 3, 4], 'ud')
    with pytest.raises(ValueError):
        Tensor(metric, [1, 2, 3], 'ud')
    with pytest.raises(ValueError):
        Tensor(metric, [1, 2], 'x')
    with pytest.raises(ValueError):
        T.with_positions('dd').contract(0, 1)
    with pytest.raises(ValueError):
        T.symmetrize(0, 1)
    with pytest.raises(ValueError):
        T + Tensor(metric, [1, 2, 3, 4], 'dd')
    with pytest.raises(IndexError):
        T[0, 5]
    with pytest.raises(ValueError):
        T.scalar()
    with pytest.raises(ValueError):
        Tensor(metric, [1, 2], 'u').trace()
    assert T[1, 0] == 3 and T.scale(2).components == (2, 4, 6, 8) and (-T)[0, 0] == -1
