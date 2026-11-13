"""Most of these tests come from the examples in Bronstein's book (on the
typed core: the polynomials of integrals.risch.tower)."""
from __future__ import annotations

from sympy_extras.core import Symbol
from sympy_extras.core.expr import Expr
from sympy.testing.pytest import raises
from sympy_extras.integrals.risch.risch import (DifferentialExtension, derivation,
    NonElementaryIntegralException)
from sympy_extras.integrals.risch.rde import OO
from sympy_extras.integrals.risch.cde import (coupled_DE_system, param_coupled_DE_system,
    coupled_DE_cancel_prim, coupled_DE_cancel_exp, coupled_DE_cancel_tan)
from sympy_extras.integrals.risch.tower import RationalFunction, TowerPoly

x, t = Symbol('x'), Symbol('t')


def P(e: object, gen: Expr) -> TowerPoly:
    if isinstance(e, int):
        return TowerPoly.from_expr(e, gen)
    assert isinstance(e, Expr)
    return TowerPoly.from_expr(e, gen)


def R(e: object) -> RationalFunction:
    if isinstance(e, int):
        return RationalFunction.from_int(e)
    assert isinstance(e, Expr)
    return RationalFunction.from_expr(e)


def _frac(f: tuple[TowerPoly, TowerPoly]) -> RationalFunction:
    return f[0].as_element()/f[1].as_element()


def _check(q1: TowerPoly, q2: TowerPoly, b1e: object, b2e: object, c1e: object, c2e: object,
        DE: DifferentialExtension) -> None:
    """(q1, q2) solves Dq1 + b1*q1 - b2*q2 == c1, Dq2 + b2*q1 + b1*q2 == c2."""
    q1e, q2e = q1.as_element(), q2.as_element()
    b1, b2, c1, c2 = R(b1e), R(b2e), R(c1e), R(c2e)
    assert (derivation(q1, DE).as_element() + b1*q1e - b2*q2e - c1).is_zero
    assert (derivation(q2, DE).as_element() + b2*q1e + b1*q2e - c2).is_zero


def test_coupled_DE_system() -> None:
    DE = DifferentialExtension.manual([P(1, x)])
    one = P(1, x)
    # From the integration of tan(x**2) (Section 5.10):
    # CoupledDESystem(0, 1, 2/x, 0) has no solution, so tan(x**2) has
    # no elementary integral
    raises(NonElementaryIntegralException, lambda: coupled_DE_system(
        (P(0, x), one), (one, one), (P(2, x), P(x, x)),
        (P(0, x), one), DE))
    # From Example 6.6.2: CoupledDESystem(1, -1, -2*x, -2*(x + 1))
    # == (0, -2*x)
    y1, y2 = coupled_DE_system((one, one), (P(-1, x), one),
        (P(-2*x, x), one), (P(-2*(x + 1), x), one), DE)
    assert _frac(y1) == 0
    assert _frac(y2) == R(-2*x)
    # The inner system of Example 8.4.1:
    # CoupledDESystem(0, 4*x - 2, 2 - 8*x**2, 4 - 4*x) == (-1, 2*x + 1)
    y1, y2 = coupled_DE_system((P(0, x), one),
        (P(4*x - 2, x), one), (P(2 - 8*x**2, x), one),
        (P(4 - 4*x, x), one), DE)
    assert _frac(y1) == -1
    assert _frac(y2) == R(2*x + 1)


def test_param_coupled_DE_system() -> None:
    DE = DifferentialExtension.manual([P(1, x)])
    one = P(1, x)
    X = R(x)
    # The inner system of Example 8.4.1, parametrized:
    # Du - (4*x - 2)*v == c1*(2 - 8*x**2), Dv + (4*x - 2)*u == c1*(4 - 4*x)
    # has exactly the solutions (u, v) == c1*(-1, 2*x + 1)
    H, A = param_coupled_DE_system((P(0, x), one), (P(4*x - 2, x), one),
        [((P(2 - 8*x**2, x), one), (P(4 - 4*x, x), one))], DE)
    found = False
    for v in A.nullspace():
        c1 = v[0, 0]
        u = RationalFunction.from_int(0)
        w = RationalFunction.from_int(0)
        for j in range(len(H)):
            u = u + v[1 + j, 0]*_frac(H[j][0])
            w = w + v[1 + j, 0]*_frac(H[j][1])
        # each nullspace vector solves the system
        assert (u.diff(x) - (4*X - 2)*w - c1*(2 - 8*X**2)).is_zero
        assert (w.diff(x) + (4*X - 2)*u - c1*(4 - 4*X)).is_zero
        if c1 == 1 and (u, w) == (R(-1), R(2*x + 1)):
            found = True
    assert found


def test_coupled_DE_cancel_prim() -> None:
    # t = log(x)
    DE = DifferentialExtension.manual([P(1, x), P(1/x, t)])
    # b == b1 + b2*sqrt(-1) == Dz/z for z == x + sqrt(-1); the system
    # was built from the solution (t, 1), and the returned solution
    # differs from it by the homogeneous solution -x/(x**2 + 1) +
    # sqrt(-1)/(x**2 + 1) == -1/(x + sqrt(-1))
    b1e = x/(x**2 + 1)
    b2e = -1/(x**2 + 1)
    c1e = (R(1/x) + R(b1e)*R(t) - R(b2e)).as_expr()
    c2e = (R(b2e)*R(t) + R(b1e)).as_expr()
    q1, q2 = coupled_DE_cancel_prim(P(b1e, t), P(b2e, t),
        P(c1e, t), P(c2e, t), DE, 1)
    _check(q1, q2, b1e, b2e, c1e, c2e, DE)
    assert q1.degree(t) == 1
    # b == sqrt(-1) is not a logarithmic derivative of k(sqrt(-1))*,
    # so this exercises the coefficient-descent loop; the solution
    # (t, 0) is unique
    q1, q2 = coupled_DE_cancel_prim(P(0, t), P(1, t),
        P(1/x, t), P(t, t), DE, 1)
    assert (q1, q2) == (P(t, t), P(0, t))
    # no solution: Dq1 == 1/x + t needs q1 == t*(t + ...)/2 of degree 2
    raises(NonElementaryIntegralException, lambda: coupled_DE_cancel_prim(
        P(0, t), P(1, t), P(1/x + t, t), P(t, t), DE, 1))


def test_coupled_DE_cancel_exp() -> None:
    # t = exp(x)
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    # b == Dz/z + m*Dt/t for z == x + sqrt(-1) and m == 1; the system
    # was built from the solution (t, 0), which is unique here
    b1e = (R(x/(x**2 + 1)) + 1).as_expr()
    b2e = -1/(x**2 + 1)
    c1e = (R(t) + R(b1e)*R(t)).as_expr()
    c2e = (R(b2e)*R(t)).as_expr()
    q1, q2 = coupled_DE_cancel_exp(P(b1e, t), P(b2e, t),
        P(c1e, t), P(c2e, t), DE, 1)
    assert (q1, q2) == (P(t, t), P(0, t))
    _check(q1, q2, b1e, b2e, c1e, c2e, DE)
    # b == sqrt(-1) is not of the form Dz/z + m*Dt/t, so this
    # exercises the coefficient-descent loop; the solution (t, t) is
    # unique
    q1, q2 = coupled_DE_cancel_exp(P(0, t), P(1, t),
        P(0, t), P(2*t, t), DE, 1)
    assert (q1, q2) == (P(t, t), P(t, t))
    # no solution: the t-free parts require Ds1 - s2 == 2/x and
    # Ds2 + s1 == 0 in k, the system whose unsolvability proves that
    # tan(x**2) has no elementary integral (Section 5.10)
    raises(NonElementaryIntegralException, lambda: coupled_DE_cancel_exp(
        P(0, t), P(1, t), P(2/x, t), P(2*t, t), DE, 1))


def test_coupled_DE_cancel_tan() -> None:
    # Example 8.4.1: t = tan(x), from the integration of (8.12); the
    # system (8.15) with b0 == 0, b2 == 4*x, n == 2 has the solution
    # (t - 1, 2*x)
    DE = DifferentialExtension.manual([P(1, x), P(t**2 + 1, t)])
    b2e = 4*x
    c1e = -t**2 + 2*t - 8*x**2 + 1
    c2e = 2*(1 - 2*x)
    q1, q2 = coupled_DE_cancel_tan(P(0, t), P(b2e, t),
        P(c1e, t), P(c2e, t), DE, 2)
    assert (q1, q2) == (P(t - 1, t), P(2*x, t))
    # the diagonal entries of the system are b0 - n*eta*t (the matrix
    # in the book's CoupledDECancelTan specification misprints the
    # bottom-right entry as b0 + n*eta*t)
    _check(q1, q2, -2*t, b2e, c1e, c2e, DE)
    # n == 0: the system for constant-degree solutions is a coupled
    # system over k; (x, 1) is the unique solution
    q1, q2 = coupled_DE_cancel_tan(P(0, t), P(1, t),
        P(0, t), P(x, t), DE, 0)
    assert (q1, q2) == (P(x, t), P(1, t))
    _check(q1, q2, 0, 1, 0, x, DE)
    # n == 0 with deg(c) > 0 has no solution
    raises(NonElementaryIntegralException, lambda: coupled_DE_cancel_tan(
        P(0, t), P(1, t), P(t, t), P(x, t), DE, 0))
    # the recursion descends from n to 0, so n == oo is not allowed
    raises(ValueError, lambda: coupled_DE_cancel_tan(
        P(0, t), P(1, t), P(0, t), P(x, t), DE, OO))
