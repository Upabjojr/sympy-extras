"""Tests of Chyzak's algorithm (D-finite creative telescoping)."""
from __future__ import annotations

from sympy import symbols, exp, besselj, sin, cos, oo, Eq, Function, diff, simplify, legendre
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy import S
from sympy_extras._testing import untyped
from sympy_extras._typing import as_expr
from sympy_extras.integrals.dfinite import (Annihilator, annihilators, normal_form, closure, chyzak, dfinite_ode,
                                            Factor)
from sympy_extras.integrals.telescoping import almkvist_zeilberger

x = symbols('x')
t = symbols('t', positive=True)


def test_annihilators() -> None:
    found = annihilators(exp(-t * x**2), x, t)
    assert found is not None
    L_x, L_t = found
    assert L_x.order == 1 and L_t.order == 1
    assert L_x.apply(exp(-t * x**2)).expand() == 0
    # a scaled Bessel argument, which expr_to_holonomic refuses directly,
    # goes through the substitution x = y/t
    found = annihilators(exp(-x**2) * besselj(0, t * x), x, t)
    assert found is not None
    L_x, L_t = found
    assert L_x.order == 2 and L_t.order == 2
    assert simplify(L_x.apply(exp(-x**2) * besselj(0, t * x))) == 0
    assert simplify(L_t.apply(exp(-x**2) * besselj(0, t * x))) == 0
    # not D-finite
    assert annihilators(exp(exp(x)) * t, x, t) is None
    raises(ValueError, lambda: Annihilator(x, [S.One]))


def test_normal_forms_are_identities() -> None:
    F = exp(-t * x) * besselj(0, x)
    found = annihilators(F, x, t)
    assert found is not None
    L_x, L_t = found
    reduce = normal_form(L_x, L_t, x, t)
    basis: dict[tuple[int, int], Expr] = {(i, 0): diff(F, x, i) for i in range(2)}
    for i, j in [(2, 0), (3, 0), (1, 1), (2, 2), (0, 3)]:
        form = reduce(i, j)
        assert all(key[0] < 2 and key[1] < 1 for key in form)
        combination: Expr = as_expr(sum((value * basis[key] for key, value in form.items()), S.Zero))
        target = diff(F, x, i, t, j) if j else diff(F, x, i)
        assert simplify(combination - target) == 0


def test_closure_of_products() -> None:
    found = closure(x * exp(-x**2) * besselj(0, t * x), x, t)
    assert found is not None
    coefficient, factors = found
    assert coefficient == x and sorted(f.order for f in factors) == [1, 2]
    assert all(isinstance(f, Factor) for f in factors)
    # the reduction of the Bessel equation: J'' = -J'/u - J
    J = [f for f in factors if f.order == 2][0]
    assert J.reduction == [-1, -1 / J.u]
    # a square is two factors; a polynomial is a coefficient
    found = closure(besselj(1, x)**2 * legendre(2, x), x, t)
    assert found is not None and len(found[1]) == 2
    assert closure(exp(exp(x)), x, t) is None
    # log(x) is D-finite (u y'' + y' = 0); a non-polynomial argument is not handled
    assert closure(sin(exp(x)) * exp(-t * x), x, t) is None


def test_telescopers_are_verified() -> None:
    for F in [exp(-t * x) * besselj(0, x), x * exp(-x**2) * besselj(0, t * x), exp(-t * x) * sin(x) / x,
              x**2 * exp(-x**2) * cos(2 * t * x), legendre(2, x) * exp(t * x)]:
        telescoper = chyzak(F, x, t)
        assert telescoper is not None, F
        assert telescoper.check(), F
    # the hyperexponential case agrees with the Almkvist–Zeilberger algorithm
    F = exp(-x**2) * exp(2 * t * x)
    hyper = almkvist_zeilberger(F, x, t)
    ours = chyzak(F, x, t)
    assert hyper is not None and ours is not None
    assert ours.coefficients == hyper.coefficients
    assert ours.order == 1
    # the bug: the two annihilators overcount the rank of J_0(t x) (its
    # t-derivative is a combination of its x-derivatives), and the first
    # order telescoper was missed; the closure of the factors has the
    # right rank
    telescoper = chyzak(x * exp(-x**2) * besselj(0, t * x), x, t)
    assert telescoper is not None and telescoper.order == 1
    assert chyzak(exp(exp(x)), x, t) is None


def test_differential_equations() -> None:
    I = Function('I')
    equation = dfinite_ode(exp(-t * x) * besselj(0, x), x, 0, oo, t)
    assert equation == Eq(t * I(t) + (t**2 + 1) * diff(I(t), t), 0)
    equation = dfinite_ode(exp(-t * x) * sin(x) / x, x, 0, oo, t)
    assert equation == Eq(diff(I(t), t), -1 / (t**2 + 1))
    # a boundary value which is not finite: no equation
    assert dfinite_ode(exp(t * x) * besselj(0, x), x, 0, oo, t) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(chyzak)([1], x, t))
