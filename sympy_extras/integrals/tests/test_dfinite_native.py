"""Tests of the native Chyzak algorithm on the objects of the core
(:mod:`sympy_extras.integrals.dfinite`)."""
from __future__ import annotations

import pytest

from sympy_extras.core import Integer, Symbol, diff, oo
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.functions import besselj, cos, erf, exp, log, sin
from sympy_extras.integrals.dfinite import Annihilator, chyzak, closure, dfinite_ode, normal_form
from sympy_extras.simplify.simplify import simplify

x = Symbol('x')
t = Symbol('t', positive=True)


def test_telescopers_satisfy_their_identity() -> None:
    for F in [exp(-t*x)*besselj(0, x), x*exp(-x**2)*besselj(0, t*x), exp(-t*x)*sin(x)/x, exp(-x**2)*cos(t*x),
              erf(t*x)*exp(-x**2), exp(-t*x)*log(x)]:
        found = chyzak(F, x, t)
        assert found is not None, F
        assert found.check(), F
        # independently of check: P F - dG/dx vanishes at a sample point
        lhs = sum((a*diff(F, (t, k)) for k, a in enumerate(found.coefficients)), Integer(0))
        difference = (lhs - diff(found.function(), x)).doit()
        value = difference.subs({x: Integer(3)/7, t: Integer(5)/3}).evalf(30)
        assert isinstance(value, Expr) and abs(complex(value)) < 1e-20, F


def test_known_telescopers() -> None:
    found = chyzak(exp(-t*x)*besselj(0, x), x, t)
    assert found is not None and found.coefficients == [t, t**2 + 1]
    found = chyzak(exp(-x**2)*exp(2*t*x), x, t)
    assert found is not None and found.coefficients == [-2*t, 1]
    # the closure of the factors has the rank of J_0(t x): order one
    found = chyzak(x*exp(-x**2)*besselj(0, t*x), x, t)
    assert found is not None and found.order == 1
    assert chyzak(exp(exp(x)), x, t) is None


def test_closure() -> None:
    found = closure(x*exp(-x**2)*besselj(0, t*x), x, t)
    assert found is not None
    coefficient, factors = found
    assert coefficient == x and sorted(f.order for f in factors) == [1, 2]
    J = [f for f in factors if f.order == 2][0]
    assert J.reduction == [-1, -1/J.u] and J.derivative(1) == -besselj(1, t*x)
    # a square is two factors
    found = closure(besselj(1, x)**2*exp(-t*x), x, t)
    assert found is not None and len(found[1]) == 3
    assert closure(sin(exp(x))*exp(-t*x), x, t) is None


def test_normal_forms() -> None:
    # F = exp(-t*x**2): L_x = D_x + 2 t x, L_t = D_t + x**2
    L_x, L_t = Annihilator(x, [2*t*x, Integer(1)]), Annihilator(t, [x**2, Integer(1)])
    reduce = normal_form(L_x, L_t, x, t)
    F = exp(-t*x**2)
    for i, j in [(2, 0), (1, 1), (3, 2)]:
        form = reduce(i, j)
        assert list(form) == [(0, 0)]
        assert simplify(form[(0, 0)]*F - diff(F, (x, i), (t, j))) == 0
    with pytest.raises(ValueError):
        Annihilator(x, [Integer(1)])


def test_equations() -> None:
    equation = dfinite_ode(exp(-t*x)*sin(x)/x, x, 0, oo, t)
    assert equation is not None and simplify(equation.rhs + 1/(t**2 + 1)) == 0
    # a boundary value which is not finite: no equation
    assert dfinite_ode(exp(t*x)*besselj(0, x), x, 0, oo, t) is None


def _annihilates(L: Annihilator, F: Expr) -> bool:
    """``L F`` vanishes at two sample points (an independent check of the
    operator, whatever its form)."""
    residual = L.apply(F).doit()
    for point in (Integer(3)/7, Integer(13)/5):
        values: dict[Basic, Basic] = {x: Integer(11)/9, t: Integer(5)/3, Symbol('n'): Integer(2)/3}
        values[L.variable] = point
        value = residual.subs(values).evalf(30)
        if not (isinstance(value, Expr) and abs(complex(value)) < 1e-20):
            return False
    return True


def test_native_annihilators() -> None:
    from sympy_extras.functions import Ei, Si, atan, asin, airyai, besseli, bessely, sqrt
    from sympy_extras.integrals.dfinite import _minimal_annihilator, annihilators
    n = Symbol('n')
    # the annihilators came from SymPy's expr_to_holonomic
    # (holonomic_of_core); the minimal ones of the closure, by their
    # orders (SymPy's are of order 3 for besseli and bessely and it has
    # none for airyai) and by applying them
    for F, order in [(besseli(0, x), 2), (besseli(n, x), 2), (bessely(n, x), 2), (airyai(x), 2), (Ei(x), 2),
                     (Si(x), 3), (atan(x), 2), (asin(x), 2), (sqrt(x), 1), (x**n, 1), (exp(1/x), 1),
                     (log(1 + x), 2), (sin(x) + exp(x), 3), (x**2 + 1, 1)]:
        found = _minimal_annihilator(F, x, t)
        assert found is not None and found.order == order, (F, found)
        assert _annihilates(found, F), F
    # SymPy's equation of Ei, the first order equation of exp(-t x**2)
    found = _minimal_annihilator(Ei(x), x, t)
    assert found is not None and found.coefficients == [0, 1 - x, x]
    pair = annihilators(exp(-t*x**2), x, t)
    assert pair is not None and pair[0].coefficients == [2*t*x, 1] and pair[1].coefficients == [x**2, 1]
    # a scaled argument needs no substitution any more
    pair = annihilators(exp(-x**2)*besselj(0, t*x), x, t)
    assert pair is not None and [L.order for L in pair] == [2, 2]
    assert all(_annihilates(L, exp(-x**2)*besselj(0, t*x)) for L in pair)
    assert annihilators(exp(exp(x))*t, x, t) is None


def test_factors_outside_the_table() -> None:
    from sympy_extras.functions import besseli
    # besseli went to SymPy's expr_to_holonomic, which gives a third order
    # equation: the second order modified Bessel equation now
    found = closure(besseli(0, x)*exp(-t*x), x, t)
    assert found is not None and sorted(f.order for f in found[1]) == [1, 2]
    telescoper = chyzak(exp(-t*x)*besseli(0, x), x, t)
    assert telescoper is not None and telescoper.check()
    assert telescoper.coefficients == [t, t**2 - 1]
