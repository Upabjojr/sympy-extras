"""Tests of the native ``meijerint_inversion`` (the inverse Laplace
transform by Meijer G-functions): each value against SymPy's
``meijerint_inversion`` (through :mod:`sympy_extras.interop`) and
numerically, by its forward Laplace transform computed by quadrature at a
sample point, the parameters at sample values."""
from __future__ import annotations

from typing import Callable, Optional

import mpmath
import pytest
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.integrals.meijerint import meijerint_inversion as sympy_meijerint_inversion

from sympy_extras.core import Rational, Symbol, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import true
from sympy_extras.functions import atan, exp, gamma, log
from sympy_extras.integrals.meijerint import meijerint_inversion
from sympy_extras.interop import sympy_expr, to_sympy

x, t = Symbol('x'), Symbol('t')
a, b = Symbol('a', positive=True), Symbol('b', positive=True)
R = Rational

#: the sample values of the parameters and the point of the forward transform
VALUES: dict[str, sympy.Rational] = {'a': sympy.Rational(3, 2), 'b': sympy.Rational(5, 2)}
S0 = sympy.Integer(3)


def _heaviside(u: mpmath.mpf, h0: mpmath.mpf = mpmath.mpf(0.5)) -> mpmath.mpf:
    return mpmath.mpf(1) if u > 0 else (h0 if u == 0 else mpmath.mpf(0))


def _forward_error(f: Expr, F: Expr) -> float:
    """The relative difference between ``F(S0)`` and the Laplace transform
    of ``f(t)`` at ``S0`` by quadrature, the parameters at ``VALUES``."""
    return _forward_error_sympy(sympy_expr(to_sympy(f)), sympy_expr(to_sympy(F)))


def _forward_error_sympy(fs: sympy.Expr, Fs: sympy.Expr) -> float:
    subs: dict[sympy.Basic | complex, sympy.Expr | complex] = {
        sym: VALUES[sym.name] for sym in fs.free_symbols | Fs.free_symbols
        if isinstance(sym, sympy.Symbol) and sym.name in VALUES}
    fs, Fs = sympy.unpolarify(fs.subs(subs)), Fs.subs(subs)
    ts, xs = sympy.Symbol('t'), sympy.Symbol('x')
    expected = complex(sympy.N(Fs.subs(xs, S0), 30))
    func: Callable[[mpmath.mpf], mpmath.mpc] = sympy.lambdify(ts, fs, modules=[{'Heaviside': _heaviside}, 'mpmath'])
    points: list[mpmath.mpf] = [mpmath.mpf(0)]
    for h in fs.atoms(sympy.Heaviside):
        points += [mpmath.mpf(float(v)) for v in sympy.solve(h.args[0], ts) if v.is_positive]
    points = sorted(set(points)) + [mpmath.inf]
    with mpmath.workdps(20):
        value = mpmath.quad(lambda u: func(u) * mpmath.exp(-S0 * u), points)
    return abs(complex(value) - expected) / max(1.0, abs(expected))


def _sympy_value(F: Expr) -> Optional[sympy.Expr]:
    """SymPy's value (its first piece), ``None`` when it fails."""
    try:
        found = sympy_meijerint_inversion(to_sympy(F), sympy.Symbol('x'), sympy.Symbol('t'))
    except NotImplementedError:
        return None
    if found is None:
        return None
    if isinstance(found, sympy.Piecewise):
        piece = found.args[0]
        assert isinstance(piece, ExprCondPair)
        return sympy_expr(piece.expr)
    return sympy_expr(found)


# the transforms on which the native value is SymPy's, unconditionally
AGREE: list[Expr] = [
    1 / x, 1 / x**2, 1 / x**3, 1 / (x + a), 1 / (x + a)**2, 1 / (x**2 + 1), x / (x**2 + 1),
    a / (x**2 + a**2), x / (x**2 + a**2), 1 / (x**2 - a**2), x / (x**2 - a**2), 1 / (x * (x + 1)),
    1 / (x**2 * (x + a)), 1 / (x * (x**2 + 1)), 1 / (x**2 + a**2)**2, x / (x**2 + a**2)**2,
    1 / (x**2 * (x**2 + a**2)), 1 / (x * (x + a)**2), 1 / x**R(3, 2), 1 / (x + a)**R(5, 2),
    1 / sqrt(x**2 + 1), exp(-sqrt(x)), exp(-a * sqrt(x)) / sqrt(x), log(x) / x, 1 / (x * sqrt(x + 1)),
    exp(-a / x) / x,
    # shifts
    exp(-a * x) / x, exp(-a * x) / x**2, exp(-a * x) / (x + b), exp(-2 * x) / (x**2 + 1),
    exp(-x) / (x * (x + a)), x**(-a - 1) * exp(-b * x),
]


@pytest.mark.parametrize('F', AGREE, ids=str)
def test_agrees_with_sympy(F: Expr) -> None:
    found = meijerint_inversion(F, x, t)
    assert found is not None
    value, cond = found
    assert cond == true
    expected = _sympy_value(F)
    assert expected is not None
    assert sympy.simplify(sympy_expr(to_sympy(value)) - expected) == 0
    assert _forward_error(value, F) < 1e-10


# gamma-type: a condition on the exponent, as in SymPy
CONDITIONAL: list[tuple[Expr, str]] = [
    (gamma(a) / x**a, 'a >= 1'), (1 / x**a, 'a >= 1'), ((x + a)**(-b), 'b >= 1'),
    (exp(-a * x) * gamma(b) / x**b, 'b >= 1'),
]


@pytest.mark.parametrize('F, condition', CONDITIONAL, ids=str)
def test_conditional(F: Expr, condition: str) -> None:
    found = meijerint_inversion(F, x, t)
    assert found is not None
    value, cond = found
    assert str(cond) == condition
    expected = _sympy_value(F)
    assert expected is not None and sympy.simplify(sympy_expr(to_sympy(value)) - expected) == 0
    assert _forward_error(value, F) < 1e-10


def test_delta() -> None:
    assert str(meijerint_inversion(exp(-a * x), x, t)) == '(DiracDelta(-a + t), True)'
    assert str(meijerint_inversion(3 * exp(-a * x - 1), x, t)) == '(3*exp(-1)*DiracDelta(-a + t), True)'


def test_not_found() -> None:
    # SymPy fails on these too: the conditions of the inversion theorems
    # are stricter than the existence of the transform
    for F in [1 / sqrt(x), 1 / sqrt(x + a), exp(-a / x) / sqrt(x), 1 / (x**2 + 2 * x + 2), log(1 + a / x)]:
        assert meijerint_inversion(F, x, t) is None
        assert _sympy_value(F) is None


# the transforms on which SymPy is wrong or finds nothing: the native value
# checked numerically
DIFFER: list[Expr] = [
    # SymPy records the shift of 2**(-x) and keeps the factor: no result
    2**(-x) / x,
    # SymPy finds nothing
    atan(1 / x),
    # SymPy's rewriting of 1/((x + a)*(x + b)) as G-functions is wrong by
    # 1/((a - b)*x): the Mellin-Barnes inversion of the sum put the pole at
    # s = 1 left of the contour for one term and right of it for the other
    1 / ((x + a) * (x + b)), 1 / ((x + 1) * (x + 2)), exp(-a * x) / ((x + 1) * (x + 2)),
    # SymPy finds nothing (or raises 'Gammas partially over the strip')
    x / ((x + a) * (x + b)), 1 / ((x + 1) * (x + 2) * (x + 3)),
]


@pytest.mark.parametrize('F', DIFFER, ids=str)
def test_where_sympy_fails(F: Expr) -> None:
    found = meijerint_inversion(F, x, t)
    assert found is not None
    value, cond = found
    assert cond == true
    assert _forward_error(value, F) < 1e-10
    expected = _sympy_value(F)
    if expected is not None:
        assert _forward_error_sympy(expected, sympy_expr(to_sympy(F))) > 1e-3


def test_explicit_values() -> None:
    assert str(meijerint_inversion(1 / ((x + 1) * (x + 2)), x, t)) == \
        '((-(t - t*exp(-t))/t + (2*t - 2*t*exp(-2*t))/(2*t))*Heaviside(t), True)'
    assert str(meijerint_inversion(2**(-x) / x, x, t)) == '(Heaviside(t - log(2)), True)'
    assert str(meijerint_inversion(atan(1 / x), x, t)) == '(sin(t)*Heaviside(t)/t, True)'
    # not analytic: None
    from sympy_extras.functions.special.delta_functions import heaviside
    assert meijerint_inversion(heaviside(x - 1) / x, x, t) is None
