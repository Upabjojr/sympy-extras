"""The summability means checked against the numerical regularised
integrals: the Abel and Gaussian means by mpmath's quadrature of the
regularised integrand at eps = 1/100 and 1/1000, the Cesàro means by the
kernel (1 - x/R)**k at R = 200 and 2000 (a partial-integral mean converges
like 1/R, the exponential means like eps or eps**2)."""
from __future__ import annotations

from typing import Callable

import mpmath
from sympy import symbols, sin, cos, exp, besselj, oo, pi, I, S, Expr, lambdify, Rational

from sympy_extras._typing import as_expr
from sympy_extras.integrals.summability import summable_integral, oscillatory_tail

x = symbols('x')


def _regularised(f: Expr, kernel: Callable[[float, mpmath.mpf], mpmath.mpf], parameter: float,
                 upper: float) -> complex:
    """``Integral(f * kernel(parameter, x), (x, 0, upper))`` numerically."""
    g = lambdify(x, f, 'mpmath')
    value = mpmath.quadosc(lambda t: g(t) * kernel(parameter, t), [0, mpmath.inf], period=2 * mpmath.pi) \
        if upper == float('inf') else mpmath.quad(lambda t: g(t) * kernel(parameter, t), mpmath.linspace(0, upper, 64))
    return complex(value)


def _check(f: Expr, method: str, expected: Expr, tolerance: float = 2e-2) -> None:
    found = summable_integral(f, x, 0, oo, method)
    assert found is not None and found.condition is S.true
    assert found.value == expected
    target = complex(as_expr(expected))
    if method == 'abel':
        numeric = _regularised(f, lambda e, t: mpmath.exp(-e * t), 1e-3, float('inf'))
        coarse = _regularised(f, lambda e, t: mpmath.exp(-e * t), 1e-2, float('inf'))
    elif method == 'gaussian':
        numeric = _regularised(f, lambda e, t: mpmath.exp(-e * t**2), 1e-3, float('inf'))
        coarse = _regularised(f, lambda e, t: mpmath.exp(-e * t**2), 1e-2, float('inf'))
    else:
        numeric = _regularised(f, lambda R, t: (1 - t / R)**2, 2000.0, 2000.0)
        coarse = _regularised(f, lambda R, t: (1 - t / R)**2, 200.0, 200.0)
    assert abs(numeric - target) < tolerance, (f, method, numeric, target)
    # the finer regularisation is the closer one
    assert abs(numeric - target) <= abs(coarse - target) + 1e-6


def test_abel_values() -> None:
    _check(sin(x), 'abel', as_expr(1))
    _check(cos(x), 'abel', as_expr(0))
    _check(x * sin(x), 'abel', as_expr(0))
    _check(x * cos(x), 'abel', as_expr(-1))
    _check(sin(2 * x), 'abel', Rational(1, 2))
    _check(exp(I * x), 'abel', I)
    _check(besselj(0, x), 'abel', as_expr(1))


def test_cesaro_values() -> None:
    _check(sin(x), 'cesaro', as_expr(1))
    _check(cos(x), 'cesaro', as_expr(0))
    _check(x * sin(x), 'cesaro', as_expr(0))
    _check(x * cos(x), 'cesaro', as_expr(-1))


def test_gaussian_values() -> None:
    _check(sin(x), 'gaussian', as_expr(1))
    _check(cos(x), 'gaussian', as_expr(0))
    _check(x * sin(x), 'gaussian', as_expr(0))


def test_tail_from_pi() -> None:
    # Integral(sin(x), (x, pi, oo)) has the Abel mean -1: the tail from
    # pi is the reflected tail from 0
    found = oscillatory_tail(sin(x), x, pi, 'abel')
    assert found is not None and found.value == -1
    numeric = _regularised(sin(x + pi), lambda e, t: mpmath.exp(-e * t), 1e-3, float('inf'))
    assert abs(numeric + 1) < 2e-2
