from __future__ import annotations

import mpmath

from sympy import zeta, pi, harmonic, polygamma, N, S, Integral, Symbol, Float, EulerGamma, lambdify
from sympy.core.expr import Expr
from sympy.abc import n

from sympy_extras.concrete.eulersums import euler_sum, polygamma_series, polygamma_integral_representation


def _euler_sum_numeric(p: int, q: int) -> Float:
    """The Euler sum by mpmath, with the harmonic numbers written through
    the Hurwitz zeta function so that the tail is extrapolated exactly."""
    mpmath.mp.dps = 20

    def term(k: mpmath.mpf) -> mpmath.mpf:
        h = mpmath.psi(0, k + 1) + mpmath.euler if p == 1 else mpmath.zeta(p) - mpmath.zeta(p, k + 1)
        return h/k**q
    return Float(mpmath.nsum(term, [1, mpmath.inf], method='euler-maclaurin'), 18)


def _close(value: Expr, number: Float) -> bool:
    return abs(N(value, 30) - number) < Float(10)**-14


def test_euler_sum_known() -> None:
    # Euler's evaluations (Flajolet-Salvy, table 1)
    assert euler_sum(1, 2) == 2*zeta(3)
    assert euler_sum(1, 3) == pi**4/72
    assert euler_sum(1, 4) == 3*zeta(5) - pi**2*zeta(3)/6
    assert euler_sum(2, 2) == 7*pi**4/360
    assert euler_sum(2, 3) == pi**2*zeta(3)/2 - 9*zeta(5)/2
    assert euler_sum(2, 4) == zeta(3)**2 - pi**6/2835
    assert euler_sum(3, 3) == (zeta(3)**2 + pi**6/945)/2
    assert euler_sum(2, 6) is None and euler_sum(1, 1) is None and euler_sum(0, 2) is None


def test_euler_sum_numeric() -> None:
    for p in range(1, 5):
        for q in range(2, 6):
            value = euler_sum(p, q)
            if value is None:
                assert (p + q) % 2 == 0 and p != q and (p, q) not in ((2, 4), (4, 2))
                continue
            assert _close(value, _euler_sum_numeric(p, q)), (p, q, value)


def _harmonic_numeric(k: mpmath.mpf, p: int = 1) -> mpmath.mpf:
    if p == 1:
        return mpmath.psi(0, k + 1) + mpmath.euler
    return mpmath.zeta(p) - mpmath.zeta(p, k + 1)


def _series_numeric(term: Expr, lower: int) -> Float:
    """The series by mpmath (Euler-Maclaurin summation), with the harmonic
    numbers through the digamma and Hurwitz zeta functions."""
    mpmath.mp.dps = 20
    f = lambdify(n, term, modules=[{'harmonic': _harmonic_numeric, 'polygamma': mpmath.psi}, 'mpmath'])
    return Float(mpmath.nsum(f, [lower, mpmath.inf], method='euler-maclaurin'), 18)


def test_polygamma_series() -> None:
    assert polygamma_series(harmonic(n)/n**2, n) == 2*zeta(3)
    assert polygamma_series(polygamma(0, n)/n**2, n) == zeta(3) - EulerGamma*pi**2/6
    assert polygamma_series(polygamma(1, n)/n**2, n) == 7*pi**4/360
    assert polygamma_series(harmonic(n - 1)/n**2, n) == zeta(3)
    assert polygamma_series(harmonic(n)/n**2, n, 2) == 2*zeta(3) - 1
    for term, lower in [(polygamma(0, n + 2)/n**3, 1), (harmonic(n + 1, 2)/n**3, 1), (polygamma(2, n + 1)/n**2, 1),
                        (3*harmonic(n, 3)/n**4, 1), (polygamma(1, n + 3)/n**3, 1), (polygamma(1, n - 1)/n**2, 2),
                        (harmonic(n - 2, 2)/n**4, 3), (polygamma(0, n)/n**4, 1)]:
        value = polygamma_series(term, n, lower)
        assert value is not None, term
        assert _close(value, _series_numeric(term, lower)), term
    assert polygamma_series(harmonic(n)**2/n**2, n) is None
    assert polygamma_series(harmonic(n)/n, n) is None
    assert polygamma_series(harmonic(n, 2)/n**6, n) is None
    assert polygamma_series(polygamma(1, n - 1)/n**2, n) is None


def test_integral_representation() -> None:
    s = Symbol('s')
    result = polygamma_integral_representation(polygamma(0, n)/n**s, n)
    assert result is not None and result.has(Integral) and result.has(EulerGamma)
    # at s = 2 the integral equals the closed form
    value = result.subs(s, 2)
    integral = [a for a in value.atoms(Integral)][0]
    numeric = integral.evalf(20)
    expected = N(zeta(3) - EulerGamma*pi**2/6 + EulerGamma*pi**2/6, 20)
    assert abs(numeric - expected) < 1e-12
    assert polygamma_integral_representation(polygamma(1, n)/n**s, n) is None
    assert S.true
