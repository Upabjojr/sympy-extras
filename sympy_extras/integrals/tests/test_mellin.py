"""Tests of the Mellin transform representation and the table matching."""
from __future__ import annotations

from sympy import (Symbol, symbols, exp, sin, cos, log, sqrt, besselj, besseli, oo, gamma,
                   Heaviside, S, atan, erf, Ei, expint, sinh, cosh)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.mellin import (GammaQuotient, mellin_kernel, mellin_transform, monomial,
                                           decompose_integrand, KERNELS)

x, s, a, b, k = symbols('x s a b k')
nu = Symbol('nu', positive=True)


def test_gamma_quotient_expression() -> None:
    q = GammaQuotient(1, [(2, -1)], [(0, 1)], [], 0, oo)
    assert q.as_expr(s) == gamma(s) / 2**s
    assert q.strip == (0, oo)
    assert q.strip_condition(a) == (a > 0)
    assert q.is_gamma_quotient()


def test_compose_shifts_and_scales_the_strip() -> None:
    q = GammaQuotient(1, [(2, -1)], [(0, S.One)], [], 0, 3)
    # M(1 + 2 s): 0 < 1 + 2 s < 3
    shifted = q.compose(1, 2)
    assert shifted.as_expr(s) == 2**(-2 * s) * gamma(2 * s + 1) / 2
    assert shifted.strip == (-S.Half, 1)
    # a negative scale reflects the strip: 0 < 1 - 2 s < 3
    reflected = q.compose(1, -2)
    assert reflected.strip == (-1, S.Half)
    raises(ValueError, lambda: q.compose(0, 0))
    raises(ValueError, lambda: q.compose(0, a))


def test_times_intersects_the_strips() -> None:
    p = GammaQuotient(1, [], [(0, 1)], [], 0, oo)
    q = GammaQuotient(1, [], [(1, -1)], [], -oo, 1)
    r = p.times(q)
    assert r.as_expr(s) == gamma(s) * gamma(1 - s)
    assert r.strip == (0, 1)


def test_cancelled_removes_common_factors() -> None:
    q = GammaQuotient(1, [], [(0, 1), (1, 1)], [(1, 1)], 0, oo)
    assert q.cancelled().as_expr(s) == gamma(s)


def test_monomial() -> None:
    assert monomial(3 * a * sqrt(x), x) == (3 * a, S.Half)
    assert monomial(x, x) == (1, 1)
    assert monomial(-x**2 / 2, x) == (-S.Half, 2)
    assert monomial(x + 1, x) is None
    assert monomial(x**a, x) is None


def test_kernel_matching() -> None:
    m = mellin_kernel(exp(-3 * x**2), x)
    assert m is not None and m.kernel is KERNELS['exp'] and (m.beta, m.gamma) == (3, 2)
    m = mellin_kernel(1 / (x + 2)**a, x)
    assert m is not None and m.beta == S.Half and m.constant == 2**(-a) and m.kernel.parameters == (a,)
    m = mellin_kernel(besselj(0, sqrt(x)), x)
    assert m is not None and m.gamma == S.Half
    m = mellin_kernel(1 / (exp(2 * x) + 1), x)
    assert m is not None and m.kernel is KERNELS['1/(exp(x) + 1)'] and m.beta == 2
    m = mellin_kernel(1 / sinh(x), x)
    assert m is not None and m.kernel is KERNELS['1/sinh(x)']
    m = mellin_kernel(1 / cosh(x), x)
    assert m is not None and m.kernel is KERNELS['1/cosh(x)']
    m = mellin_kernel(Ei(-x), x)
    assert m is not None and m.constant == -1
    m = mellin_kernel(expint(2, 3 * x), x)
    assert m is not None and m.kernel.parameters == (2,)
    m = mellin_kernel(log(1 + 2 * x), x)
    assert m is not None and m.beta == 2
    m = mellin_kernel(Heaviside(1 - x), x)
    assert m is not None and m.kernel is KERNELS['theta(1 - x)']
    m = mellin_kernel(Heaviside(x - 2), x)
    assert m is not None and m.kernel is KERNELS['theta(x - 1)'] and m.beta == S.Half
    # not kernels: a growing exponential, a power of x with a symbolic exponent
    assert mellin_kernel(exp(x), x) is None
    assert mellin_kernel(exp(-x**a), x) is None
    assert mellin_kernel(atan(x + 1), x) is None
    assert mellin_kernel(erf(x**2 + 1), x) is None


def test_cutoff_kernels() -> None:
    # (1 - x)**a on (0, 1) is the Beta kernel and carries the cutoff
    p = decompose_integrand((1 - x)**a * x**2, x, 'lower')
    assert p is not None and len(p.matches) == 1 and p.matches[0].cutoff and p.alpha == 2
    # without such a factor the step function is added
    p = decompose_integrand(exp(-x), x, 'lower')
    assert p is not None and [m.kernel.name for m in p.matches] == ['exp', 'theta(1 - x)']
    # (x**2 - 1)**a on (1, oo)
    p = decompose_integrand((x**2 - 1)**a, x, 'upper')
    assert p is not None and p.matches[0].kernel.name.startswith('(x - 1)') and p.matches[0].gamma == 2
    # (-log x)**k on (0, 1)
    p = decompose_integrand((-log(x))**k, x, 'lower')
    assert p is not None and p.matches[0].kernel.parameters == (k,)
    # (1 - x)**a alone over (0, oo) has no transform
    assert decompose_integrand((1 - x)**a, x) is None


def test_decompose_integrand() -> None:
    p = decompose_integrand(3 * x**a * log(x)**2 * exp(-x) * sin(2 * x), x)
    assert p is not None
    assert (p.constant, p.alpha, p.log_power) == (3, a, 2)
    assert [m.kernel.name for m in p.matches] == ['exp', 'sin']
    # a square of a function is two kernels
    p = decompose_integrand(besselj(nu, x)**2, x)
    assert p is not None and len(p.matches) == 2
    # three kernels are too many, an unknown factor is refused
    assert decompose_integrand(exp(-x) * sin(x) * cos(x), x) is None
    assert decompose_integrand(exp(-x) * log(x + 2) / (x + 1), x) is None
    # the bug: exp(-a x) besseli(nu, b x) with a > b is the kernel
    # exp(-b x) besseli(nu, b x) times exp(-(a - b) x)
    p = decompose_integrand(exp(-3 * x) * besseli(nu, 2 * x), x)
    assert p is not None and sorted(m.kernel.name for m in p.matches) == ['exp', 'exp(-x)*besseli']
    assert {m.beta for m in p.matches} == {1, 2}


def test_mellin_transform_lookup() -> None:
    found = mellin_transform(exp(-2 * x), x, s)
    assert found is not None and found.transform == gamma(s) / 2**s and found.strip == (0, oo)
    found = mellin_transform(1 / (1 + x)**a, x, s)
    assert found is not None and found.transform == gamma(s) * gamma(a - s) / gamma(a)
    assert found.strip == (0, a)
    found = mellin_transform(exp(-x**2) / sqrt(x), x, s)
    assert found is not None and found.strip == (S.Half, oo)
    # a product of two kernels is a G-function, not a quotient
    assert mellin_transform(besselj(0, 3 * x) / (x**2 + 4), x, s) is None
    assert mellin_transform(x**a * cos(x + 1), x, s) is None


def test_scale_condition_of_the_match() -> None:
    m = mellin_kernel(exp(-a * x), x)
    assert m is not None
    q = m.quotient()
    assert q.as_expr(s) == a**(-s) * gamma(s)
    assert q.condition.free_symbols == {a}


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(GammaQuotient)(1, [], [(0, 1)], [], 0, oo, [1]))
    raises(TypeError, lambda: untyped(GammaQuotient)([1], [], [(0, 1)], [], 0, oo))
    raises(ValueError, lambda: untyped(GammaQuotient)(1, [], [(0, a)], [], 0, oo))
