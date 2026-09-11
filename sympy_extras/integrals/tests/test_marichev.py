"""Tests of the Marichev–Adamchik integrator over (0, oo), (0, 1) and (1, oo)."""
from __future__ import annotations

from sympy import (symbols, exp, sin, cos, log, sqrt, besselj, oo, gamma, pi, S, Rational,
                   EulerGamma, simplify, zeta, Integer, I, erf)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions import element
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.marichev import (mellin_integrate, evaluate_quotient, reduce_positive_powers,
                                             integrate_product)
from sympy_extras.integrals.mellin import GammaQuotient, decompose_integrand

x, k, s = symbols('x k s')
a, b, nu = symbols('a b nu', positive=True)


def _same(u: ExprLike, v: ExprLike) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0


def test_one_kernel_is_the_transform_at_alpha_plus_one() -> None:
    found = mellin_integrate(x**2 * exp(-3 * x), x)
    assert found == ConditionalValue(Rational(2, 27))
    found = mellin_integrate(x**k / (x + 3), x)
    assert found is not None
    assert found.value == -3**k * pi / sin(pi * k)
    assert found.condition == ((k > -1) & (k < 0))
    # decided by the assumptions
    found = mellin_integrate(x**k / (x + 3), x, (k > -1) & (k < 0))
    assert found is not None and found.condition is S.true
    # refuted by them: the integral diverges, the method says nothing
    assert mellin_integrate(x**k / (x + 3), x, k > 0) is None


def test_removable_singularity_of_the_quotient() -> None:
    # Gamma(s) sin(pi s/2) at s = 0 is a limit
    found = mellin_integrate(sin(x) / x, x)
    assert found == ConditionalValue(pi / 2)
    q = GammaQuotient(1, [], [(0, S.One)], [], 0, oo)
    assert evaluate_quotient(q, Integer(2)) == 1
    assert evaluate_quotient(q, S.Zero) is None


def test_two_kernels_through_the_g_function() -> None:
    found = mellin_integrate(exp(-a * x) * sin(b * x), x)
    assert found is not None and _same(found.value, b / (a**2 + b**2)) and found.condition is S.true
    found = mellin_integrate(exp(-x) * besselj(0, x), x)
    assert found is not None and _same(found.value, sqrt(2) / 2)
    # Gradshteyn-Ryzhik 6.611.1: Integral(exp(-a x) J_nu(b x), (x, 0, oo)) = b^-nu (sqrt(a^2 + b^2) - a)^nu / sqrt(a^2 + b^2)
    found = mellin_integrate(exp(-a * x) * besselj(nu, b * x), x)
    assert found is not None
    for values in ({a: 2, b: 3, nu: S.Half}, {a: Rational(1, 2), b: 1, nu: Rational(3, 2)}):
        expected = b**(-nu) * (sqrt(a**2 + b**2) - a)**nu / sqrt(a**2 + b**2)
        assert abs(complex((found.value - expected).evalf(20, subs=values))) < 1e-15


def test_logarithms_by_differentiation() -> None:
    found = mellin_integrate(exp(-x) * log(x), x)
    assert found is not None and _same(found.value, -EulerGamma)
    found = mellin_integrate(log(x)**2 / (1 + x**2), x)
    assert found is not None and _same(found.value, pi**3 / 8)
    # Gradshteyn-Ryzhik 4.272.6: Integral(x^(m) log^n x, (x, 0, 1)) = (-1)^n n! / (m + 1)^(n + 1)
    found = mellin_integrate(x**3 * log(x)**2, x, cutoff='lower')
    assert found == ConditionalValue(Rational(2, 64))


def test_finite_ranges_with_the_cutoffs() -> None:
    found = mellin_integrate(x**(b - 1) * (1 - x)**(a - 1), x, cutoff='lower')
    assert found is not None and _same(found.value, gamma(a) * gamma(b) / gamma(a + b))
    found = mellin_integrate(x**2 * exp(-x), x, cutoff='lower')
    assert found is not None and _same(found.value, 2 - 5 * exp(-1))
    found = mellin_integrate(x**2 * exp(-x), x, cutoff='upper')
    assert found is not None and _same(found.value, 5 * exp(-1))
    found = mellin_integrate((-log(x))**k, x, cutoff='lower')
    assert found is not None and found.value == gamma(k + 1) and found.condition == (k > -1)


def test_extra_factors_are_single_kernel_only() -> None:
    found = mellin_integrate(x / (exp(x) + 1), x)
    assert found is not None and _same(found.value, pi**2 / 12)
    found = mellin_integrate(x**2 / (exp(x) - 1), x)
    assert found is not None and _same(found.value, 2 * zeta(3))
    # 1/(e^x + 1) times another kernel is not a G-function
    assert mellin_integrate(sin(x) / (exp(x) + 1), x) is None


def test_differences_of_kernels() -> None:
    # the terms diverge separately: the difference kernels carry the continued strips
    found = mellin_integrate((1 - cos(x)) / x**2, x)
    assert found == ConditionalValue(pi / 2)
    found = mellin_integrate((1 - exp(-x)) * exp(-a * x) / x, x)
    assert found is not None and _same(found.value, log((a + 1) / a))


def test_real_scales_of_unknown_sign() -> None:
    # cos is even: a real b of unknown sign is allowed; the integrand
    # exp(-a x) cos(b x) has the value a/(a^2 + b^2) for every real b
    found = mellin_integrate(exp(-a * x) * cos(k * x), x, element(k, S.Reals))
    assert found is not None and _same(found.value, a / (a**2 + k**2)) and found.condition is S.true
    found = mellin_integrate(exp(-a * x) * sin(k * x), x, element(k, S.Reals))
    assert found is not None and _same(found.value, k / (a**2 + k**2))
    # a possibly complex scale of an even kernel keeps its condition
    found = mellin_integrate(exp(-a * x) * cos(k * x), x)
    assert found is not None and found.condition is not S.true


def test_reduce_positive_powers() -> None:
    assert reduce_positive_powers(sqrt(1 + x), x) == x / sqrt(x + 1) + 1 / sqrt(x + 1)
    assert reduce_positive_powers((1 + x)**Rational(3, 2), x).count(sqrt(x + 1)) == 0
    assert reduce_positive_powers(exp(-x), x) == exp(-x)
    # a positive power over a finite range
    found = mellin_integrate(sqrt(1 + x), x, cutoff='lower')
    assert found is not None and _same(found.value, Rational(2, 3) * (2 * sqrt(2) - 1))


def test_unrecognised_integrands() -> None:
    assert mellin_integrate(exp(x), x) is None
    assert mellin_integrate(sin(x + 1), x) is None
    # three kernels none of which is trigonometric: no exponential form
    # (two Bessel functions of one argument are one kernel, so erf here)
    assert mellin_integrate(exp(-x) * besselj(0, x) * erf(x), x) is None
    assert mellin_integrate(x**k, x) is None
    # the trigonometric factors of three kernels go through exponentials
    found = mellin_integrate(exp(-x) * sin(x) * cos(x), x)
    assert found is not None and found.value == Rational(1, 5)


def test_integrate_product_with_no_kernel() -> None:
    p = decompose_integrand(x**2, x)
    assert p is not None and integrate_product(p) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(mellin_integrate)([1], x))


def test_analytic_regularisation() -> None:
    # regularize=True drops the strip of convergence and keeps the
    # analytic continuation of the gamma quotient: the Hadamard finite
    # parts of divergent Mellin-type integrals
    assert mellin_integrate(x**Rational(-3, 2) * exp(-x), x) is None
    found = mellin_integrate(x**Rational(-3, 2) * exp(-x), x, regularize=True)
    assert found is not None and found.value == -2 * sqrt(pi)
    found = mellin_integrate(sqrt(x) / (x + 1), x, regularize=True)
    assert found is not None and found.value == -pi


def test_three_kernels_through_the_exponential_form() -> None:
    # a trigonometric factor next to two kernels is written as
    # exponentials with complex scales, two kernels per term; the
    # exponentials linear in x are combined, exp(-x**2) is kept apart
    from sympy import I, N, Integral
    from sympy_extras.integrals.marichev import exponential_form, right_half_plane_powers
    form = exponential_form(exp(-a * x) * sin(b * x) / (1 + x**2), x)
    assert form.count(exp) == 2 and form.count(x**2 + 1) == 2
    assert form.expand() == (exp(-a * x) * (exp(I * b * x) - exp(-I * b * x)) / (2 * I * (x**2 + 1))).expand()
    form = exponential_form(exp(-a * x) * cos(b * x) * exp(-x**2), x)
    assert form.count(exp) == 4 and all(isinstance(e, exp) for e in form.atoms(exp))
    # sqrt((a + I b)**2) is a + I b: the real part is positive
    assert right_half_plane_powers(sqrt((a + I * b)**2) / (a + I * b)) == 1
    assert right_half_plane_powers(sqrt((-a + I * b)**2)) == sqrt((-a + I * b)**2)
    found = mellin_integrate(exp(-a * x) * cos(b * x) * exp(-x**2), x)
    assert found is not None
    numeric = found.value.subs({a: 2, b: 1})
    assert abs(N(numeric) - N(Integral(exp(-2 * x) * cos(x) * exp(-x**2), (x, 0, oo)))) < 1e-10


def test_airy_products_and_polar_incomplete_gammas() -> None:
    # sympy-extras: Integral(airyai(x)**2, (x, 0, oo)) came out as zoo, the
    # Slater series at |z| = 1 summed by Gauss's formula term by term
    # although the terms diverge there; now refused
    from sympy import airyai, hyper, lowergamma, exp_polar, I, N, Integral
    from sympy_extras.integrals.marichev import polar_lowergamma
    assert mellin_integrate(airyai(x)**2, x) is None
    assert mellin_integrate(airyai(x), x) is not None
    # the Laplace transform of Ai: hyperexpand's lowergamma of a polar
    # argument rewritten as a real confluent series
    found = mellin_integrate(exp(-a * x) * airyai(x), x)
    assert found is not None and not found.value.has(exp_polar, lowergamma) and found.value.has(hyper)
    assert abs(N(found.value.subs(a, 2)) - N(Integral(exp(-2 * x) * airyai(x), (x, 0, oo)))) < 1e-10
    w = symbols('w', positive=True)
    rewritten = polar_lowergamma(lowergamma(Rational(1, 3), w * exp_polar(I * pi)))
    assert rewritten == 3 * S.NegativeOne**Rational(1, 3) * w**Rational(1, 3) * hyper((Rational(1, 3),), (Rational(4, 3),), w)


def test_bessel_products_through_the_driver() -> None:
    from sympy import besselk, elliptic_k, exp_polar, N, Integral
    from sympy_extras.integrals import definite_integral
    from sympy_extras.integrals.marichev import polar_elliptic
    assert definite_integral(besselj(1, x)**2 / x, (x, 0, oo)) == S.Half
    assert definite_integral(x * besselk(0, x)**2, (x, 0, oo)) == S.Half
    # the Laplace transform of J_0**2 is an elliptic integral of a negative
    # parameter, written without a polar number
    found = definite_integral(exp(-a * x) * besselj(0, x)**2, (x, 0, oo))
    assert found.has(elliptic_k) and not found.has(exp_polar)
    assert abs(N(found.subs(a, 2)) - N(Integral(exp(-2 * x) * besselj(0, x)**2, (x, 0, oo)))) < 1e-10
    assert polar_elliptic(elliptic_k(4 * exp_polar(I * pi) / a**2)) == elliptic_k(-4 / a**2)
