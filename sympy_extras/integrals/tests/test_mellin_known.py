"""The Mellin transforms of the table against independent sources: the
values Mathematica's ``MellinTransform`` gives at sample points (and its
closed forms), SymPy's ``mellin_transform`` where it has one, and
numerical quadrature."""
from __future__ import annotations

from typing import Callable

import mpmath

from sympy import (Integer, Symbol, symbols, exp, sin, cos, log, sqrt, besselj, oo, gamma, pi,
                   Rational, S, atan, erf, erfc, Ei, expint, mellin_transform as sympy_mellin,
                   simplify, lambdify, sinh, cosh)

from sympy_extras.integrals import mellin as M
from sympy_extras.integrals.mellin import KERNELS, mellin_transform

x, s = symbols('x s')
a = Symbol('a', positive=True)


def _value(name: str, point: Rational) -> float:
    return float(KERNELS[name].quotient.as_expr(point).evalf(20))


def test_values_mathematica() -> None:
    # MellinTransform[...] evaluated to 20 digits by Mathematica 14
    third = Rational(1, 3)
    assert abs(float(M._bessely_kernel(third).quotient.as_expr(S.Half).evalf(20)) - (-5.3190781252801530603)) < 1e-15
    assert abs(_value('cos - 1', Rational(-1, 2)) - (-2.5066282746310005024)) < 1e-15
    assert abs(_value('cos - 1', Rational(-3, 2)) - (-1.6710855164206670016)) < 1e-15
    assert abs(_value('sin - x', Rational(-2)) - (-0.78539816339744830962)) < 1e-15
    assert abs(_value('Si', -S.Half) - 5.0132565492620010048) < 1e-15
    # the bug: after the poles were moved to the right sides of the strips
    # the constants of cos - 1, sin - x and Si were off by a factor 2
    assert abs(_value('atan', -S.Half) - 4.4428829381583662470) < 1e-15
    assert abs(_value('exp - 1', -S.Half) - (-3.5449077018110320546)) < 1e-15
    assert abs(_value('log(1 + x)', -S.Half) - 6.2831853071795864769) < 1e-15
    assert abs(_value('theta(x - 1)', Rational(-7, 10)) - 1.4285714285714285714) < 1e-15
    assert abs(float(M._exp_besseli_kernel(third).quotient.as_expr(Rational(3, 10)).evalf(20))
               - 3.0372021556683050603) < 1e-15
    assert abs(float(M._besselk_kernel(third).quotient.as_expr(S.Half).evalf(20)) - 8.6499275138179498683) < 1e-15
    assert abs(float(M._expint_kernel(S.Half).quotient.as_expr(Rational(7, 10)).evalf(20))
               - 6.4902766632377889284) < 1e-15


def test_closed_forms_mathematica() -> None:
    # MellinTransform[BesselY[1/3, x], x, s] ==
    #   -2^(s-1) Cos[Pi/6 - Pi s/2] Gamma[s/2 - 1/6] Gamma[s/2 + 1/6] / Pi
    third = Rational(1, 3)
    ours = M._bessely_kernel(third).quotient.as_expr(s)
    theirs = -2**(s - 1) * cos(pi / 6 - pi * s / 2) * gamma(s / 2 - Rational(1, 6)) * gamma(s / 2 + Rational(1, 6)) / pi
    assert abs(complex((ours - theirs).evalf(20, subs={s: 0.7}))) < 1e-15
    # MellinTransform[Exp[-x] BesselI[1/3, x], x, s] == Gamma[1/2 - s] Gamma[1/3 + s] / (2^s Sqrt[Pi] Gamma[4/3 - s])
    ours = M._exp_besseli_kernel(third).quotient.as_expr(s)
    theirs = gamma(S.Half - s) * gamma(third + s) / (2**s * sqrt(pi) * gamma(Rational(4, 3) - s))
    assert simplify(ours - theirs) == 0
    # MellinTransform[Cos[x] - 1, x, s] == Cos[Pi s / 2] Gamma[s]
    ours = KERNELS['cos - 1'].quotient.as_expr(s)
    assert abs(complex((ours - cos(pi * s / 2) * gamma(s)).evalf(20, subs={s: -0.7}))) < 1e-15


def test_against_sympy_mellin_transform() -> None:
    # SymPy's own transforms, where it has them (checked here, not copied)
    for f, name in [(exp(-x), 'exp'), (log(1 + x), 'log(1 + x)'), (sin(x), 'sin'), (cos(x), 'cos'),
                    (erfc(x), 'erfc')]:
        theirs, strip, _ = sympy_mellin(f, x, s)
        ours = KERNELS[name].quotient
        assert (ours.lower, ours.upper) == strip, name
        point = Rational(1, 3) if strip[0] < Rational(1, 3) < strip[1] else (strip[0] + strip[1]) / 2 \
            if strip[1] != oo else strip[0] + 1
        assert abs(complex((ours.as_expr(point) - theirs.subs(s, point)).evalf(20))) < 1e-15, name
    theirs, strip, _ = sympy_mellin(besselj(a, x), x, s)
    ours = M._besselj_kernel(a).quotient
    assert simplify(ours.as_expr(s) - theirs) == 0
    theirs, strip, _ = sympy_mellin(1 / (1 + x)**a, x, s)
    assert simplify(M._power_kernel(a).quotient.as_expr(s) - theirs) == 0


def test_against_quadrature() -> None:
    mpmath.mp.dps = 20
    cases = [(exp(-x) - 1, 'exp - 1', -0.5), (erf(x), 'erf', -0.5), (-Ei(-x), 'E1', 0.7),
             (1 / (exp(x) - 1), '1/(exp(x) - 1)', 1.5), (1 / (exp(x) + 1), '1/(exp(x) + 1)', 0.5),
             (1 / sinh(x), '1/sinh(x)', 1.5), (1 / cosh(x), '1/cosh(x)', 0.5), (atan(x) - pi / 2, 'atan - pi/2', 0.5)]
    for f, name, point in cases:
        g = lambdify(x, f, 'mpmath')
        approx = mpmath.quad(lambda t: t**(point - 1) * g(t), [0, 1, 10, mpmath.inf])
        exact = KERNELS[name].quotient.as_expr(Rational(point)).evalf(20)
        assert abs(complex(exact) - complex(approx)) < 1e-8 * (1 + abs(complex(approx))), name
    # the kernels with parameters and the cutoffs
    parametric = [((1 + x)**(-Rational(7, 3)), M._power_kernel(Rational(7, 3)), 2.0, (0, mpmath.inf)),
                  ((1 - x)**Rational(1, 2), M._beta_kernel(Rational(3, 2)), 0.5, (0, 1)),
                  ((x - 1)**(-Rational(2, 3)), M._beta_upper_kernel(Rational(1, 3)), -0.5, (1, mpmath.inf)),
                  (expint(2, x), M._expint_kernel(Integer(2)), 0.5, (0, mpmath.inf)),
                  ((-log(x))**Rational(1, 2), M._log_power_lower_kernel(Rational(1, 2)), 0.7, (0, 1)),
                  (S.One, M._theta_lower_kernel(), 0.7, (0, 1)), (S.One, M._theta_upper_kernel(), -0.7, (1, mpmath.inf))]
    for f, kernel, point, (lo, hi) in parametric:
        g = lambdify(x, f, 'mpmath')
        approx = mpmath.quad(lambda t: t**(point - 1) * g(t), [lo, hi] if hi != mpmath.inf else [lo, lo + 1, lo + 10, hi])
        exact = kernel.quotient.as_expr(Rational(point)).evalf(20)
        assert abs(complex(exact) - complex(approx)) < 1e-6 * (1 + abs(complex(approx))), kernel.name


def test_scaled_argument() -> None:
    # M[f(b x^g)](s) = b^(-s/g) F(s/g) / g, checked on exp(-2 x^3) against quadrature
    found = mellin_transform(exp(-2 * x**3), x, s)
    assert found is not None
    approx = mpmath.quad(lambda t: t**0.5 * mpmath.exp(-2 * t**3), [0, 1, mpmath.inf])
    assert abs(float(found.transform.subs(s, Rational(3, 2)).evalf(20)) - float(approx)) < 1e-12
    # a power of x is a shift of s
    found = mellin_transform(x**2 * exp(-x), x, s)
    assert found is not None and found.transform == gamma(s + 2) and found.strip == (-2, oo)


def test_poles_lie_on_the_right_sides_of_the_strips() -> None:
    # the bug: erf's 1/s was written Gamma(s)/Gamma(1 + s), whose pole at 0
    # lies to the right of the strip (-1, 0); the G-function built from
    # the product with exp(-v x) then missed the residue at 0 and
    # Integral(exp(-v*x)*erf(x), (x, 0, oo)) came out with a spurious -1/v
    from sympy_extras.integrals.mellin import poles_separated
    for name, kernel in KERNELS.items():
        assert poles_separated(kernel.quotient) is True, name
    third = Rational(1, 3)
    for kernel in [M._power_kernel(third), M._beta_kernel(third), M._beta_upper_kernel(third),
                   M._expint_kernel(Integer(2)), M._besselj_kernel(third), M._bessely_kernel(third),
                   M._besselk_kernel(third), M._exp_besseli_kernel(third), M._log_power_lower_kernel(third),
                   M._polylog_kernel(Integer(1)), M._polylog_kernel(Integer(3))]:
        assert poles_separated(kernel.quotient) is True, kernel.name


def test_airy_polylog_fresnel_and_erfc_kernels() -> None:
    # Mathematica's MellinTransform (Ai, Li_2, S, C, erfc(x) exp(x^2)) and
    # quadrature: the transform of Li_n(-x) is negative on its strip for
    # every n (the bug: the first drafts had a sign alternating with n,
    # from the (-s)^n of the closed form read as s^n)
    mpmath.mp.dps = 20

    def check(kernel: M.Kernel, f: Callable[[mpmath.mpf], mpmath.mpf], point: Rational, tolerance: float,
              cuts: list[object]) -> None:
        exact = float(kernel.quotient.as_expr(point).evalf(20))
        approx = mpmath.quad(lambda t: t**(float(point) - 1) * f(t), cuts)
        assert abs(exact - float(approx)) < tolerance, (kernel.name, point, exact, approx)

    check(M._airyai_kernel(), mpmath.airyai, Rational(17, 10), 1e-12, [0, 1, mpmath.inf])
    assert abs(_value('airyai', Rational(3, 10)) - 1.0703613413320555124) < 1e-12
    check(M._polylog_kernel(Integer(1)), lambda t: -mpmath.log(1 + t), -S.Half, 1e-9, [0, 1, mpmath.inf])
    check(M._polylog_kernel(Integer(2)), lambda t: mpmath.polylog(2, -t), -S.Half, 1e-8, [0, 1, mpmath.inf])
    check(M._polylog_kernel(Integer(3)), lambda t: mpmath.polylog(3, -t), -S.Half, 1e-6, [0, 1, mpmath.inf])
    assert simplify(M._polylog_kernel(Integer(2)).quotient.as_expr(s) - pi / (s**2 * sin(pi * s))) == 0
    cuts = mpmath.linspace(0, 60, 121) + [mpmath.inf]
    check(M._fresnels_kernel(), mpmath.fresnels, Rational(-7, 10), 1e-4, cuts)
    check(M._fresnelc_kernel(), mpmath.fresnelc, Rational(-7, 10), 1e-4, cuts)
    # erfc(t) exp(t^2) ~ (1 - 1/(2 t^2)) / (sqrt(pi) t): the tail beyond 30 in closed form
    point = Rational(3, 10)
    head = mpmath.quad(lambda t: t**(float(point) - 1) * mpmath.exp(t**2 + mpmath.log(mpmath.erfc(t))), [0, 1, 10, 30])
    tail = (30**(float(point) - 1) / (1 - float(point)) - 30**(float(point) - 3) / (2 * (3 - float(point)))) / mpmath.sqrt(mpmath.pi)
    assert abs(float(M._erfc_exp_kernel().quotient.as_expr(point).evalf(20)) - float(head + tail)) < 1e-6


def test_log_one_minus_kernel() -> None:
    # Erdelyi, Tables of Integral Transforms I, 6.6 (7): the Mellin transform of
    # log(1 - x) on (0, 1) is -(psi(1 + s) + EulerGamma)/s
    q = M._log_one_minus_kernel().quotient
    approx = mpmath.quad(lambda t: t**0.5 * mpmath.log(1 - t), [0, 1])
    assert abs(float(q.as_expr(Rational(3, 2)).evalf(20)) - float(approx)) < 1e-12
    from sympy_extras.integrals.mellin import decompose_integrand
    p = decompose_integrand(log(1 - x**2), x, 'lower')
    assert p is not None and p.matches[0].kernel.name.startswith('log(1 - x)') and p.matches[0].gamma == 2
