"""Tests of the native Mellin transforms (:mod:`sympy_extras.integrals.mellin`)
on the objects of the core: the representation, the matching, the table
against quadrature (mpmath, an independent source) and the agreement with
the SymPy face; the integrators test the face."""
from __future__ import annotations

from typing import Callable

import mpmath
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.assumptions.facts import element
from sympy_extras.core import Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.functions import (Abs, Ei, arg, Heaviside, airyai, atan, besseli, besselj, besselk, bessely, cos, cosh,
                                    erf, erfc, exp, expint, gamma, hyper, log, polygamma, polylog, sin,
                                    sinh)
from sympy_extras.integrals import mellin as M
from sympy_extras.integrals.mellin import (KERNELS, GammaQuotient, decompose_integrand, mellin_kernel,
                                           mellin_transform, monomial, poles_separated)
from sympy_extras.sets import Reals

x, s, a, b, k = Symbol('x'), Symbol('s'), Symbol('a'), Symbol('b'), Symbol('k')
nu = Symbol('nu', positive=True)


def _value(e: Expr) -> complex:
    found = numeric_value(e, 80)
    assert found is not None, e
    return complex(found)


def test_gamma_quotient_operations() -> None:
    q = GammaQuotient(1, [(2, -1)], [(0, 1)], [], 0, oo)
    assert q.as_expr(s) == gamma(s) / 2**s
    assert q.strip == (0, oo) and q.strip_condition(a) == (a > 0) and q.is_gamma_quotient()
    # M(1 + 2 s): 0 < 1 + 2 s < 3; a negative scale reflects the strip
    q = GammaQuotient(1, [(2, -1)], [(0, S.One)], [], 0, 3)
    assert q.compose(1, 2).as_expr(s) == 2**(-2 * s) * gamma(2 * s + 1) / 2
    assert q.compose(1, 2).strip == (-S.Half, 1) and q.compose(1, -2).strip == (-1, S.Half)
    raises(ValueError, lambda: q.compose(0, 0))
    raises(ValueError, lambda: q.compose(0, a))
    p = GammaQuotient(1, [], [(0, 1)], [], 0, oo).times(GammaQuotient(1, [], [(1, -1)], [], -oo, 1))
    assert p.as_expr(s) == gamma(s) * gamma(1 - s) and p.strip == (0, 1)
    assert GammaQuotient(1, [], [(0, 1), (1, 1)], [(1, 1)], 0, oo).cancelled().as_expr(s) == gamma(s)
    assert GammaQuotient(3, [], [], [], 0, a).scaled(b).constant == 3 * b
    # symbolic ends of the strips meet in Max and Min
    r = GammaQuotient(1, [], [], [], a, 1).times(GammaQuotient(1, [], [], [], b, k))
    assert str(r.strip) == '(Max(a, b), Min(1, k))'


def test_poles_separated() -> None:
    assert poles_separated(GammaQuotient(1, [], [(0, 1)], [(1, 1)], 0, 1)) is True
    assert poles_separated(GammaQuotient(1, [], [(0, 1)], [(1, 1)], -1, 0)) is False
    assert poles_separated(GammaQuotient(1, [], [(a, 1)], [], 0, 1)) is None
    for name, kernel in KERNELS.items():
        assert poles_separated(kernel.quotient) is True, name


def test_monomial() -> None:
    assert monomial(3 * a * sqrt(x), x) == (3 * a, S.Half)
    assert monomial(x, x) == (1, 1)
    assert monomial(-x**2 / 2, x) == (-S.Half, 2)
    assert monomial(-a * x - 2 * b * x, x) == (-a - 2 * b, 1)
    assert monomial(x + 1, x) is None and monomial(x**a, x) is None


def test_kernel_matching() -> None:
    found = mellin_kernel(exp(-3 * x**2), x)
    assert found is not None and found.kernel is KERNELS['exp'] and (found.beta, found.gamma) == (3, 2)
    found = mellin_kernel(1 / (x + 2)**a, x)
    assert found is not None and found.beta == S.Half and found.constant == 2**(-a)
    found = mellin_kernel(besselj(Integer(0), sqrt(x)), x)
    assert found is not None and found.gamma == S.Half and found.kernel.name == 'besselj'
    for f, name in [(1 / (exp(2 * x) + 1), '1/(exp(x) + 1)'), (1 / sinh(x), '1/sinh(x)'), (1 / cosh(x), '1/cosh(x)'),
                    (Heaviside(1 - x), 'theta(1 - x)'), (Heaviside(x - 2), 'theta(x - 1)'),
                    (exp(-x) - 1, 'exp - 1'), (1 - cos(x), 'cos - 1'), (sin(x) - x, 'sin - x'),
                    (atan(x) - pi / 2, 'atan - pi/2')]:
        found = mellin_kernel(f, x)
        assert found is not None and found.kernel is KERNELS[name], f
    found = mellin_kernel(Ei(-x), x)
    assert found is not None and found.constant == -1
    found = mellin_kernel(expint(Integer(2), 3 * x), x)
    assert found is not None and found.kernel.parameters == (2,)
    found = mellin_kernel(polylog(Integer(2), -3 * x**2), x)
    assert found is not None and (found.beta, found.gamma) == (3, 2)
    found = mellin_kernel(hyper([a], [b], -x), x)
    assert found is not None and found.kernel.parameters == (a, b)
    for f in [exp(x), exp(-x**a), atan(x + 1), erf(x**2 + 1), polylog(S.Half, -x), polylog(Integer(2), 3 * x)]:
        assert mellin_kernel(f, x) is None, f


def test_decompose_integrand() -> None:
    p = decompose_integrand(3 * x**a * log(x)**2 * exp(-x) * sin(2 * x), x)
    assert p is not None and (p.constant, p.alpha, p.log_power) == (3, a, 2)
    assert [m.kernel.name for m in p.matches] == ['exp', 'sin']
    p = decompose_integrand(sin(x)**2, x)
    assert p is not None and len(p.matches) == 2
    # the products which are one kernel come after the other kernels
    p = decompose_integrand(exp(-x) * besselj(Integer(0), 2 * x) * besselj(Integer(1), 2 * x), x)
    assert p is not None and [m.kernel.name for m in p.matches] == ['exp', 'besselj*besselj']
    p = decompose_integrand(x * besselk(Integer(0), x)**2, x)
    assert p is not None and [m.kernel.name for m in p.matches] == ['besselk*besselk'] and p.alpha == 1
    p = decompose_integrand(exp(-3 * x) * besseli(nu, 2 * x), x)
    assert p is not None and [m.kernel.name for m in p.matches] == ['exp', 'exp(-x)*besseli']
    assert [m.beta for m in p.matches] == [1, 2]
    p = decompose_integrand(erfc(x) * exp(x**2 / 2) / x, x)
    assert p is not None and [m.kernel.name for m in p.matches] == ['exp', 'erfc(x)*exp(x**2)']
    p = decompose_integrand(besselj(Integer(0), x) * bessely(Integer(1), x), x)
    assert p is not None and len(p.matches) == 2
    assert decompose_integrand(exp(-x) * sin(x) * cos(x), x) is None
    assert decompose_integrand(exp(-x) * log(x + 2) / (x + 1), x) is None
    # the cutoffs of the finite ranges
    p = decompose_integrand((1 - x)**a * x**2, x, 'lower')
    assert p is not None and len(p.matches) == 1 and p.matches[0].cutoff and p.alpha == 2
    p = decompose_integrand(exp(-x), x, 'lower')
    assert p is not None and [m.kernel.name for m in p.matches] == ['exp', 'theta(1 - x)']
    p = decompose_integrand((x**2 - 1)**a, x, 'upper')
    assert p is not None and p.matches[0].kernel.name.startswith('(x - 1)') and p.matches[0].gamma == 2
    p = decompose_integrand(log(1 - x)**2 * x**a, x, 'lower')
    assert p is not None and p.one_minus_power == 2 and p.matches[0].kernel.name == M.BETA_LOWER
    assert decompose_integrand((1 - x)**a, x) is None


def test_scale_conditions() -> None:
    found = mellin_kernel(exp(-a * x), x)
    assert found is not None
    q = found.quotient()
    assert q.as_expr(s) == a**(-s) * gamma(s) and q.condition == (Abs(arg(a)) < pi / 2)
    # a real scale of unknown sign of an odd kernel: |a| and the sign
    found = mellin_kernel(sin(a * x), x)
    assert found is not None
    q = found.quotient(element(a, Reals))
    assert q.condition == S.true and q.constant == sqrt(pi) * a / (2 * Abs(a))
    assert q.bases[-1][0] == Abs(a)


def _quadrature(f: Callable[[mpmath.mpf], mpmath.mpf], point: float, cuts: list[object]) -> complex:
    return complex(mpmath.quad(lambda t: t**(point - 1) * f(t), cuts))


def test_transforms_against_quadrature() -> None:
    mpmath.mp.dps = 20
    cases: list[tuple[Expr, Callable[[mpmath.mpf], mpmath.mpf], Rational]] = [
        (exp(-2 * x**3), lambda t: mpmath.exp(-2 * t**3), Rational(3, 2)),
        (1 / cosh(x), lambda t: 1 / mpmath.cosh(t), S.Half),
        (1 / (1 + x)**Rational(7, 3), lambda t: (1 + t)**(-mpmath.mpf(7) / 3), S.Half),
        (exp(-x) * log(x)**2, lambda t: mpmath.exp(-t) * mpmath.log(t)**2, Rational(3, 2)),
        (besselk(Rational(1, 3), x)**2, lambda t: mpmath.besselk(mpmath.mpf(1) / 3, t)**2, Integer(2)),
        (erfc(x), mpmath.erfc, Rational(1, 2)),
        (1 / (exp(x) + 1), lambda t: 1 / (mpmath.exp(t) + 1), Rational(1, 2)),
        (airyai(x), mpmath.airyai, Rational(17, 10)),
        (exp(-x) - 1, lambda t: mpmath.exp(-t) - 1, Rational(-1, 2)),
    ]
    for f, g, point in cases:
        found = mellin_transform(f, x, s)
        assert found is not None, f
        exact = _value(found.transform.subs(s, point))
        approx = _quadrature(g, float(point), [0, 1, 10, mpmath.inf])
        assert abs(exact - approx) < 1e-7 * (1 + abs(approx)), (f, exact, approx)


def test_logarithm_powers_are_derivatives() -> None:
    # the second derivative of gamma(s) in SymPy's form: SymPy's Derivative
    # writes an order >= 2 as factor_terms(signsimp(...)), the core's does
    # not (the first native draft gave the expanded sum)
    found = mellin_transform(exp(-x) * log(x)**2, x, s)
    assert found is not None
    assert found.transform == (polygamma(0, s)**2 + polygamma(1, s)) * gamma(s)
    found = mellin_transform(exp(-x) * log(x), x, s)
    assert found is not None and found.transform == gamma(s) * polygamma(0, s)


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(GammaQuotient)(1, [], [(0, 1)], [], 0, oo, [1]))
    raises(TypeError, lambda: untyped(GammaQuotient)([1], [], [(0, 1)], [], 0, oo))
    raises(ValueError, lambda: untyped(GammaQuotient)(1, [], [(0, a)], [], 0, oo))


def test_face_agrees_with_the_native_module() -> None:
    import sympy
    from sympy_extras._numeric_sympy import sympy_expr
    from sympy_extras.integrals import mellin_sympy as F
    X, S_ = sympy.symbols('x s')
    for f, native in [(sympy.exp(-2 * X), exp(-2 * x)), (sympy.sin(X) / X, sin(x) / x),
                      (sympy.besselj(sympy.Symbol('nu', positive=True), X), besselj(nu, x)),
                      (sympy.exp(-X) * sympy.log(X)**2, exp(-x) * log(x)**2)]:
        face, ours = F.mellin_transform(f, X, S_), mellin_transform(native, x, s)
        assert face is not None and ours is not None
        assert face.transform == sympy_expr(ours.transform) and face.strip == tuple(sympy_expr(e) for e in ours.strip)
    assert F.KERNELS['exp'].quotient.to_core().as_expr(s) == KERNELS['exp'].quotient.as_expr(s)
    found = F.mellin_kernel(sympy.exp(-3 * X), X)
    assert found is not None and found.kernel is F.KERNELS['exp']
