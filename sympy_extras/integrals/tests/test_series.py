"""Tests of the series expansion with termwise integration."""
from __future__ import annotations

from typing import Callable

from sympy import symbols, log, exp, oo, pi, S, Dummy, atan, Catalan, polygamma
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.series import expansions, moments, sum_series, series_integral
from sympy_extras.settings import configure

x = symbols('x')
p = symbols('p', positive=True)


def test_expansions_of_the_factors() -> None:
    found = list(expansions(log(1 - x) / x, x, S.Zero, S.One))
    assert [(e.factor, rest) for e, rest in found] == [(log(1 - x), 1 / x), (log(1 - x) / x, 1)]
    # the formula -1/j of log(1 - x) is infinite at j = 0: the series starts at 1
    assert found[0][0].start == 1
    # the geometric series of exponentials on (0, oo)
    found = list(expansions(x / (exp(x) - 1), x, S.Zero, oo))
    assert found and found[0][0].factor == 1 / (exp(x) - 1) and found[0][0].start == 1
    assert found[0][0].coefficient == 1
    found = list(expansions(x / (exp(x) + 1), x, S.Zero, oo))
    assert found and found[0][0].coefficient.subs(found[0][0].index, 2) == -1
    # log(x) has no power series about 0, a monomial is left to the rest
    assert [e.factor for e, _ in expansions(x**p * log(x), x, S.Zero, S.One)] == []


def test_moments_and_sums() -> None:
    found = list(expansions(log(1 - x) / x, x, S.Zero, S.One))
    expansion, rest = found[0]
    moment = moments(expansion, rest, x, S.Zero, S.One)
    assert moment is not None and moment.subs(expansion.index, 3) == S(1) / 3
    j = Dummy('j', integer=True, nonnegative=True)
    assert sum_series((-1)**j / (2 * j + 1)**2, j, 0) == Catalan
    assert sum_series(1 / (j + 1)**2, j, 0) == pi**2 / 6
    assert sum_series(1 / (j + p + 1)**2, j, 0) == polygamma(1, p + 1)
    # a divergent series has no sum
    assert sum_series(S(2)**j, j, 0) is None


def test_series_integral() -> None:
    assert series_integral(log(1 - x) / x, x, 0, 1) == ConditionalValue(-pi**2 / 6)
    assert series_integral(x / (exp(x) - 1), x, 0, oo) == ConditionalValue(pi**2 / 6)
    assert series_integral(atan(x) / x, x, 0, 1) == ConditionalValue(Catalan)
    # a chosen factor
    found = series_integral(log(x) / (1 + x), x, 0, 1, expand=1 / (1 + x))
    assert found == ConditionalValue(-pi**2 / 12)
    # (0, c) is mapped onto (0, 1)
    found = series_integral(log(1 - x / 2) / x, x, 0, 2)
    assert found == ConditionalValue(-pi**2 / 6)
    # other ranges are not handled
    assert series_integral(log(1 - x) / x, x, S.Half, 1) is None


def test_unjustified_interchange_is_refused() -> None:
    # the terms of exp(-x)*log(1 + x) integrate to (-1)**(j-1) (j-1)!, a
    # divergent series: no value
    assert series_integral(exp(-x) * log(1 + x), x, 0, oo) is None
    # without numerical checks an undecided interchange is refused too
    with configure(numerical_checks=False):
        value = series_integral(log(1 - x) / x, x, 0, 1)
    assert value is None or value == ConditionalValue(-pi**2 / 6)


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(series_integral)([1], x, 0, 1))


def test_fourier_expansion_table() -> None:
    from sympy import sin, cos, tan, Abs, sign, zeta
    from sympy_extras.integrals.series import fourier_expansion
    two_pi = 2 * pi
    e = fourier_expansion(log(2 * sin(x / 2)), x, 0, two_pi)
    assert e is not None
    k = e.index
    assert (e.coefficient, e.basis, e.start, e.constant) == (-1 / k, cos(k * x), 1, 0)
    # the mean of log(sin x) and a positive constant in the logarithm
    e = fourier_expansion(log(sin(x)), x, 0, pi)
    assert e is not None and e.constant == -log(2) and e.basis == cos(2 * e.index * x)
    e = fourier_expansion(log(3 * sin(x)), x, 0, pi)
    assert e is not None and e.constant == log(3) - log(2)
    # rescaled arguments and the range of validity
    e = fourier_expansion(log(cos(3 * x)), x, 0, pi / 6)
    assert e is not None and e.basis == cos(6 * e.index * x)
    assert fourier_expansion(log(sin(x)), x, 0, 2 * pi) is None
    assert fourier_expansion(log(cos(x)), x, 0, pi) is None
    assert fourier_expansion(log(tan(x / 2)), x, 0, pi) is not None
    # the sawtooth waves on the two ranges
    e = fourier_expansion(x, x, 0, two_pi)
    assert e is not None and e.constant == pi and e.basis == sin(e.index * x)
    e = fourier_expansion(x, x, -pi, pi)
    assert e is not None and e.constant == 0
    for g in (Abs(sin(x)), Abs(cos(x)), sign(sin(x)), sign(cos(x)), x**2, log(1 + cos(x)), log(1 - cos(x))):
        assert fourier_expansion(g, x, 0, pi) is not None
    # not in the table
    assert fourier_expansion(log(sin(x)) + 1, x, 0, pi) is None
    assert fourier_expansion(exp(x), x, 0, pi) is None
    assert fourier_expansion(log(sin(x) * cos(x)), x, 0, pi / 2) is None
    assert fourier_expansion(log(-sin(x)), x, 0, pi) is None
    assert zeta(2) == pi**2 / 6


def test_fourier_table_numerically() -> None:
    """Every tabulated series has the right mean and the right first
    coefficients: the function integrated against the harmonics."""
    import mpmath
    from sympy import lambdify, Symbol, sin, cos
    from sympy_extras.integrals.series import _FOURIER_TABLE, _harmonic_parts, _u, _k
    u = Symbol('u', real=True)
    for entry in _FOURIER_TABLE:
        lower, upper = entry.lower, entry.upper
        if lower == -oo:
            lower, upper = S.Zero, 2 * pi
        points = [mpmath.mpf(float(lower + (upper - lower) * i / 8)) for i in range(9)]
        evaluate = lambdify(u, entry.function.xreplace({_u: u}), 'mpmath')

        def function(v: mpmath.mpf, evaluate: Callable[..., object] = evaluate) -> mpmath.mpf:
            # the quadrature may hit the logarithmic singularity at an endpoint
            value = mpmath.re(evaluate(v))
            return value if mpmath.isfinite(value) else mpmath.mpf(0)

        parts = _harmonic_parts(entry.basis.xreplace({_u: u}), u)
        assert parts is not None
        cosine, sine, frequency = parts
        mean = mpmath.quad(function, points) / float(upper - lower)
        assert abs(mean - float(entry.constant)) < 1e-8, entry.function
        for m in (1, 2, 3):
            w = frequency.subs(_k, m)
            for amplitude, harmonic in ((cosine, cos(w * u)), (sine, sin(w * u))):
                expected = float(entry.coefficient.subs(_k, m) * amplitude.subs(_k, m))
                projection = lambdify(u, harmonic, 'mpmath')
                found = mpmath.quad(lambda v: function(v) * projection(v), points)
                norm = mpmath.quad(lambda v: projection(v)**2, points)
                assert abs(found / norm - expected) < 1e-8, (entry.function, m, harmonic)


def test_fourier_products_and_harmonics() -> None:
    from sympy import sin, cos, symbols, zeta, Abs
    from sympy_extras.integrals.series import fourier_integral
    n = symbols('n', integer=True, positive=True)
    # orthogonality reduces the double series to a single one
    found = fourier_integral(log(sin(x))**2, x, 0, pi)
    assert found is not None and found.value.expand() == pi * log(2)**2 + pi**3 / 12
    assert fourier_integral(x**2 * log(2 * sin(x / 2)), x, 0, 2 * pi) == ConditionalValue(-4 * pi * zeta(3))
    # a harmonic picks one coefficient
    assert fourier_integral(log(2 * sin(x / 2)) * cos(n * x), x, 0, 2 * pi) == ConditionalValue(-pi / n)
    assert fourier_integral(log(sin(x)) * cos(2 * x), x, 0, pi) == ConditionalValue(-pi / 2)
    assert fourier_integral(log(sin(x)) * cos(3 * x), x, 0, pi) == ConditionalValue(0)
    assert fourier_integral(Abs(sin(x)) * cos(2 * x), x, 0, 2 * pi) == ConditionalValue(-S(4) / 3)
    # a harmonic that is not orthogonal on the range goes through the moments
    assert fourier_integral(log(sin(x)) * sin(x), x, 0, pi) == ConditionalValue(2 * log(2) - 2)
    # a rest with a possibly resonant harmonic is refused rather than
    # integrated with the resonant term lost
    assert fourier_integral(log(sin(x)) * cos(x)**2, x, 0, pi) is None
    # not a Fourier integral
    assert fourier_integral(exp(-x) * log(1 + x), x, 0, 1) is None
    assert fourier_integral(log(sin(x)), x, pi, 0) is None


def test_sum_series_rational_and_clausen() -> None:
    import mpmath
    from sympy import sin, symbols, EulerGamma, I, polylog
    k = Dummy('k', integer=True, positive=True)
    t = symbols('t', positive=True)
    # summation returns nan for this one (the partial fractions diverge separately)
    assert sum_series(2 / (k * (4 * k**2 - 1)), k, 1) == 4 * log(2) - 2
    found = sum_series(1 / (k * (4 * k**2 + 1)), k, 1)
    assert found is not None and found.has(EulerGamma) and abs(found.n() - 0.248329307672074) < 1e-12
    assert sum_series(1 / (k * (4 * k**2 + 1)), k, 3) is not None
    # the Clausen function
    found = sum_series(sin(k * t) / k**2, k, 1)
    assert found is not None and found.has(polylog, I)
    assert abs(found.subs(t, 1).n() - mpmath.clsin(2, 1)) < 1e-12


def test_fourier_through_series_integral() -> None:
    from sympy import sin
    assert series_integral(log(sin(x)), x, 0, pi) == ConditionalValue(-pi * log(2))
    assert series_integral(x * log(sin(x)), x, 0, pi, expand=log(sin(x))) == ConditionalValue(-pi**2 * log(2) / 2)
    assert series_integral(x * log(sin(x)), x, 0, pi, expand=x) == ConditionalValue(-pi**2 * log(2) / 2)
    from sympy_extras.integrals.series import fourier_integral
    assert fourier_integral(x * log(sin(x)), x, 0, pi, expand=exp(x)) is None


def test_parametric_and_computed_fourier_expansions() -> None:
    from sympy import sin, cos, symbols, Rational
    from sympy_extras.integrals.series import fourier_expansion, _computed_expansion
    a = symbols('a', positive=True)
    # the Poisson kernel and its logarithm, GR 1.447-1.448, with the root
    # inside the unit circle; numeric P, Q give the root directly
    e = fourier_expansion(1 / (1 - 2 * a * cos(x) + a**2), x, 0, pi, a < 1)
    assert e is not None and e.basis == cos(e.index * x) and e.condition is S.true
    assert (e.coefficient * (1 - a**2) / a**e.index).simplify() == 2
    assert (e.constant * (1 - a**2)).simplify() == 1
    e = fourier_expansion(log(1 - 2 * a * cos(x) + a**2), x, -pi, pi, a < 1)
    assert e is not None and e.coefficient == -2 * a**e.index / e.index and e.constant == 0
    # without the assumption the condition P > |Q| is carried along
    e = fourier_expansion(1 / (1 - 2 * a * cos(x) + a**2), x, 0, pi)
    assert e is not None and e.condition is not S.true
    e = fourier_expansion(1 / (5 + 3 * cos(x)), x, 0, 2 * pi)
    assert e is not None and e.coefficient == Rational(-1, 3)**e.index / 2 and e.constant == Rational(1, 4)
    assert fourier_expansion(1 / (3 + 5 * cos(x)), x, 0, pi) is None
    # computed coefficients: the half-range cosine series of x**2 on (0, pi)
    e = fourier_expansion(x**2, x, 0, pi, compute=True)
    assert e is not None and e.basis == cos(e.index * x) and e.constant == pi**2 / 3
    assert e.coefficient == 4 * (-1)**e.index / e.index**2
    # the full series on (-pi, pi) is the same as the table's
    e = _computed_expansion(x, x, -pi, pi, None)
    assert e is not None and e.basis == sin(e.index * x) and (e.coefficient + 2 * (-1)**e.index / e.index) == 0
    # a factor with a harmonic would resonate with the basis: refused
    assert _computed_expansion(x * sin(x), x, S.Zero, pi, None) is None
    assert _computed_expansion(x**2, x, S.Zero, oo, None) is None


def test_computed_fourier_integrals() -> None:
    from sympy import sin, cos, symbols, polylog, zeta
    from sympy_extras.integrals.series import fourier_integral
    a = symbols('a', positive=True)
    n = symbols('n', integer=True, positive=True)
    # a polynomial against a harmonic, GR 2.633
    assert fourier_integral(x**2 * cos(n * x), x, -pi, pi) == ConditionalValue(4 * (-1)**n * pi / n**2)
    # the Poisson kernel: GR 3.613.2 and the series 1.447
    found = fourier_integral(cos(n * x) / (1 - 2 * a * cos(x) + a**2), x, 0, pi, a < 1)
    assert found is not None and (found.value - pi * a**n / (1 - a**2)).simplify() == 0
    found = fourier_integral(x**2 / (1 - 2 * a * cos(x) + a**2), x, 0, pi, a < 1)
    assert found is not None and (found.value - pi * (pi**2 / 3 + 4 * polylog(2, -a)) / (1 - a**2)).simplify() == 0
    # log(1 - 2 a cos x + a**2): GR 4.224.14-15, both sides of |a| = 1
    assert fourier_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi, a < 1) == ConditionalValue(0)
    assert fourier_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi, a > 1) == ConditionalValue(2 * pi * log(a))
    found = fourier_integral(x * log(1 - 2 * a * cos(x) + a**2), x, 0, pi, a < 1)
    assert found is not None and found.value.expand() == 2 * polylog(3, a) - 2 * polylog(3, -a)
    # without the assumption on a, the condition of the expansion is reported
    found = fourier_integral(cos(n * x) / (1 - 2 * a * cos(x) + a**2), x, 0, pi)
    assert found is None or found.condition is not S.true
    # tabulated and computed factors paired: (pi - x)**2 against log(sin(x))
    assert fourier_integral((pi - x)**2 * log(sin(x)), x, 0, pi) == ConditionalValue(-pi**3 * log(2) / 3 - pi * zeta(3) / 2)
    # a factor with a harmonic next to two others is refused (resonance)
    assert fourier_integral(x * sin(x) * log(sin(x)), x, 0, pi) is None
