"""Tests of the definite integration driver."""
from __future__ import annotations

from sympy import (symbols, exp, sin, cos, log, sqrt, oo, pi, S, Rational, Abs, Heaviside, Piecewise, Integral, I, Eq,
                   sign, Max, Min, simplify, gamma, DiracDelta, erf, EulerGamma, atan, asin, besselj, Si, E, tan, arg, cosh,
                   sinh)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions import element
from sympy_extras.integrals import definite_integral, conditional_integral, verify_numerically, ConditionalValue
from sympy_extras.settings import configure

x, k, t = symbols('x k t')
a, b, s = symbols('a b s', positive=True)


def _same(u: ExprLike, v: ExprLike) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0


def test_trivial_ranges() -> None:
    assert definite_integral(exp(-x), (x, 2, 2)) == 0
    # swapped bounds change the sign
    assert definite_integral(exp(-x), (x, oo, 0)) == -1
    # a constant integrand
    assert definite_integral(a, (x, 1, 3)) == 2 * a
    assert definite_integral(3 * exp(-x), (x, 0, oo)) == 3


def test_conditions_are_reported_or_decided() -> None:
    value = definite_integral(x**k / (x + 3), (x, 0, oo))
    assert isinstance(value, Piecewise)
    first, condition = value.args[0].args
    assert first == -3**k * pi / sin(pi * k) and condition == ((k > -1) & (k < 0))
    assert value.args[1].args[0] == Integral(x**k / (x + 3), (x, 0, oo))
    assert definite_integral(x**k / (x + 3), (x, 0, oo), (k > -1) & (k < 0)) == -3**k * pi / sin(pi * k)
    assert definite_integral(x**k / (x + 3), (x, 0, oo), conds='none') == -3**k * pi / sin(pi * k)
    # refuted: the integral diverges, nothing is claimed
    value = definite_integral(x**k / (x + 3), (x, 0, oo), k > 1)
    assert isinstance(value, Integral)
    found = conditional_integral(x**k / (x + 3), x, S.Zero, oo)
    assert isinstance(found, ConditionalValue) and found.condition == ((k > -1) & (k < 0))
    raises(ValueError, lambda: definite_integral(exp(-x), (x, 0, oo), conds='maybe'))


def test_the_bug_sympy_drops_the_upper_condition() -> None:
    # SymPy's integrate gives Piecewise((-3**k*pi/sin(pi*k), k > -1), ...):
    # the antiderivative is evaluated at 0 with its condition and the
    # divergence at oo for k >= 0 is missed
    found = conditional_integral(x**k / (x + 3), x, S.Zero, oo)
    assert found is not None and found.condition.subs(k, S.Half) is S.false


def test_splitting_at_the_kinks() -> None:
    # Integral(|x - 1|/sqrt(x), (x, 0, 2)) = 4/3 + (4/3 - 2 sqrt(2)/3)
    assert _same(definite_integral(Abs(x - 1) / sqrt(x), (x, 0, 2)), Rational(8, 3) - 2 * sqrt(2) / 3)
    # -1/2 over (0, 1) and 4 over (1, 3)
    assert definite_integral(sign(x - 1) * x, (x, 0, 3)) == Rational(7, 2)
    assert definite_integral(Heaviside(x - 1) * exp(-x), (x, 0, oo)) == exp(-1)
    assert definite_integral(Max(x, 1), (x, 0, 2)) == Rational(5, 2)
    assert definite_integral(Min(x, 1), (x, 0, 2)) == Rational(3, 2)
    assert definite_integral(Piecewise((x, x < 1), (1, True)), (x, 0, 2)) == Rational(3, 2)
    # sqrt(u**2) is |u|
    assert definite_integral(sqrt((x - 1)**2), (x, 0, 2)) == 1
    # the kinks of a periodic function inside a finite range
    assert _same(definite_integral(sqrt(1 - cos(x)), (x, 0, 2 * pi)), 4 * sqrt(2))


def test_singularities_inside_the_range() -> None:
    # the bug (SymPy): integrate(1/(x*sqrt((x + 1)**2)), (x, -oo, -2)) is
    # unevaluated and the antiderivative approach ignores |x + 1|
    assert definite_integral(1 / (x * sqrt((x + 1)**2)), (x, -oo, -2)) == -log(2)
    assert definite_integral(1 / (x * sqrt((x + 1)**2)), (x, 1, oo)) == log(2)
    # a non-integrable singularity inside: nothing is claimed (SymPy's
    # integrate returns log(2) + I*pi)
    with configure(numerical_checks=True):
        value = definite_integral(1 / x, (x, -1, 2))
    assert value.has(Integral)
    # an integrable one
    assert definite_integral(1 / sqrt(Abs(x)), (x, -1, 1)) == 4


def test_ranges_are_mapped() -> None:
    assert _same(definite_integral(x**2 * exp(-x), (x, 1, oo)), 5 * exp(-1))
    assert _same(definite_integral(exp(-x**2), (x, 1, 3)), sqrt(pi) * (erf(3) - erf(1)) / 2)
    assert _same(definite_integral(exp(-Abs(x)), (x, -oo, oo)), 2)
    assert definite_integral(exp(-x), (x, -oo, 0)) == oo
    assert _same(definite_integral(exp(x), (x, -oo, 0)), 1)
    # x = log(u) for a function of exp(x) on the real line
    assert _same(definite_integral(exp(x / 4) / (9 * exp(x / 2) + 4), (x, -oo, oo)), pi / 3)
    value = definite_integral(x * exp(x) * exp(k * x) / (exp(x) + 3), (x, -oo, oo), (k > -1) & (k < 0))
    assert verify_numerically(value, x * exp(x) * exp(k * x) / (exp(x) + 3), x, -oo, oo, (k > -1) & (k < 0))
    # x = 1/u between (1, oo) and (0, 1)
    assert _same(definite_integral(log(1 + 1 / x**2), (x, 1, oo)), pi / 2 - log(2))


def test_trigonometric_beta_integrals() -> None:
    assert _same(definite_integral(sqrt(sin(x)), (x, 0, pi / 2)), 2 * sqrt(pi) * gamma(Rational(3, 4)) / gamma(Rational(1, 4)))
    assert _same(definite_integral(sin(x)**Rational(1, 3) * cos(x)**Rational(1, 3), (x, 0, pi / 2)),
                 gamma(Rational(2, 3))**2 / (2 * gamma(Rational(4, 3))))
    assert _same(definite_integral(1 / sqrt(sin(x) * cos(x)), (x, 0, pi / 2)), gamma(Rational(1, 4))**2 / (2 * sqrt(pi)))
    # a range of several quarter periods
    assert _same(definite_integral(sin(x)**2, (x, 0, 2 * pi)), pi)
    assert _same(definite_integral(Abs(cos(x)), (x, 0, pi)), 2)
    # log(sin x) over (0, pi/2) is a Beta integral differentiated
    assert _same(definite_integral(log(sin(x)), (x, 0, pi / 2)), -pi * log(2) / 2)


def test_logarithms_and_powers() -> None:
    assert _same(definite_integral(x**a * log(x), (x, 0, 1)), -1 / (a + 1)**2)
    assert _same(definite_integral(log(1 - x) / x, (x, 0, 1)), -pi**2 / 6)
    assert _same(definite_integral((-log(x))**k, (x, 0, 1), k > -1), gamma(k + 1))
    assert _same(definite_integral(x**Rational(1, 3) / sqrt(-log(x)), (x, 0, 1)), sqrt(3 * pi) / 2)
    assert _same(definite_integral(x**k * (1 - x)**k, (x, 0, 1), k > -1), gamma(k + 1)**2 / gamma(2 * k + 2))
    assert _same(definite_integral(exp(-x**2) * log(x), (x, 0, oo)), -sqrt(pi) * (EulerGamma + 2 * log(2)) / 4)


def test_parameters_with_assumptions() -> None:
    assert _same(definite_integral(exp(-s * x) * sin(a * x) / x, (x, 0, oo)), atan(a / s))
    assert _same(definite_integral(exp(-a * x) * cos(k * x), (x, 0, oo), element(k, S.Reals)), a / (a**2 + k**2))
    value = definite_integral(exp(-a * x) * cos(k * x), (x, 0, oo))
    assert isinstance(value, Piecewise)


def test_sympy_fallback_is_checked_numerically() -> None:
    # an integral no method here handles goes to SymPy...
    assert definite_integral(DiracDelta(x - 1) * exp(-x), (x, 0, oo)) == exp(-1)
    # ...whose answer is checked: integrate(1/x, (x, -1, 2)) = log(2) + I*pi
    # would be accepted without the check (the quadrature cannot settle
    # the principal value, but the singularity inside the range stops the
    # fallback anyway)
    with configure(numerical_checks=False):
        assert definite_integral(1 / x, (x, -1, 2)).has(Integral)


def test_verify_numerically() -> None:
    assert verify_numerically(S.One, exp(-x), x, S.Zero, oo) is True
    assert verify_numerically(S(2), exp(-x), x, S.Zero, oo) is False
    assert verify_numerically(1 / a, exp(-a * x), x, S.Zero, oo) is True
    assert verify_numerically(1 / a**2, exp(-a * x), x, S.Zero, oo) is False
    # an oscillating integrand: the plain rules are not trusted, the
    # oscillatory quadrature (period by period) is
    assert verify_numerically(pi / 2, sin(x) / x, x, S.Zero, oo) is True
    assert verify_numerically(pi / 2, sin(x) / x, x, S(2), oo) is False
    assert verify_numerically(pi / 2 - Si(2), sin(x) / x, x, S(2), oo) is True


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(definite_integral)([1], (x, 0, 1)))
    raises(TypeError, lambda: untyped(definite_integral)(exp(-x), (x, 0, [1])))


def test_complex_segments_go_to_sympy_and_are_checked() -> None:
    # a complex bound is the straight segment between the bounds, which
    # only SymPy's antiderivatives handle; the numerical check
    # parametrises the segment
    z = symbols('z')
    from sympy import I
    assert definite_integral(z**2, (z, -I, I)) == -2 * I / 3
    assert verify_numerically(-2 * I / 3, z**2, z, -I, I) is True
    # the bug: SymPy's integrate gives 1 + pi/2 + I + I*pi for the second
    # integral, off by 2*I*pi from the value 2.5708 - 2.1416*I of the
    # segment; the check rejects it and the integral is returned
    assert definite_integral(log(z) / z**2, (z, -I, -1)).has(Integral)


def test_leaked_constants_are_rejected() -> None:
    # the bug: the creative telescoping route returned 16*I*C1/3 for the
    # inner integral of the ball volume, the constant of integration of
    # dsolve unresolved; a value with symbols absent from the input is
    # not an answer and the next method is tried
    from sympy_extras.integrals import integrate_by_ranges
    y, z = symbols('y z')
    assert integrate_by_ranges(1, x**2 + y**2 + z**2 < 1) == 4 * pi / 3


def test_logarithms_of_negative_arguments_are_made_real() -> None:
    # the bug: SymPy's value I*c**2*(log(c**2) - log(-c**2)) of the area
    # of the disc of radius -c (c < 0) was kept as it was, and the region
    # integral then refined it to 0
    c = symbols('c')
    from sympy_extras.integrals.marichev import real_logarithms
    assert real_logarithms(I * c**2 * (log(c**2) - log(-c**2)), c < 0) == pi * c**2
    assert _same(definite_integral(2 * sqrt(c**2 - x**2), (x, c, -c), c < 0), pi * c**2)
    from sympy_extras.integrals import integrate_by_ranges
    y = symbols('y')
    assert _same(integrate_by_ranges(1, x**2 + y**2 < c**2, [x, y], c < 0), pi * c**2)


def test_unconfirmed_fallback_answers_are_dropped() -> None:
    # the bug: integrate(log(z**I)**2, (z, 0, 1)) gives -2 from
    # log(z**I) = I*log(z), which fails below z = exp(-pi) (the principal
    # branch); the quadrature could not confirm it and the answer was
    # kept. Mathematica's NIntegrate disagrees with -2.
    z = symbols('z')
    with configure(numerical_checks=True):
        value = definite_integral(log(z**I)**2, (z, 0, 1))
    assert value != -2


def test_trigonometric_products_hyperbolic_and_polynomials() -> None:
    from sympy import sinh, cosh, laguerre, atanh
    # products and powers of sin and cos are written as sums (TR8)
    assert _same(definite_integral(exp(-a * x) * sin(x) * cos(x), (x, 0, oo)), 1 / (a**2 + 4))
    assert _same(definite_integral(exp(-a * x) * sin(x)**2, (x, 0, oo)), 2 / (a * (a**2 + 4)))
    # hyperbolic functions as exponentials, with the condition a > b
    assert _same(definite_integral(exp(-a * x) * sinh(b * x), (x, 0, oo), a > b), b / (a**2 - b**2))
    assert _same(definite_integral(exp(-x) * cosh(x / 2), (x, 0, oo)), Rational(4, 3))
    # orthogonal polynomials expanded: Integral(exp(-x) L_2(x)) = 0 (orthogonality to L_0)
    assert definite_integral(exp(-x) * laguerre(2, x), (x, 0, oo)) == 0
    # inverse hyperbolic functions as logarithms, and the log(1 - x) kernel on (0, 1)
    assert _same(definite_integral(x * atanh(x), (x, 0, 1)), S.Half)
    assert definite_integral(log(1 - x), (x, 0, 1)) == -1


def test_principal_values() -> None:
    # Cauchy principal values from the antiderivative, the symmetric
    # limits at the pole taken as one limit
    assert definite_integral(1 / x, (x, -1, 2), principal_value=True) == log(2)
    assert definite_integral(1 / (x - 1), (x, 0, 3), principal_value=True) == log(2)
    assert _same(definite_integral(x / (x**2 - 1), (x, 0, 2), principal_value=True), log(3) / 2)
    # a double pole has no principal value
    # a non-integrable singularity of one sign has no principal value: the
    # integral diverges to oo, which is what is reported
    assert definite_integral(1 / x**2, (x, -1, 1), principal_value=True) == oo
    # without the flag nothing is claimed
    assert definite_integral(1 / x, (x, -1, 2)).has(Integral)


def test_symbolic_endpoints() -> None:
    # symbolic endpoints go through the antiderivative with the limits
    # under the assumptions
    assert _same(definite_integral(exp(-x), (x, a, b)), exp(-a) - exp(-b))
    assert _same(definite_integral(1 / x, (x, a, b), a < b), log(b / a))
    assert _same(definite_integral(cos(x), (x, a, 2 * a)), sin(2 * a) - sin(a))


def test_equalities_among_the_assumptions_are_substituted() -> None:
    # the bug: Integral(sin(m x) sin(n x), (x, 0, 2 pi)) under Eq(n, m) came
    # out as -pi*m/(2*n) from the generic antiderivative sin((m - n) x)/(m - n)
    m, n = symbols('m n', integer=True)
    value = definite_integral(sin(m * x) * sin(n * x), (x, 0, 2 * pi), Eq(n, m))
    assert value.subs(m, 3) == pi


def test_sympy_internal_failures_are_contained() -> None:
    # SymPy 1.14 raises inside these: an AssertionError of the LRA solver
    # (lra_theory.py, a term without symbols) reached through ask, an
    # AttributeError of the cache wrapper of meijerint (a lazy exception
    # message) through integrate, a ZeroDivisionError of mpmath through
    # evalf; each was a crash of the driver
    from sympy import Heaviside
    T, w, u, v, n = symbols('T w u v n')
    value = definite_integral(sin(x)**2 / (sin(x) + cos(x)), (x, 0, pi / 2))
    assert verify_numerically(value, sin(x)**2 / (sin(x) + cos(x)), x, S.Zero, pi / 2) is not False
    value = definite_integral(cos(T + w) / (cos(T) / 2 + 1)**2, (T, 0, 2 * pi))
    assert _same(value, -8 * sqrt(3) * pi * cos(w) / 9) or value.has(Integral)
    definite_integral((-a + u)**n * exp(-u * v) * Heaviside(-a + u, 0), (u, 0, oo))


def test_nested_complex_powers_keep_their_branches() -> None:
    # the bug: (z**(I/2))**I substituted with a positive variable becomes
    # z**(-1/2) in SymPy, and the integral over (0, 1) came out as 2; on
    # the principal branches the integrand is exp(-2 pi) z**(-1/2) below
    # z = exp(-2 pi), and Mathematica's NIntegrate gives 1.9963
    z = symbols('z')
    value = definite_integral((z**(I / 2))**I, (z, 0, 1))
    assert value != 2


def test_argument_conditions_of_complex_scales_are_decided() -> None:
    # sympy-extras: |arg z| <= 2 pi holds for every z, |arg w**2| < pi
    # unless w is imaginary, |arg w| < pi/2 when Re w > 0; a disjunction
    # of settled alternatives is settled
    from sympy import arg, And, Or, true
    from sympy_extras.integrals.conditions import decide, real_form
    w = a + I * b
    assert real_form(Abs(arg(w**2)) < 2 * pi, ()) is true
    assert real_form(Abs(arg(w**2)) <= pi, ()) is true
    assert real_form(Abs(arg(w**2)) < pi, ()) is true
    assert real_form(Abs(arg(w)) < pi / 2, ()) is true
    assert real_form(Abs(arg(-w)) < pi / 2, ()) is S.false
    assert decide(And(Or(Abs(arg(w**2)) <= 2 * pi, Abs(arg(w**2)) < 2 * pi), Abs(arg(w)) < pi / 2), ()) is true
    # the three-kernel products through the exponential form
    value = definite_integral(exp(-a * x) * sin(b * x) * besselj(0, x), (x, 0, oo))
    assert value == I / (2 * sqrt((a + I * b)**2 + 1)) - I / (2 * sqrt((a - I * b)**2 + 1))
    # a divergent integral has no value, whatever ran before
    assert definite_integral(x**Rational(-3, 2) * exp(-x), (x, 0, oo)) == oo
    assert definite_integral(x**Rational(-3, 2) * exp(-x), (x, 0, oo), regularize=True) == -2 * sqrt(pi)


def test_summability_and_unsigned_infinities() -> None:
    # the summability keyword routes to the Abel, Cesaro or Gaussian means;
    # a value zoo or nan from a method is refused
    from sympy import airyai, zoo
    assert definite_integral(sin(x), (x, 0, oo), summability='abel') == 1
    assert definite_integral(x * sin(x), (x, 0, oo), summability='cesaro') == 0
    assert definite_integral(sin(x)**2, (x, 0, oo), summability='abel') == Integral(sin(x)**2, (x, 0, oo))
    assert definite_integral(exp(-x), (x, 0, oo), summability='gaussian') == 1
    raises(ValueError, lambda: definite_integral(sin(x), (x, 0, oo), summability='borel'))
    value = definite_integral(airyai(x)**2, (x, 0, oo))
    assert value != zoo and not value.has(zoo)


def test_sums_are_integrated_term_by_term() -> None:
    # the antiderivative SymPy finds for the difference of the two arcs is
    # a complex Piecewise whose branches give nan at the endpoints, and no
    # other method takes a sum of two distinct radicals; each term is
    # elementary, and the slice of the union of two discs between their
    # intersection points is exactly this integrand
    from sympy_extras.integrals.definite import _Integrator
    g = sqrt(x) * sqrt(2 - x) - sqrt(1 - x) * sqrt(x + 1)
    assert _same(definite_integral(g, (x, S.Half, 1)), sqrt(3) / 4 - pi / 12)
    assert _same(definite_integral(sqrt(x) * sqrt(2 - x) + sqrt(1 - x) * sqrt(x + 1), (x, S.Half, 1)), pi / 4)
    # a product which expands to a few terms
    assert _same(definite_integral((sqrt(x) + sqrt(1 - x))**2, (x, 0, 1)), 1 + pi / 4)
    # terms which diverge separately and cancel in the sum give no
    # termwise value; the whole integrand still has one
    integrator = _Integrator(())
    assert integrator._termwise(1 / x - 1 / (x + x**2), x, S.Zero, S.One, 0) is None
    assert definite_integral(1 / x - 1 / (x + x**2), (x, 0, 1)) == log(2)


def test_sampling_under_irrational_bounds() -> None:
    # the bug: sample_values had no point for x > -sqrt(2)/2 (the solver
    # takes rational coefficients), verify_numerically reported nothing
    # and the driver rejected a correct value
    import random
    from sympy_extras.integrals.conditions import sample_values, _rational_relation
    values = sample_values([x], [x > -sqrt(2) / 2, x < 0], random.Random(0))
    assert values is not None and -sqrt(2) / 2 < values[x] < 0
    from sympy import Gt
    rewritten = _rational_relation(x > -sqrt(2) / 2)
    assert isinstance(rewritten, Gt) and rewritten.rhs.is_rational and rewritten.rhs > -sqrt(2) / 2
    assert _rational_relation(x < 1) == (x < 1)


def test_powers_and_logarithms_split_by_the_signs_of_their_factors() -> None:
    # (u v)**r = u**r v**r and log(u v) = log(u) + log(v) for u > 0 on the
    # piece: the square roots of squares (Maxima's test suite), the
    # trigonometric squares, the logarithm of a product
    assert _same(definite_integral((1 - cos(x))**Rational(3, 2), (x, 0, 2 * pi)), 16 * sqrt(2) / 3)
    cardioid = a**2 * (1 - cos(t))**2 * sqrt(a**2 * (1 - cos(t))**2 + a**2 * sin(t)**2)
    assert _same(definite_integral(cardioid, (t, 0, 2 * pi)), 256 * a**3 / 15)
    assert definite_integral(sqrt(tan(x)**2 + 1) * sin(x), (x, 0, pi / 3)) == log(2)
    assert definite_integral(sqrt(x**2 + 2 * x + 1) / x, (x, 1, E)) == E
    assert definite_integral(sqrt(x**2 + 2 * x + 1) / x, (x, -E, -1)) == 2 - E
    assert _same(definite_integral(sqrt(x - 2 + 1 / x), (x, 0, 2)), 2 * (4 - sqrt(2)) / 3)
    assert _same(definite_integral(sqrt(1 + (x - 474)**2 / (107669 - (x - 474)**2)), (x, 181, 474)),
                 sqrt(107669) * asin(293 / sqrt(107669)))
    assert _same(definite_integral(log(sin(x) / x), (x, 0, pi / 2)), pi * (1 - log(pi)) / 2)
    assert _same(definite_integral(log(x**2) / sqrt(1 - x**2), (x, -1, 1)), -2 * pi * log(2))
    assert _same(definite_integral((x - x**2)**k, (x, 0, 1), k > -1), gamma(k + 1)**2 / gamma(2 * k + 2))
    # a factor of unknown sign stays in the remainder: nothing is claimed
    from sympy_extras.integrals.definite import _Integrator
    integrator = _Integrator(None)
    piece = integrator._branch(sqrt((x - k) * x), x, S(2), S(3), frozenset(), True)
    assert piece == sqrt(x) * sqrt(x - k)
    piece = integrator._branch(log(x**2 * (x - 1)), x, S(0), S(1), frozenset(), True)
    assert piece == 2 * log(x) + log(1 - x) + log(-1)


def test_exponentials_are_combined() -> None:
    # the bug: exp(-a t) exp(-s t) went through Parseval's formula as two
    # kernels, with the condition a > 0 instead of a + s > 0
    assert definite_integral(exp(-k * t) * exp(-s * t), (t, 0, oo), k + s > 0) == 1 / (k + s)
    found = definite_integral(t**(a - 1) * exp(-k * t) * exp(-s * t), (t, 0, oo), k + s > 0)
    assert found == gamma(a) / (k + s)**a
    assert definite_integral((exp(4 * x) - exp(-4 * x))**2, (x, 0, 1)) == sinh(8) / 4 - 2


def test_oscillatory_tails_are_verified() -> None:
    # sin(x)/x over (2, oo): the antiderivative is Si, and the numerical
    # check needs the oscillatory quadrature
    assert definite_integral(sin(x) / x, (x, 2, oo)) == pi / 2 - Si(2)
    # the series route summed a wrong formal power series of SymPy for
    # 1/(x**2 + x + 1) to log(4) - 1 (HOL-Py's euler_log_sin06)
    assert _same(definite_integral((1 - x) / (x**2 + x + 1), (x, 0, 1)), sqrt(3) * pi / 6 - log(3) / 2)


def test_the_antiderivative_route_takes_the_verified_methods() -> None:
    # 1/(cosh(n t)**2 + 1): the Risch port gives up and integrate answers a
    # Piecewise in tanh; the trigonometric integrator has a logarithmic
    # form, which checks under n > 0 (HOL-Py's Chapter3Practice)
    n = symbols('n')
    value = definite_integral(1 / (cosh(n * t)**2 + 1), (t, 0, 1), n > 0)
    assert not value.has(Integral) and verify_numerically(value, 1 / (cosh(n * t)**2 + 1), t, S.Zero, S.One, n > 0)
    # the common factor of the denominator comes out before the constants:
    # log(t + 1)/(a**2 t**2 + a**2) is log(t + 1)/(t**2 + 1) over a**2
    # (HOL-Py's trick2e)
    assert _same(definite_integral(log(a + t) / (a**2 + t**2), (t, 0, a)), pi * (2 * log(a) + log(2)) / (8 * a))


def test_periodic_laplace_rule_with_a_symbolic_frequency() -> None:
    # the period 2 pi/Abs(k) of Abs(sin(k t)) under k > 0, and the condition
    # Abs(arg(s/k + I)) < pi/2 of the pieces decided by the real part s/k
    from sympy import coth
    from sympy_extras.integrals.conditions import decide
    assert decide((Abs(arg(I + s / k)) < pi / 2) & (Abs(arg(-I + s / k)) < pi / 2), [s > 0, k > 0]) is S.true
    value = definite_integral(exp(-s * t) * Abs(sin(k * t)), (t, 0, oo), k > 0)
    assert _same(value, k * coth(pi * s / (2 * k)) / (k**2 + s**2))


def test_mellin_conditions_on_combined_exponentials_and_two_scales() -> None:
    # the expanded form kept exp(-(a + s)*t) whole (Mathematica:
    # s/(a + s)**(3/2) under a + s > 0, no condition on a)
    from sympy import acosh, besselk
    from sympy_extras.integrals.conditions import decide
    value = definite_integral((1 - 2 * k * t) * exp(-k * t) * exp(-s * t) / (sqrt(pi) * sqrt(t)), (t, 0, oo), k + s > 0)
    assert _same(value, s / (k + s)**Rational(3, 2))
    # an alternative which is a conjunction is settled (the conditions of
    # Parseval's formula on two scales)
    condition = ((Abs(arg(k)) <= pi / 2) & (Abs(arg(s)) < pi / 2)) | ((Abs(arg(s)) <= pi / 2) & (Abs(arg(k)) < pi / 2))
    assert decide(condition, [k > 0, s > k]) is S.true
    value = definite_integral(exp(-s * t) * besselk(0, k * t), (t, 0, oo), [k > 0, s > k])
    assert not value.has(Piecewise)
    assert verify_numerically(value, exp(-s * t) * besselk(0, k * t), t, S.Zero, oo, [k > 0, s > k]) is True
    assert _same(value.subs({k: 1, s: 2}), acosh(2) / sqrt(3))



def test_sums_of_sines_as_products_and_radical_factors_of_logarithms() -> None:
    # (sin(19x) + sin(20x))/(cos(19x) + cos(20x)) is tan(39x/2) by the
    # sum-to-product formulas (HOL-Py's MIT 2019; Mathematica agrees with
    # the recorded value)
    f = (sin(19 * x) + sin(20 * x)) / (cos(19 * x) + cos(20 * x))
    assert _same(definite_integral(f, (x, 0, pi / 100)), -2 * log(cos(39 * pi / 200)) / 39)
    # log(1/cos(x)) is -log(cos(x)) on (0, pi/2), and -log(sqrt(1 - u)) after
    # the Beta substitution is -log(1 - u)/2: the radical factor of the
    # argument keeps its exponent (HOL-Py's LogFunction02; Mathematica: pi**2/24)
    assert _same(definite_integral(log(1 / cos(x)) * cos(x) / sin(x), (x, 0, pi / 2)), pi**2 / 24)



def test_divergence_to_a_signed_infinity() -> None:
    # an infinite one-sided limit of the antiderivative at an endpoint or
    # a singularity, of one sign: the integral diverges to it (Maxima's
    # rtestint answers "divergent" for these)
    assert definite_integral(1 / x, (x, 0, 1)) == oo
    assert definite_integral(-1 / x, (x, 0, 1)) == -oo
    assert definite_integral(x**-2, (x, -1, 1)) == oo
    assert definite_integral(x**-2, (x, 0, oo)) == oo
    assert definite_integral(exp(x), (x, 0, oo)) == oo
    assert definite_integral(1 / (x + 1), (x, 0, oo)) == oo
    assert definite_integral(x / (x**2 + 1), (x, 0, oo)) == oo
    # SymPy's limit of the antiderivative of x*Shi(x) at oo is -oo where the
    # function grows like x*exp(x)/4: the sign of the integrand near the
    # point refuses the claim (the census: Reduce's defint 28 came out -oo)
    from sympy import Shi
    assert definite_integral(x * Shi(x), (x, 0, oo)) != -oo
    # infinities of both signs, and an oscillation: unevaluated
    assert definite_integral(1 / x, (x, -1, 1)).has(Integral)
    assert definite_integral(cos(x), (x, 0, oo)).has(Integral)
    assert definite_integral(sin(x) + cos(x), (x, 0, oo)).has(Integral)



def test_sums_whose_terms_diverge_separately_are_regularised() -> None:
    # Maxima's specint 45: each term is t**(-3/2) at 0, the difference is
    # not; the Riesz regularisation of each Mellin transform gives
    # gamma(-1/2) times a power, and the sum is checked numerically
    f = (exp(-b * t) - exp(-a * t)) * exp(-s * t) / (2 * sqrt(pi) * sqrt(t**3))
    assert _same(definite_integral(f, (t, 0, oo)), sqrt(a + s) - sqrt(b + s))
    # a sum which diverges as a whole is not regularised (the check fails):
    # it diverges to oo, which is what is reported
    assert definite_integral((exp(-a * t) + exp(-b * t)) / t**2, (t, 0, oo)) == oo
    # specint 18: the three exponentials, each combined with exp(-s*t) in
    # the expanded form, under a + s > 0 and the like only
    c = symbols('c')
    f = ((-a + c) * exp(-b * t) + (a - b) * exp(-c * t) + (b - c) * exp(-a * t)) * exp(-s * t) / ((-a + b) * (-a + c) * (b - c))
    value = definite_integral(f, (t, 0, oo), c + s > 0)
    assert not value.has(Piecewise) and _same(value, 1 / ((a + s) * (b + s) * (c + s)))



def test_laplace_transforms_of_laguerre_polynomials_of_symbolic_degree() -> None:
    # Maxima's specint 42: laguerre(n, t) is exp(t)*1F1(n + 1; 1; -t) by
    # Kummer's transformation, and the confluent kernel of the Mellin table
    # gives (s - 1)**n/s**(n + 1) for s > 1 (Mathematica agrees)
    from sympy import laguerre
    n = symbols('n', positive=True)
    value = definite_integral(exp(-s * t) * laguerre(n, t), (t, 0, oo), s > 1)
    assert _same(value.subs({n: 3, s: 5}), Rational(4**3, 5**4))
    assert _same(value.subs({n: Rational(5, 2), s: 3}), (2**Rational(5, 2)) / 3**Rational(7, 2))



def test_powers_of_log_of_one_minus_x_are_derivatives_of_the_beta_exponent() -> None:
    # Maxima's rtestint 206 (Mathematica: (12135541 - 200 pi**2 (3739 + 30 pi**2)
    # - 3384000 zeta(3))/16200000), and the simplest cases
    from sympy import zeta
    assert _same(definite_integral(t * (1 - t) * log(1 - t), (t, 0, 1)), -Rational(5, 36))
    assert _same(definite_integral(log(1 - t)**2, (t, 0, 1)), 2)
    assert _same(definite_integral(log(t) * log(1 - t), (t, 0, 1)), 2 - pi**2 / 6)
    value = definite_integral(t**2 * (1 - t)**2 * log(t)**2 * log(1 - t)**2, (t, 0, 1))
    assert _same(value, (12135541 - 200 * pi**2 * (3739 + 30 * pi**2) - 3384000 * zeta(3)) / 16200000)
