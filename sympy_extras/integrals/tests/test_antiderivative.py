"""Tests of the antiderivative-with-limits method."""
from __future__ import annotations

from sympy import symbols, cos, pi, oo, sqrt, exp, log, atan, tan, S, integrate, floor, Rational

from sympy_extras.integrals.antiderivative import (antiderivative, discontinuities, one_sided_limit,
                                                   antiderivative_integral)
from sympy_extras.integrals.conditions import ConditionalValue

x = symbols('x')
a = symbols('a', positive=True)


def test_antiderivative_is_found_or_none() -> None:
    F = antiderivative(exp(-a * x), x)
    assert F is not None and F.diff(x).equals(exp(-a * x))
    # no elementary antiderivative
    assert antiderivative(exp(-x**2) * log(x + 2), x) is None


def test_discontinuities_of_the_antiderivative() -> None:
    # atan(tan(x/2)) jumps at x = pi
    F = 2 * atan(tan(x / 2))
    assert discontinuities(F, x, S.Zero, 2 * pi) == [pi]
    # a logarithm of a factor vanishing inside, and nothing for a continuous one
    assert discontinuities(log(x - 1), x, S.Zero, S(3)) == [1]
    assert discontinuities(atan(x), x, -oo, oo) == []


def test_one_sided_limits() -> None:
    F = 2 * atan(tan(x / 2))
    assert one_sided_limit(F, x, pi, '-') == pi
    assert one_sided_limit(F, x, pi, '+') == -pi
    assert one_sided_limit(atan(x), x, oo, '-') == pi / 2
    # an infinite limit of one sign is returned: the integral diverges to it
    assert one_sided_limit(-1 / x, x, S.Zero, '+') == -oo
    # an oscillation is None
    from sympy import sin
    assert one_sided_limit(sin(x), x, oo, '-') is None


def test_jumps_are_subtracted() -> None:
    # the bug (of the antiderivative route in general): F(2 pi) - F(0)
    # with F = 2 atan(tan(x/2))/sqrt(3)*... would be 0; the one-sided
    # limits at the jump give the correct value
    found = antiderivative_integral(1 / (2 + cos(x)), x, S.Zero, 2 * pi)
    assert found == ConditionalValue(2 * sqrt(3) * pi / 3)
    found = antiderivative_integral(1 / (5 - 4 * cos(x)), x, S.Zero, 2 * pi)
    assert found == ConditionalValue(2 * pi / 3)


def test_divergent_integrals_are_signed_infinities_or_none() -> None:
    # integrate(1/x**2, (x, -1, 1)) evaluated at the endpoints would give -2:
    # the one-sided limits at 0 are infinite, of one sign
    assert antiderivative_integral(1 / x**2, x, S.NegativeOne, S.One) == ConditionalValue(oo)
    assert antiderivative_integral(-1 / x**2, x, S.NegativeOne, S.One) == ConditionalValue(-oo)
    # infinities of both signs: nothing is claimed (the principal value is
    # another question)
    assert antiderivative_integral(1 / x, x, S.NegativeOne, S(2)) is None


def test_infinite_ranges_and_parameters() -> None:
    assert antiderivative_integral(1 / (1 + x**2), x, -oo, oo) == ConditionalValue(pi)
    found = antiderivative_integral(exp(-a * x), x, S.Zero, oo, a > 0)
    assert found == ConditionalValue(1 / a)
    # SymPy's own continuous antiderivative (with floor) is used as it is
    F = integrate(1 / (2 + cos(x)), x)
    assert F.has(floor)


def test_piecewise_antiderivatives_take_the_branch_of_the_range() -> None:
    # the bug: SymPy's antiderivative of sqrt(t)*sqrt(2*t*c - 2*c) is a
    # Piecewise on t > 1, and limit() of it at t = 1 gave 0 from the wrong
    # branch, so the integral came out as 0 (and with it the area of the
    # disc x**2 + y**2 < c**2 for c < 0); the branch is now selected first
    # and the infinite limit of the right one refuses the antiderivative
    t, c = symbols('t c')
    from sympy_extras.integrals.antiderivative import select_branch
    from sympy import Piecewise
    F = Piecewise((t**2, t > 1), (t**3, True))
    assert select_branch(F, t, S(2), S(3)) == t**2
    assert select_branch(F, t, S.Zero, S.One) == t**3
    assert select_branch(F, t, S.Zero, S(2)) is None
    found = antiderivative_integral(sqrt(t) * sqrt(2 * t * c - 2 * c), t, S.Zero, S.One, c < 0)
    assert found is None or found.value != 0


def test_hadamard_finite_parts() -> None:
    # Estrada-Kanwal ch. 2: the finite part drops the divergent terms of
    # the symmetric excision; a simple pole gives the principal value
    from sympy_extras.integrals.antiderivative import finite_part_integral
    assert finite_part_integral(1 / x**2, x, S.NegativeOne, S.One) == ConditionalValue(-2)
    assert finite_part_integral(1 / x, x, S.NegativeOne, S(2)) == ConditionalValue(log(2))
    # [-1/(2 (x - 1)**2)] between 0 and 3: -1/8 + 1/2
    assert finite_part_integral(1 / (x - 1)**3, x, S.Zero, S(3)) == ConditionalValue(S(3) / 8)
    # a convergent integral is unchanged
    assert finite_part_integral(exp(-a * x), x, S.Zero, oo, a > 0) == ConditionalValue(1 / a)


def test_limits_at_symbolic_points() -> None:
    # the bug: the leak guard of one_sided_limit refused every value with
    # a symbol of the point itself, and singularities() could not form an
    # interval with symbolic ends
    from sympy import asin, sqrt, symbols
    x, y = symbols('x y')
    F = y * sqrt(1 - y**2) + asin(y)
    assert one_sided_limit(F, y, x, '-', [x > -sqrt(2) / 2, x < 0]) == x * sqrt(1 - x**2) + asin(x)
    found = antiderivative_integral(2 * sqrt(1 - y**2), y, -sqrt(1 - x**2), x, [x > -sqrt(2) / 2, x < 0])
    assert found is not None and found.value.has(asin)


def test_polynomials_in_elementary_functions_go_to_integrate_first() -> None:
    # the Risch port spent seconds on the gcds over pi in the arguments of
    # sin(pi*t/4 + pi/4)**3, on every quarter period of a trigonometric
    # integral; integrate does these at once
    from sympy import sin, cos, sinh, diff, simplify
    from sympy_extras.integrals.antiderivative import _elementary_polynomial
    t = symbols('t')
    assert _elementary_polynomial(sin(pi * t / 4 + pi / 4)**3, t)
    assert _elementary_polynomial(t**2 * exp(2 * t) * cos(t) + sinh(3 * t - 1), t)
    assert not _elementary_polynomial(sin(t**2), t)
    assert not _elementary_polynomial(sin(t) / t, t)
    assert not _elementary_polynomial(log(t) * cos(t), t)
    F = antiderivative(sin(pi * t / 4 + pi / 4)**3, t)
    assert F is not None and simplify(diff(F, t) - sin(pi * t / 4 + pi / 4)**3) == 0


def test_infinite_limits_absorb_finite_terms() -> None:
    # sin(x)/x**3 over (0, 1): the limits give -Si(1)/2 + oo, which SymPy
    # keeps as a sum (the finiteness of Si(1) is not known to it)
    from sympy import sin
    assert antiderivative_integral(sin(x) / x**3, x, 0, 1) == ConditionalValue(oo)


def test_logarithms_of_real_arguments_are_taken_of_their_absolute_values() -> None:
    # sin(x)**2*cot(x - 1) over (0, 1): the antiderivative has
    # log(sin(x)/tan(1) - cos(x)), of a negative argument on the range,
    # whose infinite limit at 1 the complex values made zoo
    from sympy import sin, cot
    assert antiderivative_integral(sin(x)**2 * cot(x - 1), x, 0, 1) == ConditionalValue(-oo)
    assert antiderivative_integral(1 / x, x, -3, -1) == ConditionalValue(-log(3))


def test_a_logarithm_times_a_polynomial_exponential_by_parts() -> None:
    # u**3*exp(-u)*log(u): G*log(u) - Integral(G/u), the last an
    # exponential integral (integrate took six seconds)
    import time
    from sympy import Ei
    started = time.monotonic()
    F = antiderivative(x**3 * exp(-x) * log(x), x)
    assert F is not None and F.has(Ei) and time.monotonic() - started < 5
    assert abs((F.diff(x) - x**3 * exp(-x) * log(x)).subs(x, Rational(3, 2)).evalf(20)) < 1e-15


def test_the_error_function_times_a_polynomial_exponential_by_parts() -> None:
    # sinh(u - w)*erf(w) over (0, u) (Maxima's laplace 57): G*erf(w) minus
    # the Gaussian remainder, term by term with the exponentials combined
    from sympy import sinh, erf, Rational, Integral
    from sympy_extras.integrals import verify_numerically
    u, w = symbols('u w', positive=True)
    F = antiderivative(sinh(u - w) * erf(w), w)
    assert F is not None
    assert abs((F.diff(w) - sinh(u - w) * erf(w)).subs({u: Rational(13, 10), w: Rational(3, 7)}).evalf(20)) < 1e-15
    from sympy_extras.integrals import definite_integral
    value = definite_integral(sinh(u - w) * erf(w), (w, 0, u))
    assert not value.has(Integral) and verify_numerically(value, sinh(u - w) * erf(w), w, S.Zero, u) is True
