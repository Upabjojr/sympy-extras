"""Tests of the antiderivative-with-limits method on the objects of the
core (:mod:`sympy_extras.integrals.antiderivative`): the endpoint,
singularity and limit logic with a native antiderivative (the native
``ratint``), and the default step through the face. The values are hand
computations: ``log`` and ``atan`` of the rational integrals, Hadamard's
finite parts (Estrada–Kanwal ch. 2)."""
from __future__ import annotations

from typing import Optional

from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core import I, Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.functions import Piecewise, atan, cos, exp, log, sin, tan
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.integrals.antiderivative import (antiderivative, antiderivative_integral, by_parts_shape,
                                                   discontinuities, elementary_polynomial, finite_part_integral,
                                                   one_sided_limit, principal_value_integral, quick_shape,
                                                   real_logarithms, select_branch, singularities)
from sympy_extras.integrals.conditions import ConditionalValue, numerically_equal
from sympy_extras.integrals.risch.rationaltools import ratint

x, t = Symbol('x'), Symbol('t')
a = Symbol('a', positive=True)


def rational(f: Expr, x: Symbol, assumptions: Assumptions, late: bool) -> Optional[Expr]:
    """A native antiderivative step: the rational integrator."""
    if f.is_rational_function(x) is not True:
        return None
    return ratint(f, x)


def test_native_step_runs_the_whole_method() -> None:
    assert antiderivative_integral(1 / (1 + x**2), x, -oo, oo, step=rational) == ConditionalValue(pi)
    found = antiderivative_integral(1 / (x**2 - 4), x, S.Zero, S.One, step=rational)
    assert found is not None and numerically_equal(found.value, -log(3) / 4)
    # divergences of one sign are reported, of both signs refused
    assert antiderivative_integral(1 / x**2, x, -S.One, S.One, step=rational) == ConditionalValue(oo)
    assert antiderivative_integral(-1 / x**2, x, -S.One, S.One, step=rational) == ConditionalValue(-oo)
    assert antiderivative_integral(1 / x, x, -S.One, Integer(2), step=rational) is None
    # log(Abs(x)) on a negative range: real limits
    assert antiderivative_integral(1 / x, x, Integer(-3), -S.One, step=rational) == ConditionalValue(-log(3))
    # a parameter in the bounds and in the integrand
    found = antiderivative_integral(1 / (x**2 + a**2), x, S.Zero, oo, a > 0, step=rational)
    assert found is not None and numerically_equal(found.value, pi / (2 * a), a > 0)
    # no antiderivative from the step: None
    assert antiderivative_integral(sin(x), x, S.Zero, pi, step=rational) is None


def test_principal_values_and_finite_parts_with_a_native_step() -> None:
    assert principal_value_integral(1 / x, x, -S.One, Integer(2), step=rational) == ConditionalValue(log(2))
    assert principal_value_integral(1 / x**2, x, -S.One, S.One, step=rational) is None
    assert finite_part_integral(1 / x**2, x, -S.One, S.One, step=rational) == ConditionalValue(Integer(-2))
    # [-1/(2 (x - 1)**2)] between 0 and 3: -1/8 + 1/2
    assert finite_part_integral(1 / (x - 1)**3, x, S.Zero, Integer(3), step=rational) \
        == ConditionalValue(Rational(3, 8))


def test_singularities_and_discontinuities() -> None:
    assert singularities(1 / (x**2 - 1), x) == [-1, 1]
    assert singularities(1 / (x**2 - 1), x, [x > 0]) == [1]
    assert singularities(log(x + 2) + sqrt(x), x) == [-2]
    assert discontinuities(log(x - 1), x, S.Zero, Integer(3)) == [1]
    assert discontinuities(atan(x), x, -oo, oo) == []
    # atan of a fraction jumps at the pole of its argument
    assert discontinuities(atan(1 / (x - 2)), x, S.Zero, Integer(3)) == [2]
    # atan(tan(x/2)) jumps at x = pi (the zero of the cosine is solved
    # through the face of the solver)
    assert discontinuities(2 * atan(tan(x / 2)), x, S.Zero, 2 * pi) == [pi]


def test_one_sided_limits_and_branches() -> None:
    assert one_sided_limit(atan(x), x, oo, '-') == pi / 2
    assert one_sided_limit(-1 / x, x, S.Zero, '+') == -oo
    assert one_sided_limit(sin(x), x, oo, '-') is None
    # a symbolic point: the value there
    y = Symbol('y')
    assert one_sided_limit(log(x + 2), x, y, '-', [y > 0]) == log(y + 2)
    F = Piecewise((t**2, t > 1), (t**3, True))
    assert select_branch(F, t, Integer(2), Integer(3)) == t**2
    assert select_branch(F, t, S.Zero, S.One) == t**3
    assert select_branch(F, t, S.Zero, Integer(2)) is None


def test_shapes_and_real_logarithms() -> None:
    assert elementary_polynomial(sin(pi * t / 4 + pi / 4)**3, t)
    assert elementary_polynomial(t**2 * exp(2 * t) * cos(t), t)
    assert not elementary_polynomial(sin(t**2), t) and not elementary_polynomial(sin(t) / t, t)
    assert by_parts_shape(x**3 * exp(-x) * log(x), x) and not by_parts_shape(x * log(x), x)
    assert quick_shape(x**3 * exp(-x) * log(x), x) and not quick_shape(log(x) * cos(x), x)
    assert real_logarithms(log(x - 1), x) == log(Abs(x - 1))
    # a complex argument keeps its logarithm
    assert real_logarithms(log(x + I), x) == log(x + I)


def test_default_step_through_the_face() -> None:
    # the radical table is native; integrate and the verified methods are
    # reached through the face
    F = antiderivative(exp(-a * x), x)
    assert F is not None and (F.diff(x) - exp(-a * x)).simplify() == 0
    assert antiderivative_integral(exp(-a * x), x, S.Zero, oo, a > 0) == ConditionalValue(1 / a)
    found = antiderivative_integral(1 / (2 + cos(x)), x, S.Zero, 2 * pi)
    assert found is not None and numerically_equal(found.value, 2 * sqrt(3) * pi / 3)


def test_a_complex_limit_of_a_real_antiderivative_is_refused() -> None:
    # the bug: this antiderivative of u*log(1 - u**2)*atanh(u) (another
    # form of the one the verified methods usually give) has the limit
    # log(2)/2 - I*pi at 1 from below by SymPy's and the core's limit
    # (the value is -3/2 + log(2)), and the integral over (0, 1) came out
    # as -3/2 + log(2) - I*pi
    from sympy_extras.functions import atanh
    u = Symbol('u', positive=True)
    f = u * (log(1 - u) + log(u + 1)) * atanh(u)
    F = (u**2 * log(1 - u) * atanh(u) / 2 + u**2 * log(u + 1) * atanh(u) / 2 - u**2 * atanh(u) / 2
         + u * log(1 - u) / 2 + u * log(u + 1) / 2 - 3 * u / 2 - log(1 - u) * atanh(u) / 2 - log(1 - u)
         - log(u + 1) * atanh(u) / 2 + log(u + 1) - atanh(u) / 2)
    found = antiderivative_integral(f, u, S.Zero, S.One, known=F)
    assert found is None or numerically_equal(found.value, Rational(-3, 2) + log(2))


def test_the_generic_branch_of_cases_at_isolated_values() -> None:
    # the bug: the antiderivative x/2 - Piecewise((x, Eq(b, 0)), ...)/2 of
    # sin(b*x)**2 had no branch selected for a plain b, and the methods
    # spent 5 s on the Mellin route of the quarter periods over (0, pi)
    # before the last method found the value; the case a = b of
    # sin(a*x)*sin(b*x) missed the budget of the special values
    from sympy_extras.core import Eq
    from sympy_extras.integrals.antiderivative import generic_antiderivative
    from sympy_extras.integrals.definite import _Integrator
    b, c = Symbol('b'), Symbol('c')
    F = x / 2 - Piecewise((x, Eq(b, 0)), (sin(2 * b * x) / (2 * b), True)) / 2
    assert generic_antiderivative(F, x) == x / 2 - sin(2 * b * x) / (4 * b)
    # a case on the variable, or a set of positive measure, is kept
    kept = Piecewise((x, x > 1), (x**2, True)) + Piecewise((x, b > 0), (x**2, True))
    assert generic_antiderivative(kept, x) == kept
    # nested cases of isolated values, a conjunction among them
    nested = Piecewise((x, Eq(b, 0) & Eq(c, 0)), (Piecewise((x**2, Eq(c, 0)), (x**3, True)), True))
    assert generic_antiderivative(nested, x) == x**3
    found = _Integrator(None)._quick_antiderivative(sin(b * x)**2, x, S.Zero, pi, 0)
    assert found is not None and found.condition == True
    assert numerically_equal(found.value, pi / 2 - sin(2 * pi * b) / (4 * b))
