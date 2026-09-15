"""Tests of the heuristic Risch integrator."""
from __future__ import annotations

from sympy import (symbols, exp, sin, cos, tan, log, sqrt, erf, Ei, Si, Ci, li, asin, asinh, atan, pi, I, Abs, sign,
                   Float, Piecewise, Ne, besselj, LambertW, sinh, cosh, tanh, Rational, S, diff, cancel, simplify)
from typing import Optional, Sequence

from sympy.core.expr import Expr
from sympy.logic.boolalg import Boolean

from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.heurisch import (components, heurisch_antiderivative, heurisch_cases, _real_forms,
                                              _verified)

x, y, n = symbols('x y n')


def _antiderivative_of(F: object, f: ExprLike, facts: Optional[Sequence[Boolean]] = None) -> bool:
    """Whether ``F`` differentiates to ``f``, symbolically or numerically
    where the facts hold (the radicals are real there)."""
    if not isinstance(F, Expr):
        return False
    difference = as_expr(diff(F, x) - as_expr(f))
    if cancel(difference) == 0 or simplify(difference) == 0:
        return True
    return numerically_equal(as_expr(diff(F, x)), as_expr(f), [as_boolean(c) for c in (facts or [])])


def test_components() -> None:
    assert components(sin(x) * cos(x)**2, x) == {x, sin(x), cos(x)}
    assert components(sqrt(x + 1)**3, x) == {x, sqrt(x + 1)}
    assert components(x**y, x) == {x, x**y}
    assert components(exp(sin(x)), x) == {x, sin(x), exp(sin(x))}
    assert components(y, x) == set()


def test_elementary_families() -> None:
    # rational, exponential, logarithmic, trigonometric and hyperbolic
    # integrands: every antiderivative checked by differentiation
    for f in [1 / (x**2 + 1), 1 / (x**3 + 1), x / (x**2 - 1), (x + 1) / (x**2 + 2 * x + 5), 1 / (x**4 + 1),
              1 / (x**2 + 1)**2, x**3 / (x**2 + 1), x**2 * exp(x), exp(x) * sin(x), exp(x) / (exp(x) + 1),
              exp(x) / (exp(2 * x) + 1), log(x), x * log(x), log(x) / x, log(x)**2 / x**3, 1 / (x * log(x)),
              exp(x) * log(x) + exp(x) / x, sin(x) * cos(x), sin(x)**2, x * sin(x), sin(x)**3, cos(x)**4,
              sin(2 * x) * cos(3 * x), x * cos(x)**2, tan(x)**2, tan(x)**3, y * tan(x), 1 / (1 + tan(x)**2),
              sinh(x) * cosh(x), tanh(x), 1 / cosh(x)**2, x**Rational(1, 3) / (x + 1), 1 / (x**Rational(1, 3) + 1),
              sqrt(x) * exp(-sqrt(x)), x * besselj(0, x), LambertW(x), atan(x), asin(x), x * atan(x),
              cos(n * x), exp(y * x) * sin(x), 1 / (x**2 + y**2)]:
        F = heurisch_antiderivative(f, x)
        assert _antiderivative_of(F, f), f


def test_half_angle_rewriting() -> None:
    # the bug: the sine and cosine integrals were added as candidates for
    # every sin(x), which multiplied the types of components and the
    # permutations tried before the rewriting in tangents, so that
    # 1/sin(x) never reached log(tan(x/2)); the candidates are added only
    # where x divides the denominator
    assert heurisch_antiderivative(1 / sin(x), x) == log(tan(x / 2))
    for f in [1 / cos(x), 1 / (1 + sin(x)), 1 / (2 + cos(x))]:
        F = heurisch_antiderivative(f, x)
        assert F is not None and numerically_equal(diff(F, x), f), f


def test_mapping_symbols_are_shared() -> None:
    # the bug: fresh dummies for the components on each call left x/_x1
    # as the result, the mapping of the retried call being unknown to the
    # outer one; the symbols are a cache local to the module as in SymPy
    F = heurisch_antiderivative(1 / cosh(x)**2, x)
    assert F is not None and not (F.free_symbols - {x}) and numerically_equal(diff(F, x), 1 / cosh(x)**2)


def test_special_candidates() -> None:
    assert heurisch_antiderivative(exp(-x**2), x) == sqrt(pi) * erf(x) / 2
    assert heurisch_antiderivative(x**2 * exp(-x**2), x) == -x * exp(-x**2) / 2 + sqrt(pi) * erf(x) / 4
    assert heurisch_antiderivative(exp(-x**2) * erf(x), x) == sqrt(pi) * erf(x)**2 / 4
    assert heurisch_antiderivative(exp(x) / x, x) == Ei(x)
    assert heurisch_antiderivative(exp(2 * x) / x, x) == Ei(2 * x)
    assert heurisch_antiderivative(sin(x) / x, x) == Si(x)
    assert heurisch_antiderivative(cos(x) / x, x) == Ci(x)
    assert heurisch_antiderivative(x * Ei(x), x) == x**2 * Ei(x) / 2 - x * exp(x) / 2 + exp(x) / 2
    assert heurisch_antiderivative(li(x), x) == x * li(x) - Ei(2 * log(x))
    assert heurisch_antiderivative(1 / sqrt(x**2 + 1), x) == asinh(x)
    assert heurisch_antiderivative(1 / sqrt(x**2 - 4), x) == log(x + sqrt(x**2 - 4))
    # the completed square
    assert heurisch_antiderivative(1 / sqrt(2 * x - x**2), x) == asin(x - 1)
    assert heurisch_antiderivative(1 / sqrt(x**2 + 2 * x + 5), x) == asinh(x / 2 + S.Half)
    F = heurisch_antiderivative(sqrt(2 * x - x**2), x)
    assert _antiderivative_of(F, sqrt(2 * x - x**2), [x > 0, x < 2])
    F = heurisch_antiderivative(sqrt(1 - x**2), x)
    assert F == x * sqrt(1 - x**2) / 2 + asin(x) / 2


def test_real_forms() -> None:
    # -I/2 log(x - I) + I/2 log(x + I) differentiates to 1/(x**2 + 1), and
    # its conjugate pair to -1/(x**2 + 1): the real forms keep the sign
    # (the bug: the first draft of the test expected atan(x) for the
    # latter, which is -atan(x) up to a constant, as the value at x = 2 shows)
    assert _real_forms(-I * log(x - I) / 2 + I * log(x + I) / 2) == atan(x)
    assert _real_forms(I * log(x - I) / 2 - I * log(x + I) / 2) == -atan(x)
    assert heurisch_antiderivative(1 / (x**2 + 1), x) == atan(x)
    for f in [1 / (x**4 + 1), 1 / (x**3 + 1), exp(x) / (exp(2 * x) + 1), 1 / (x**2 + y**2)]:
        F = heurisch_antiderivative(f, x)
        assert F is not None and not F.has(I) and _antiderivative_of(F, f), f


def test_refusals_and_verification() -> None:
    # not elementary in the sense of the derivation, or not exact
    assert heurisch_antiderivative(Abs(x), x) is None
    assert heurisch_antiderivative(sign(x) * x, x) is None
    assert heurisch_antiderivative(Float(1.5) * x, x) is None
    # no elementary antiderivative: the error function the tower allows
    # (exp(x**2 + x) is exp((x + 1/2)**2 - 1/4)), or refused, never an
    # unevaluated integral
    from sympy import erfi, sqrt, pi
    assert heurisch_antiderivative(exp(x**2) * exp(x), x) == sqrt(pi) * exp(-S(1) / 4) * erfi(x + S(1) / 2) / 2
    assert heurisch_antiderivative(exp(x) / log(x), x) is None
    # an integrand free of x
    assert heurisch_antiderivative(y, x) == x * y
    # the verification refuses a candidate which does not differentiate back
    assert _verified(x**3, x**2, x) is None
    assert _verified(x**3 / 3, x**2, x) == x**3 / 3


def test_cases_over_the_parameters() -> None:
    found = heurisch_cases(cos(n * x), x)
    assert isinstance(found, Piecewise)
    assert found.args[0] == (sin(n * x) / n, Ne(n, 0)) and found.args[1] == (x, True)
    assert heurisch_cases(cos(x), x) == sin(x)
    assert heurisch_cases(y, x) == x * y
    found = heurisch_cases(exp(n * x), x)
    assert isinstance(found, Piecewise) and found.subs(n, 0).doit() == x
    assert found.args[1].args[0] == x and found.args[0].args[1] == Ne(n, 0)


def test_beyond_the_method() -> None:
    # the derivation takes sqrt(Q) as an independent component, so the
    # antiderivatives whose derivatives match only modulo sqrt(Q)**2 = Q
    # are out of reach (Mathematica: -atanh(sqrt(1 + x**2))); the radical
    # table and Trager's algorithm take them. With the completed-square
    # candidate asinh(x) and the raised degree bound, x**2 sqrt(x**2 + 1)
    # does come out, as ((x + 2 x**3) sqrt(1 + x**2) - asinh(x))/8
    assert heurisch_antiderivative(1 / (x * sqrt(x**2 + 1)), x) is None
    F = heurisch_antiderivative(x**2 * sqrt(x**2 + 1), x)
    assert F is not None and simplify(F - ((x + 2 * x**3) * sqrt(1 + x**2) - asinh(x)) / 8) == 0


def test_the_special_functions_of_the_tower() -> None:
    # Cherry's structure: Ei(theta + c) for a factor theta + c of the
    # denominator (the logarithmic integral for a logarithm), erf(u) for a
    # perfect square -theta = u**2 + c with exp(c) in the field, the
    # dilogarithm for exp(theta) + 1 in the denominator; the FriCAS suite's
    # cases which FriCAS integrates in these functions
    from sympy import erfi, polylog
    from sympy_extras.integrals.indefinite import is_antiderivative
    expected = {
        exp(x) / (x + 1)**2: Ei, 1 / (log(x) + 1): Ei, x / (log(x) + 1): Ei,
        (2 * exp(2 * x) + exp(x)) / log(exp(2 * x) + exp(x)): Ei, exp(-x) / (x + 2): Ei,
        x / (exp(x) + 1): polylog, exp(x) * exp(-exp(2 * x)): erf, (exp(x) + 1) * exp(-(x + exp(x))**2): erf,
        x * exp(-x**2 + 2 * x): erf, exp(x**2 + 1): erfi, exp(-log(x)**2) / x: erf}
    for f, function in expected.items():
        F = heurisch_antiderivative(f, x)
        assert F is not None and F.has(function) and is_antiderivative(F, f, x) is True, f
    assert heurisch_antiderivative(1 / (log(x) + 1), x) == exp(-1) * Ei(log(x) + 1)
    assert heurisch_antiderivative(exp(x) * exp(-exp(2 * x)), x) == sqrt(pi) * erf(exp(x)) / 2
    # the components from the largest to the smallest: log(log(x))/x and the
    # nested exponentials need log(x), exp(2*x) left inside their outer
    # functions until those are mapped
    assert heurisch_antiderivative(log(log(x)) / x, x) == log(x) * log(log(x)) - log(x)
