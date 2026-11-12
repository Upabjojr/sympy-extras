"""Tests of the native verified indefinite integration driver, on core
objects (:mod:`sympy_extras.integrals.indefinite`)."""
from __future__ import annotations

import pytest

from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core import I, Rational, S
from sympy_extras.core.expr import Expr
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import asin, atan, cos, exp, log, sign, sin, sqrt, tan
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.integrals import indefinite
from sympy_extras.integrals.indefinite import (conjugate_logarithms, indefinite_integral, is_antiderivative, real_form,
                                               verified_antiderivative)
from sympy_extras.integrals.integral import Integral, is_unevaluated

x, k, a = Symbol('x'), Symbol('k'), Symbol('a')


def test_the_typed_methods_answer_on_core_objects() -> None:
    expected = {
        sqrt(1 - x**2): 'radicals', x / (x**2 + 1): 'rational', x * exp(x): 'exponential',
        tan(x)**3: 'trigonometric', 1 / (x * (log(x)**2 + 1)): 'risch', 1 / (x**3 + 1): 'rational',
        x / sqrt(x**4 + 1): 'trager', exp(x) * sin(x): 'trigonometric', log(x)**2: 'risch',
        sqrt(1 + x**3): 'binomial'}
    for f, method in expected.items():
        found = verified_antiderivative(f, x)
        assert found is not None and found[1] == method, (f, found)
        assert is_antiderivative(found[0], f, x) is True
    assert indefinite_integral(sqrt(1 - x**2), x) == x * sqrt(1 - x**2) / 2 + asin(x) / 2


def test_sympys_routes_come_last_and_are_checked() -> None:
    # SymPy's integrate (the face, loaded at run time) gives a core expression
    found = verified_antiderivative(sin(x), x, methods=['sympy'])
    assert found == (-cos(x), 'sympy')
    assert indefinite.METHODS[-len(indefinite.SYMPY_METHODS):] == [
        (name, method) for name, method in indefinite.METHODS if name in indefinite.SYMPY_METHODS]


def test_the_unevaluated_integral_is_a_core_node() -> None:
    F = indefinite_integral(x**x, x)
    assert isinstance(F, Integral) and F.function == x**x and F.variable == x
    assert is_unevaluated(F) and not is_unevaluated(x**2)
    assert F.diff(x) == x**x and str(F) == 'Integral(x**x, x)'
    # the derivative in a parameter goes under the integral sign
    G = Integral(exp(a * x**2), x)
    assert G.diff(a) == Integral(x**2 * exp(a * x**2), x) and G.diff(k) == S.Zero
    assert G.subs(a, 2) == Integral(exp(2 * x**2), x) and G.subs(x, k) == G


def test_the_unevaluated_integral_converts_to_sympys() -> None:
    import sympy
    from sympy_extras.interop import to_sympy
    converted = to_sympy(Integral(x**x, x))
    sx = sympy.Symbol('x')
    assert converted == sympy.Integral(sx**sx, sx)


def test_wrong_candidates_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def wrong(f: Expr, v: Symbol, assumptions: Assumptions) -> Expr:
        return cos(v)

    monkeypatch.setattr(indefinite, 'METHODS', [('wrong', wrong)])
    assert indefinite_integral(sin(x), x) == Integral(sin(x), x)
    assert verified_antiderivative(sin(x), x) is None


def test_is_antiderivative() -> None:
    assert is_antiderivative(log(x**2 + 1) / 2, x / (x**2 + 1), x) is True
    assert is_antiderivative(sin(x), sin(x), x) is False
    assert is_antiderivative(Integral(exp(-x**2), x), exp(-x**2), x) is None
    # the numerical fallback: an identity cancel does not see
    assert is_antiderivative(x * sqrt(1 - x**2) / 2 + asin(x) / 2, sqrt(1 - x**2), x) is True
    # the jump of sign(x) is not a test point
    assert is_antiderivative(x**2 * sign(x) / 2, x * sign(x), x) is True
    # right for x > 0 only: the points of both signs see it
    from sympy_extras.functions import asinh
    assert is_antiderivative(-asinh(1 / x), 1 / (x * sqrt(x**2 + 1)), x) is False
    assert is_antiderivative(-asinh(1 / x), 1 / (x * sqrt(x**2 + 1)), x, [x > 0]) is not False


def test_real_forms() -> None:
    F = -I * log(x - I) / 2 + I * log(x + I) / 2
    assert conjugate_logarithms(F, x) == atan(x)
    assert real_form(F, 1 / (x**2 + 1), x) == atan(x)
    # the antiderivative of -1/(x**2 + 1): no real rewriting checks against 1/(x**2 + 1)
    assert real_form(-F, 1 / (x**2 + 1), x).has(I)
    assert not indefinite_integral(1 / (x**4 + 1), x).has(I)
    # the dummy of the real variable has a name of its own (the typed
    # polynomials identify symbols by name)
    assert conjugate_logarithms(-I * log(x - I) / 2 + I * log(x + I) / 2 + log(x), x) == atan(x) + log(x)


def test_special_values_in_sympys_form() -> None:
    F = indefinite_integral(exp(k * x), x)
    assert isinstance(F, Piecewise) and str(F) == 'Piecewise((exp(k*x)/k, Ne(k, 0)), (x, True))'
    assert indefinite_integral(exp(k * x), x, special_values=False) == exp(k * x) / k


def test_the_assumptions_reach_the_typed_methods() -> None:
    b, c = Symbol('b'), Symbol('c')
    Q = a + b * x + c * x**2
    facts = [a > 0, b > 0, c > 0, 4 * a * c - b**2 > 0]
    found = verified_antiderivative(sqrt(Q) / x, x, facts)
    assert found is not None and found[1] == 'radicals'
    assert is_antiderivative(found[0], sqrt(Q) / x, x, facts) is True


def test_substitutions_and_rewritings() -> None:
    f = 1 / (x**Rational(1, 3) + sqrt(x))
    found = verified_antiderivative(f, x, methods=['rewriting'])
    assert found is not None and found[1] == 'rewriting' and is_antiderivative(found[0], f, x) is True
