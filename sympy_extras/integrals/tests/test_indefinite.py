"""Tests of the verified indefinite integration driver."""
from __future__ import annotations

from sympy import symbols, sqrt, exp, sin, cos, log, tan, atan, asin, Integral, I, diff, simplify, erf

from sympy_extras._typing import as_expr
from sympy_extras.integrals import indefinite_integral, verified_antiderivative, is_antiderivative
from sympy_extras.integrals import indefinite as module

x = symbols('x')


def test_classic_integrands_by_the_typed_methods() -> None:
    # the typed methods answer these, SymPy's routes are never reached
    expected = {
        sqrt(1 - x**2): 'radicals', x / (x**2 + 1): 'rational', x * exp(x): 'risch',
        tan(x)**3: 'trigonometric', 1 / (x * (log(x)**2 + 1)): 'risch', 1 / (x**3 + 1): 'rational',
        sqrt(x**2 + 1) / x: 'trager', x**2 * atan(x): 'trigonometric', 1 / sqrt(x**2 + 1): 'radicals',
        x / sqrt(x**4 + 1): 'trager', exp(x) * sin(x): 'trigonometric', log(x)**2: 'risch'}
    for f, method in expected.items():
        found = verified_antiderivative(f, x)
        assert found is not None and found[1] == method, (f, found)
        assert simplify(diff(found[0], x) - f) == 0 or is_antiderivative(found[0], f, x) is True
    assert indefinite_integral(sqrt(1 - x**2), x) == x * sqrt(1 - x**2) / 2 + asin(x) / 2
    # the error function from the heuristic integrator's candidate table
    found = verified_antiderivative(exp(-x**2), x)
    assert found is not None and found[1] == 'heurisch' and found[0].has(erf)
    # SymPy's methods stay the last resort, still checked
    found = verified_antiderivative(sin(x), x, methods=['sympy'])
    assert found == (-cos(x), 'sympy')


def test_is_antiderivative() -> None:
    assert is_antiderivative(log(x**2 + 1) / 2, x / (x**2 + 1), x) is True
    assert is_antiderivative(sin(x), sin(x), x) is False
    assert is_antiderivative(Integral(exp(-x**2), x), exp(-x**2), x) is None
    # the numerical fallback: an identity cancel does not see
    assert is_antiderivative(x * sqrt(1 - x**2) / 2 + asin(x) / 2, sqrt(1 - x**2), x) is True


def test_wrong_candidates_are_refused(monkeypatch: object) -> None:
    # a method returning a wrong antiderivative is not believed: the
    # driver moves on, and returns the Integral when nothing checks
    import pytest
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    wrong = [('wrong', lambda f, v: as_expr(cos(v)))]
    monkeypatch.setattr(module, 'METHODS', wrong)
    assert indefinite_integral(sin(x), x) == Integral(sin(x), x)
    assert verified_antiderivative(sin(x), x) is None


def test_real_forms_are_preferred() -> None:
    # a candidate with I for a real integrand is rewritten when a real
    # antiderivative checks (the logarithms of conjugates as an arctangent)
    F = module.real_form(-I * log(x - I) / 2 + I * log(x + I) / 2, 1 / (x**2 + 1), x)
    assert F == atan(x)
    # the sign matters: this pair is an antiderivative of -1/(x**2 + 1), and
    # no real rewriting checks against 1/(x**2 + 1), so the form is kept
    kept = module.real_form(I * log(x - I) / 2 - I * log(x + I) / 2, 1 / (x**2 + 1), x)
    assert kept.has(I)
    assert not indefinite_integral(1 / (x**2 + 1), x).has(I)
    assert not indefinite_integral(1 / (x**4 + 1), x).has(I)


def test_antiderivatives_wrong_for_negative_x_are_refused() -> None:
    # the census of SymPy's mistakes: -asinh(1/x) is an antiderivative of
    # 1/(x*sqrt(x**2 + 1)) for x > 0 only (Maxima's reference is
    # -asinh(1/Abs(x))), and the sampler of the check drew positive
    # points only
    from sympy import asinh
    assert is_antiderivative(-asinh(1 / x), 1 / (x * sqrt(x**2 + 1)), x) is False
    assert is_antiderivative(-asinh(1 / x), 1 / (x * sqrt(x**2 + 1)), x, [x > 0]) is not False
    found = indefinite_integral(1 / (x * sqrt(x**2 + 1)), x)
    assert found.has(Integral) or is_antiderivative(found, 1 / (x * sqrt(x**2 + 1)), x) is True
