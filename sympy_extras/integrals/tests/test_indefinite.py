"""Tests of the verified indefinite integration driver."""
from __future__ import annotations

from sympy import symbols, sqrt, exp, sin, cos, log, tan, atan, asin, Integral, I, diff, simplify, erf

from sympy_extras._typing import as_expr
from sympy_extras.integrals import indefinite_integral, verified_antiderivative, is_antiderivative
from sympy_extras.integrals import indefinite as module

x = symbols('x')


def test_classic_integrands_by_the_typed_methods() -> None:
    expected = {
        sqrt(1 - x**2): 'radicals', x / (x**2 + 1): 'rational', x * exp(x): 'risch', tan(x)**3: 'risch',
        1 / (x * (log(x)**2 + 1)): 'risch', 1 / (x**3 + 1): 'rational', sqrt(x**2 + 1) / x: 'trager',
        x**2 * atan(x): 'risch', 1 / sqrt(x**2 + 1): 'radicals', x / sqrt(x**4 + 1): 'trager'}
    for f, method in expected.items():
        found = verified_antiderivative(f, x)
        assert found is not None and found[1] == method, (f, found)
        assert simplify(diff(found[0], x) - f) == 0 or is_antiderivative(found[0], f, x) is True
    assert indefinite_integral(sqrt(1 - x**2), x) == x * sqrt(1 - x**2) / 2 + asin(x) / 2
    # SymPy's methods stay the last resort, still checked
    found = verified_antiderivative(exp(-x**2), x)
    assert found is not None and found[1] == 'manual' and found[0].has(erf)


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
