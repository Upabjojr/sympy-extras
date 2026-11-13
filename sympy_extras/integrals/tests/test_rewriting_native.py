"""Tests of the canonical forms and substitutions on the objects of the
core (:mod:`sympy_extras.integrals.rewriting`): every form checked equal
to the integrand and every substitution checked by ``g(t(x)) t'(x) =
f(x)``, at random points where the integrand is real."""
from __future__ import annotations

from sympy_extras.core import Rational, Symbol, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean
from sympy_extras.core.symbol import Dummy
from sympy_extras.functions import Abs, Cosh, Sinh, acosh, asinh, atanh, cosh, exp, log, sinh
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.rewriting import (Substitution, _extracted_powers, implied_assumptions,
                                              power_substitutions, rewritten_forms, substitute_back)

x, z = Symbol('x'), Symbol('z')
a, c, d = Symbol('a', positive=True), Symbol('c', positive=True), Symbol('d', positive=True)
p = Symbol('p', real=True)


def _holds(s: Substitution, f: Expr, facts: list[Boolean]) -> bool:
    """``g(t) dt = f(x) dx`` with ``t = back(x)``."""
    left = s.integrand.subs(s.variable, s.back) * s.back.diff(x)
    return numerically_equal(left, f, facts)


def test_forms() -> None:
    for f in (a**(d * z) * exp(z**2), exp(c * z**2)**p * exp(d * z), 2**z * cosh(z)):
        forms = rewritten_forms(f, z)
        assert forms and all(form != f for form in forms)
        assert all(numerically_equal(form, f, [z > 0]) for form in forms), (f, forms)
        assert all(not form.has(Cosh) for form in forms)
    forms = rewritten_forms(x / sinh(x + 2), x)
    assert forms and all(not form.has(Sinh) and numerically_equal(form, x / sinh(x + 2), [x > 0]) for form in forms)
    assert rewritten_forms(asinh(x) / sqrt(x**2 + 1), x) == [log(x + sqrt(x**2 + 1)) / sqrt(x**2 + 1)]
    forms = rewritten_forms(atanh(x), x)
    assert forms and all(numerically_equal(form, atanh(x), [x > 0, x < 1]) for form in forms)
    # SymPy's branch of acosh: z + sqrt(z - 1)*sqrt(z + 1), right for z < -1 too
    assert rewritten_forms(exp(acosh(z)), z) == [z + sqrt(z - 1) * sqrt(z + 1)]


def test_signs_and_radicals() -> None:
    k, n = Symbol('k'), Symbol('n', negative=True)
    assert rewritten_forms(k**z, z) == [exp(z * log(k))]
    assert rewritten_forms(k**z, z, [k < 0]) == [] and rewritten_forms(n**z, z) == []
    assert implied_assumptions(k**(d * z), z) == [k > 0] and implied_assumptions(a**z, z) == []
    assert rewritten_forms(x * Abs(x), x, [x > 0]) == [x**2]
    assert rewritten_forms(x * Abs(x), x, [x < 0]) == [-x**2]
    assert rewritten_forms(x * Abs(x), x) == []
    forms = rewritten_forms(sqrt(x**2 + 1) * sqrt(x**2 + 4), x)
    assert forms and forms[0] == sqrt((x**2 + 1) * (x**2 + 4))
    # (z**r)**p is z**(r*p) for z > 0 only
    r = Symbol('r')
    assert rewritten_forms((z**r)**p, z, [z > 0]) == [z**(p * r)]
    assert rewritten_forms((z**r)**p, z) == []


def test_power_and_exponential_substitutions() -> None:
    for f in (sqrt(x) / (1 + x), 1 / (sqrt(x) + x**Rational(1, 3)), x * sqrt(2 * x + 1), exp(x) / (exp(2 * x) - 1),
              exp(x / 2) / (exp(x) + 1), 1 / (x * log(x)**2), x**2 * log(x)**3, sqrt(x**4 + 1) / x**5,
              x**5 * sqrt(x**3 + 1)):
        found = power_substitutions(f, x)
        assert found, f
        assert all(_holds(s, f, [x > 0, x < 1]) for s in found), (f, found)
    assert power_substitutions(exp(x**2), x) == [] and power_substitutions(1 / (x + 1), x) == []
    assert power_substitutions(2 * x * exp(x) / (exp(2 * x) - 1), x) == []
    # the new variables are dummies with names of their own (the typed
    # polynomials identify symbols by name)
    t = Symbol('t')
    found = power_substitutions(sqrt(x) * t / (1 + x), x)
    assert found and isinstance(found[0].variable, Dummy) and found[0].variable.name != 't'
    assert _holds(found[0], sqrt(x) * t / (1 + x), [x > 0, t > 0])


def test_regions_and_nested_radicals() -> None:
    f = ((x - 1)**2 * (x + 1))**Rational(1, 3) / x**2
    regions = _extracted_powers(f, x)
    assert [facts for _, facts in regions] == [[x - 1 > 0], [1 - x > 0]]
    assert numerically_equal(regions[0][0], f, [x > 1]) and numerically_equal(regions[1][0], f, [x < 1, x > 0])
    found = power_substitutions(f, x)
    assert [s.facts for s in found] == [[x - 1 > 0], [1 - x > 0]]
    assert all(s.integrand.is_rational_function(s.variable) for s in found)
    assert _holds(found[0], f, [x > 1]) and _holds(found[1], f, [x > 0, x < 1])
    assert _extracted_powers(sqrt(x**2 + 1) / x, x) == []
    g = sqrt(1 - sqrt(x)) / (x**2 - 1)
    found = power_substitutions(g, x)
    assert len(found) == 1 and found[0].back == sqrt(1 - sqrt(x))
    assert found[0].integrand.is_rational_function(found[0].variable) and _holds(found[0], g, [x > 0, x < 1])


def test_substitute_back() -> None:
    t = Symbol('t')
    assert substitute_back(2 * t - 2 * log(t + 1), t, sqrt(x)) == 2 * sqrt(x) - 2 * log(sqrt(x) + 1)
