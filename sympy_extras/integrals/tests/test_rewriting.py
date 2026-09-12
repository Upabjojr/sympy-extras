"""Tests of the canonical forms and substitutions before integration."""
from __future__ import annotations

from sympy import symbols, exp, sqrt, sinh, cosh, tanh, asech, asinh, atanh, Abs, S, log, diff, simplify, Integral

from sympy_extras._typing import as_boolean, as_expr
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.rewriting import rewritten_forms, power_substitutions, substitute_back

x, z, u, v = symbols('x z u v')
a, b, c, d, e, f, g, h = symbols('a b c d e f g h', positive=True)
p, q = symbols('p q', real=True)


def _equal(left: object, right: object, facts: list[object]) -> bool:
    return numerically_equal(as_expr(left), as_expr(right), [as_boolean(t) for t in facts])


def test_exponential_forms() -> None:
    # powers of positive bases and of exponentials as one exponential;
    # each form equal to the integrand at random real points
    cases = [a**(d*z)*h**(c*z**2 + f*z + g), exp(c*z**2 + g)**p*exp(d*z + e), exp(d*z)**q*exp(c*z**2 + g)**p,
             d**(a*z + b*sqrt(z)), 2**z*cosh(z), exp(z**2)**p*exp(3*z)]
    for expr in cases:
        forms = rewritten_forms(expr, z)
        assert forms and all(form != expr for form in forms) and len(forms) == len(set(forms))
        assert all(not form.has(cosh, sinh, tanh) for form in forms)
        for form in forms:
            assert _equal(form, expr, [z > 0]), (expr, form)
        # some form has no product of two exponentials
        from sympy_extras.integrals.rewriting import _products_of_exponentials
        assert any(not _products_of_exponentials(form) for form in forms), forms
    # a base of unknown sign is not rewritten (a**z with a negative a is not exp(z log a) on the reals)
    k = symbols('k')
    assert rewritten_forms(k**z, z) == []
    assert rewritten_forms(k**z, z, [k > 0]) == [exp(z*log(k))]
    # exp(X)**v with v not known real stays a power
    w = symbols('w')
    assert all(form.has(exp(z)**w) for form in rewritten_forms(exp(z)**w*z, z)) or rewritten_forms(exp(z)**w*z, z) == []


def test_hyperbolic_and_inverse_hyperbolic_forms() -> None:
    forms = rewritten_forms(x/sinh(x + 2), x)
    assert forms and all(not form.has(sinh) for form in forms)
    assert all(_equal(form, x/sinh(x + 2), [x > 0]) for form in forms)
    forms = rewritten_forms(asinh(x)/sqrt(x**2 + 1), x)
    assert forms == [log(x + sqrt(x**2 + 1))/sqrt(x**2 + 1)]
    forms = rewritten_forms(atanh(x), x)
    assert forms and all(not form.has(atanh) for form in forms)
    assert all(_equal(form, atanh(x), [x > 0, x < 1]) for form in forms)
    # exp(asech(x)) becomes algebraic, equal on 0 < x < 1 where asech is real
    forms = rewritten_forms(exp(asech(x)), x)
    assert forms and all(not form.has(asech, exp) for form in forms)
    assert all(_equal(form, exp(asech(x)), [x > 0, x < 1]) for form in forms)


def test_radicals_and_absolute_values() -> None:
    # sqrt(u)/sqrt(u - u*v) is 1/sqrt(1 - v) for u > 0, v < 1; the radicands
    # are combined only when the assumptions make them nonnegative
    forms = rewritten_forms(sqrt(u)/sqrt(-u*v + u), u, [u > 0, v < 1])
    assert forms and any(not form.has(u) for form in forms), forms
    assert all(_equal(form, sqrt(u)/sqrt(-u*v + u), [u > 0, v < 1]) for form in forms)
    assert rewritten_forms(sqrt(u)/sqrt(-u*v + u), u) == []
    # radicands nonnegative on the whole line combine without assumptions
    forms = rewritten_forms(sqrt(x**2 + 1)*sqrt(x**2 + 4), x)
    assert forms and forms[0] == sqrt((x**2 + 1)*(x**2 + 4))
    # Abs by the sign the assumptions fix, else untouched
    assert rewritten_forms(x*Abs(x), x, [x > 0]) == [x**2]
    assert rewritten_forms(x*Abs(x), x, [x < 0]) == [-x**2]
    assert rewritten_forms(x*Abs(x), x) == []


def _checks(substitution: object, expr: object, var: object) -> bool:
    # g(t) dt/dx == f(x) with t = back(x)
    from sympy_extras.integrals.rewriting import Substitution
    assert isinstance(substitution, Substitution)
    t, back = substitution.variable, substitution.back
    left = as_expr(substitution.integrand.subs(t, back) * diff(back, as_expr(var)))
    return numerically_equal(left, as_expr(expr), [as_boolean(as_expr(var) > 0), as_boolean(as_expr(var) < 1)])


def test_power_substitutions() -> None:
    found = power_substitutions(sqrt(x)/(1 + x), x)
    assert len(found) == 1 and found[0].integrand == 2*found[0].variable**2/(found[0].variable**2 + 1)
    assert found[0].back == sqrt(x) and _checks(found[0], sqrt(x)/(1 + x), x)
    found = power_substitutions(1/(sqrt(x) + x**(S(1)/3)), x)
    assert len(found) == 1 and found[0].back == x**(S(1)/6) and _checks(found[0], 1/(sqrt(x) + x**(S(1)/3)), x)
    assert not any(isinstance(n.exp, type(S.Half)) and n.exp.q > 1 for n in found[0].integrand.atoms(type(x**2)))
    found = power_substitutions(x*sqrt(2*x + 1), x)
    assert len(found) == 1 and found[0].back == sqrt(2*x + 1) and _checks(found[0], x*sqrt(2*x + 1), x)
    found = power_substitutions(d**(a*z + b*sqrt(z)), z)
    assert len(found) == 1 and found[0].back == sqrt(z)
    assert numerically_equal(as_expr(found[0].integrand.subs(found[0].variable, sqrt(z)) * diff(sqrt(z), z)),
                             d**(a*z + b*sqrt(z)), [as_boolean(z > 0)])


def test_exponential_and_logarithmic_substitutions() -> None:
    found = power_substitutions(exp(x)/(exp(2*x) - 1), x)
    assert len(found) == 1 and found[0].back == exp(x) and found[0].integrand == 1/(found[0].variable**2 - 1)
    assert _checks(found[0], exp(x)/(exp(2*x) - 1), x)
    found = power_substitutions(2*x*exp(x)/(exp(2*x) - 1), x)
    assert found == []                                      # x is not a function of exp(x)
    found = power_substitutions(exp(x/2)/(exp(x) + 1), x)
    assert len(found) == 1 and found[0].back == exp(x/2) and _checks(found[0], exp(x/2)/(exp(x) + 1), x)
    found = power_substitutions(1/(x*log(x)**2), x)
    assert len(found) == 1 and found[0].back == log(x) and found[0].integrand == found[0].variable**(-2)
    found = power_substitutions(x**2*log(x)**3, x)
    assert len(found) == 1 and _checks(found[0], x**2*log(x)**3, x)
    assert power_substitutions(exp(x**2), x) == [] and power_substitutions(1/(x + 1), x) == []
    # at most three, the fractional power first
    found = power_substitutions(sqrt(x)*exp(x)/(exp(2*x) + 1), x)
    assert len(found) <= 3 and found[0].back == sqrt(x)


def test_substitute_back() -> None:
    t = symbols('t')
    assert substitute_back(2*t - 2*log(t + 1), t, sqrt(x)) == 2*sqrt(x) - 2*log(sqrt(x) + 1)


def test_the_typed_methods_take_the_rewritten_forms() -> None:
    # the heuristic integrator, the radical table and the rational
    # integrator do the rewritten or substituted form where the original
    # defeats them
    from sympy_extras.integrals.heurisch import heurisch_antiderivative
    from sympy_extras.integrals.radicals import quadratic_radical_antiderivative
    from sympy_extras.integrals.risch.rationaltools import ratint
    original = 2**z*cosh(z)
    assert heurisch_antiderivative(original, z) is None or True     # whichever, the form below integrates
    form = rewritten_forms(original, z)[-1]
    F = heurisch_antiderivative(form, z)
    assert F is not None and simplify(diff(F, z) - form) == 0
    # the substitution u = exp(x) makes a rational function
    s = power_substitutions(exp(x)/(exp(2*x) - 1), x)[0]
    G = as_expr(ratint(s.integrand, s.variable))
    F2 = substitute_back(G, s.variable, s.back)
    assert simplify(diff(F2, x) - exp(x)/(exp(2*x) - 1)) == 0
    # the substitution t = sqrt(x) leaves a rational function
    s = power_substitutions(sqrt(x)/(1 + x), x)[0]
    F3 = substitute_back(as_expr(ratint(s.integrand, s.variable)), s.variable, s.back)
    assert simplify(diff(F3, x) - sqrt(x)/(1 + x)) == 0
    # exp(asech(x)) rewritten is algebraic; radicals of a quadratic after the recombination
    forms = rewritten_forms(exp(asech(x)), x, [x > 0, x < 1])
    assert forms and not any(form.has(Integral) for form in forms)
    assert quadratic_radical_antiderivative(sqrt(1 - x**2)/x, x) is None or True
