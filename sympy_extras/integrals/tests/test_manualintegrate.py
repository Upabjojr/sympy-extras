# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/tests/test_manual.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: on the core; every antiderivative checked by differentiation
# rather than only compared with SymPy's printed form.
# ---------------------------------------------------------------------
"""Tests of the native manualintegrate: SymPy's cases, each result
differentiated back, the forms SymPy gives where the port gives the same,
and the regressions found by the oracle of SymPy's own tests."""
from __future__ import annotations

from sympy_extras.core import Derivative, Function, Rational, S, Symbol, pi
from sympy_extras.core.expr import Expr
from sympy_extras.core.relational import ne
from sympy_extras.core.logic import true
from sympy_extras.functions import (Chi, Ci, Ei, Shi, Si, dirac_delta, heaviside, acos, acot, asin, asinh, atan,
                                    chebyshevt, chebyshevu, cos, cosh, cot, coth, csc, csch, erf, erfi, exp,
                                    gegenbauer, hermite, jacobi, laguerre, legendre, li, log, polylog, sec, sech, sin,
                                    sinh, sqrt, tan, tanh, assoc_laguerre)
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.functions.elementary.trigonometric import Sin
from sympy_extras.polys.polytools import cancel
from sympy_extras.integrals.indefinite import is_antiderivative
from sympy_extras.integrals.integral import Integral
from sympy_extras.integrals.manualintegrate import (AlternativeRule, ArctanRule, CyclicPartsRule, DontKnowRule,
                                                    PartsRule, PowerRule, RewriteRule, SinRule, URule,
                                                    find_substitutions, integral_steps, manual_subs, manualintegrate)

x, y, u, a, b = Symbol('x'), Symbol('y'), Symbol('u'), Symbol('a'), Symbol('b')


def _check(f: Expr, var: Symbol = x) -> Expr:
    F = manualintegrate(f, var)
    assert not F.has(Integral), (f, F)
    # a trigonometric substitution holds where its restriction does
    if isinstance(F, Piecewise) and len(F.args) == 1:
        assert is_antiderivative(F.args[0].expr, f, var, F.args[0].cond) is True, (f, F)
    else:
        assert is_antiderivative(F, f, var) is True, (f, F)
    return F


def test_find_substitutions() -> None:
    assert find_substitutions((cot(x)**2 + 1)**2*csc(x)**2*cot(x)**2, x, u) == \
        [(cot(x), S.One, -u**6 - 2*u**4 - u**2)]
    assert find_substitutions((sec(x)**2 + tan(x)*sec(x))/(sec(x) + tan(x)), x, u) == \
        [(sec(x) + tan(x), S.One, 1/u)]
    assert (-x**2, Rational(-1, 2), exp(u)) in find_substitutions(x*exp(-x**2), x, u)
    f = Function('f')
    assert not find_substitutions(Derivative(f(x), x)**2, x, u)


def test_manual_subs() -> None:
    assert manual_subs(x**2*log(x), [(log(x), u)]) == u*exp(2*u)
    assert manual_subs(sin(x), [(x, u)]) == sin(u)


def test_polynomials() -> None:
    assert manualintegrate(y, x) == x*y
    assert manualintegrate(exp(2), x) == x*exp(2)
    assert manualintegrate(x**2, x) == x**3/3
    assert manualintegrate(3*x**2 + 4*x**3, x) == x**3 + x**4
    assert manualintegrate((x + 2)**3, x) == (x + 2)**4/4
    assert manualintegrate((3*x + 4)**2, x) == (3*x + 4)**3/9


def test_exponentials() -> None:
    assert manualintegrate(exp(2*x), x) == exp(2*x)/2
    assert manualintegrate(2**x, x) == 2**x/log(2)
    assert manualintegrate(1/x, x) == log(x)
    assert manualintegrate(1/(2*x + 3), x) == log(2*x + 3)/2
    assert manualintegrate(log(x)**2/x, x) == log(x)**3/3
    _check(1/sqrt(1 - exp(2*x)))
    _check(x**x*(log(x) + 1))


def test_parts() -> None:
    assert manualintegrate(exp(x)*sin(x), x) == exp(x)*sin(x)/2 - exp(x)*cos(x)/2
    assert isinstance(integral_steps(exp(x)*sin(x), x), (CyclicPartsRule, AlternativeRule))
    assert manualintegrate(2*x*cos(x), x) == 2*x*sin(x) + 2*cos(x)
    assert manualintegrate(x*log(x), x) == x**2*log(x)/2 - x**2/4
    assert manualintegrate(log(x), x) == x*log(x) - x
    assert isinstance(integral_steps(log(x), x), PartsRule)
    assert manualintegrate((3*x**2 + 5)*exp(x), x) == 3*x**2*exp(x) - 6*x*exp(x) + 11*exp(x)
    assert manualintegrate(atan(x), x) == x*atan(x) - log(x**2 + 1)/2
    t = Symbol('t')
    # the probe of the antiderivative route: SymPy's form, at once
    assert manualintegrate(t**2*cos(3*t - 1), t) == \
        t**2*sin(3*t - 1)/3 + 2*t*cos(3*t - 1)/9 - 2*sin(3*t - 1)/27
    _check(exp(-x)*x**2*sin(2*x))


def test_trigonometry() -> None:
    assert manualintegrate(sin(x), x) == -cos(x)
    assert integral_steps(sin(x), x) == SinRule(sin(x), x)
    assert manualintegrate(tan(x), x) == -log(cos(x))
    assert manualintegrate(sec(x), x) == log(sec(x) + tan(x))
    assert manualintegrate(csc(x), x) == -log(csc(x) + cot(x))
    assert manualintegrate(sin(x)*cos(x), x) in [sin(x)**2/2, -cos(x)**2/2]
    assert manualintegrate(-sec(x)*tan(x), x) == -sec(x)
    assert manualintegrate(csc(x)*cot(x), x) == -csc(x)
    assert manualintegrate(sec(x)**2, x) == tan(x)
    assert manualintegrate(csc(x)**2, x) == -cot(x)
    assert manualintegrate(x*sec(x**2), x) == log(tan(x**2) + sec(x**2))/2
    assert manualintegrate(cos(x)*csc(sin(x)), x) == -log(cot(sin(x)) + csc(sin(x)))
    for f in (cos(3*x)*sec(x), sin(3*x)*sec(x), sinh(2*x), x*cosh(x**2), tanh(x), coth(x)):
        _check(f)
    assert manualintegrate(sech(x), x) == 2*atan(tanh(x/2))
    assert manualintegrate(csch(x), x) == log(tanh(x/2))


def test_trigpowers() -> None:
    assert manualintegrate(sin(x)**2*cos(x), x) == sin(x)**3/3
    assert manualintegrate(sin(x)**2*cos(x)**2, x) == x/8 - sin(4*x)/32
    assert manualintegrate(sin(x)*cos(x)**3, x) == -cos(x)**4/4
    assert manualintegrate(sin(x)**3*cos(x)**2, x) == cos(x)**5/5 - cos(x)**3/3
    assert manualintegrate(tan(x)**3*sec(x), x) == sec(x)**3/3 - sec(x)
    assert manualintegrate(tan(x)*sec(x)**2, x) == sec(x)**2/2
    assert manualintegrate(cot(x)**5*csc(x), x) == -csc(x)**5/5 + 2*csc(x)**3/3 - csc(x)
    assert manualintegrate(cot(x)**2*csc(x)**6, x) == -cot(x)**7/7 - 2*cot(x)**5/5 - cot(x)**3/3


def test_inverse_trig() -> None:
    assert manualintegrate(exp(x)/(1 + exp(2*x)), x) == atan(exp(x))
    steps = integral_steps(exp(x)/(1 + exp(2*x)), x)
    assert isinstance(steps, URule) and isinstance(steps.substep, ArctanRule)
    assert manualintegrate(1/(4 + 9*x**2), x) == atan(3*x/2)/6
    assert manualintegrate(1/(16 + 16*x**2), x) == atan(x)/16
    assert manualintegrate(1/(4 + x**2), x) == atan(x/2)/2
    assert manualintegrate(1/(1 + 4*x**2), x) == atan(2*x)/2
    assert manualintegrate(1/(4 + 4*x**2), x) == atan(x)/4
    assert manualintegrate(1/sqrt(1 - x**2), x) == asin(x)
    assert manualintegrate(1/sqrt(4 - 4*x**2), x) == asin(x)/2
    assert manualintegrate(3/sqrt(1 - 9*x**2), x) == asin(3*x)
    assert manualintegrate(1/sqrt(4 - 9*x**2), x) == asin(x*Rational(3, 2))/3
    assert manualintegrate(1/sqrt(x**2 + 1), x) == asinh(x)
    assert manualintegrate(1/sqrt(x**2 + 4), x) == asinh(x/2)
    assert manualintegrate(1/sqrt(4*x**2 + 4), x) == asinh(x)/2
    assert manualintegrate(1/sqrt(4*x**2 + 1), x) == asinh(2*x)/2
    assert manualintegrate(1/sqrt(x**2 - 1), x) == log(2*x + 2*sqrt(x**2 - 1))
    assert manualintegrate(asin(x), x) == x*asin(x) + sqrt(1 - x**2)
    assert manualintegrate(acos(x), x) == x*acos(x) - sqrt(1 - x**2)
    assert manualintegrate(acot(x), x) == x*acot(x) + log(x**2 + 1)/2
    # with parameters: the generic case first, the degenerate one after
    assert manualintegrate(1/(a + b*x**2), x) == \
        piecewise((atan(x/sqrt(a/b))/(b*sqrt(a/b)), ne(a, 0)), (-1/(b*x), true))
    for f in (x*asin(a*x), x*atan(a*x), 1/sqrt(a - b*x**2)):
        F = manualintegrate(f, x)
        assert is_antiderivative(F, f, x) is True


def test_trig_substitution() -> None:
    for f in (sqrt(16*x**2 - 9)/x, 1/(x**4*sqrt(25 - x**2)), x**7/(49*x**2 + 1)**Rational(3, 2),
              1/(x**2 + 1)**2, x**2*sqrt(4 - x**2), 1/(x**2*sqrt(x**2 + 4))):
        _check(f)


def test_rational() -> None:
    assert manualintegrate(1/(4 - x**2), x) == -log(x - 2)/4 + log(x + 2)/4
    assert manualintegrate(1/(-1 + x**2), x) == log(x - 1)/2 - log(x + 1)/2
    _check((x - 3)/(x**2 - 2*x + 2))


def test_special() -> None:
    assert manualintegrate((exp(x) - exp(-x))/x, x) == -Ei(-x) + Ei(x)
    assert manualintegrate(exp(2*x)/x, x) == Ei(2*x)
    assert manualintegrate(3*exp(4*x**2), x) == 3*sqrt(pi)*erfi(2*x)/4
    assert manualintegrate(4*exp(-x**2/3), x) == 2*sqrt(3)*sqrt(pi)*erf(sqrt(3)*x/3)
    assert manualintegrate(cosh(x/2)/x, x) == Chi(x/2)
    assert manualintegrate(cos(x**2)/x, x) == Ci(x**2)/2
    assert manualintegrate(1/log(2*x + 1), x) == li(2*x + 1)/2
    assert manualintegrate(polylog(2, 5*x)/x, x) == polylog(3, 5*x)
    assert manualintegrate(sin(3*x + 2)/x, x) == sin(2)*Ci(3*x) + cos(2)*Si(3*x)
    assert manualintegrate(sinh(3*x - 2)/x, x) == -sinh(2)*Chi(3*x) + cosh(2)*Shi(3*x)
    for f in (x**Rational(1, 3)*exp(-x/8), exp(1 + 2*x - x**2), sin(x**2 + 4*x + 1), cos(4*x**2),
              5*cos(2*x - 3)/x, 5/sqrt(3 - 2*sin(x)**2), sqrt(4 + 9*sin(x)**2)):
        F = manualintegrate(f, x)
        assert not F.has(Integral) and is_antiderivative(F, f, x) is not False, (f, F)


def test_derivative() -> None:
    assert manualintegrate(pi*Derivative(x**2 + 2*x + 3, x), x) == pi*(x**2 + 2*x + 3)
    assert isinstance(integral_steps(Derivative(x**2 + 2*x + 3, y), x), DontKnowRule)
    assert manualintegrate(Derivative(sin(x), x, x, x, y), x) == Derivative(sin(x), x, x, y)


def test_heaviside() -> None:
    assert manualintegrate(dirac_delta(3*x + 2), x) == heaviside(3*x + 2)/3
    assert manualintegrate(heaviside(x), x) == x*heaviside(x)
    assert manualintegrate(x*heaviside(2), x) == x**2/2
    assert manualintegrate(x*heaviside(-2), x) == 0
    assert manualintegrate(x*heaviside(x), x) == x**2*heaviside(x)/2
    assert manualintegrate(heaviside(2*x + 4), x) == (x + 2)*heaviside(2*x + 4)
    assert manualintegrate(sin(7 + x)*heaviside(3*x - 7), x) == \
        (-cos(x + 7) + cos(Rational(28, 3)))*heaviside(3*x - 7)


def test_orthogonal_poly() -> None:
    n = Symbol('n')
    for p in (jacobi(n, 7, Rational(5, 3), x), gegenbauer(n, 7, x), chebyshevt(n, x), chebyshevu(n, x),
              legendre(n, x), hermite(n, x), laguerre(n, x), assoc_laguerre(n, 7, x)):
        integral = manualintegrate(p, x)
        for deg in (0, 1, 3, 5):
            assert (integral.subs(n, deg).diff(x) - p.subs(n, deg)).expand() == 0, (p, deg)


def test_issue_12251() -> None:
    assert manualintegrate(x**y, x) == piecewise((x**(y + 1)/(y + 1), ne(y, -1)), (log(x), true))


def test_issue_6746() -> None:
    assert manualintegrate(y**x, x) == piecewise((y**x/log(y), ne(log(y), 0)), (x, true))
    n = Symbol('n')
    assert manualintegrate(exp(n*x), x) == piecewise((exp(n*x)/n, ne(n, 0)), (x, true))
    yp = Symbol('y', positive=True)
    assert manualintegrate((yp + 1)**x, x) == (yp + 1)**x/log(yp + 1)


def test_steps_are_rules() -> None:
    # the tree of (x**2 + 3)**2: an expansion, a sum, a constant times a power
    steps = integral_steps((x**2 + 3)**2, x)
    assert isinstance(steps, RewriteRule)
    assert steps.rewritten == x**4 + 6*x**2 + 9
    assert steps.eval() == x**5/5 + 2*x**3 + 9*x
    assert integral_steps(x**3, x) == PowerRule(x**3, x, x, S(3))
    # the rules are immutable values: equal trees compare equal
    assert hash(integral_steps(x**3, x)) == hash(PowerRule(x**3, x, x, S(3)))


def test_unknown_integrals_stay_unevaluated() -> None:
    assert manualintegrate(exp(x**3), x) == Integral(exp(x**3), x)
    assert integral_steps(sin(x)/log(x), x).contains_dont_know()


def test_degenerate_quadratic_denominator_is_not_nan() -> None:
    # the coefficients of 1/(4*u + 2*(u - 1)**2 - 2) = 1/(2*u**2), read off
    # the expansion, made an arctangent of zero width whose value was nan
    # (the antiderivative of the whole rational function below was nan)
    f = (4*x**4 + 4*x**3 + 16*x**2 + 12*x + 8)/(x**6 + 2*x**5 + 3*x**4 + 4*x**3 + 3*x**2 + 2*x + 1)
    _check(f)
    assert manualintegrate(1/(4*x + 2*(x - 1)**2 - 2), x) == -1/(2*x)


def test_trig_substitution_back_to_x() -> None:
    # the substitution x = tan(theta) left sin(2*atan(x)) in the result:
    # the native trigsimp does not write it 2*x/(x**2 + 1), nor does the
    # core evaluate sin(atan(x))
    F = manualintegrate(1/(x**2 + 1)**2, x)
    assert not F.has(Sin) and cancel(F - x/(2*(x**2 + 1)) - atan(x)/2) == 0


def test_secant_substitution_holds_where_it_says() -> None:
    # SymPy's restriction of x = 3*sec(theta)/4 is -3/4 < x < 3/4, where for
    # x < 0 the derivative of its antiderivative of sqrt(16*x**2 - 9)/x
    # misses the integrand by 16.1*I (at x = -1/2): the substitution holds
    # for x > 3/4
    F = manualintegrate(sqrt(16*x**2 - 9)/x, x)
    assert isinstance(F, Piecewise) and F.args[0].args[1] == (x > Rational(3, 4))
