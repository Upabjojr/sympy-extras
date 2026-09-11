"""Tests of the summability methods for divergent oscillatory integrals."""
from __future__ import annotations

from sympy import symbols, sin, cos, exp, besselj, oo, pi, S, I, Dummy
from sympy.testing.pytest import raises

from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.summability import summable_integral, oscillatory_tail, _cesaro_mean, METHODS

x = symbols('x')
a = symbols('a', positive=True)
k = symbols('k')


def test_abel_means() -> None:
    # the regularised integrals: 1/(1 + eps**2), eps/(1 + eps**2),
    # 2 eps/(1 + eps**2)**2 and (eps**2 - 1)/(1 + eps**2)**2 (Hardy 5.14)
    assert summable_integral(sin(x), x, 0, oo, 'abel') == ConditionalValue(1)
    assert summable_integral(cos(x), x, 0, oo, 'abel') == ConditionalValue(0)
    assert summable_integral(x * sin(x), x, 0, oo, 'abel') == ConditionalValue(0)
    assert summable_integral(x * cos(x), x, 0, oo, 'abel') == ConditionalValue(-1)
    assert summable_integral(x**2 * cos(x), x, 0, oo, 'abel') == ConditionalValue(0)
    assert summable_integral(exp(I * x), x, 0, oo, 'abel') == ConditionalValue(I)
    assert summable_integral(sin(a * x), x, 0, oo, 'abel') == ConditionalValue(1 / a)


def test_convergent_integrals_keep_their_values() -> None:
    # the consistency theorem: a convergent integral has every mean equal
    # to its value
    for method in METHODS:
        assert summable_integral(besselj(0, x), x, 0, oo, method) == ConditionalValue(1)
        assert summable_integral(exp(-x), x, 0, oo, method) == ConditionalValue(1)


def test_means_which_do_not_exist() -> None:
    # sin(x)**2: the regularised integral is 2/(eps (eps**2 + 4)) and grows
    # like 1/(4 eps); the powers of x give k!/eps**(k + 1); never a finite
    # value for these
    for method in METHODS:
        assert summable_integral(sin(x)**2, x, 0, oo, method) is None
        assert summable_integral(S.One, x, 0, oo, method) is None
        assert summable_integral(x, x, 0, oo, method) is None
        assert summable_integral(x**2, x, 0, oo, method) is None
        assert summable_integral(exp(x), x, 0, oo, method) is None


def test_cesaro_orders() -> None:
    # (C, 1) of x sin(x) oscillates: (2 - 2 cos R - R sin R)/R; (C, 2)
    # exists (Hardy 5.14: Cesàro means of every order agree where they
    # exist, and the order needed grows with the power of x)
    t = Dummy('t', positive=True)
    assert _cesaro_mean(t * sin(t), t, 1, None) is None
    assert _cesaro_mean(t * sin(t), t, 2, None) == ConditionalValue(0)
    assert _cesaro_mean(t * sin(t), t, 3, None) == ConditionalValue(0)
    assert _cesaro_mean(t * cos(t), t, 2, None) == ConditionalValue(-1)
    assert summable_integral(sin(x), x, 0, oo, 'cesaro') == ConditionalValue(1)
    assert summable_integral(cos(x), x, 0, oo, 'cesaro') == ConditionalValue(0)
    assert summable_integral(x * sin(x), x, 0, oo, 'cesaro') == ConditionalValue(0)
    assert summable_integral(sin(a * x), x, 0, oo, 'cesaro') == ConditionalValue(1 / a)


def test_gaussian_means() -> None:
    assert summable_integral(sin(x), x, 0, oo, 'gaussian') == ConditionalValue(1)
    assert summable_integral(cos(x), x, 0, oo, 'gaussian') == ConditionalValue(0)
    assert summable_integral(x * sin(x), x, 0, oo, 'gaussian') == ConditionalValue(0)


def test_ranges() -> None:
    # the real line is cut at 0 and both tails summed; a range toward
    # -oo is reflected; swapped bounds change the sign; a bounded range is
    # the ordinary integral
    assert summable_integral(cos(x), x, -oo, oo, 'abel') == ConditionalValue(0)
    assert summable_integral(sin(x), x, -oo, 0, 'abel') == ConditionalValue(-1)
    assert summable_integral(sin(x), x, oo, 0, 'abel') == ConditionalValue(-1)
    assert summable_integral(sin(x), x, 0, pi, 'abel') == ConditionalValue(2)
    assert summable_integral(sin(x), x, 2, 2, 'abel') == ConditionalValue(0)
    # a tail from pi: Integral(sin(t + pi) exp(-eps t)) = -1/(1 + eps**2)
    assert oscillatory_tail(sin(x), x, pi, 'abel') == ConditionalValue(-1)
    assert oscillatory_tail(sin(x), x, pi, 'cesaro') == ConditionalValue(-1)


def test_conditions_on_the_parameters() -> None:
    # the value 1/k holds for k > 0 (and for k < 0 too, but the inner
    # integral reports the condition it was found under); the assumption
    # settles it
    found = summable_integral(sin(k * x), x, 0, oo, 'abel', k > 0)
    assert found == ConditionalValue(1 / k)
    found = summable_integral(sin(k * x), x, 0, oo, 'abel')
    assert found is not None and found.condition != S.true
    assert found.condition.subs(k, 2) == S.true or found.condition.subs(k, 2).simplify() == S.true
    assert found.value.subs(k, 2) == S.Half


def test_wrong_method() -> None:
    raises(ValueError, lambda: summable_integral(sin(x), x, 0, oo, 'borel'))
    raises(ValueError, lambda: oscillatory_tail(sin(x), x, 0, 'borel'))
    raises(ValueError, lambda: summable_integral(sin(x), x, 0, oo, 'abel '))
