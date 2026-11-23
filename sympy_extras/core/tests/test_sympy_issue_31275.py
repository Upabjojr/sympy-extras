"""Regression tests for two errors of the series code found while fixing
the findings of sympy/sympy#31275 in the functions (both shared with
SymPy 1.14)."""
from __future__ import annotations

import mpmath

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.functions import Abs, SingularityFunction, sin

x = Symbol('x')
p = Symbol('p', positive=True)


def test_power_of_a_base_expanded_by_its_series() -> None:
    # the base, which has no leading term of its own (Abs), is replaced by its series: to the
    # relative order that the power needs, not to O(x**n). SymPy: 1/x + x/6 + x**3/36 + O(x**4),
    # and the same for sqrt(sin(x)**2)**(-1) in any symbol, which series() first makes positive
    expected = 1/p + p/6 + 7*p**3/360 + 31*p**5/15120
    assert (1/Abs(sin(p))).series(p, n=6).removeO() == expected
    assert (1/Abs(sin(p))).series(p, n=4).removeO() == 1/p + p/6 + 7*p**3/360
    assert ((sin(x)**2)**Rational(-1, 2)).series(x, n=6).removeO() == expected.subs(p, x)
    assert abs(1/mpmath.sin(mpmath.mpf(1)/10) - (10 + 1/60 + 7/360000 + 31/1512000000)) < 1e-9
    assert (Abs(sin(p))**Rational(-3, 2)).series(p, n=4).removeO() == (
        p**Rational(-3, 2) + p**Rational(1, 2)/4 + 19*p**Rational(5, 2)/480)


def test_leading_term_of_a_sum_which_vanishes_on_one_side() -> None:
    # every term cancels to any order: the loop which raises the order of the series ends with 0
    length = Symbol('l', positive=True)
    e = (SingularityFunction(x + length, 0, 1)/2 - SingularityFunction(x + length, length/2, 1)
         + SingularityFunction(x + length, length, 1)/2)
    assert e.as_leading_term(x) == Integer(0)
