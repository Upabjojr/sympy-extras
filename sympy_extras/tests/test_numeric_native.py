"""Tests of the native reliable values (:mod:`sympy_extras._numeric`) on the
expressions of the core; ``test_numeric.py`` tests the SymPy face."""
from __future__ import annotations

import time

from sympy_extras._numeric import reliable_complex, reliable_form, reliable_value
from sympy_extras.core import Integer, Rational, I, Symbol, pi, sqrt
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.numbers import dps_to_prec
from sympy_extras.core.expr import Expr
from sympy_extras.functions import Log, asin, atan, cos, exp, gamma, log, sin

x = Symbol('x')

#: a sum which is exactly zero, proved by its minimal polynomial
Z = -6 + (-2 + sqrt(2))**2 + 4 * sqrt(2)
#: one which is not proved zero
W = sin(Integer(1))**2 + cos(Integer(1))**2 - 1
#: log(2*sqrt(2)) + I*pi, on the cut when Z is evaluated
L = Log(-2 * sqrt(2) - 2 * sqrt(Z))
RIGHT = log(2 * sqrt(2)) + I * pi


def _value(e: Expr, digits: int = 20) -> complex:
    value = reliable_complex(e, digits)
    assert value is not None, e
    return value


def test_hidden_zeros_are_written_zero() -> None:
    assert reliable_form(L) == RIGHT
    assert reliable_form(sqrt(Z) + 1) == 1
    for digits in (15, 20, 30, 50):
        assert abs(_value(L, digits) - _value(RIGHT)) < 1e-12


def test_an_undecided_sum_is_refused_where_it_matters() -> None:
    assert reliable_form(sqrt(W) + 1) is None
    assert reliable_value(log(-1 - sqrt(W)), 20) is None
    # a value which is itself zero is not exposed to a cut
    assert reliable_form(W) is not None
    assert reliable_form(asin(Integer(-1)) + pi / 2) is not None


def test_a_sum_which_cancels_exactly_in_binary_is_a_suspect() -> None:
    # mpmath evaluates Z, and log(2) + log(3) - log(6), to an exact 0.0 at
    # twenty digits (to a rounding error at other precisions): an exact zero
    # of a sum whose terms are not zero is no evidence. Z is proved zero,
    # the sum of logarithms is not (and sqrt of it is refused)
    assert numeric_value(Z, dps_to_prec(20) + 20) == 0
    assert reliable_form(Log(-1 - sqrt(Z))) == I * pi
    logarithms = log(Integer(2)) + log(Integer(3)) - log(Integer(6))
    assert numeric_value(logarithms, dps_to_prec(20) + 20) == 0
    assert reliable_form(sqrt(logarithms) + 1) is None
    assert reliable_form(logarithms) is not None


def test_a_part_which_is_exactly_zero() -> None:
    base = I * (log(1 + 3 * I) - log(1 - 3 * I)) / 2
    third = Rational(1, 3)
    assert abs(_value(base**third - atan(Integer(-3))**third, 30)) < 1e-25
    # a small part which is there is not a rounding error
    assert reliable_value(log(-1 + I * exp(Integer(-200))), 20) is not None
    assert abs(_value(sqrt(exp(Integer(-200)) - exp(Integer(-201)))) - 2.9577e-44) < 1e-47


def test_a_small_value_which_needs_more_precision_is_kept() -> None:
    # at the twenty digits asked for, (1 + exp(-100)) - 1 cancels to noise;
    # the evaluations at more digits agree on exp(-100): it is not a
    # rounding error (SymPy's evalf raises the precision by itself)
    tiny = (1 + exp(Integer(-100))) - 1
    value = _value(Log(-1 + I * tiny))
    assert abs(value.imag - 3.141592653589793) < 1e-12


def test_symbols_and_their_values() -> None:
    e = Log(-2 * sqrt(2) * x - 2 * sqrt(Z))
    value = reliable_complex(e, 20, {x: Integer(3)})
    assert value is not None and abs(value - _value(log(6 * sqrt(2)) + I * pi)) < 1e-12
    assert reliable_value(e, 20) is None             # not a number
    assert reliable_complex(sqrt(x), 15, {x: Integer(-4)}) == 2j


def test_extreme_magnitudes_are_cheap() -> None:
    # the bug: mpmath converts the denominator of 1e-1737200, an integer of
    # 5.8 million bits, in about a minute; the floats are now built from
    # their mantissa and exponent
    start = time.perf_counter()
    value = reliable_value(exp(-1000000 * x**2 - 50), 30, {x: Integer(2)})
    assert str(value) == '2.27857274099872593085040751845e-1737200'
    assert time.perf_counter() - start < 10


def test_a_value_which_is_zero_is_not_refined_forever() -> None:
    # x*gamma(x) - gamma(x + 1) is zero: the evaluations at doubling
    # precisions never agree, and stop at four times the precision (the
    # first version doubled four times: gamma at 2400 bits, 1.4 s)
    start = time.perf_counter()
    value = reliable_complex(x * gamma(x) - gamma(x + 1), 40, {x: Rational(-7, 50)})
    assert value is not None and abs(value) < 1e-40
    assert time.perf_counter() - start < 1.0
