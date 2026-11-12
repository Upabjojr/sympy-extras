"""Tests of the ported handlers of the predicates and of the last steps
of the core's ``ask`` (:mod:`sympy_extras.assumptions.handlers`)."""
from __future__ import annotations

from sympy_extras.core import I, Integer, Q, Rational, Symbol, ask, pi, sqrt
from sympy_extras.functions import Abs, acos, asin, atan, exp, factorial, log, re
from sympy_extras.functions.elementary.exponential import ExpPolar


def test_real_and_imaginary() -> None:
    x, k = Symbol('x'), Symbol('k')
    assert ask(Q.real(re(k) + 1)) is True
    assert ask(Q.real(Abs(x) - 1)) is True
    assert ask(Q.real(I * x), Q.real(x) & Q.nonzero(x)) is False
    assert ask(Q.imaginary(I * x), Q.real(x) & Q.nonzero(x)) is True
    assert ask(Q.real(log(x)), Q.positive(x)) is True
    assert ask(Q.real(exp(x)), Q.real(x)) is True


def test_real_root_of_a_base_of_unknown_sign() -> None:
    # SymPy's handler answers True (ask(Q.real(sqrt(x)), Q.real(x)) and
    # ask(Q.real(sqrt(-1 - x**2)), Q.real(x)) in SymPy 1.14): wrong at
    # x = -1, where both roots are imaginary (the second one is, for every
    # real x)
    x = Symbol('x')
    assert ask(Q.real(sqrt(x)), Q.real(x)) is None
    assert ask(Q.real(sqrt(-1 - x**2)), Q.real(x)) is False
    assert ask(Q.real(sqrt(x)), Q.positive(x)) is True
    assert ask(Q.real(x**3), Q.real(x)) is True


def test_order_of_functions() -> None:
    x = Symbol('x')
    assert ask(Q.positive(log(x)), Q.positive(x) & Q.negative(x - 1)) is False
    assert ask(Q.positive(log(x)), Q.real(x) & Q.positive(x - 1)) is True
    assert ask(Q.positive(atan(x)), Q.positive(x)) is True
    assert ask(Q.positive(asin(x)), Q.positive(x) & Q.nonpositive(x - 1)) is True
    assert ask(Q.positive(acos(x)), Q.nonpositive(x - 1) & Q.nonnegative(x + 1)) is True
    n = Symbol('n')
    assert ask(Q.positive(factorial(n)), Q.integer(n) & Q.positive(n)) is True
    assert ask(Q.negative(x * Symbol('y')), Q.positive(x) & Q.negative(Symbol('y'))) is True


def test_number_theory() -> None:
    n, m = Symbol('n'), Symbol('m')
    assert ask(Q.even(2 * n), Q.integer(n)) is True
    assert ask(Q.odd(n + m), Q.odd(n) & Q.even(m)) is True
    assert ask(Q.prime(n**2), Q.integer(n)) is False
    assert ask(Q.integer(n * m), Q.integer(n) & Q.integer(m)) is True
    assert ask(Q.rational(n / 3), Q.integer(n)) is True


def test_numbers_below_the_precision_are_not_decided() -> None:
    # exp_polar(I*pi) evaluates to -1 + 1e-44*I: the imaginary part is
    # below the precision, it was taken as nonzero (Q.negative False,
    # where SymPy answers True)
    e = ExpPolar(I * pi)
    assert ask(Q.negative(e), Q.real(Symbol('x'))) is not False
    assert ask(Q.positive(sqrt(2) - 1)) is True and ask(Q.negative(Rational(1, 3) - pi)) is True
    assert ask(Q.integer(sqrt(2) + 1)) is False and ask(Q.real(Integer(2) + 3 * I)) is False


def test_sat_step() -> None:
    x = Symbol('x')
    assert ask(Q.nonzero(x), Q.positive(x) | Q.negative(x)) is True
    assert ask(Q.zero(Abs(x)), Q.zero(x)) is True


def test_numbers_equal_to_integers() -> None:
    # (2*sqrt(2) + 3)/(1 + sqrt(2))**2 is 1: proved by its minimal
    # polynomial (SymPy proves it with equals)
    one = (2 * sqrt(2) + 3) / (1 + sqrt(2))**2
    assert ask(Q.integer(one)) is True and ask(Q.odd(one)) is True
    assert ask(Q.even(one)) is False and ask(Q.prime(one)) is False
    seven = (sqrt(2) + 1) * (sqrt(2) - 1) * 7
    assert ask(Q.prime(seven)) is True


def test_signs_of_scaled_subjects() -> None:
    # Q.positive(a*(1 - a)) under Q.negative(a - 1) with a positive was not
    # decided (SymPy decides it by lra_satask): 1 - a is -(a - 1)
    a = Symbol('a', positive=True)
    assert ask(Q.positive(a * (1 - a)), Q.real(a) & Q.negative(a - 1)) is True
    assert ask(Q.negative(a * (1 - a)), Q.real(a) & Q.negative(a - 1)) is False
    x = Symbol('x')
    assert ask(Q.nonpositive(2 - 2 * x), Q.nonnegative(x - 1)) is True
    assert ask(Q.positive(x + 1), Q.negative(x - 1)) is None


def test_algebraic_zero_is_proved() -> None:
    # (-12 - 8*sqrt(2))*(2*sqrt(2) + 3) + 24*sqrt(2) + 34 + 2*(2*sqrt(2) + 3)**2
    # is 0: Q.zero was not decided (SymPy proves it with equals)
    r = sqrt(2)
    e = (-12 - 8 * r) * (2 * r + 3) + 24 * r + 34 + 2 * (2 * r + 3)**2
    assert ask(Q.zero(e)) is True and ask(Q.nonzero(e)) is False
