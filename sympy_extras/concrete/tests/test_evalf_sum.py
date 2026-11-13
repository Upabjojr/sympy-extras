"""The native numerical values of sums (the port of SymPy's evalf_sum),
against SymPy's and against known values."""
from __future__ import annotations

import mpmath
import sympy

from sympy.testing.pytest import raises

from sympy_extras.core import Integer, Rational, Symbol, oo, pi, sqrt
from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.functions import binomial, exp, factorial, harmonic, log, sin

i = Symbol('i', integer=True)


def _value(e: Sum) -> mpmath.mpc:
    value = e._numeric_node(80)
    assert value is not None, e
    return value


def test_convergent_hypergeometric_sums() -> None:
    # the factorial, geometric and power rates; the values are known
    with mpmath.workprec(80):
        cases = [(Sum(1 / factorial(i), (i, 0, oo)), mpmath.e), (Sum(1 / i**2, (i, 1, oo)), mpmath.pi**2 / 6),
                 (Sum((-1)**i / i, (i, 1, oo)), -mpmath.log(2)), (Sum(Rational(1, 2)**i, (i, 0, oo)), mpmath.mpf(2)),
                 (Sum((-1)**i / (2 * i + 1), (i, 0, oo)), mpmath.pi / 4),
                 (Sum(binomial(2 * i, i) / 5**i, (i, 0, oo)), mpmath.sqrt(5)),
                 (Sum(2**i / factorial(i), (i, 3, oo)), mpmath.e**2 - 5), (Sum(i / 3**i, (i, 1, oo)), mpmath.mpf(3) / 4)]
        for e, known in cases:
            assert abs(_value(e) - known) < mpmath.mpf(10)**-20, e


def test_divergent_sums_raise_sympy_value_error() -> None:
    # Sum(1, (i, 1, oo)) went to SymPy's evalf through the bridge for its
    # ValueError
    for e, message in [(Sum(Integer(1), (i, 1, oo)), 'n^0'), (Sum(1 / i, (i, 1, oo)), 'n^-1'),
                       (Sum(2**i, (i, 0, oo)), '(2)^n'), (Sum(factorial(i), (i, 0, oo)), '(n!)^1'),
                       (Sum((-1)**i, (i, 0, oo)), 'n^0')]:
        raises(ValueError, lambda: e._numeric_node(53))
        try:
            e._numeric_node(53)
        except ValueError as error:
            assert str(error) == 'Sum diverges like ' + message, (e, error)


def test_finite_sums_and_the_empty_sum() -> None:
    with mpmath.workprec(80):
        assert _value(Sum(i**2, (i, 1, 10))) == 385
        assert abs(_value(Sum(sin(i), (i, 1, 5))) - sum(mpmath.sin(k) for k in range(1, 6))) < mpmath.mpf(10)**-20
        assert _value(Sum(1 / i, (i, 5, 4))) == 0


def test_irrational_ratios_are_not_truncated() -> None:
    # SymPy 1.14 truncates the values of the coefficients of the ratio to
    # integers: Sum(pi**i/factorial(i), (i, 0, oo)).evalf() is exp(3) =
    # 20.0855..., Sum(sqrt(2)**i/factorial(i), ...) is E (sympy-extras#25)
    with mpmath.workprec(80):
        assert abs(_value(Sum(pi**i / factorial(i), (i, 0, oo))) - mpmath.exp(mpmath.pi)) < mpmath.mpf(10)**-20
        assert abs(_value(Sum(sqrt(2)**i / factorial(i), (i, 0, oo))) - mpmath.exp(mpmath.sqrt(2))) \
            < mpmath.mpf(10)**-20
    s = sympy.Symbol('i', integer=True)
    assert abs(sympy.Sum(sympy.pi**s / sympy.factorial(s), (s, 0, sympy.oo)).evalf() - sympy.exp(3).evalf()) < 1e-12


def test_numeric_values_match_sympy() -> None:
    from sympy_extras.interop import expr_from_sympy
    s = sympy.Symbol('i', integer=True)
    for e in [sympy.Sum(1 / (s**2 + 1), (s, 0, sympy.oo)), sympy.Sum(1 / (s * (s + 1)), (s, 1, sympy.oo)),
              sympy.Sum(1 / (s + sympy.Rational(1, 2))**2, (s, 0, sympy.oo)),
              sympy.Sum(sympy.Rational(3, 7)**s * (s + 1)**2, (s, 0, sympy.oo)), sympy.Sum(sympy.cos(s), (s, 2, 9))]:
        native = expr_from_sympy(e)
        assert isinstance(native, Sum)
        value = native._numeric_node(80)
        assert value is not None
        assert abs(complex(value) - complex(e.evalf(25))) < 1e-15, e


def test_euler_maclaurin_sums() -> None:
    # these went to SymPy's evalf through the bridge (the Euler-Maclaurin
    # summation was not ported, the native value was None)
    with mpmath.workprec(80):
        theta = (1 + mpmath.jtheta(3, 0, mpmath.exp(-1))) / 2
        cases = [(Sum(i / factorial(i), (i, 0, oo)), mpmath.e), (Sum(exp(-i**2), (i, 0, oo)), theta),
                 (Sum(1 / i**Rational(3, 2), (i, 1, oo)), mpmath.zeta(1.5)),
                 (Sum(harmonic(i) / i**3, (i, 1, oo)), mpmath.pi**4 / 72),
                 (Sum(log(i) / i**3, (i, 1, oo)), -mpmath.zeta(3, derivative=1)),
                 (Sum(1 / i, (i, 1, 1000)), mpmath.harmonic(1000)),
                 (Sum(1 / (i**2 + log(i)), (i, 1, oo)),
                  mpmath.nsum(lambda k: 1 / (k**2 + mpmath.log(k)), [1, mpmath.inf], method='euler-maclaurin'))]
        for e, known in cases:
            assert abs(_value(e) - known) < mpmath.mpf(10)**-18, e


def test_euler_maclaurin_symbolic() -> None:
    # SymPy's docstring examples of Sum.euler_maclaurin
    k = Symbol('k', integer=True)
    b = Symbol('b')
    s, e = Sum(1 / k, (k, 2, 5)).euler_maclaurin()
    assert abs(complex(s.evalf()) - 1.26629073187415) < 1e-12 and e == Rational(7, 400)
    assert Sum(k, (k, 2, b)).euler_maclaurin() == (b**2 / 2 + b / 2 - 1, 0)
    # a reversed range is the negated sum over the swapped one; an empty one is zero
    assert Sum(k, (k, 5, 4)).euler_maclaurin() == (0, 0)
    assert Sum(k**2, (k, 5, 1)).euler_maclaurin() == (Rational(-86, 3), Rational(1, 3))
    assert Sum(k**2, (k, 5, 1)).euler_maclaurin(m=10) == (-Sum(k**2, (k, 2, 4)).doit(), 0)


def test_zero_term_ratio_pole_is_summed() -> None:
    # SymPy's hypsum divides by zero (i**2/(i**4 + 1) from i = 0: the first
    # term is zero, the term ratio has a pole at 0), sympy-extras#25
    with mpmath.workprec(80):
        known = mpmath.nsum(lambda k: k**2 / (k**4 + 1), [1, mpmath.inf])
        assert abs(_value(Sum(i**2 / (i**4 + 1), (i, 0, oo))) - known) < mpmath.mpf(10)**-18


def test_symmetric_finite_sum_is_not_zero() -> None:
    # SymPy's evalf gives 0 for Sum(exp(-i**2), (i, -50, 50)) (its
    # Euler-Maclaurin integral over (-50, 50) evaluated as 0, sympy-extras#25)
    with mpmath.workprec(80):
        known = mpmath.jtheta(3, 0, mpmath.exp(-1))
        assert abs(_value(Sum(exp(-i**2), (i, -50, 50))) - known) < mpmath.mpf(10)**-18


def test_wide_finite_sums_with_a_peak() -> None:
    # SymPy's Euler-Maclaurin summation stops at the first term below eps
    # (exp(-250000) at i = -500) and gives 0; over (-5000, 5000) it would
    # give the integral sqrt(pi): the terms are added one by one up to
    # 4000 of them, and a summand with a peak is not summed by
    # Euler-Maclaurin at all
    with mpmath.workprec(80):
        known = mpmath.jtheta(3, 0, mpmath.exp(-1))
        assert abs(_value(Sum(exp(-i**2), (i, -500, 500))) - known) < mpmath.mpf(10)**-18
    assert Sum(exp(-(i - 3)**2), (i, -5000, 5000))._numeric_node(53) is None


def test_inaccurate_sums_are_left_to_the_caller() -> None:
    # an oscillating summand: the quadrature of the tail does not reach the
    # precision, so no value (SymPy prints a few digits of a wrong value)
    assert Sum(sin(i) / i**2, (i, 1, oo))._numeric_node(53) is None
