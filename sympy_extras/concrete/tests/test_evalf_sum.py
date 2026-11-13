"""The native numerical values of sums (the port of SymPy's evalf_sum),
against SymPy's and against known values."""
from __future__ import annotations

import mpmath
import sympy

from sympy.testing.pytest import raises

from sympy_extras.core import Integer, Rational, Symbol, oo, pi, sqrt
from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.functions import binomial, exp, factorial, sin

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


def test_sums_without_a_native_rule_are_left_to_the_caller() -> None:
    # SymPy's Euler-Maclaurin summation is not ported
    assert Sum(i / factorial(i), (i, 0, oo))._numeric_node(53) is None
    assert Sum(exp(-i**2), (i, 0, oo))._numeric_node(53) is None
