"""The order in which ``sstr`` (and with it the LaTeX and the pretty
printer) writes the terms of an evaluated sum and the factors of an
evaluated product is SymPy's.

Regression tests of the differences found while the printers were
unified: every expression below was printed in another order than
SymPy's ``str`` before the fix of ``ordered_terms``."""
from __future__ import annotations

import random

from sympy.testing.pytest import raises

from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.core import Add, Derivative, Float, Function, I, Integer, Mul, Rational, Symbol, pi, sqrt, sstr
from sympy_extras.core.expr import Expr, factors_of, terms_of
from sympy_extras.core.limits import Limit
from sympy_extras.core.printing import ordered_terms, printed_factors, printed_terms
from sympy_extras.functions.elementary.exponential import exp, log
from sympy_extras.functions.elementary.trigonometric import cos, sin
from sympy_extras.interop import to_sympy

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
A, B = Symbol('A', commutative=False), Symbol('B', commutative=False)
f = Function('f')


def _same_as_sympy(e: Expr) -> bool:
    return sstr(e) == str(to_sympy(e))


def test_a_power_with_a_rational_coefficient_in_its_exponent() -> None:
    # the generator of exp(-2*z/3) is exp(z/3), with the degree -2 (SymPy's
    # decompose_power): it was the power itself, with the degree 1
    assert sstr(exp(-2*z/3) + cos(x)) == 'cos(x) + exp(-2*z/3)'
    assert sstr(exp(-z/3) + cos(x)) == 'cos(x) + exp(-z/3)'
    assert sstr(x**(-2*z/3) + cos(x)) == 'cos(x) + x**(-2*z/3)'
    assert sstr(exp(-y/2) + sin(x)) == 'sin(x) + exp(-y/2)'
    assert sstr(exp(2*z/3) + cos(x)) == 'exp(2*z/3) + cos(x)'
    # the sign of a float is taken out (SymPy's as_coeff_Mul(rational=True))
    assert sstr(exp(-2.5*z) + cos(x)) == 'cos(x) + exp(-2.5*z)'
    for e in (exp(-2*z/3) + cos(x), exp(-2*z/3) + exp(z/3) + exp(-z/3) + 1, x**(y/2) + x**y + x**(-y/2) + x**(3*y/2),
              sqrt(x)*exp(-x/3) + exp(-2*x/3) + x, exp(Rational(-2, 3)*x*y) + exp(x*y/3) + x*y,
              exp(-2.5*z) + cos(x), exp(2.5*z) + cos(x), exp(-pi*z/3) + cos(x)):
        assert _same_as_sympy(e), e


def test_a_factor_which_is_not_known_to_commute() -> None:
    # the commutativity of a Limit is unknown (None, in SymPy too): SymPy's
    # as_terms and args_cnc put it with the noncommutative factors, the
    # test here was "is False"
    limit = Limit(f(x), x, Integer(0))
    assert limit.is_commutative is None
    assert sstr(limit + 1) == "1 + Limit(f(x), x, 0, dir='+')"
    assert sstr(limit + x) == "x + Limit(f(x), x, 0, dir='+')"
    series = Sum(f(x), (x, Integer(0), Integer(1)))
    assert sstr(limit*series) == "Sum(f(x), (x, 0, 1))*Limit(f(x), x, 0, dir='+')"
    for e in (limit + 1, limit*y + y, limit*series, 2*limit*series*y, limit + series,
              limit + 2*Limit(f(y), y, Integer(0))**2 + 1 + y):
        assert _same_as_sympy(e), e


def test_a_factor_without_free_symbols_and_without_a_value_is_a_generator() -> None:
    # Sum(f(k), (k, 0, 1)) has no free symbols and no numerical value:
    # SymPy's as_terms makes it a generator, so that a number comes after it
    series = Sum(f(y), (y, Integer(0), Integer(1)))
    assert sstr(sqrt(Integer(2))*series - Rational(4, 3)) == 'sqrt(2)*Sum(f(y), (y, 0, 1)) - 4/3'
    assert sstr(Float(2.5)*series - pi) == '2.5*Sum(f(y), (y, 0, 1)) - pi'
    for e in (sqrt(Integer(2))*series - Rational(4, 3), Float(2.5)*series - pi, series + 1,
              -y*series - 3*sqrt(Integer(2))*Sum(f(z), (z, Integer(0), Integer(1))) + series*A,
              Derivative(f(x), x)/2 - Float(2.5)*Derivative(f(x), x)*series*A + cos(y)):
        assert _same_as_sympy(e), e


def test_a_number_before_a_negative_float_multiple() -> None:
    # SymPy's special case of two terms (1 - x) holds for a float coefficient too
    assert sstr(Float(6.25) - Float(2.5)*x) == '6.25 - 2.5*x'
    assert sstr(1 - x) == '1 - x' and sstr(-pi*x + 1) == '-pi*x + 1'
    assert _same_as_sympy(Float(6.25) - Float(2.5)*x) and _same_as_sympy(Float(1.25) - Float(3.5)*x**2)


def _random_term(rng: random.Random) -> Expr:
    coefficients: list[Expr] = [Integer(1), Integer(-1), Integer(2), Rational(-2, 3), Rational(1, 2), Integer(-3),
                                Float(2.5), Float(-2.5), pi, sqrt(Integer(2)), I]
    term = rng.choice(coefficients)
    for _ in range(rng.randint(1, 3)):
        v = rng.choice([x, y, z])
        kind = rng.random()
        factor: Expr
        if kind < 0.2:
            factor = v
        elif kind < 0.35:
            factor = v**rng.choice([Integer(2), Integer(-1), Rational(1, 2), Rational(-3, 2), Rational(2, 3), y, -y, 2*z,
                                    Rational(-2, 3)*z, Float(-2.5)*z])
        elif kind < 0.55:
            factor = exp(rng.choice(coefficients)*rng.choice([x, y, z, x*y]))
        elif kind < 0.65:
            factor = rng.choice([sin(v), cos(v), log(v), f(v)])
        elif kind < 0.72:
            factor = Limit(f(v), v, Integer(0))
        elif kind < 0.78:
            factor = Sum(f(v), (v, Integer(0), Integer(1)))
        elif kind < 0.84:
            factor = rng.choice([A, B])
        elif kind < 0.9:
            factor = Derivative(f(v), v)
        else:
            factor = rng.choice(coefficients)
        term = term*factor
    return term


def test_random_sums_are_printed_in_the_order_of_sympy() -> None:
    # (108 of 3000 such sums differed before the fix)
    rng = random.Random(20261123)
    different: list[str] = []
    for _ in range(400):
        e = _random_term(rng)
        for _ in range(rng.randint(1, 3)):
            e = e + _random_term(rng)
        if not _same_as_sympy(e):
            different.append(sstr(e))
    assert different == []


def test_printed_terms_and_factors() -> None:
    terms = terms_of(x + y**2 + x*y + 1)
    assert printed_terms(terms) == ordered_terms(terms) == [x*y, x, y**2, Integer(1)]
    assert printed_terms(terms, 'lex') == printed_terms(terms)
    assert printed_terms(terms, 'grlex') == [x*y, y**2, x, Integer(1)]
    assert printed_terms(terms, 'none') == list(terms)
    # an unevaluated sum keeps the order of its arguments
    assert printed_terms(Add(Integer(1), y, x).args) == [Integer(1), y, x]
    factors = factors_of(2*y*x*sqrt(Integer(2)))
    assert printed_factors(factors) == [Integer(2), sqrt(Integer(2)), x, y]
    assert printed_factors(factors, 'none') == list(factors)
    assert printed_factors(Mul(y, x, Integer(2)).args) == [y, x, Integer(2)]
    raises(ValueError, lambda: printed_terms(terms, 'wrong'))
