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
from sympy_extras.core.order import Order
from sympy_extras.core.printing import PRINT_ORDERS, ordered_terms, printed_factors, printed_terms
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


#: the values of the setting ``order`` of SymPy's printers
_ORDERS = ['lex', 'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex', 'rev-lex', 'rev-grlex', 'rev-grevlex', 'rev-ilex',
           'rev-igrlex', 'rev-igrevlex', 'old', 'none']


def _order_corpus() -> list[Expr]:
    w = Symbol('w')
    series = x - x**3/6 + x**5/120 + Order(x**6)
    return [x + y**2 + x*y + 1, x**2 + y**2 + z**2 + x*y*z + x + y + z + 1, x**3*y + x*y**3 + x**2*y**2 + z**4 + x*z + 3,
            2*x**2 - 3*x*y + y - Rational(1, 2), sin(x) + x**2 + exp(y)*x + 1, x + I*x + sqrt(Integer(2))*x + y, 1 - x,
            1 - 2*x**2, x**2 + x + Order(x**3), series, A*B + B*A + x*A + 1, x/y + y/x + 1/(x*y) + x*y,
            sqrt(x) + x**Rational(3, 2) + 1/x + x, (x + 1)**2 + (x + 1) + y*(x + 1)**3, x*y*z + x*y + y*z + x*z + w**3,
            pi + exp(Integer(1)) + x + sqrt(Integer(2)), x**y + y**x + x + y, f(x) + f(y)**2 + x*f(x) + 1, -x - y - 1,
            x**2*y**2*z + x*y*z**3 + x**3 + y**4*z, exp(x) + exp(2*x) + exp(-x), 3*x*y**2 - Float(2.5)*x**2*y + x - 7,
            x + 2*y + 3*z + 4*w, log(x)*x + log(x)**2 + x**2, (x*y + 1)*(z + x**2), sin(x + y**2 + x*y),
            x*y**2*sin(x)**3 + x**2*y + sin(x), Rational(1, 2) + x/3 + y**2/5, x*exp(y) + sin(x) + x**2 + 1]


def test_the_orders_of_sympy_printers() -> None:
    # the setting `order` of SymPy's printers: sstr had none, and printed_terms knew None, 'lex', 'grlex', 'grevlex'
    # and 'none' only ('rev-lex', 'old', ... raised ValueError), with the special case 1 - x under 'lex' as well
    # (SymPy has it for order=None only: -x + 1 under 'lex')
    import sympy
    for e in _order_corpus():
        reference = to_sympy(e)
        for order in _ORDERS:
            assert sstr(e, order=order) == sympy.sstr(reference, order=order), (str(e), order)
    assert sstr(1 - x) == '1 - x' and sstr(1 - x, order='lex') == '-x + 1' and sstr(1 - x, order='rev-lex') == '1 - x'
    e = x**2 + y**2 + z**2 + x*y*z + x + y + z + 1
    assert sstr(e, order='grlex') == 'x*y*z + x**2 + y**2 + z**2 + x + y + z + 1'
    assert sstr(e, order='rev-grlex') == '1 + z + y + x + z**2 + y**2 + x**2 + x*y*z'
    assert sstr(e, order='old') == '1 + x + y + z + x*y*z + x**2 + y**2 + z**2'
    # with an order term the powers increase whatever the order (as in SymPy)
    series = x - x**3/6 + Order(x**5)
    assert sstr(series, order='rev-lex') == sstr(series, order='grlex') == 'x - x**3/6 + O(x**5)'
    raises(ValueError, lambda: sstr(e, order='bogus'))
    raises(ValueError, lambda: printed_terms(terms_of(e), 'rev-old'))
    assert set(_ORDERS) | {None} == set(PRINT_ORDERS)


def test_random_sums_in_every_order() -> None:
    import sympy
    rng = random.Random(30844)
    different: list[tuple[str, str]] = []
    for _ in range(60):
        e: Expr = Integer(0)
        for _ in range(rng.randint(2, 6)):
            e = e + _random_term(rng)
        reference = to_sympy(e)
        for order in _ORDERS:
            if order in ('old', 'none') and e.has(Limit):
                # (these orders print the factors of a product as they are stored, and the core stores a Limit,
                # whose commutativity is unknown, before the commutative factors, SymPy after them)
                continue
            if sstr(e, order=order) != sympy.sstr(reference, order=order):
                different.append((sstr(e), order))
    assert different == []


def test_ordered_counts_the_nodes_as_sympy() -> None:
    # `ordered` counted every node 1: SymPy counts a float a half and a derivative as its function plus the number
    # of differentiations, so that the factors of these products were ordered otherwise (found by the 'old' order
    # of the printers, which takes the exponent of the last factor)
    import sympy
    from sympy_extras.core.sorting import ordered
    w = [exp(Float(-2.5)*x*y), x**(2*z), f(z), pi, Derivative(f(y), y), exp(z/2), log(y), Derivative(f(x), x, x), x*y*z]
    reference = [str(s) for s in sympy.ordered([to_sympy(e) for e in w])]
    assert [str(to_sympy(e)) for e in ordered(w)] == reference
    assert ordered([x**(2*z), exp(Float(-2.5)*x*y)])[-1] == x**(2*z)
    assert ordered([exp(z/2), Derivative(f(y), y)])[-1] == exp(z/2)


def test_old_order_takes_the_exponent_of_the_last_factor() -> None:
    # SymPy's 'old' order compares the exponents p3 of the matches of p1*p2**p3; its matcher gives the power the
    # last factor in the order of `ordered` (x*exp(y) is p1 = x, p2 = E, p3 = y), the core's matcher the first
    # one (p2 = x, p3 = 1): the order is computed without the matcher
    from sympy_extras.core.printing import _old_exponent
    assert _old_exponent(x*exp(y)) == y and _old_exponent(x*y**2*sin(x)**3) == 3 and _old_exponent(sqrt(x)*y) == Rational(1, 2)
    assert _old_exponent(Integer(1)) == 0 and _old_exponent(Rational(1, 2)) == 1 and _old_exponent(A*B) is None
    assert sstr(x*exp(y) + sin(x) + x**2 + 1, order='old') == '1 + sin(x) + x**2 + x*exp(y)'
