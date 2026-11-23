"""Tests of the ported handlers of the predicates and of the last steps
of the core's ``ask`` (:mod:`sympy_extras.assumptions.handlers`)."""
from __future__ import annotations

from sympy_extras.core import I, Integer, Q, Rational, Symbol, ask, pi, sqrt
from sympy_extras.functions import Abs, acos, asin, atan, exp, factorial, log, re
from sympy_extras.functions.elementary.exponential import ExpPolar


def test_real_and_imaginary() -> None:
    x, k = Symbol('x'), Symbol('k')
    assert ask(Q.real(re(k) + 1), Q.complex(k)) is True
    assert ask(Q.real(Abs(x) - 1), Q.complex(x)) is True
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


def test_product_with_a_factor_which_may_be_zero() -> None:
    # SymPy 1.14 answers ask(Q.real(x*y), Q.real(x) & Q.imaginary(y)) with
    # False (and True for Q.imaginary): x = 0 makes x*y = 0, which is real
    # and not imaginary. The same holds for an integer (rational) factor
    # times an irrational one, in the handlers and in the facts of the SAT
    # step
    x, y = Symbol('x'), Symbol('y')
    assert ask(Q.real(x*y), Q.real(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x*y), Q.real(x) & Q.imaginary(y)) is None
    assert ask(Q.real(x*y), Q.zero(x) & Q.imaginary(y)) is True
    assert ask(Q.imaginary(x*y), Q.zero(x) & Q.imaginary(y)) is False
    assert ask(Q.real(x*y), Q.nonzero(x) & Q.imaginary(y)) is False
    assert ask(Q.imaginary(x*y), Q.nonzero(x) & Q.imaginary(y)) is True
    assert ask(Q.integer(x*y), Q.integer(x) & Q.irrational(y)) is None
    assert ask(Q.integer(x*y), Q.zero(x) & Q.irrational(y)) is True
    assert ask(Q.integer(x*y), Q.integer(x) & Q.nonzero(x) & Q.irrational(y)) is False
    assert ask(Q.rational(x*y), Q.rational(x) & Q.irrational(y)) is None
    assert ask(Q.irrational(x*y), Q.rational(x) & Q.irrational(y)) is None
    assert ask(Q.irrational(x*y), Q.rational(x) & Q.nonzero(x) & Q.irrational(y)) is True


def test_order_of_functions() -> None:
    x = Symbol('x')
    assert ask(Q.positive(log(x)), Q.positive(x) & Q.negative(x - 1)) is False
    assert ask(Q.positive(log(x)), Q.real(x) & Q.positive(x - 1)) is True
    assert ask(Q.positive(atan(x)), Q.positive(x)) is True
    assert ask(Q.positive(asin(x)), Q.positive(x) & Q.nonpositive(x - 1)) is True
    # acos(1) = 0: positive on [-1, 1) only (SymPy 1.14 answers True on [-1, 1])
    assert ask(Q.positive(acos(x)), Q.nonpositive(x - 1) & Q.nonnegative(x + 1)) is None
    assert ask(Q.positive(acos(x)), Q.negative(x - 1) & Q.nonnegative(x + 1)) is True
    assert ask(Q.positive(acos(x)), Q.zero(x - 1)) is False
    assert ask(Q.positive(acos(x)), Q.positive(x - 1)) is False     # not real
    # asin(0) = 0, asin(1) = pi/2
    assert ask(Q.positive(asin(x)), Q.nonnegative(x) & Q.nonpositive(x - 1)) is None
    assert ask(Q.positive(asin(x)), Q.zero(x)) is False
    assert ask(Q.positive(asin(x)), Q.negative(x + 1)) is False     # not real
    # atan(oo) = pi/2 (SymPy 1.14 answers False), atan(0) = 0
    assert ask(Q.positive(atan(x)), Q.positive_infinite(x)) is True
    assert ask(Q.positive(atan(x)), Q.zero(x)) is False
    assert ask(Q.negative(atan(x)), Q.negative(x)) is True
    assert ask(Q.negative(atan(x)), Q.nonnegative(x)) is False
    assert ask(Q.real(atan(x) + atan(1/x)), Q.negative(x)) is True
    # log(1) = 0
    assert ask(Q.positive(log(x)), Q.zero(x - 1)) is False
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


def test_hidden_zero_has_no_sign() -> None:
    # -12*sqrt(2) + 4*sqrt(2)/(2*sqrt(2) + 3) + 16 is 0 and evaluates to
    # -1e-87: the product with another factor was taken for positive
    # (elliptic_integral of x/sqrt((x**2 + 1)*(x - 2)*(x - 3)) over (4, oo)
    # then returned a value with zoo instead of None)
    r = sqrt(2)
    zero = -12 * r + 4 * r / (2 * r + 3) + 16
    e = zero * (-12 * r + 8 * r / (4 * r + 9) + 16)
    assert ask(Q.positive(e)) is False and ask(Q.negative(e)) is False
    assert ask(Q.positive(zero)) is False and ask(Q.nonzero(zero)) is False


def test_sympy_issue_31275_functions_of_a_possibly_infinite_argument() -> None:
    # V5-V8, V28: SymPy answers True. exp(oo), Abs(oo) and re(oo) are oo,
    # log(0) and 0**(-1) are zoo: not complex, not real
    from sympy_extras.functions import cos, im, sin
    x, y = Symbol('x'), Symbol('y')
    for f in (exp(x), Abs(x), re(x), im(x), sin(x), cos(x), log(x)):
        assert ask(Q.complex(f)) is None, f
    for f in (exp(x), Abs(x), re(x), im(x), sin(x), cos(x)):
        assert ask(Q.complex(f), Q.complex(x)) is True, f
    assert ask(Q.complex(log(x)), Q.complex(x)) is None
    assert ask(Q.complex(log(x)), Q.positive(x)) is True
    for f in (Abs(x), re(x), im(x)):
        assert ask(Q.real(f)) is None, f
        assert ask(Q.real(f), Q.complex(x)) is True, f
    assert ask(Q.finite(Abs(x))) is None
    assert ask(Q.nonnegative(Abs(x))) is None
    assert ask(Q.extended_nonnegative(Abs(x))) is True
    assert ask(Q.nonnegative(Abs(x)), Q.complex(x)) is True
    assert ask(Q.complex(x**y), Q.complex(x) & Q.complex(y)) is None
    assert ask(Q.complex(x**y), Q.nonzero(x) & Q.complex(y)) is True
    assert ask(Q.complex(x**y), Q.complex(x) & Q.positive(y)) is True
    # (-1)**(1/2) is I
    assert ask(Q.extended_real(x**y), Q.extended_real(x) & Q.extended_real(y)) is None
    assert ask(Q.extended_real(x**y), Q.positive(x) & Q.real(y)) is True


def test_sympy_issue_31275_imaginary_sums_and_powers() -> None:
    # V17-V21: SymPy answers False with one real term (it may be 0) and True
    # without real terms (I - I is 0, which is not imaginary)
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    assert ask(Q.imaginary(x + I), Q.real(x)) is None
    assert ask(Q.imaginary(x + I), Q.imaginary(x)) is None
    assert ask(Q.imaginary(x + y), Q.imaginary(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x + y), Q.imaginary(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x + y + z), Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) is None
    assert ask(Q.imaginary(x + I), Q.nonzero(x)) is False
    assert ask(Q.imaginary(x + y), Q.imaginary(x) & Q.zero(y)) is True
    assert ask(Q.imaginary(x + y), Q.real(x) & Q.real(y)) is False
    assert ask(Q.imaginary(I*x + I*y), Q.positive(x) & Q.positive(y)) is True
    # V24: Q.integer(2*y) allows an integer y, and then x**y is real
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.rational(y) & Q.integer(2*y)) is None
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.integer(2*y) & ~Q.integer(y)) is True
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.integer(y)) is False
    # 5**x at the imaginary x = I*pi/log(5) is -1, at half of it I
    assert ask(Q.real(5**x), Q.imaginary(x)) is None
    assert ask(Q.imaginary(5**x), Q.imaginary(x)) is None
    assert ask(Q.real(I**x), Q.imaginary(x)) is True
    assert ask(Q.imaginary(I**x), Q.imaginary(x)) is False


def test_sympy_issue_31275_nonzero_power_and_inverse_functions() -> None:
    from sympy_extras.core import Pow
    from sympy_extras.functions import acot
    x, y, n = Symbol('x'), Symbol('y'), Symbol('n')
    # V26, V27: Q.nonzero is real and not zero; 5**(I*n) is 1 at n = 0 and
    # not real at the other integers (SymPy: False)
    assert ask(Q.nonzero(Pow(5, 2*I*n*pi)), Q.integer(n)) is None
    assert ask(Q.nonzero(Pow(5, I*n)), Q.integer(n)) is None
    assert ask(Q.nonzero(x**2), Q.nonzero(x)) is True
    assert ask(Q.nonzero(x**y), Q.positive(x) & Q.real(y)) is True
    assert ask(Q.nonzero(x**y), Q.zero(x) & Q.positive(y)) is False
    # V31: acot is odd, acot(-1) = -pi/4 (SymPy: True for every real x)
    assert acot(Integer(-1)) == -pi/4 and acot(Integer(0)) == pi/2
    assert ask(Q.positive(acot(x)), Q.real(x)) is None
    assert ask(Q.positive(acot(x)), Q.nonnegative(x)) is True
    assert ask(Q.positive(acot(x)), Q.negative(x)) is False
