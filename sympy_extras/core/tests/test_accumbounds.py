"""Oracle tests of the accumulation bounds against SymPy."""
import sympy

from sympy_extras.interop import expr_from_sympy, to_sympy
from sympy_extras.core.symbol import Symbol

s = sympy
AB = s.AccumBounds
pi = s.pi
q = s.Symbol('q')
Q = Symbol('q')

BOUNDS = [AB(-1, 1), AB(0, pi), AB(pi/3, 2*pi/3), AB(0, s.oo), AB(-s.oo, s.oo), AB(1, 2), AB(-pi, pi/4),
          AB(pi/2, 3*pi/2), AB(2, 7), AB(-3, -1), AB(0, 1), AB(s.Rational(1, 2), 4)]
TEMPLATES = ['sin(q)', 'cos(q)', 'tan(q)', 'cot(q)', 'exp(q)', 'log(q)', 'atan(q)', 'q**2', 'q**3', '1/q', 'q**(-2)',
             'Abs(q)', 'sqrt(q)', '2**q', 'q + 4', '3*q', 'q - 1/2', '-q', 'frac(q)']


def test_functions_of_bounds_match_sympy() -> None:
    for b in BOUNDS:
        for text in TEMPLATES:
            template = s.sympify(text)
            want = str(template.subs(q, b))
            if text == 'q**(-2)' and s.Lt(b.args[0], 0) == s.true and s.Gt(b.args[1], 0) == s.true:
                # SymPy loses the lower bound when 0 is inside
                # (sympy/sympy#31275, V34): test_negative_even_power_around_zero
                continue
            got = str(to_sympy(expr_from_sympy(template).subs(Q, expr_from_sympy(b)), evaluate=False))
            assert got == want, (b, text)


def test_arithmetic_of_bounds_matches_sympy() -> None:
    for a in BOUNDS:
        for b in BOUNDS[:6]:
            for op in (lambda u, v: u + v, lambda u, v: u - v, lambda u, v: u * v, lambda u, v: u / v):
                try:
                    want = str(op(a, b))
                except Exception as error:
                    want = type(error).__name__
                try:
                    got = str(to_sympy(op(expr_from_sympy(a), expr_from_sympy(b)), evaluate=False))
                except Exception as error:
                    got = type(error).__name__
                assert got == want, (a, b)


def test_nan_in_a_sum_of_bounds() -> None:
    from sympy_extras.core.add import add
    from sympy_extras.core.numbers import Integer, nan
    from sympy_extras.functions import log
    b = expr_from_sympy(AB(-1, 1))
    # the sum recursed without end: nan was not absorbed in the
    # coefficient of the bounds (SymPy's Add.flatten returns nan at once)
    assert add(nan, b) == nan and add(nan, -b, Integer(1)) == nan
    assert str(to_sympy((log(Q) - Q + 1).subs(Q, expr_from_sympy(AB(-1, 1))))) == str(s.nan)


def test_evalf_of_a_function_of_bounds() -> None:
    b = expr_from_sympy(AB(0, s.oo))
    from sympy_extras.functions import atan
    # evalf recursed without end on the product -atan(AccumBounds(0, oo)),
    # a number without a value; SymPy leaves it as it is
    e = b - atan(b)
    assert str(e.evalf(30)) == 'AccumBounds(0, oo) - atan(AccumBounds(0, oo))'
    assert str((-atan(b)).evalf(30)) == str((-s.atan(AB(0, s.oo))).evalf(30)) == '-atan(AccumBounds(0, oo))'


def test_negative_even_power_around_zero() -> None:
    # sympy/sympy#31275, V34: on [-1, 3] without 0 the infimum of x**(-2) is
    # 1/9, at x = 3; SymPy answers AccumBounds(0, oo)
    from sympy_extras.core import Integer, Rational, oo
    from sympy_extras.core.accumbounds import accum_bounds
    assert accum_bounds(-1, 3)**(-2) == accum_bounds(Rational(1, 9), oo)
    assert accum_bounds(-2, 3)**(-4) == accum_bounds(Rational(1, 81), oo)
    assert accum_bounds(-3, 1)**(-2) == accum_bounds(Rational(1, 9), oo)
    assert accum_bounds(0, 3)**(-2) == accum_bounds(Rational(1, 9), oo)
    assert accum_bounds(-oo, oo)**(-2) == accum_bounds(0, oo)
    assert accum_bounds(-1, 3)**(-3) == accum_bounds(-oo, oo)
    # the values on a grid lie in the bounds, and reach the lower one
    values = [Rational(k, 100)**(-2) for k in range(-100, 301) if k != 0]
    assert Rational(1, 9) in values and Integer(1) in values
    assert all((v - Rational(1, 9)).is_nonnegative is True for v in values)
