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


# an even negative power of bounds with zero inside: from the smaller endpoint value up; SymPy goes
# through 1/bounds, the whole line, and has (0, oo) (sympy/sympy#31275, V222)
TIGHTER = {('AccumBounds(-1, 1)', 'q**(-2)'): 'AccumBounds(1, oo)',
           ('AccumBounds(-pi, pi/4)', 'q**(-2)'): 'AccumBounds(pi**(-2), oo)'}


def test_functions_of_bounds_match_sympy() -> None:
    for b in BOUNDS:
        for text in TEMPLATES:
            template = s.sympify(text)
            want = TIGHTER.get((str(b), text), str(template.subs(q, b)))
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


def test_even_negative_power_of_bounds_with_zero_inside() -> None:
    # sympy/sympy#31275, V222: the values of a**(-2) on [-1, 3] without 0 are [1/9, oo)
    from sympy_extras.core.accumbounds import accum_bounds
    from sympy_extras.core.numbers import Integer, Rational, oo
    assert accum_bounds(-1, 3)**(-2) == accum_bounds(Rational(1, 9), oo)
    assert accum_bounds(-2, 1)**(-4) == accum_bounds(Rational(1, 16), oo)
    assert accum_bounds(-oo, 1)**(-2) == accum_bounds(Integer(0), oo)
    assert accum_bounds(0, 3)**(-2) == accum_bounds(Rational(1, 9), oo)
    # an odd power takes every value but those near zero: the whole line stays
    assert accum_bounds(-1, 2)**(-3) == accum_bounds(-oo, oo)
