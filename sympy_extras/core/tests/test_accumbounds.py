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
