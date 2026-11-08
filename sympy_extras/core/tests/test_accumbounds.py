"""Oracle tests of the accumulation bounds against SymPy."""
import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
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
