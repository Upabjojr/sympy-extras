"""The native parser of sympify against SymPy's parse_expr (evaluating)."""
from __future__ import annotations

import pytest
import sympy
from sympy.parsing.sympy_parser import parse_expr

from sympy_extras.core.parsing import ParseUnsupported, parse
from sympy_extras.interop import to_sympy

TEXTS = ['(x**2 - 1)/(x - 1)', 'Abs(x)', 'cos(3)', 'cos(a_)', 'exp(-1)', 'exp(-x)', 'exp(2*x)', 'f(2*x + 4*y)', 'f(x)',
         'gamma(x + 1)', 'log(1/(4096*x**4))', 'log(2)', 'log(3*z/8 - 3*pi/8)', 'log(p*q)', 'sin(x)**2 + cos(x)**2',
         'sin(x)*cos(x)', 'x + I*y', 'x**2 + 1', '2/3', '1/2 + 1/3', '-x**2', '2**-1', '2**3**2', '-2**2', 'x - y - z',
         'x/y/z', '2.5*x', '1e-3', '.5', '3.', 'log(8, 2)', 'sqrt(8)', 'besselj(0, x)', 'Max(x, 2, 3)', 'oo - oo',
         'E**(I*pi)', 'zoo', 'x*(y + 1)**2', '+x', '--x', 'x**-2', 'sin(cos(tan(x)))', 'f(x, y)', 'GoldenRatio**2',
         'EulerGamma + Catalan', 'x - 2*x', 'nan', '(x)', 'x*2*y', 'atan2(y, x)', 'binomial(n, k)', 'factorial(5)',
         'Rational(1, 2)', 'Integer(7)']


def test_parse_matches_sympy() -> None:
    for text in TEXTS:
        assert sympy.srepr(to_sympy(parse(text), evaluate=False)) == sympy.srepr(parse_expr(text)), text


def test_outside_the_grammar() -> None:
    for text in ['not an expression', 'x if y else z', 'x +', '(x', 'x == y', "f'(x)", 'lambda x: x']:
        with pytest.raises(ParseUnsupported):
            parse(text)
