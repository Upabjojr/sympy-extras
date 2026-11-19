"""The Rust port of ``integrals/heurisch.py`` against the Python one on
the integrands of its tests (read with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.parsing import parse
from sympy_extras.integrals.heurisch import heurisch_antiderivative, heurisch_cases

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


CASES: list[tuple[str, str, str]] = [
    ('antiderivative', '1/(p**2 + x**2)', 'x'),
    ('antiderivative', '1/(-p**2 + x**2)', 'x'),
    ('antiderivative', 'x/(p**2 + x**2)**2', 'x'),
    ('antiderivative', '1/(x**3 + 1)', 'x'),
    ('antiderivative', '1/(x**4 + 1)', 'x'),
    ('antiderivative', 'x**2/(x**4 + 1)', 'x'),
    ('antiderivative', '(x**2 + 1)/(x**4 + x**2 + 1)', 'x'),
    ('antiderivative', '1/(x**6 + 1)', 'x'),
    ('antiderivative', '1/(x*(x**2 + 1))', 'x'),
    ('antiderivative', '1/sqrt(p**2 + x**2)', 'x'),
    ('antiderivative', '1/sqrt(p**2 - x**2)', 'x'),
    ('antiderivative', 'x/sqrt(p**2 + x**2)', 'x'),
    ('antiderivative', 'sqrt(p**2 - x**2)', 'x'),
    ('antiderivative', 'x**2/sqrt(1 - x**2)', 'x'),
    ('antiderivative', 'sqrt(1 - x**2)/x**2', 'x'),
    ('antiderivative', 'x*sqrt(x**2 - 1)', 'x'),
    ('antiderivative', '1/sqrt(-x**2 + 2*x)', 'x'),
    ('antiderivative', 'x**(1/3)/(x + 1)', 'x'),
    ('antiderivative', 'sqrt(x)/(x + 1)', 'x'),
    ('antiderivative', 'exp(a*x)', 'x'),
    ('antiderivative', 'x*exp(a*x)', 'x'),
    ('antiderivative', 'x**2*exp(a*x)', 'x'),
    ('antiderivative', '1/(exp(x) + 1)', 'x'),
    ('antiderivative', 'exp(x)*sin(x)', 'x'),
    ('antiderivative', 'exp(a*x)*cos(b*x)', 'x'),
    ('antiderivative', 'x**3*exp(-x**2)', 'x'),
    ('antiderivative', 'exp(sqrt(x))/sqrt(x)', 'x'),
    ('antiderivative', 'sinh(x)**2', 'x'),
    ('antiderivative', 'cosh(x)**3', 'x'),
    ('antiderivative', '1/(cosh(x) + 1)', 'x'),
    ('antiderivative', 'sinh(x)/(cosh(x)**2 + 1)', 'x'),
    ('antiderivative', 'x*sinh(x)', 'x'),
    ('antiderivative', 'exp(x)*sinh(x)', 'x'),
    ('antiderivative', 'sin(x)**2', 'x'),
    ('antiderivative', '1/sin(x)', 'x'),
    ('antiderivative', '1/(cos(x) + 1)', 'x'),
    ('antiderivative', '1/(cos(x) + 2)', 'x'),
    ('antiderivative', 'sin(x)**4', 'x'),
    ('antiderivative', 'sin(x)**2*cos(x)**3', 'x'),
    ('antiderivative', 'sin(a*x)*cos(b*x)', 'x'),
    ('antiderivative', 'x*sin(x)**2', 'x'),
    ('antiderivative', 'x**2*cos(x)', 'x'),
    ('antiderivative', '1/(sin(x) + cos(x))', 'x'),
    ('antiderivative', 'log(x)**2', 'x'),
    ('antiderivative', 'x**2*log(x)', 'x'),
    ('antiderivative', '1/(x*log(x)**2)', 'x'),
    ('antiderivative', 'log(x)**3', 'x'),
    ('antiderivative', 'log(x)/x**3', 'x'),
    ('antiderivative', 'log(x + 1)/x**2', 'x'),
    ('antiderivative', 'x*log(x**2 + 1)', 'x'),
    ('antiderivative', 'log(x)**2/x', 'x'),
    ('antiderivative', 'log(x)**2/x**2', 'x'),
    ('antiderivative', 'log(log(x))/x', 'x'),
    ('antiderivative', 'x*y*cos(x**2)', 'x'),
    ('antiderivative', 'x*exp(-x**2)', 'x'),
    ('antiderivative', '1/(x**2 + 1)', 'x'),
    ('antiderivative', 'exp(x)/x', 'x'),
    ('antiderivative', 'y*tan(x)', 'x'),
    ('antiderivative', 'log(x)**2/x**3', 'x'),
    ('antiderivative', 'exp(x)*exp(x**2)', 'x'),
    ('antiderivative', 'x*exp(x)', 'x'),
    ('antiderivative', 'sin(x)*cos(x)', 'x'),
    ('antiderivative', 'exp(-x**2)*erf(x)', 'x'),
    ('antiderivative', 'x*besselj(0, x)', 'x'),
    ('antiderivative', 'sinh(x)*cosh(x)', 'x'),
    ('antiderivative', 'pi*x/(x**2 + pi)', 'x'),
    ('antiderivative', 'sqrt(2)*x/(x**2 + 2)', 'x'),
    ('antiderivative', 'exp(I*x)', 'x'),
    ('antiderivative', '1/(x + I)', 'x'),
    ('antiderivative', 'exp(p*x)*cos(x)', 'x'),
    ('antiderivative', 'y', 'x'),
    ('antiderivative', 'Abs(x)', 'x'),
    ('cases', 'cos(n*x)', 'x'),
    ('cases', '1/(a + x)', 'x'),
    ('cases', 'y', 'x'),
    ('cases', '1/(x + sin(y))', 'x'),
    ('cases', '1/(x + exp(y))', 'x'),
    ('cases', 'exp(x)/(exp(x) + sin(y))', 'x'),
    ('cases', 'log(x)/(x*(log(x) + sin(y)))', 'x'),
    ('antiderivative', 'x/(x**2 - 1)', 'x'),
    ('antiderivative', '(x + 1)/(x**2 + 2*x + 5)', 'x'),
    ('antiderivative', '(x**2 + 1)**(-2)', 'x'),
    ('antiderivative', 'x**3/(x**2 + 1)', 'x'),
    ('antiderivative', 'x**2*exp(x)', 'x'),
    ('antiderivative', 'exp(x)/(exp(x) + 1)', 'x'),
    ('antiderivative', 'exp(x)/(exp(2*x) + 1)', 'x'),
    ('antiderivative', 'log(x)', 'x'),
    ('antiderivative', 'x*log(x)', 'x'),
    ('antiderivative', 'log(x)/x', 'x'),
    ('antiderivative', '1/(x*log(x))', 'x'),
    ('antiderivative', 'exp(x)*log(x) + exp(x)/x', 'x'),
    ('antiderivative', 'x*sin(x)', 'x'),
    ('antiderivative', 'sin(x)**3', 'x'),
    ('antiderivative', 'cos(x)**4', 'x'),
    ('antiderivative', 'sin(2*x)*cos(3*x)', 'x'),
    ('antiderivative', 'x*cos(x)**2', 'x'),
    ('antiderivative', 'tan(x)**2', 'x'),
    ('antiderivative', 'tan(x)**3', 'x'),
    ('antiderivative', '1/(tan(x)**2 + 1)', 'x'),
    ('antiderivative', 'tanh(x)', 'x'),
    ('antiderivative', 'cosh(x)**(-2)', 'x'),
    ('antiderivative', '1/(x**(1/3) + 1)', 'x'),
    ('antiderivative', 'sqrt(x)*exp(-sqrt(x))', 'x'),
    ('antiderivative', 'LambertW(x)', 'x'),
    ('antiderivative', 'atan(x)', 'x'),
    ('antiderivative', 'asin(x)', 'x'),
    ('antiderivative', 'x*atan(x)', 'x'),
    ('antiderivative', 'cos(n*x)', 'x'),
    ('antiderivative', 'exp(x*y)*sin(x)', 'x'),
    ('antiderivative', '1/(x**2 + y**2)', 'x'),
    ('antiderivative', '1/cos(x)', 'x'),
    ('antiderivative', '1/(sin(x) + 1)', 'x'),
    ('antiderivative', 'exp(-x**2)', 'x'),
    ('antiderivative', 'x**2*exp(-x**2)', 'x'),
    ('antiderivative', 'exp(2*x)/x', 'x'),
    ('antiderivative', 'sin(x)/x', 'x'),
    ('antiderivative', 'cos(x)/x', 'x'),
    ('antiderivative', 'x*Ei(x)', 'x'),
    ('antiderivative', '1/sqrt(x**2 + 1)', 'x'),
    ('antiderivative', '1/sqrt(x**2 - 4)', 'x'),
    ('antiderivative', '1/sqrt(x**2 + 2*x + 5)', 'x'),
    ('antiderivative', 'sqrt(-x**2 + 2*x)', 'x'),
    ('antiderivative', 'sqrt(1 - x**2)', 'x'),
    ('antiderivative', 'x*sign(x)', 'x'),
    ('cases', 'cos(x)', 'x'),
    ('cases', 'exp(n*x)', 'x'),
    ('antiderivative', '1/(x*sqrt(x**2 + 1))', 'x'),
    ('antiderivative', 'x**2*sqrt(x**2 + 1)', 'x'),
    ('antiderivative', 'exp(x)/(x + 1)**2', 'x'),
    ('antiderivative', '1/(log(x) + 1)', 'x'),
    ('antiderivative', 'x/(log(x) + 1)', 'x'),
    ('antiderivative', '(2*exp(2*x) + exp(x))/log(exp(2*x) + exp(x))', 'x'),
    ('antiderivative', 'exp(-x)/(x + 2)', 'x'),
    ('antiderivative', 'x/(exp(x) + 1)', 'x'),
    ('antiderivative', 'exp(x)*exp(-exp(2*x))', 'x'),
    ('antiderivative', '(exp(x) + 1)*exp(-(x + exp(x))**2)', 'x'),
    ('antiderivative', 'x*exp(-x**2 + 2*x)', 'x'),
    ('antiderivative', 'exp(x**2 + 1)', 'x'),
    ('antiderivative', 'exp(-log(x)**2)/x', 'x'),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_heurisch(i: int) -> None:
    kind, text, variable = CASES[i]
    x = Symbol(variable)
    f = parse(text)
    if kind == 'cases':
        got = outcome(lambda: rc.logic_call('heurisch', [Symbol('cases'), f, x]))
        expected = outcome(lambda: heurisch_cases(f, x))
    else:
        got = outcome(lambda: rc.logic_call('heurisch', [Symbol('antiderivative'), f, x, Tuple(), Integer(0), Integer(2)]))
        expected = outcome(lambda: heurisch_antiderivative(f, x))
    if got != 'SKIP':
        assert got == expected, text
