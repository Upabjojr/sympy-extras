"""The Rust port of ``integrals/recognize.py`` against the Python one:
closed forms of constants (PSLQ) and of definite integrals (two quadrature
rules, then PSLQ), compared as strings."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol
from sympy_extras.core.parsing import parse
from sympy_extras.integrals.recognize import recognize_constant, recognize_integral

from ..conftest import load

rc = load("sympy_extras_rs.core")
x = Symbol('x')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


INTEGRALS: list[tuple[str, str, str]] = [
    ('1/(x**3 + 1)', '0', '1'), ('atan(x)/x', '0', '1'), ('log(1 + x)/(1 + x**2)', '0', '1'),
    ('exp(-x**2)', '0', 'oo'), ('1/(1 + x**2)', '-oo', 'oo'), ('x*exp(-x)*sin(x)', '0', 'oo'),
    ('sin(x)/x', '0', 'pi'), ('log(x)/(1 + x)', '0', '1'), ('x/(exp(x) - 1)', '0', 'oo'),
    ('log(x)**2', '0', '1'), ('1/(1 + x**4)', '0', 'oo'), ('x**2/(exp(x) + 1)', '0', 'oo'),
    ('log(1 + x)/x', '0', '1'), ('atan(x)', '0', '1'), ('sqrt(1 - x**2)', '-1', '1'),
    ('log(sin(x))', '0', 'pi/2'), ('x*log(1 + x)', '0', '1'), ('exp(-x)*log(x)', '0', 'oo'),
    ('1/(2 + cos(x))', '0', 'pi'), ('x**3/(exp(x) - 1)', '0', 'oo'),
]

CONSTANTS: list[str] = ['pi**2/6 + log(2)', 'atan(1)*4', 'zeta(3)*3 - 1', 'catalan*2 + pi', 'sqrt(2) + sqrt(3)',
                        'EulerGamma - 1/2', 'log(6)', 'exp(1)/3', 'gamma(1/4)**2/sqrt(pi)', 'pi*log(2) + 1',
                        'sqrt(7)', 'log(5)']


@pytest.mark.parametrize('case', INTEGRALS, ids=[' '.join(c) for c in INTEGRALS])
def test_integral(case: tuple[str, str, str]) -> None:
    f, a, b = (parse(s) for s in case)
    rust = outcome(lambda: rc.logic_call('recognize', [Symbol('integral'), f, x, a, b, Integer(30)]))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: recognize_integral(f, (x, a, b)))


@pytest.mark.parametrize('case', CONSTANTS)
def test_constant(case: str) -> None:
    e = parse(case)
    rust = outcome(lambda: rc.logic_call('recognize', [Symbol('constant'), e, Integer(30)]))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: recognize_constant(e))
