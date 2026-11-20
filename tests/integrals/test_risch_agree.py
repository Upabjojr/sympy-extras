"""The Rust port of the Risch algorithm (``integrals/risch/risch.py``)
against the Python one. The integrands of the Python tests of the Risch package are read with
the core's parser; the trigonometric ones need ``restore_sincos``, not
ported yet, and are skipped."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.parsing import parse
from sympy_extras.functions import exp, log, sqrt
from sympy_extras.integrals.risch.risch import risch_integrate

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
    return re.sub(r'\b_?([a-zA-Z]+)_?\d+\b', r'\1', str(r))


INTEGRANDS: list[Expr] = [
    1 / (x**2 + 1), x / (x**2 - 1), (x**3 + 1) / (x**2 - 2), 1 / (x**3 + x), x**5, 1 / (x**4 + 1),
    (2 * x + 3) / (x**2 + 3 * x + 2)**2, 1 / (x**2 - 2)**3, (x**4 - 3) / (x**2 + x + 1), x**2 / (x - 1)**4,
    1 / (x**6 - 1), (x - sqrt(2)) / (x**2 - 2), 1 / ((x - 1) * (x - 2) * (x - 3)),
    log(x), exp(x), x * exp(x**2), log(x) / x, exp(x) / (exp(x) + 1), 1 / (x * log(x)),
]

RECORDED: list[str] = [
    '1',
    'x',
    'x**2',
    't0*exp(x)',
    '(x**4 + 2*x**3*exp(2*x**2) + 2*x**2 + 1)/(x**4*exp(x**2) + 2*x**2*exp(x**2) + exp(x**2))',
    '0',
    'log(x*exp(-x) + 1)',
    'log(x**x)',
    'log(x**y)',
    'log(sqrt(x))',
    'exp(x) + exp(x/2 + 1) + log(x) + log(x**2)',
    'exp(x/2) + sqrt(exp(x))',
    'sqrt(exp(x))',
    '(-x**2 + 2*x + exp(x))*exp(1/(x + exp(x)) + (x**2 - 1)/x)/(x**2*(x + exp(x))**2)',
    'exp(x - exp(x)) + exp(x + exp(x))',
    'exp(x)/2 - exp(-x)/2',
    '-I*(exp(I*x) - exp(-I*x))/2',
    '(-2*x*log(x) - 2*x - 2)/(x**3*log(x)**2 + 2*x**2*log(x)**2 + x*log(x)**2 - x)',
    'tan(x)',
    'tan(x)**2',
    'tan(x)**5',
    'x*tan(x) + tan(x)**2 + 1',
    '(x**2*tan(x) + tan(x)**5 + tan(x)**3 + 1)/(tan(x)**2 + 1)**3',
    '(tan(x)**2 + 2)/((x + tan(x))**2 + 1)',
    '(x*(tan(x)**2 + 1) + 1)*exp(tan(x))',
    'atan(x)',
    'x*atan(x)',
    '1/((x**2 + 1)*atan(x))',
    'atan(x)/(x**2 + 1)',
    'atan(1/x) + atan(x)',
    'acot(x)',
    '(tan(atan(x)/3)**2 + 1)/(x**2 + 1)',
    'cot(x)',
    'exp(x*y)',
    'exp(x)/((exp(x) + 1)**2 + 1)',
    'exp(x)/(exp(2*x) + 1)',
    '1/(x*(log(x)**2 + 1))',
    '1/(x**2 + 1)',
    '(exp(2*x) + 2*exp(x) + 7)*exp(x)/((2*exp(x) + 6)*(exp(2*x) + 1))',
    '(exp(x) + 4)*exp(x)/((exp(x) + 1)**2 + 1)',
    'exp(x)/(exp(2*x) - 2)',
    'exp(x)*log(exp(x) + 1)',
    '(x**2*exp(x) + x*exp(x)*log(exp(x) + log(x)) - x*exp(x) + x*log(x)*log(exp(x) + log(x)) + x - 1)/(x*exp(x) + x*log(x))',
    '(x*exp(2*x)*log(exp(x) + log(x)) + x*exp(2*x) + x*exp(x)*log(x)*log(exp(x) + log(x)) + exp(x))/(x*exp(x) + x*log(x))',
    '(-exp(1/log(x))*log(x) - exp(1/log(x)))/(x*log(x)**3)',
    '(-2*exp(1/log(x))*log(x) - exp(1/log(x)))/(x*log(x)**3)',
    '(exp(x/log(x))*log(x) - exp(x/log(x)))/log(x)**2',
    '(-exp(1/log(x))*log(x)**2 - exp(1/log(x))*log(x) - exp(1/log(x)))/(x**2*log(x)**3)',
    '(-exp(1/(x*log(x)))*log(x) - exp(1/(x*log(x))))/(x**2*log(x)**2)',
    'exp(x**2)',
    'x**x*log(x)',
    'x**x',
    '((4*x**6 - 42*x**5 + 144*x**4 - 108*x**3 - 324*x**2 + 486*x + (-2*x**5 + 24*x**4 - 108*x**3 + 216*x**2 - 162*x)*exp(x) + (4*x**5 - 44*x**4 + 168*x**3 - 216*x**2 - 108*x + 324)*log(x))*log(-x**2 - 2*x*log(x) - x + exp(x) - log(x)**2 + log(5)) + (-2*x**7 + 26*x**6 - 116*x**5 + 180*x**4 + 54*x**3 - 270*x**2 + (-2*x**5 + 28*x**4 - 144*x**3 + 324*x**2 - 270*x)*log(x)**2 + (2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*exp(x) + (2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*log(5) + (-4*x**6 + 56*x**5 - 288*x**4 + 648*x**3 - 540*x**2)*log(x))*log(-x**2 - 2*x*log(x) - x + exp(x) - log(x)**2 + log(5))**2)/(2*x**2*exp(2*x)*log(x) - x*exp(3*x) + x*exp(2*x)*log(x)**2 + (x**3 + x**2 - x*log(5))*exp(2*x))',
    'exp(v*log(-log(x)))',
    '(a*x + a)*exp(a*log(x))/(a*x*log(x) + b*x)',
    'exp(b*x*log(a) + (v - 1)*log(x))',
    'x*exp(v*log(-log(x**2 + 1)))',
    'x*tan(x)',
    '2*x*exp(x**2)',
    '1/(x*log(x))',
    'exp(x)*log(x) + exp(x)/x',
    'exp(x)/x',
    '1/log(x)',
    'sqrt(x)',
    'x*exp(x)',
]
INTEGRANDS += [parse(s) for s in RECORDED if parse(s) not in INTEGRANDS]


@pytest.mark.parametrize('k', range(len(INTEGRANDS)))
def test_agrees(k: int) -> None:
    f = INTEGRANDS[k]
    py = outcome(lambda: risch_integrate(f, x))
    rs = outcome(lambda: rc.logic_call('risch_integrate', [f, x]))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
