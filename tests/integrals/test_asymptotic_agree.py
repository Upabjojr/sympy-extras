"""The Rust port of ``integrals/asymptotic.py`` against the Python one, on
the calls of the Python tests and the module documentation (the core's
objects passed as they are, with the parameter ``t`` positive)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import I, Integer, Symbol, Tuple, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.functions import cos, cosh, exp, log, sin
import sympy_extras.integrals.asymptotic as asy

from ..conftest import load

rc = load("sympy_extras_rs.core")

x = Symbol('x')
t = Symbol('t', positive=True)
a = Symbol('a', positive=True)
T = Symbol('True')
F = Symbol('False')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    if r is None or str(r) == 'NONE':
        return 'None'
    if isinstance(r, tuple):
        r = Tuple(*[Symbol(v) if isinstance(v, str) else v for v in r])
    return re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


CASES: list[tuple[str, tuple[Basic, ...]]] = [
    ('integral', (exp(-t*x)*sqrt(1 + x), x, Integer(0), oo, t, Integer(2), F)),
    ('integral', (exp(-t*(x**2 - 1)**2), x, -oo, oo, t, Integer(1), F)),
    ('integral', (exp(-t*x**2 + I*t*x), x, -oo, oo, t, Integer(3), F)),
    ('integral', (exp(-2*t*x)*cos(x), x, Integer(0), Integer(5), t, Integer(2), F)),
    ('integral', (exp(t*(x - x**2))/(1 + x), x, Integer(0), Integer(1), t, Integer(2), F)),
    ('integral', (cos(t*sin(x))/pi, x, Integer(0), pi, t, Integer(3), F)),
    ('integral', (exp(t*(a*x - x**2)), x, -oo, oo, t, Integer(2), F)),
    ('integral', (exp(-x**2), x, Integer(0), oo, t, Integer(3), F)),
    ('integral', (exp(-t*x)/(1 + x), x, Integer(0), oo, t, Integer(3), F)),
    ('integral', (exp(t*(log(x) - x)), x, Integer(0), oo, t, Integer(2), F)),
    ('integral', (exp(-t*cosh(x)), x, -oo, oo, t, Integer(2), F)),
    ('integral', (exp(-t*x)*cos(x)/(1 + x**2), x, Integer(0), oo, t, Integer(4), F)),
    ('integral', (exp(-t*x**4), x, -oo, oo, t, Integer(1), F)),
    ('integral', (exp(t*cos(x)), x, Integer(0), 2*pi, t, Integer(2), F)),
    ('integral', (exp(t*(-x**4 + I*x)), x, -oo, oo, t, Integer(3), F)),
    ('integral', (exp(I*t*(x**3/3 - a*x)), x, -oo, oo, t, Integer(1), T)),
    ('integral', (exp(I*t*x**2), x, a, oo, t, Integer(1), T)),
    ('integral', (sin(t*x**2)*x, x, Integer(-1), Integer(2), t, Integer(1), F)),
    ('watson', (1/(1 + x), x, t, Integer(3))),
    ('watson', (sqrt(x), x, t, Integer(3))),
    ('watson', (1/x, x, t, Integer(3))),
    ('watson', (sin(x), x, t, Integer(2))),
    ('laplace', (Integer(1), log(x) - x, x, Integer(0), oo, t, Integer(3))),
    ('laplace', (Integer(1), -cosh(x), x, -oo, oo, t, Integer(2))),
    ('laplace', (Integer(1), -(x**2 - 1)**2, x, -oo, oo, t, Integer(1))),
    ('laplace', (Integer(1), -x**4, x, -oo, oo, t, Integer(1))),
    ('laplace', (Integer(1), cos(x), x, Integer(0), 2*pi, t, Integer(2))),
    ('laplace', (1/(1 + x), x - x**2, x, Integer(0), Integer(1), t, Integer(2))),
    ('steepest', (Integer(1), -x**2 + I*x, x, t)),
    ('steepest', (Integer(1), -x**4 + I*x, x, t)),
    ('steepest', (Integer(1), I*x - cosh(x), x, t)),
    ('stationary', (Integer(1), x**2, x, Integer(-1), Integer(1), t, Symbol('exp'))),
    ('stationary', (1/pi, sin(x), x, Integer(0), pi, t, Symbol('cos'))),
    ('uniform', (Integer(1), x**3/3 - a*x, x, -oo, oo, t, Symbol('exp'))),
    ('uniform', (Integer(1), x**2, x, a, oo, t, Symbol('exp'))),
    ('uniform', (Integer(1), x**2, x, -oo, oo, t, Symbol('exp'))),
    ('exponential_part', (exp(-t*x)/(1 + x), x, t)),
    ('exponential_part', (exp(t*(log(x) - x) - x), x, t)),
    ('exponential_part', (exp(-t**2*x), x, t)),
    ('exponential_part', (x**t*exp(-t*x)/(1 + x), x, t)),
    ('oscillatory_part', (exp(I*t*x**2)*x, x, t)),
    ('oscillatory_part', (cos(t*x + 1), x, t)),
    ('oscillatory_part', (cos(t*sin(x))/pi, x, t)),
]


def python_call(kind: str, args: tuple[Basic, ...]) -> object:
    if kind == 'integral':
        f, x_, lo, hi, p, order, uniform = args
        return asy.asymptotic_integral(f, x_, lo, hi, p, order=int(order), uniform=uniform == T)  # type: ignore[arg-type]
    if kind == 'watson':
        return asy.watson_lemma(args[0], args[1], args[2], int(args[3]))  # type: ignore[arg-type]
    if kind == 'laplace':
        return asy.laplace_method(*args[:6], int(args[6]))  # type: ignore[arg-type]
    if kind in ('stationary', 'uniform'):
        method = asy.stationary_phase if kind == 'stationary' else asy.uniform_expansion
        return method(*args[:6], str(args[6]))  # type: ignore[arg-type]
    if kind == 'steepest':
        return asy.steepest_descent(*args)  # type: ignore[arg-type]
    if kind == 'exponential_part':
        return asy.exponential_part(*args)  # type: ignore[arg-type]
    return asy.oscillatory_part(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize('case', CASES, ids=[c[0] + ' ' + str(c[1][0])[:60] for c in CASES])
def test_agree(case: tuple[str, tuple[Basic, ...]]) -> None:
    kind, args = case
    rust = outcome(lambda: rc.logic_call('asymptotic', [Symbol(kind)] + list(args)))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: python_call(kind, args))
