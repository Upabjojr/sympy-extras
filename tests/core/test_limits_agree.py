"""The Rust port of ``core/limits.py`` and ``core/gruntz.py`` against the
Python modules: limits from both sides, at the infinities and two-sided,
compared as strings (a part the Rust core does not have yet counts as
``'SKIP'``)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.core.basic import Basic
from sympy_extras.core.limits import limit
from sympy_extras.functions import Abs, asin, atan, cos, cosh, exp, log, sign, sin, sqrt, tan, tanh

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = Symbol('x'), Symbol('y')
n = Symbol('n', positive=True)


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as e:
        return type(e).__name__


from sympy_extras.functions import Ei, erf, erfc, erfi, expint, li  # noqa: E402

CASES: list[tuple[Basic, Basic]] = [
    # the helpers _erfs and _eis of the Gruntz algorithm
    (erf(x), oo), (erf(x), -oo), (x*exp(x**2)*erfc(x), oo), (exp(x**2)*erfc(x)*x**3 - x**2/sqrt(3), oo),
    (erfi(x)*exp(-x**2)*x, oo), (Ei(x)*exp(-x)*x, oo), (Ei(x)/exp(x), oo), (li(x)*log(x)/x, oo),
    (x*exp(x)*expint(2, x), oo), (exp(x)*expint(1, x)*x, oo), (erf(x) - 1, oo), ((erf(x) - 1)*exp(x**2)*x, oo),
    (atan(x) - 1, oo), (x*(atan(x) - 3*atan(Integer(1))*2/3), oo),
    (sin(x)/x, Integer(0)), (1/x, Integer(0)), (x*log(x), Integer(0)), (x**x, Integer(0)), (exp(1/x), Integer(0)),
    ((1 + 1/x)**x, oo), (log(x)/x, oo), (exp(x)/x**10, oo), (x*sin(1/x), Integer(0)), ((1 - cos(x))/x**2, Integer(0)),
    (atan(x), oo), (exp(-x)*x**5, oo), (sqrt(x**2 + x) - x, oo), (tanh(x), oo), (log(1 + x)/x, Integer(0)),
    ((exp(x) - 1)/x, Integer(0)), (x**(1/x), oo), (sin(x), oo), (1/x, oo), (x/(x + 1), oo),
    ((x**2 + 3)/(2*x**2 - 1), oo), (log(x), Integer(0)), (exp(x), -oo), (x*exp(-x), oo), (Abs(x)/x, Integer(0)),
    (sign(x), Integer(0)), (exp(x)*(exp(1/x - exp(-x)) - exp(1/x)), oo), (log(log(x))/log(x), oo),
    ((3**x + 5**x)**(1/x), oo), (x**2*exp(-x**2), oo), (cosh(x)/exp(x), oo), (y/x, Integer(0)), (x*y, Integer(0)),
    (sin(y*x)/x, Integer(0)), (1/sqrt(x), Integer(0)), (x**Rational(1, 3), Integer(0)), (asin(x)/x, Integer(0)),
    (tan(x)/x, Integer(0)), (exp(1/x)*x, Integer(0)), (log(x)**2/x, oo), ((1 + 2/x)**(3*x), oo), (n**(1/n), oo),
    (exp(-1/x**2)/x**10, Integer(0)), (1/x, -oo), (x**3 - x**5, oo),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_limit(i: int) -> None:
    e, p = CASES[i]
    for d in (['+', '-'] if p == 0 else ['+']):
        r = outcome(lambda: rc.logic_call('series', [Symbol('limit'), e, x, p, Symbol(d)]))
        if r != 'SKIP':
            assert r == outcome(lambda: limit(e, x, p, d)), (e, p, d)


def test_two_sided() -> None:
    for e in [1/x, x**2, 1/x**2, sin(x)/x, exp(1/x)]:
        r = outcome(lambda: rc.logic_call('series', [Symbol('limit'), e, x, Integer(0), Symbol('+-')]))
        if r != 'SKIP':
            assert r == outcome(lambda: limit(e, x, 0, '+-'))
