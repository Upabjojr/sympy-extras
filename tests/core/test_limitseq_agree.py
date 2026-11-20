"""The Rust port of ``series/limitseq.py`` (``limit_seq``,
``difference_delta``, ``dominant``) against the Python one on the
calls of the tests (read with the core's parser: plain symbols, no
Sum; the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
from sympy_extras.series.limitseq import difference_delta, dominant, limit_seq

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


#: (operation, expression, variable, the trials or the step)
CASES: list[tuple[str, ...]] = [
    ('difference', 'n*(n + 1)', 'n', '1'),
    ('difference', 'k*n*(n + 1)', 'n', '2'),
    ('difference', 'k*n*(n + 1)', 'n', 'oo'),
    ('difference', '4**n', 'n', '1'),
    ('difference', '4**n', 'n', '2'),
    ('difference', '4**(2*n)', 'n', '1'),
    ('difference', '4**(2*n)', 'n', '2'),
    ('difference', 'n**4', 'n', '1'),
    ('difference', 'n**n', 'n', '1'),
    ('difference', 'n*(n + 1) + n', 'n', '1'),
    ('difference', 'n*(n + 1) + n', 'n', '2'),
    ('dominant', '5*n**3 + 4*n**2 + n + 1', 'n'),
    ('dominant', '2**n + 3**n + n**5', 'n'),
    ('limit', '(5*n**3 + 3*n**2 + 4)/(3*n**3 + 4*n - 5)', 'n', '5'),
    ('limit', 'binomial(2*n, n)/4**n', 'm', '5'),
    ('limit', '3', 'n', '5'),
    ('limit', '(-1)**n/n**2', 'n', '5'),
    ('limit', '(-2)**(n + 1)/(3**n + n)', 'n', '5'),
    ('limit', '((-1)**n + 2*n)/(n + 1)', 'n', '5'),
    ('limit', '0', 'n', '5'),
    ('limit', '1', 'n', '5'),
    ('limit', '(-1/5)**n', 'n', '5'),
    ('limit', '(I/3)**n', 'n', '5'),
    ('limit', 'sqrt(n)*(I/2)**n', 'n', '5'),
    ('limit', 'n**7*(I/3)**n', 'n', '5'),
    ('limit', 'n/(n + 1) + (I/2)**n', 'n', '5'),
    ('limit', '(-1)**n', 'n', '5'),
    ('limit', 'sin(pi*n/2)**2', 'n', '5'),
    ('limit', '2*(-3)**n/(3**n + n)', 'n', '5'),
    ('limit', '2*(-1)**n + 3*n/(n + 1)', 'n', '5'),
    ('limit', '(-2)**n', 'n', '5'),
    ('limit', 'subfactorial(n)/factorial(n)', 'n', '5'),
    ('limit', '5**n/factorial(n)', 'n', '5'),
    ('limit', '2**n/factorial(n)', 'n', '5'),
    ('limit', '1/gamma(n + 1/2)', 'n', '5'),
    ('limit', 'binomial(2*n, n)/4**n', 'n', '5'),
    ('limit', 'fibonacci(n + 1)/fibonacci(n)', 'n', '5'),
    ('limit', '(1 + 1/n)**n', 'n', '5'),
    ('limit', '_x**n/factorial(n)', 'n', '5'),
    ('limit', '0**n/factorial(n)', 'n', '5'),
    ('limit', '(-_x)**n/factorial(n)', 'n', '5'),
    ('limit', '(_x/n + 1)**n', 'n', '5'),
    ('limit', 'exp(-_q*n)', 'n', '5'),
    ('limit', 'exp(_q*n)', 'n', '5'),
    ('limit', '(-_a)**n', 'n', '5'),
    ('limit', 'n**_a', 'n', '5'),
    ('limit', 'n**2', 'n', '5'),
    ('limit', 'sqrt(n)', 'n', '5'),
    ('limit', 'n**3', 'n', '5'),
    ('limit', 'n', 'n', '5'),
    ('limit', '1/n', 'n', '5'),
    ('limit', 'n**(-5/2)', 'n', '5'),
    ('limit', 'n**(-3)', 'n', '5'),
    ('limit', '-log(n) + harmonic(n)', 'n', '5'),
    ('limit', '(-1)**n/n', 'n', '5'),
    ('limit', 'sin(pi*n)', 'n', '5'),
    ('limit', 'n*exp(-_q*n)', 'n', '5'),
    ('limit', 'exp(_q*n)/n', 'n', '5'),
    ('limit', '_a**n', 'n', '5'),
    ('limit', '2**n', 'n', '5'),
    ('limit', '3**n', 'n', '5'),
    ('limit', '5**n', 'n', '5'),
    ('limit', '(1/2)**n', 'n', '5'),
    ('limit', '(-5/2)**n', 'n', '5'),
    ('limit', '0**n', 'n', '5'),
    ('limit', '(5/2)**n', 'n', '5'),
    ('limit', '(-1/2)**n', 'n', '5'),
    ('limit', '(-3)**n', 'n', '5'),
]


@pytest.mark.parametrize('case', CASES)
def test_agrees(case: tuple[str, ...]) -> None:
    kind, source = case[0], case[1]
    e = parse(source)
    args: list[Basic]
    if kind == 'limit':
        n = None if case[2] == 'None' else parse(case[2])
        assert n is None or isinstance(n, Symbol)
        trials = int(case[3])
        py = outcome(lambda: limit_seq(e, n, trials))
        args = [Symbol('limit'), e, Symbol('NONE') if n is None else n, Integer(trials)]
    elif kind == 'difference':
        var, step = parse(case[2]), parse(case[3])
        assert isinstance(var, Symbol)
        v: Symbol = var
        py = outcome(lambda: difference_delta(e, v, step))
        args = [Symbol('difference'), e, var, step]
    else:
        var = parse(case[2])
        assert isinstance(var, Symbol)
        w: Symbol = var
        py = outcome(lambda: dominant(e, w))
        args = [Symbol('dominant'), e, var]
    rs = outcome(lambda: rc.logic_call('limit_seq', args))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
