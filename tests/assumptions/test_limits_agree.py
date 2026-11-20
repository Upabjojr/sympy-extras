"""The Rust port of ``assumptions/limits.py`` (``limit``, ``limit_seq``,
``series`` under assumptions on the parameters) against the Python one on
the calls of the tests (read with the core's parser and a reader of the
relations: plain symbols; the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from sympy_extras.assumptions.limits import limit, limit_seq, series
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, and_
from sympy_extras.core.parsing import parse
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


def expr(text: str) -> Expr:
    e = parse(text)
    assert isinstance(e, Expr)
    return e


def symbol(text: str) -> Symbol:
    s = parse(text)
    assert isinstance(s, Symbol)
    return s


def relation(text: str) -> Boolean:
    """A relation as the core prints it: ``Eq(a, b)``, ``a > b`` and the
    other inequalities (the core's parser reads expressions only)."""
    text = text.strip()
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    if text.startswith('Eq(') and text.endswith(')'):
        lhs, rhs = text[3:-1].split(', ', 1)
        found = Eq(expr(lhs), expr(rhs))
        assert isinstance(found, Boolean)
        return found
    for op, build in (('>=', Ge), ('<=', Le), ('>', Gt), ('<', Lt)):
        if ' ' + op + ' ' in text:
            lhs, rhs = text.split(' ' + op + ' ', 1)
            made = build(expr(lhs), expr(rhs))
            assert isinstance(made, Boolean)
            return made
    raise ValueError(text)


def booleans(texts: list[str]) -> list[Boolean]:
    return [and_(*[relation(part) for part in t.split(' & ')]) for t in texts]


#: (operation, its arguments as strings, the assumptions last)
CASES: list[list[Any]] = [
    ['limit', 'exp(a*x)', 'x', 'oo', '+', ['a < 0']],
    ['limit', 'exp(a*x)', 'x', 'oo', '+', ['a > 0']],
    ['limit', 'exp(a*x)', 'x', 'oo', '+', ['Eq(a, 0)']],
    ['limit', 'x**a', 'x', 'oo', '+', ['a > 0']],
    ['limit', 'x**a', 'x', 'oo', '+', ['a < 0']],
    ['limit', 'x**a*log(x)', 'x', '0', '+', ['a > 0']],
    ['limit', 'sin(a*x)/x', 'x', '0', '+', []],
    ['limit', '(a/x + 1)**x', 'x', 'oo', '+', []],
    ['limit', 'a**x', 'x', 'oo', '+', ['(a > 0) & (a < 1)']],
    ['limit', 'a**x', 'x', 'oo', '+', ['a > 1']],
    ['limit', 'exp(a*x)', 'x', 'oo', '+', []],
    ['limit', 'x**a', 'x', 'oo', '+', []],
    ['limit', 'exp(a*x)', 'x', 'oo', '+', ['a >= 0']],
    ['limit', 'a*exp(-x**2)', 'x', 'oo', '+', []],
    ['series', 'sqrt(a**2 + x)', 'x', '0', 2, ['a > 0']],
    ['series', 'sqrt(a**2 + x)', 'x', '0', 2, ['a < 0']],
    ['series', 'log(a*x)', 'x', '1', 2, ['a > 0']],
    ['series', 'exp(a*x)', 'x', '0', 3, []],
    ['limit_seq', 'x**n/factorial(n)', 'n', []],
    ['limit_seq', '(1 + x/n)**n', 'n', []],
    ['limit_seq', '-log(n) + harmonic(n)', 'n', []],
    ['limit_seq', '(-1)**n/n', 'n', []],
    ['limit_seq', 'sin(pi*n)', 'n', []],
    ['limit_seq', 'a**n', 'n', ['(a > 0) & (a < 1)']],
    ['limit_seq', 'a**n', 'n', ['a > 1']],
    ['limit_seq', 'a**n', 'n', ['Eq(a, 1)']],
    ['limit_seq', 'a**n*n', 'n', ['(a > 0) & (a < 1)']],
    ['limit_seq', 'a**n/n', 'n', ['a > 1']],
    ['limit_seq', 'a**n', 'n', ['(a > -1) & (a < 0)']],
    ['limit_seq', 'a**n', 'n', ['a < -1']],
    ['limit_seq', 'a**n', 'n', ['a > 0']],
    ['limit_seq', 'a**n', 'n', []],
    ['limit_seq', 'n**a', 'n', ['a < 0']],
    ['limit_seq', 'n**a', 'n', []],
    ['limit', 'a**x', 'x', 'oo', '+', ['a < -1']],
    ['limit', 'a**x', 'x', 'oo', '+', ['Eq(a, -1)']],
    ['limit', 'a**x', 'x', 'oo', '+', ['(a > -1) & (a < 0)']],
    ['limit', 'a**x', 'x', 'oo', '+', ['Eq(a, 0)']],
    ['limit', '2*a**x', 'x', 'oo', '+', ['a < -1']],
    ['limit', 'a**x', 'x', 'oo', '+', []],
    ['limit', 'a**x', 'x', '-oo', '+', ['a < -1']],
    ['limit', 'a**(-x)', 'x', 'oo', '+', ['a < -1']],
    ['limit', 'a**(-x)', 'x', 'oo', '+', ['(a > -1) & (a < 0)']],
    ['limit', 'a**(2*x)', 'x', 'oo', '+', ['a < -1']],
    ['limit', 'a**x*x', 'x', 'oo', '+', []],
    ['limit', 'x**a*sin(x)', 'x', 'oo', '+', []],
    ['limit', 'sin(x)/x', 'x', 'oo', '+', []],
    ['limit', 'exp(x**2 + I*x)', 'x', 'oo', '+', []],
    ['limit', 'sqrt(x)*(x**(1/x) - 1)', 'x', '0', '-', []],
    ['limit', 'sqrt(x)*(x**(1/x) - 1)', 'x', '0', '+', []],
    ['limit', '(a*x + 1)/(b*x + 2)', 'x', 'oo', '+', []],
    ['limit', 'log(x**a + x**b)/log(x)', 'x', 'oo', '+', []],
    ['limit', 'atan(a*x)', 'x', 'oo', '+', []],
    ['limit', 'atan(x)', 'x', 'oo', '+', []],
    ['limit', 'x*exp(-x)', 'x', 'oo', '+', []],
    ['limit', '-a*(-I*log(-I*a + sqrt(-_t**2*a**2 + a**2)/(_t + 1))/2 + I*log(I*a + sqrt(-_t**2*a**2 + a**2)/(_t + 1))/2)', '_t', '0', '+', ['a < 0']],
    ['limit', '-a*(-I*log(-I*a + sqrt(-_t**2*a**2 + a**2)/(_t + 1))/2 + I*log(I*a + sqrt(-_t**2*a**2 + a**2)/(_t + 1))/2)', '_t', '0', '+', ['a > 0']],
]


@pytest.mark.parametrize('case', CASES)
def test_agrees(case: list[Any]) -> None:
    kind = case[0]
    facts = booleans(case[-1])
    args: list[Basic]
    if kind == 'limit':
        e, x, x0, d = expr(case[1]), symbol(case[2]), expr(case[3]), str(case[4])
        py = outcome(lambda: limit(e, x, x0, d, facts or None))
        args = [Symbol('limit'), e, x, x0, Symbol(d), Tuple(*facts)]
    elif kind == 'limit_seq':
        e, n = expr(case[1]), symbol(case[2])
        py = outcome(lambda: limit_seq(e, n, facts or None))
        args = [Symbol('limit_seq'), e, n, Tuple(*facts)]
    else:
        e, x, x0, k = expr(case[1]), symbol(case[2]), expr(case[3]), int(case[4])
        py = outcome(lambda: series(e, x, x0, k, facts or None))
        args = [Symbol('series'), e, x, x0, Integer(k), Tuple(*facts)]
    rs = outcome(lambda: rc.logic_call('assumption_limits', args))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
