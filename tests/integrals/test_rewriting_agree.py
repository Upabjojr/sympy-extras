"""The Rust port of ``integrals/rewriting.py`` (``rewritten_forms``,
``power_substitutions``, ``implied_assumptions``) against the Python one on
the calls of the tests (read with the core's parser and a reader of the
relations: plain symbols; the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from sympy_extras.core import Symbol, Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean
from sympy_extras.core.parsing import parse
from sympy_extras.core.relational import Gt, Lt
from sympy_extras.integrals.rewriting import implied_assumptions, power_substitutions, rewritten_forms

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


def relation(text: str) -> Boolean:
    """``a > b`` or ``a < b`` as the core prints it."""
    for op, build in (('>', Gt), ('<', Lt)):
        if ' ' + op + ' ' in text:
            lhs, rhs = text.split(' ' + op + ' ', 1)
            made = build(expr(lhs), expr(rhs))
            assert isinstance(made, Boolean)
            return made
    raise ValueError(text)


#: (operation, integrand, variable, assumptions)
CASES: list[list[Any]] = [
    ['forms', 'a**(d*z)*exp(z**2)', 'z', []],
    ['forms', 'exp(c*z**2)**p*exp(d*z)', 'z', []],
    ['forms', '2**z*cosh(z)', 'z', []],
    ['forms', 'x/sinh(x + 2)', 'x', []],
    ['forms', 'asinh(x)/sqrt(x**2 + 1)', 'x', []],
    ['forms', 'atanh(x)', 'x', []],
    ['forms', 'exp(acosh(z))', 'z', []],
    ['forms', 'k**z', 'z', []],
    ['forms', 'k**z', 'z', ['k < 0']],
    ['forms', 'n**z', 'z', []],
    ['implied', 'k**(d*z)', 'z', []],
    ['implied', 'a**z', 'z', []],
    ['forms', 'x*Abs(x)', 'x', ['x > 0']],
    ['forms', 'x*Abs(x)', 'x', ['x < 0']],
    ['forms', 'x*Abs(x)', 'x', []],
    ['forms', 'sqrt(x**2 + 1)*sqrt(x**2 + 4)', 'x', []],
    ['forms', '(z**r)**p', 'z', ['z > 0']],
    ['forms', '(z**r)**p', 'z', []],
    ['subs', 'sqrt(x)/(x + 1)', 'x', []],
    ['subs', '1/(x**(1/3) + sqrt(x))', 'x', []],
    ['subs', 'x*sqrt(2*x + 1)', 'x', []],
    ['subs', 'exp(x)/(exp(2*x) - 1)', 'x', []],
    ['subs', 'exp(x/2)/(exp(x) + 1)', 'x', []],
    ['subs', '1/(x*log(x)**2)', 'x', []],
    ['subs', 'x**2*log(x)**3', 'x', []],
    ['subs', 'sqrt(x**4 + 1)/x**5', 'x', []],
    ['subs', 'x**5*sqrt(x**3 + 1)', 'x', []],
    ['subs', 'exp(x**2)', 'x', []],
    ['subs', '1/(x + 1)', 'x', []],
    ['subs', '2*x*exp(x)/(exp(2*x) - 1)', 'x', []],
    ['subs', 't*sqrt(x)/(x + 1)', 'x', []],
    ['subs', '((x - 1)**2*(x + 1))**(1/3)/x**2', 'x', []],
    ['subs', 'sqrt(1 - sqrt(x))/(x**2 - 1)', 'x', []],
    ['forms', 'besselk(7, x)', 'x', []],
    ['implied', 'besselk(7, x)', 'x', []],
    ['subs', 'besselk(7, x)', 'x', []],
    ['forms', 'x**2*asin(sqrt(1 - x**2)/(3*sqrt(1/9 - x**2/9)))', 'x', []],
    ['implied', 'x**2*asin(sqrt(1 - x**2)/(3*sqrt(1/9 - x**2/9)))', 'x', []],
    ['forms', '1/(x**(1/3) + sqrt(x))', 'x', []],
    ['implied', '1/(x**(1/3) + sqrt(x))', 'x', []],
    ['forms', '6*_t25**3/(_t25 + 1)', '_t25', []],
    ['forms', '2**x*cosh(x)', 'x', []],
    ['implied', '2**x*cosh(x)', 'x', []],
    ['forms', 'x**(1/3)*sqrt(x**2)', 'x', []],
    ['implied', 'x**(1/3)*sqrt(x**2)', 'x', []],
    ['subs', 'x**(1/3)*sqrt(x**2)', 'x', []],
    ['forms', '3*_t28**6', '_t28', []],
    ['forms', 'log(1 - sqrt(x)) - acoth(sqrt(x))', 'x', []],
    ['implied', 'log(1 - sqrt(x)) - acoth(sqrt(x))', 'x', []],
    ['subs', 'log(1 - sqrt(x)) - acoth(sqrt(x))', 'x', []],
    ['subs', 'log(1 - 1/sqrt(x))/2 - log(1 + 1/sqrt(x))/2 + log(1 - sqrt(x))', 'x', []],
    ['forms', '2*_t31*log(1 - _t31) - 2*_t31*acoth(_t31)', '_t31', []],
    ['forms', 'acosh(x)', 'x', []],
    ['implied', 'acosh(x)', 'x', []],
    ['subs', 'acosh(x)', 'x', []],
    ['subs', 'log(x + sqrt(x - 1)*sqrt(x + 1))', 'x', []],
]


@pytest.mark.parametrize('case', CASES)
def test_agrees(case: list[Any]) -> None:
    kind = case[0]
    f, x = expr(case[1]), parse(case[2])
    assert isinstance(x, Symbol)
    facts = [relation(t) for t in case[3]]
    if kind == 'forms':
        py = outcome(lambda: Tuple(*rewritten_forms(f, x, facts or None)))
        rs = outcome(lambda: rc.logic_call('rewriting', [Symbol('forms'), f, x, Tuple(*facts)]))
    elif kind == 'implied':
        py = outcome(lambda: Tuple(*implied_assumptions(f, x, facts or None)))
        rs = outcome(lambda: rc.logic_call('rewriting', [Symbol('implied'), f, x, Tuple(*facts)]))
    else:
        py = outcome(lambda: Tuple(*[Tuple(s.integrand, s.variable, s.back, s.forward, Tuple(*s.facts))
                                     for s in power_substitutions(f, x)]))
        rs = outcome(lambda: rc.logic_call('rewriting', [Symbol('subs'), f, x]))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
