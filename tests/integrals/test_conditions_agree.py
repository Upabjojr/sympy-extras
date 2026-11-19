"""The Rust port of ``integrals/conditions.py`` against the Python one: the
conditions decided under assumptions, the sample points of a region and
the numerical comparisons. ``numerically_equal`` of values which agree
asks the structure theorem, which is not ported yet ("not ported",
skipped)."""
from __future__ import annotations

import random
from typing import Callable, Optional

import pytest

from sympy_extras.core import I, Integer, Symbol, Tuple, pi, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import true
from sympy_extras.functions import Abs, arg, cos, exp, log, sin
from sympy_extras.sets import Reals
from sympy_extras.assumptions.facts import element
from sympy_extras.integrals.conditions import (canonical, decide, numerical_verdict, numerically_equal, positive,
                                               real_form, sample_values)

from ..conftest import load

rc = load("sympy_extras_rs.core")
a, b, k, x, y = symbols('a b k x y')
n = Symbol('n', positive=True, integer=True)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


def call(*args: Basic) -> Callable[[], object]:
    return lambda: rc.logic_call('conditions', list(args))


DECIDE: list[tuple[Basic, list[Basic]]] = [
    ((a > 0) & (b > 1), [a > 1]), (a < 0, [a > 1]), (a + 1 > 0, [a > 0]), ((a > 0) | (b > 0), [a > 1]),
    (Abs(arg(a)) < pi/2, [element(a, Reals)]), (Abs(arg(a)) < pi/2, [a > 0]), (a*b > 0, [a > 0, b > 0]),
]

VERDICT: list[tuple[Basic, Basic, list[Basic], Basic]] = [
    (sqrt(a**2), a, [a > 0], true), (sqrt(a**2), a, [a < 0], true), (log(-a), log(a) + I*pi, [], a > 0),
    (sin(x)**2 + cos(x)**2, Integer(1), [], true), (exp(x)*exp(y), exp(x + y), [x > 0, y < 1], true),
    (sqrt(x*y), sqrt(x)*sqrt(y), [x < 0, y < 0], true), (Abs(x), x, [x > -1], true),
]


@pytest.mark.parametrize('i', range(len(DECIDE)))
def test_decide(i: int) -> None:
    c, assumptions = DECIDE[i]
    got = outcome(call(Symbol('decide'), c, Tuple(*assumptions)))
    if got != 'SKIP':
        assert got == outcome(lambda: decide(c, assumptions)), (c, assumptions)


def test_small_functions() -> None:
    assert outcome(call(Symbol('canonical'), k + 1 < 1)) == outcome(lambda: canonical(k + 1 < 1))
    for c, assumptions in [(Abs(arg(a)) < pi/2, [element(a, Reals)]), (Abs(arg(a)) < 2*pi, [])]:
        assert outcome(call(Symbol('real_form'), c, Tuple(*assumptions))) == outcome(lambda: real_form(c, assumptions))
    for e, assumptions in [(a*Abs(b), [a > 0]), (a, []), (a**2, [element(a, Reals)])]:
        assert outcome(call(Symbol('positive'), e, Tuple(*assumptions))) == outcome(lambda: positive(e, assumptions))


@pytest.mark.parametrize('i', range(len(VERDICT)))
def test_numerical_verdict(i: int) -> None:
    p, q, assumptions, condition = VERDICT[i]
    got = outcome(call(Symbol('verdict'), p, q, Tuple(*assumptions), condition, Integer(3)))
    if got != 'SKIP':
        assert got == outcome(lambda: numerical_verdict(p, q, assumptions, condition)), (p, q)


def test_numerically_equal() -> None:
    for p, q, assumptions in [(sqrt(x**2), x, [x < 0]), (sin(x)**2 + cos(x)**2, Integer(1), []),
                              (exp(-10**6*x**2), Integer(0), [x > 1])]:
        got = outcome(call(Symbol('equal'), p, q, Tuple(*assumptions), Integer(3)))
        if got != 'SKIP':
            assert got == outcome(lambda: numerically_equal(p, q, assumptions)), (p, q)


@pytest.mark.parametrize('seed', ['a', 'region', 'xy'])
def test_sample_values(seed: str) -> None:
    def rust(syms: list[Basic], assumptions: list[Basic]) -> object:
        r = rc.logic_call('conditions', [Symbol('sample_values'), Tuple(*syms), Tuple(*assumptions), Symbol(seed)])
        return None if str(r) == 'NONE' else {s: v for s, v in r.args}
    for syms, assumptions in [([x], []), ([x, y], [x > 0, y < x]), ([n, x], [x > n]), ([x], [x > -sqrt(2)/2, x < 0])]:
        got = outcome(lambda: rust(syms, assumptions))
        if got != 'SKIP':
            assert got == outcome(lambda: sample_values(syms, assumptions, random.Random(seed))), (syms, assumptions)
