"""The Rust port of ``assumptions/ask.py`` against the Python ``ask``: the
predicates, the polynomial formulas (virtual substitution, the CAD), the
realness of roots and logarithms, calculus and the bounds of the
functions. The queries the port does not decide yet (the structure
theorem, the sample points, the quantifiers) raise "not ported" and are
skipped."""
from __future__ import annotations

from typing import Optional

import pytest

from sympy_extras.core import Eq, Q, Symbol, Tuple, pi, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.functions import abs_, atan, exp, floor, log, sin
from sympy_extras.sets import Integers, Interval, Reals
from sympy_extras.assumptions.ask import ask
from sympy_extras.assumptions.facts import element

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = symbols('x y')
a = Symbol('a', positive=True)


def rust(query: Basic, assumptions: list[Basic], domain: Optional[Basic]) -> str:
    try:
        r = rc.logic_call('assume_ask', [query, Tuple(*assumptions), domain if domain is not None else Symbol('NONE')])
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if str(r) == 'NONE' else str(r)


def python(query: Basic, assumptions: list[Basic], domain: Optional[Basic]) -> str:
    try:
        return str(ask(query, assumptions, domain))
    except Exception as e:
        return type(e).__name__


CASES: list[tuple[Basic, list[Basic], Optional[Basic]]] = [
    (x**3 > 0, [x > 0], None), (x**2 - 2*x + 1 >= 0, [x > 0], None), (x > 1, [x > 2], None), (x > 3, [x > 2], None),
    (x*y > 0, [x > 0, y < 0], None), (element(x + 1, Integers), [element(x, Integers)], None),
    (x**2 >= 0, [], None), (x**2 >= 0, [], Reals), (Q.positive(x), [Eq(x, 3)], None),
    (x + y > 0, [x > 0, y > 0], None), (x - y > 0, [x > y], None), (x**2 > 4, [x > 2], None),
    (Q.real(sqrt(x - 2)), [x > 3], None), (Q.real(sqrt(x - 2)), [x < 1], None), (Q.real(log(x)), [x > 0], None),
    (Q.real(log(x)), [x < 0], None), (exp(x) > 0, [element(x, Reals)], None), (sin(x) <= 1, [element(x, Reals)], None),
    (atan(x) < 2, [element(x, Reals)], None), (abs_(x) >= 0, [element(x, Reals)], None), (log(x) < x, [x > 0], None),
    (exp(x) > x, [element(x, Reals)], None), (floor(x) <= x, [element(x, Reals)], None), (pi > 3, [], None),
    (element(x, Interval(0, 1)), [x > 0, x < 1], None), (a > 0, [], None), (a*x > 0, [x > 0], None),
    ((x > 0) | (x < 1), [element(x, Reals)], None), ((x > 0) & (x < 0), [], Reals), (x > 0, [x < -1], None),
    (Eq(x**2, 1), [Eq(x, 1)], None), (x**2 + y**2 >= 2*x*y, [], Reals), (Q.integer(x**2), [Q.integer(x)], None),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_ask(i: int) -> None:
    q, assumptions, domain = CASES[i]
    got = rust(q, assumptions, domain)
    if got != 'SKIP':
        assert got == python(q, assumptions, domain), (q, assumptions, domain)
