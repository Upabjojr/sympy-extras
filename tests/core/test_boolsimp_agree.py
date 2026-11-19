"""The Rust ports of ``logic/boolsimp.py`` (``simplify_boolean``,
``simplify_univariate``) and of ``simplify`` under assumptions
(``assumptions/refine.py``) against the Python ones."""
from __future__ import annotations

from typing import Callable, Optional

import pytest

from sympy_extras.core import Eq, Ne, Symbol, Tuple, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import and_, not_, or_
from sympy_extras.functions import abs_, cos, exp, log, sin
from sympy_extras.sets import Reals
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.refine import simplify
from sympy_extras.logic.boolsimp import simplify_boolean, simplify_univariate

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = symbols('x y')
r = Symbol('r', real=True)


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__


BOOLEANS: list[Basic] = [
    and_(x > 1, x < 0), and_(x > 0, Ne(x, 0)), or_(x > 2, x < -2), and_(Eq(x, 1), y > x),
    not_(and_(r > 0, r < 1)), and_(r <= 1, r <= 2), or_(x < 1, and_(x >= 1, x < 2)), or_(x >= 1, Eq(x, 0)),
    and_(x > y, y > x), or_(x <= y, y <= x), and_(x >= y, y >= x), and_(x < 3, x < y, y < 3),
    and_(Eq(x, 2), Eq(y, x + 1), y > 0), or_(x**2 > 4, x < 0), and_(x**2 < 1, x > 0),
]

UNIVARIATE: list[Basic] = [
    and_(x > 1, x < 3), or_(x**2 > 4, x < 0), and_(x > 1, x < 0), and_(sin(x) > 0, x < 0), or_(x <= -3, x < 1),
    and_(x**2 - 2 > 0, x < 5), or_(x**3 - x > 0, x > 10),
]

SIMPLIFY: list[tuple[Basic, list[Basic], Optional[Basic]]] = [
    ((x**2 - 1)/(x - 1), [x > 1], None), (Eq(x**2, 1), [x > 0], None), (or_(x > 1, x**2 > 1), [], Reals),
    (sqrt(x**2) + abs_(x)*sin(x)**2 + abs_(x)*cos(x)**2, [x < 0], None), (log(exp(x)), [element(x, Reals)], None),
    (and_(x > 0, x > 1), [], Reals), (x**2 + 2*x + 1, [], None),
]


@pytest.mark.parametrize('i', range(len(BOOLEANS)))
def test_simplify_boolean(i: int) -> None:
    f = BOOLEANS[i]
    got = outcome(lambda: rc.logic_call('simplify_boolean', [f]))
    if got != 'SKIP':
        assert got == outcome(lambda: simplify_boolean(f)), f


@pytest.mark.parametrize('i', range(len(UNIVARIATE)))
def test_simplify_univariate(i: int) -> None:
    f = UNIVARIATE[i]
    got = outcome(lambda: rc.logic_call('simplify_univariate', [f]))
    if got != 'SKIP':
        assert got == outcome(lambda: simplify_univariate(f)), f


@pytest.mark.parametrize('i', range(len(SIMPLIFY)))
def test_simplify_under_assumptions(i: int) -> None:
    e, assumptions, domain = SIMPLIFY[i]
    got = outcome(lambda: rc.logic_call('assume_simplify',
                                        [e, Tuple(*assumptions), domain if domain is not None else Symbol('NONE')]))
    if got != 'SKIP':
        assert got == outcome(lambda: simplify(e, assumptions, domain)), (e, assumptions)
