"""The Rust ports of ``assumptions/refine.py`` (``refine``) and
``assumptions/resolve.py`` (``resolve``) against the Python ones."""
from __future__ import annotations

from typing import Callable, Optional

import pytest

from sympy_extras.core import Eq, Symbol, Tuple, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.core.mod import mod
from sympy_extras.functions import abs_, atan2, floor, log, max_, piecewise, sign
from sympy_extras.sets import Complexes, Integers, Reals
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.quantifiers import Exists, ForAll
from sympy_extras.assumptions.refine import refine
from sympy_extras.assumptions.resolve import resolve

from ..conftest import load

rc = load("sympy_extras_rs.core")
a, b, c, x, y, n = symbols('a b c x y n')


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__


def rust(arm: str, e: Basic, assumptions: list[Basic], domain: Optional[Basic]) -> str:
    return outcome(lambda: rc.logic_call(arm, [e, Tuple(*assumptions), domain if domain is not None else Symbol('NONE')]))


REFINE: list[tuple[Basic, list[Basic], Optional[Basic]]] = [
    (abs_(x - 1) + sqrt(x**2), [x > 2], None), (sqrt(x**2 - 2*x + 1), [x > 1], None),
    (log(x**2) + sqrt(x**2*y**2), [x > 0, y > 0], None), (sign(x*y), [x > 0, y < 0], None),
    (max_(x, x**2), [x > 0, x < 1], None), (floor(n + 1), [element(n, Integers)], None),
    (atan2(y, x - 1), [x > 1], None), ((x > 0) & (y > 0), [x > 1], None), (x*y > 0, [x > 0, y > 0], None),
    (mod(n**2 + n, 2), [element(n, Integers)], None), (abs_(x**2 - 1), [x > 1], None),
    (piecewise((x, x > 0), (-x, True)), [x > 1], None), (sqrt(x**2), [], Reals),
    (ForAll(y, x*y**2 >= 0), [x > 0], None), (abs_(x*y), [x < 0, y > 0], None),
]

RESOLVE: list[tuple[Basic, list[Basic], Optional[Basic]]] = [
    (ForAll(x, x**2 + b*x + c > 0), [], None), (Exists(y, Eq(x**2 + y**2, 1)), [], None),
    (ForAll(x, Exists(y, y > x)), [], None), (Exists(y, ForAll(x, y > x)), [], None),
    (Exists(x, Eq(2*x, y)), [], Integers), (Exists(x, Eq(a*x, 1)), [], Complexes),
    (Exists(x, x**2 < a), [a > -1], None), (ForAll(x, x**2 >= 0), [], Integers), (x**2 < 4, [], None),
    (Exists(x, (x > a) & (x < b)), [], None),
]


@pytest.mark.parametrize('i', range(len(REFINE)))
def test_refine(i: int) -> None:
    e, assumptions, domain = REFINE[i]
    got = rust('assume_refine', e, assumptions, domain)
    if got != 'SKIP':
        assert got == outcome(lambda: refine(e, assumptions, domain)), (e, assumptions)


@pytest.mark.parametrize('i', range(len(RESOLVE)))
def test_resolve(i: int) -> None:
    f, assumptions, domain = RESOLVE[i]
    got = rust('assume_resolve', f, assumptions, domain)
    if got != 'SKIP':
        assert got == outcome(lambda: resolve(f, domain, assumptions)), (f, assumptions, domain)
