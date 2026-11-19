"""The Rust port of ``assumptions/sat.py`` (``satisfiable``, ``tautology``,
``find_instance``) against the Python one. The cases have one
propositional structure, so the first model does not depend on the order
of the atoms (Python's ``atoms()`` set, which changes with the hash seed)."""
from __future__ import annotations

from typing import Callable, Optional

import pytest

from sympy_extras.core import Eq, Integer, Symbol, Tuple, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import BooleanFalse, BooleanTrue
from sympy_extras.sets import Integers, Reals
from sympy_extras.assumptions.sat import find_instance, satisfiable, tautology

from ..conftest import load

rc = load("sympy_extras_rs.core")
p, q, x, y = symbols('p q x y')


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__


def dom(d: Optional[Basic]) -> Basic:
    return d if d is not None else Symbol('NONE')


def model(t: Basic) -> object:
    if isinstance(t, (BooleanTrue, BooleanFalse)) or str(t) == 'NONE':
        return None if str(t) == 'NONE' else t
    return {k: (True if isinstance(v, BooleanTrue) else False if isinstance(v, BooleanFalse) else v) for k, v in t.args}


SATISFIABLE: list[tuple[Basic, Optional[Basic]]] = [
    (p & ~q, None), (p & ~p, None), ((x**2 + y**2 < 1) & (x + y > 1), None),
    ((p | (x > 1)) & (~p | (x < 0)) & (x > 0), None), (Eq(x**2, 2), Integers), ((x > 1) & (x < 3), Integers),
    (Eq(x**2, -1), Reals), (Eq(x**2, -1), None), ((x > 0) & (x < 1) & (y > x), None), (Eq(x**3, 8), Reals),
]

TAUTOLOGY: list[tuple[Basic, Optional[Basic]]] = [
    (p | ~p, None), (p | q, None), (x**2 + y**2 >= 2*x*y, Reals), (x**2 > 0, Reals), ((x > 1) >> (x > 0), Reals),
]

FIND: list[tuple[Basic, list[Basic], Optional[Basic], int]] = [
    ((x**2 + y**2 < 1) & (x > y), [x, y], Reals, 1), (Eq(x**2, 4), [x], Reals, 2), (Eq(x**2, 2), [x], Integers, 1),
    ((x**2 > 5) & (x < 0), [x], Integers, 1), (x**2 < 0, [x], Reals, 1), ((x > 2) & (x < 3), [x], Integers, 1),
]


@pytest.mark.parametrize('i', range(len(SATISFIABLE)))
def test_satisfiable(i: int) -> None:
    f, d = SATISFIABLE[i]
    got = outcome(lambda: model(rc.logic_call('sat', [Symbol('satisfiable'), f, Tuple(), dom(d)])))
    if got != 'SKIP':
        assert got == outcome(lambda: satisfiable(f, domain=d)), (f, d)


@pytest.mark.parametrize('i', range(len(TAUTOLOGY)))
def test_tautology(i: int) -> None:
    f, d = TAUTOLOGY[i]
    got = outcome(lambda: model(rc.logic_call('sat', [Symbol('tautology'), f, Tuple(), dom(d)])))
    if got != 'SKIP':
        assert got == outcome(lambda: tautology(f, domain=d)), (f, d)


@pytest.mark.parametrize('i', range(len(FIND)))
def test_find_instance(i: int) -> None:
    f, vs, d, n = FIND[i]

    def rust() -> object:
        r = rc.logic_call('sat', [Symbol('find_instance'), f, Tuple(), dom(d), Tuple(*vs), Integer(n)])
        return None if str(r) == 'NONE' else [{k: v for k, v in w.args} for w in r.args]
    got = outcome(rust)
    if got != 'SKIP':
        assert got == outcome(lambda: find_instance(f, vs, d, count=n)), (f, vs, d)
