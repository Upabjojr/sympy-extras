"""The Rust ``solve_inequality`` (``solvers/inequalities.py``) against the
Python one."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Eq, Gt, Ne, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.functions import Abs, exp, log, sin, sqrt
from sympy_extras.solvers.inequalities import solve_inequality

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, a, b = Symbol('x'), Symbol('a', positive=True), Symbol('b')


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as e:
        return type(e).__name__


CASES: list[Basic] = [
    x - 2 > 0, 2*x >= 4, a*x < 1, x**2 < 4, x**2 > 2, b*x > 1, Eq(2*x, 6), Ne(x, 3), x < x*(2/x - 1), 1/x > 1,
    exp(x) > 2, log(x) < 1, Abs(x - 1) < 2, sqrt(x) > 1, x**3 - x >= 0, Gt(3, x), x + 1 > x, (x - 1)/(x + 2) <= 0,
    sin(x) > 2, x**2 + 1 > 0, exp(x) < -1,
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_solve_inequality(i: int) -> None:
    ie = CASES[i]
    got = outcome(lambda: rc.logic_call('transcendental', [Symbol('inequality'), ie, x]))
    if got != 'SKIP':
        assert got == outcome(lambda: solve_inequality(ie, x)), ie
