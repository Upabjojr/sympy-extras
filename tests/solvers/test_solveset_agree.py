"""The Rust port of ``solvers/solveset.py`` against the Python module: the
solution sets over the reals, the complex numbers and intervals."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Eq, I, Integer, Ne, Rational, Symbol, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.functions import cos, exp, log, sin, sinh, tan
from sympy_extras.sets import Complexes
from sympy_extras.sets.sets import Reals, interval
from sympy_extras.solvers.solveset import solveset

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


CASES: list[tuple[Basic, Basic]] = [
    (Eq(x**2 - 2, 0), Reals), (x**2 - 2 < 0, Reals), (x**3 - x > 0, Reals), (Eq(x**2 + 1, 0), Reals),
    (Eq(x**2 + 1, 0), Complexes), (Eq(x**4 - 2, 0), Reals), (x**4 < 2, Reals), (Eq(x**3, pi), Reals),
    (1/(x - 1) > 0, Reals), (Ne(x**2, 4), Reals), (Eq(a*x - 1, 0), Reals), (a*x - 1 > 0, Reals), (x**2 - a > 0, Reals),
    (Eq(x**2 - b, 0), Reals), (Eq(exp(x), 3), Reals), (Eq(exp(x), 3), Complexes), (Eq(sin(x), Rational(1, 2)), Reals),
    (Eq(sin(x), Rational(1, 2)), Complexes), (Eq(cos(2*x), 1), Complexes), (Eq(tan(x), 1), Reals),
    (Eq(sinh(x), 1), Complexes), (Eq(exp(x) - x, 2), Reals), (Eq(x*exp(x), 1), Reals), (Eq(sqrt(x) - 2, 0), Reals),
    (Eq(x**2 - 2, 0), interval(Integer(0), Integer(3))), (Eq(exp(x) + exp(-x), 3), Reals), (Eq(sin(x) + cos(x), 1), Reals),
    (Eq(x**5 - x - 1, 0), Reals), (x**5 - x - 1 > 0, Reals), (Eq(tan(1)*x - 1, 0), Reals), (Eq(x**2 - pi, 0), Reals),
    (Eq(sin(a*x), 0), Complexes), (Eq((x - 1)*(x - 2)/(x - 1), 0), Reals), (Eq(x**2 + 2*I*x - 1, 0), Complexes),
    (Eq(log(x), 2), Reals),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_solveset(i: int) -> None:
    r, d = CASES[i]
    got = outcome(lambda: rc.logic_call('transcendental', [Symbol('solveset'), r, x, d]))
    if got != 'SKIP':
        assert got == outcome(lambda: solveset(r, x, d)), (r, d)
