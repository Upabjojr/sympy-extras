"""The Rust port of ``assumptions/solve.py`` against the Python ``solve``:
equations and inequalities in one unknown (polynomial, transcendental,
periodic, under assumptions and domains), systems (regular chains, the CAD,
integer linear systems) and the parametric cases."""
from __future__ import annotations

from typing import Callable, Optional

import pytest

from sympy_extras.core import Eq, Q, Rational, Symbol, Tuple, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import false, true
from sympy_extras.functions import cos, exp, sin
from sympy_extras.sets import Complexes, Integers, Interval, Naturals, Naturals0, Rationals, Reals
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.solve import solve

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y, z, a, b, c, n, u, v = symbols('x y z a b c n u v')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


Case = tuple[list[Basic], list[Basic], Optional[Basic], Optional[Basic], bool]
CASES: list[Case] = [
    ([x**2 - 2], [x], None, None, False), ([x**2 - 2], [x], x > 0, None, False), ([Eq(x**2, 2)], [x], x > 2, None, False),
    ([x**2 + 1], [x], None, Reals, False), ([x**2 + 1], [x], None, None, False),
    ([(x < 3) & (x**2 - 2 > 0)], [x], None, Reals, False), ([x**3 - x - 1], [x], None, Reals, False),
    ([x**2 < 10], [x], None, Integers, False), ([x**2 < 10], [x], None, Naturals, False),
    ([x**2 - 4], [x], element(x, Naturals), None, False), ([(x - 3)*(x - 2)*(x - 1)], [x], (x > 1) & (x < 3), None, False),
    ([x**2 - a], [x], (a > 0) & (x > 0), None, False), ([x**2 - a], [x], None, None, False), ([a*x - 1], [x], a > 0, None, False),
    ([exp(x) - 2], [x], x > 1, None, False), ([exp(x) - 2], [x], x > 0, None, False), ([sin(x)], [x], (x > 0) & (x < 4), None, False),
    ([exp(x) > 2], [x], None, Reals, False), ([sin(x)], [x], None, Reals, False),
    ([x**2 + y**2 - 1, x - y], [x, y], x > 0, None, False), ([x**2 + y**2 - 1, x - y], [x, y], None, None, False),
    ([x**2 + y**2 + 1, x - y], [x, y], None, Reals, False), ([x + y - 3, x - y - 1], [x, y], element(x, Integers), None, False),
    ([x > y, y > 0], [x, y], None, Reals, False), ([x**2 + y**2 <= 1, x + y >= 1], [x, y], None, Reals, False),
    ([x**2 + y**2 < a], [x, y], None, Reals, False), ([Eq(x**2 + y**2, 1), x > 0], [x, y], None, Reals, False),
    ([n**2 - 4], [n], Q.prime(n), None, False), ([n**2 - 9], [n], element(n, Naturals0), None, False),
    ([2*x - 1], [x], element(x, Integers), None, False), ([2*x - 1], [x], element(x, Rationals), None, False),
    ([2*x**2 + 3*sqrt(x + 6) - 1], [x], None, Reals, False), ([-x + sqrt(x + 2)], [x], None, Reals, False),
    ([sin(x) > 0], [x], None, Reals, False), ([sin(x) > 0], [x], (x > 0) & (x < 10), None, False),
    ([cos(x) < Rational(1, 3)], [x], (x > 0) & (x < 2), None, False), ([Eq(sin(x), 0)], [x], None, Reals, False),
    ([x**5 - x - y - 1, y**2 - 2], [x, y], None, Reals, False), ([x*z - y, -x + y*z], [x, y, z], None, None, False),
    ([Eq(x**3, y)], [x, y], None, Interval(-10, 10), False),
    ([a*x - 1], [x], None, None, True), ([a*x - b], [x], a > 0, None, True), ([a*x**2 + b*x + c], [x], None, None, True),
    ([a*x + y - 1, a*y + x - 1], [x, y], None, None, True), ([(x > b) & Eq(x**2, a)], [x], None, Reals, True),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_solve(i: int) -> None:
    items, unknowns, assumptions, domain, cases = CASES[i]
    got = outcome(lambda: rc.logic_call('assume_solve', [
        Tuple(*items), Tuple(*unknowns), Tuple(*([assumptions] if assumptions is not None else [])),
        domain if domain is not None else Symbol('NONE'), true if cases else false]))
    if got != 'SKIP':
        expected = outcome(lambda: solve(items, unknowns, assumptions, domain, cases))
        assert got == expected, (items, unknowns, assumptions, domain, cases)
