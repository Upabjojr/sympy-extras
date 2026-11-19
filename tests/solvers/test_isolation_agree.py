"""The Rust port of ``solvers/isolation.py`` against the Python module: the
isolated real roots of transcendental functions (their exact dyadic
isolating intervals), the refinement and approximation of root objects,
the comparison of points and the rational points between them."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple, oo
from sympy_extras.core.basic import Basic
from sympy_extras.core.numbers import number_from_fraction
from sympy_extras.functions import atan, cos, exp, log, sin, tan
from sympy_extras.sets.sets import Reals, interval
from sympy_extras.solvers.isolation import TranscendentalRoot, compare_points, isolate_real_roots, point_between

from ..conftest import load

rc = load("sympy_extras_rs.core")
x = Symbol('x')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
        return 'None' if r is None else str(r)
    except NotImplementedError:
        return 'SKIP'
    except Exception as e:
        return type(e).__name__


def call(op: str, *args: Basic) -> Callable[[], object]:
    def f() -> object:
        r = rc.logic_call('isolation', [Symbol(op), *args])
        if str(r) == 'NONE':
            return None
        return list(r.args) if isinstance(r, Tuple) else r
    return f


CASES: list[tuple[Basic, Basic]] = [
    (exp(x) - x - 2, Reals), (sin(x), interval(Integer(-1), Integer(1))), (x - cos(x), Reals),
    (sin(x) - x/2, interval(Integer(0), Integer(3))), (log(x) - 1, interval(Integer(0), oo, True, True)),
    (x**4 - 5*x**2 + 6, Reals), (x**3 - x - 1, Reals), (exp(x) - 3*x, Reals), (cos(x) - x**2, Reals),
    (x*exp(x) - 1, Reals), (atan(x) - x/2, Reals), (tan(x) - x, interval(Integer(-1), Integer(1))),
    (exp(-x) - sin(x), interval(Integer(0), Integer(10))), (x**5 - 3*x + 1, Reals),
    (log(x) + x, interval(Integer(0), oo, True, True)),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_isolate_real_roots(i: int) -> None:
    f, d = CASES[i]
    got = outcome(call('isolate', f, x, d, Integer(4)))
    if got != 'SKIP':
        assert got == outcome(lambda: isolate_real_roots(f, x, d)), (f, d)


def test_root_objects() -> None:
    r = TranscendentalRoot(x - cos(x), x, 0, 1)
    r2 = TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    assert outcome(call('refined', r, Integer(3))) == outcome(lambda: r.refined(3))
    assert outcome(call('approximation', r, Integer(40))) == outcome(lambda: number_from_fraction(r.approximation(40)))
    assert outcome(call('compare', r, Integer(1))) == outcome(lambda: compare_points(r, Integer(1)))
    assert outcome(call('between', r, r2)) == outcome(lambda: point_between(r, r2))
    assert outcome(call('between', r, oo)) == outcome(lambda: point_between(r, oo))
    v = rc.logic_call('reliable', [Symbol('value'), r, Integer(20), Tuple()])
    assert abs(complex(float(v.args[0]), float(v.args[1])) - 0.7390851332151607) < 1e-15
