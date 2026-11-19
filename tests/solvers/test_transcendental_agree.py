"""The Rust port of ``solvers/transcendental.py`` against the Python
module: the real solution sets of transcendental equations and
inequalities, and the reductions to kernel variables."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Eq, Ne, Rational, Symbol, true
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import And, Or
from sympy_extras.functions import Abs, cos, cosh, exp, log, sin, sinh, sqrt, tan
from sympy_extras.solvers.transcendental import polynomialize, solve_transcendental

from ..conftest import load

rc = load("sympy_extras_rs.core")
x = Symbol('x')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
        return 'NONE' if r is None else str(r)
    except NotImplementedError:
        return 'SKIP'
    except Exception as e:
        return type(e).__name__


CASES: list[Basic] = [
    Eq(exp(2*x) - 3*exp(x) + 2, 0), exp(x) + exp(-x) > 3, Eq(sin(x) + cos(x), 1), Eq(log(x)**2 - 3*log(x) + 2, 0),
    Eq(exp(x), 2), exp(x) < 2, Eq(sqrt(x) - 2, 0), sqrt(x) + x > 2, Eq(sin(x), Rational(1, 2)), sin(x)**2 - sin(x) > 0,
    Eq(2**x, 8), Eq(sinh(x), 1), Eq(cosh(x), 2), Eq(Abs(exp(x) - 2), 1), Eq(x*exp(x), 1), Eq(x + log(x), 2),
    Eq(tan(x), 1), Eq(exp(x) + 1, 0), Eq(cos(2*x), cos(x)), Eq(exp(3*x) - exp(x), 0), Eq(log(x + 1), 1),
    Eq(x - sqrt(x) - 2, 0), Eq(exp(x)*(x**2 - 1), 0), And(exp(x) > 1, exp(x) < 3), Or(Eq(exp(x), 1), Eq(exp(x), 4)),
    Eq(sin(2*x), sin(x)), Eq(1/(exp(x) - 1), 2), Eq(exp(x) - x, 2), Eq(log(x), -1), Ne(exp(x), 3),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_solve_transcendental(i: int) -> None:
    f = CASES[i]
    r = outcome(lambda: rc.logic_call('transcendental', [Symbol('solve'), f, x, true]))
    if r != 'SKIP':
        assert r == outcome(lambda: solve_transcendental(f, x)), f


def test_polynomialize() -> None:
    for f in [Eq(exp(2*x) - 3*exp(x) + 2, 0), Eq(sin(x) + cos(x), 1), Eq(sqrt(x) + x, 2), Eq(log(x)**2, 1), Eq(exp(x), x)]:
        p = polynomialize(f, x)
        r = rc.logic_call('transcendental', [Symbol('polynomialize'), f, x])
        if p is None:
            assert str(r) == 'NONE'
        else:
            # the kernel variables are dummies: compared by the formula's shape
            assert str(r.args[3]) == p.family and str(r.args[2]) == str(p.argument) and int(r.args[4]) == p.order
