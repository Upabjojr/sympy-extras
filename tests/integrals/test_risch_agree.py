"""The Rust port of the Risch algorithm (``integrals/risch/risch.py``)
against the Python one. The Risch differential equations (``rde``,
``prde``, ``cde``) are not ported yet: the integrands which need them
are skipped."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import exp, log, sqrt
from sympy_extras.integrals.risch.risch import risch_integrate

from ..conftest import load

rc = load("sympy_extras_rs.core")

x = Symbol('x')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return re.sub(r'\b_?([a-zA-Z]+)_?\d+\b', r'\1', str(r))


INTEGRANDS: list[Expr] = [
    1 / (x**2 + 1), x / (x**2 - 1), (x**3 + 1) / (x**2 - 2), 1 / (x**3 + x), x**5, 1 / (x**4 + 1),
    (2 * x + 3) / (x**2 + 3 * x + 2)**2, 1 / (x**2 - 2)**3, (x**4 - 3) / (x**2 + x + 1), x**2 / (x - 1)**4,
    1 / (x**6 - 1), (x - sqrt(2)) / (x**2 - 2), 1 / ((x - 1) * (x - 2) * (x - 3)),
    log(x), exp(x), x * exp(x**2), log(x) / x, exp(x) / (exp(x) + 1), 1 / (x * log(x)),
]


@pytest.mark.parametrize('k', range(len(INTEGRANDS)))
def test_agrees(k: int) -> None:
    f = INTEGRANDS[k]
    py = outcome(lambda: risch_integrate(f, x))
    rs = outcome(lambda: rc.logic_call('risch_integrate', [f, x]))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
