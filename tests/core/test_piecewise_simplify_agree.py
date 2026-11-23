"""The Rust port of ``piecewise_simplify`` (and ``piecewise_intervals``,
the ``sympy_intervals`` univariate simplification of the conditions)
against the Python one on random ``Piecewise`` of linear and a few
nonlinear conditions, in one variable and in two, and with ``Xor``."""
from __future__ import annotations

import random

import pytest

from sympy_extras.core import Integer, Rational, Symbol, and_, or_
from sympy_extras.core.logic import not_, true, xor
from sympy_extras.core.relational import ge, gt, le, lt, ne
from sympy_extras.functions import sin
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise, piecewise_simplify

from ..conftest import load

rc = load("sympy_extras_rs.core")

x, y = Symbol('x'), Symbol('y')


def outcome(f: object) -> str:
    try:
        return str(f())  # type: ignore[operator]
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__


def linear_cases(seed: int, count: int) -> list[Piecewise]:
    rng = random.Random(seed)
    consts = [Integer(-2), Integer(-1), Integer(0), Rational(1, 2), Integer(1), Integer(2), Integer(3)]
    exprs = [x, x**2, Integer(1), Integer(0), 2*x + 1, sin(x), x*y, y]

    def cond() -> object:
        rel = rng.choice([lt, le, gt, ge, ne])
        lhs = rng.choice([x, 2*x, -x, x + 1, 3 - x] + ([x**2] if rng.random() < 0.1 else []) + [y])
        r = rel(lhs, rng.choice(consts))
        k = rng.random()
        if k < 0.25:
            return and_(r, rng.choice([lt, gt])(x, rng.choice(consts)))
        if k < 0.35:
            return or_(r, gt(x, rng.choice(consts)))
        return r
    found: list[Piecewise] = []
    for _ in range(count):
        pairs = [(rng.choice(exprs), cond()) for _ in range(rng.randint(1, 3))] + [(rng.choice(exprs), true)]
        e = piecewise(*pairs)
        if isinstance(e, Piecewise):
            found.append(e)
    return found


def xor_cases(seed: int, count: int) -> list[Piecewise]:
    rng = random.Random(seed)
    consts = [Integer(-2), Integer(-1), Integer(0), Rational(1, 2), Integer(1), Integer(2)]

    def rel() -> object:
        r = rng.choice([lt, le, gt, ge, ne])
        return r(rng.choice([x, 2*x, -x, y]), rng.choice(consts))

    def cond() -> object:
        k = rng.random()
        c = xor(rel(), rel())
        if k < 0.3:
            return not_(c)
        if k < 0.5:
            return and_(c, rel())
        return c
    found: list[Piecewise] = []
    for _ in range(count):
        e = piecewise((x, cond()), (y, true))
        if isinstance(e, Piecewise):
            found.append(e)
    return found


CASES: list[Piecewise] = linear_cases(1, 120) + linear_cases(7, 80) + xor_cases(2, 40)


@pytest.mark.parametrize('e', CASES, ids=[str(c)[:80] for c in CASES])
def test_agree(e: Piecewise) -> None:
    rust = outcome(lambda: rc.logic_call('piecewise_simplify', [e]))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: piecewise_simplify(e))
