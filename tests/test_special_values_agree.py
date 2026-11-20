"""The Rust port of ``_special_values.py`` against the Python one: the
special points of the parameters, the points of conditions, the
assumptions at a point, the undefined values and ``with_special_values``
(its ``compute`` substitutes the point into a second expression)."""
from __future__ import annotations

from typing import Callable, Optional

import pytest

from sympy_extras._special_values import (Point, condition_point, defined_problem, equality_points, point_assumptions,
                                          special_points, undefined_at, with_special_values)
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, false, to_boolean, true
from sympy_extras.core.relational import Eq, Gt, Lt, Ne
from sympy_extras.functions import DiracDelta, Piecewise, exp, log, sin, sqrt, zeta
from sympy_extras.integrals.integral import Integral

from .conftest import load

rc = load("sympy_extras_rs.core")

x, a, b, n, s = Symbol('x'), Symbol('a'), Symbol('b'), Symbol('n'), Symbol('s')
k = Symbol('k', integer=True)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


def point_tuple(p: Point) -> Tuple:
    return Tuple(*[Tuple(q, v) for q, v in sorted(p.items(), key=lambda item: item[0].name)])


def items(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, Boolean):
        return [assumptions]
    if isinstance(assumptions, bool):
        return [to_boolean(assumptions)]
    return [to_boolean(c) for c in assumptions]


#: (value, parameters, assumptions)
VALUES: list[tuple[Expr, list[Symbol], Assumptions]] = [
    (x**(a + 1) / (a + 1), [a], None),
    (log(a) * x, [a], None),
    (1 / (a**2 - 1), [a], None),
    (1 / (a - b), [a, b], None),
    (zeta(s), [s], None),
    (x**(n + 1) / (n + 1), [n], [Gt(n, 0)]),
    (x**(n + 1) / (n + 1), [n], [Lt(n, 0)]),
    (1 / (a**3 - 2), [a], None),
    (1 / (a**3 - 8), [a], None),
    (1 / sin(a), [a], None),
    (1 / (exp(a) - 1), [a], None),
    (Piecewise((1 / a, Gt(a, 0)), (Integer(0), true)), [a], None),
    (1 / (a * b), [a, b], None),
    (x / (a**2 + 1), [a], None),
    (sqrt(a) / (a - 4), [a], None),
    (Piecewise((Integral(x**a, x), Eq(a, 2)), (x / (a - 1), true)), [a], None),
    (log(a - b) / (b + 1), [a, b], [Gt(a, b)]),
    (1 / (k + Integer(1) / 2), [k], None),
    (x**a, [a], None),
    (1 / (a**2 + 2 * a + 1), [a], None),
    (x * DiracDelta(a) / a, [a], None),
    (x * DiracDelta(x - a), [a], None),
]

#: the second expressions of with_special_values
ALTERNATIVES: list[Expr] = [log(x), x**2, a + x, Integer(1) / (a + 2), Integral(log(x) * a, x)]

CONDITIONS: list[Boolean] = [
    Eq(a, 1), Eq(a, 1) & Eq(b, a), Eq(a, b + 1) & Eq(b, 2), Eq(a, 1) | Eq(a, 2), true, Gt(a, 1),
    Eq(a, b) & Eq(b, a), Eq(a + b, 1), Eq(Integer(1), a) & Ne(b, 0), Eq(a, x**2),
]

POINTS: list[tuple[Point, Assumptions]] = [
    ({a: Integer(-1)}, [Gt(a, 0)]), ({a: Integer(1)}, [Gt(a, 0)]), ({a: Integer(0)}, [Gt(a, 0) | Lt(b, 0)]),
    ({a: Integer(2), b: Integer(3)}, [Gt(a, b)]), ({a: b}, [Gt(a, b)]), ({a: Integer(1)}, None),
]

CASES: list[tuple[str, int, int]] = (
    [('points', i, 0) for i in range(len(VALUES))]
    + [('equality', i, 0) for i in range(len(VALUES))]
    + [('condition', i, 0) for i in range(len(CONDITIONS))]
    + [('assumptions', i, 0) for i in range(len(POINTS))]
    + [('undefined', i, j) for i in range(len(VALUES)) for j in range(len(POINTS))]
    + [('defined', i, 0) for i in range(len(VALUES))]
    + [('with', i, j) for i in range(len(VALUES)) for j in range(len(ALTERNATIVES) * 4)]
)


@pytest.mark.parametrize('op, i, j', CASES)
def test_agrees(op: str, i: int, j: int) -> None:
    args: list[Basic]
    if op in ('points', 'equality', 'defined', 'with', 'undefined'):
        value, parameters, assumptions = VALUES[i]
    if op == 'points':
        py = outcome(lambda: Tuple(*[point_tuple(p) for p in special_points(value, parameters, assumptions)]))
        args = [value, Tuple(*parameters), Tuple(*items(assumptions))]
    elif op == 'equality':
        py = outcome(lambda: Tuple(*[point_tuple(p) for p in equality_points(value, parameters)]))
        args = [value, Tuple(*parameters)]
    elif op == 'condition':
        c = CONDITIONS[i]
        found_point = condition_point(c)
        py = 'None' if found_point is None else str(point_tuple(found_point))
        args = [c]
    elif op == 'assumptions':
        point, given = POINTS[i]
        found = point_assumptions(point, given)
        py = 'None' if found is None else str(Tuple(*found))
        args = [point_tuple(point), Tuple(*items(given))]
    elif op == 'undefined':
        point = POINTS[j][0]
        py = str(undefined_at(value, point)).lower()
        args = [value, point_tuple(point)]
    elif op == 'defined':
        py = str(defined_problem(value, [x])).lower()
        args = [value, Tuple(x)]
    else:
        alt = ALTERNATIVES[j // 4]
        sympy_style, keep = bool(j % 2), bool((j // 2) % 2)

        def compute(point: Point, at: Assumptions) -> Optional[Expr]:
            rule: dict[Basic, Basic] = {q: v for q, v in point.items()}
            replaced = alt._xreplace_evaluated(rule)
            assert isinstance(replaced, Expr)
            return replaced
        py = outcome(lambda: with_special_values(value, parameters, assumptions, compute, sympy_style=sympy_style,
                                                 keep_unevaluated=keep))
        args = [value, Tuple(*parameters), Tuple(*items(assumptions)), alt, true if sympy_style else false,
                true if keep else false]
    rs = outcome(lambda: rc.logic_call('special_values', [Symbol(op)] + args))
    if op in ('undefined', 'defined'):
        rs = rs.lower()
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
