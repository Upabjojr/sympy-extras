"""A check of the cases of the isolated values of the parameters of a sum
against the sum computed term by term, shared by the tests of the
summation functions."""
from __future__ import annotations

from typing import Optional, Sequence

from sympy import Add, Expr, Or, Piecewise, Rational, Symbol

from sympy_extras._numeric import reliable_value
from sympy_extras._special_values import Point, condition_point
from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols


def case_points(value: Expr) -> list[Point]:
    """The points of the cases of ``value`` (every branch but the last),
    each disjunct of an ``Or`` for its own."""
    assert isinstance(value, Piecewise), value
    points: list[Point] = []
    for pair in value.args[:-1]:
        condition = as_boolean(pair.args[1])
        for disjunct in (condition.args if isinstance(condition, Or) else (condition,)):
            point = condition_point(as_boolean(disjunct))
            assert point is not None, disjunct
            points.append(point)
    return points


def _samples(symbols: Sequence[Symbol]) -> Point:
    return {s: Rational(7 + 4 * i, 3) for i, s in enumerate(symbols)}


def _close(got: Expr, expected: Expr) -> bool:
    difference = reliable_value(as_expr(got - expected), 15)
    size = reliable_value(expected, 15)
    return difference is not None and size is not None and float(abs(difference)) < 1e-10 * (1 + float(abs(size)))


def check_sum_cases(value: Optional[Expr], term: Expr, k: Symbol, lower: int, upper: Expr, n: Symbol,
                    tops: Sequence[int] = (0, 1, 2, 3)) -> int:
    """Every case of ``value``, the sum of ``term`` from ``lower`` to
    ``upper`` (in ``n``), agrees at its point with the sum computed term
    by term for ``n`` in ``tops``, the other parameters at sample values;
    the whole ``Piecewise`` is evaluated at the point, so that the case is
    the branch taken there. Returns the number of points checked."""
    assert value is not None
    points = case_points(value)
    for point in points:
        at_value = as_expr(value.xreplace(point))
        at_term = as_expr(term.xreplace(point))
        others = sorted_symbols((free_symbols(at_value) | free_symbols(at_term)) - {k, n})
        samples = _samples(others)
        for m in tops:
            top = int(as_expr(upper.subs(n, m)))
            direct = as_expr(Add(*[at_term.xreplace(samples).subs(n, m).subs(k, i) for i in range(lower, top + 1)]))
            got = as_expr(at_value.xreplace(samples).subs(n, m).doit())
            assert _close(got, direct), (point, m, got, direct)
    return len(points)


def check_antidifference_cases(value: Optional[Expr], term: Expr, k: Symbol) -> int:
    """Every case of ``value``, an antidifference of ``term`` in ``k``,
    has at its point the difference ``term`` at a few values of ``k``,
    the other parameters at sample values. Returns the number of points
    checked."""
    assert value is not None
    points = case_points(value)
    for point in points:
        at_value = as_expr(value.xreplace(point))
        at_term = as_expr(term.xreplace(point))
        samples = _samples(sorted_symbols((free_symbols(at_value) | free_symbols(at_term)) - {k}))
        for i in range(2, 5):
            got = as_expr((at_value.subs(k, i + 1) - at_value.subs(k, i)).xreplace(samples))
            assert _close(got, as_expr(at_term.subs(k, i).xreplace(samples))), (point, i)
    return len(points)
