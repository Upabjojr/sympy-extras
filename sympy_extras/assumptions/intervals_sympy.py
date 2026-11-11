"""The SymPy face of the rigorous interval arithmetic
(:mod:`sympy_extras.assumptions.intervals`).

The functions take and return SymPy objects (expressions, boxes of SymPy
symbols, SymPy intervals), converted by :mod:`sympy_extras.interop`
around the native functions; the enclosures are mpmath intervals in both.

Examples
========

>>> from sympy import S, exp, sin, Interval, pi
>>> from sympy.abc import x, y
>>> from sympy_extras.assumptions.intervals_sympy import evaluate, interval_sign, signs_on_box, isolate_signs
>>> interval_sign(evaluate(exp(x) + 1, {x: (S(-1), S(1))}))
1
>>> signs_on_box(x**2 + y**2 + 1, {x: (S(-1), S(1)), y: (S(-1), S(1))})
frozenset({1})
>>> isolate_signs(sin(x), x, Interval.open(0, pi))
frozenset({1})
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

from mpmath.ctx_iv import ivmpf

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.sets.sets import Interval, Set

from sympy_extras._typing import Sign, as_expr, as_set
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import SymPySetNode, from_sympy, to_sympy
from sympy_extras.sets.sets import Interval as CoreInterval
from sympy_extras.sets.sets import Set as CoreSet

from . import intervals as native

__all__ = ['Box', 'Openness', 'Signs', 'evaluate', 'interval_sign', 'signs_on_box', 'range_on_box',
    'corner_signs', 'corner_value', 'monotone_range', 'isolate_signs']

#: a box: for every variable its closed range, with exact endpoints
#: (``-oo`` and ``oo`` allowed)
Box = dict[Symbol, tuple[Expr, Expr]]
#: the possible signs of an expression on a set
Signs = frozenset[Sign]
#: a way of finding the signs of an expression (a partial derivative)
SignFinder = Callable[[Expr], Optional[Signs]]
#: for every variable of a box, whether its range is open at the left and
#: at the right end
Openness = dict[Symbol, tuple[bool, bool]]

interval_sign = native.interval_sign


# ---------------------------------------------------------------------------
# the conversions at the boundary

def core_expr(e: Basic) -> CoreExpr:
    """A SymPy expression as an expression of the core."""
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def core_symbol(s: Basic) -> CoreSymbol:
    """A SymPy symbol as a symbol of the core."""
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def core_set(s: Set) -> Optional[CoreSet]:
    """A SymPy set as a set of the core, ``None`` when it is not ported
    (the wrapper of a set of SymPy is not)."""
    converted = from_sympy(s)
    return converted if isinstance(converted, CoreSet) and not isinstance(converted, SymPySetNode) else None


def sympy_expr(e: CoreBasic) -> Expr:
    """An expression of the core as a SymPy expression."""
    return as_expr(to_sympy(e))


def sympy_set(s: CoreSet) -> Set:
    """A set of the core as a SymPy set."""
    return as_set(to_sympy(s))


def core_box(box: Mapping[Symbol, tuple[Expr, Expr]]) -> native.Box:
    return {core_symbol(v): (core_expr(lo), core_expr(hi)) for v, (lo, hi) in box.items()}


def core_openness(openness: Openness) -> native.Openness:
    return {core_symbol(v): ends for v, ends in openness.items()}


def _core_finder(partial_sign: Optional[SignFinder]) -> Optional[native.SignFinder]:
    if partial_sign is None:
        return None
    finder = partial_sign

    def on_core(derivative: CoreExpr) -> Optional[Signs]:
        return finder(sympy_expr(derivative))
    return on_core


# ---------------------------------------------------------------------------
# the functions

def evaluate(expr: Expr, box: Mapping[Symbol, tuple[Expr, Expr]]) -> Optional[ivmpf]:
    """An enclosure of the values of ``expr`` on the box, or ``None``
    (:func:`sympy_extras.assumptions.intervals.evaluate`)."""
    return native.evaluate(core_expr(expr), core_box(box))


def signs_on_box(expr: Expr, box: Box, max_boxes: int = 120, weak: bool = False) -> Optional[Signs]:
    """The signs of ``expr`` on the box by branch and bound
    (:func:`sympy_extras.assumptions.intervals.signs_on_box`)."""
    return native.signs_on_box(core_expr(expr), core_box(box), max_boxes, weak)


def range_on_box(expr: Expr, box: Box) -> Optional[Set]:
    """A set containing the values of ``expr`` on the box
    (:func:`sympy_extras.assumptions.intervals.range_on_box`)."""
    result = native.range_on_box(core_expr(expr), core_box(box))
    return None if result is None else sympy_set(result)


def monotone_range(expr: Expr, box: Box, openness: Openness, partial_sign: Optional[SignFinder] = None,
                   max_boxes: int = 40) -> Optional[Set]:
    """The exact range of ``expr`` on the box when it is monotone in every
    variable (:func:`sympy_extras.assumptions.intervals.monotone_range`)."""
    result = native.monotone_range(core_expr(expr), core_box(box), core_openness(openness),
                                   _core_finder(partial_sign), max_boxes)
    return None if result is None else sympy_set(result)


def corner_signs(expr: Expr, box: Box, openness: Openness, partial_sign: Optional[SignFinder] = None,
                 max_boxes: int = 40) -> Optional[Signs]:
    """The signs of ``expr`` on the box when it is monotone in every
    variable (:func:`sympy_extras.assumptions.intervals.corner_signs`)."""
    return native.corner_signs(core_expr(expr), core_box(box), core_openness(openness),
                               _core_finder(partial_sign), max_boxes)


def corner_value(expr: Expr, corner: Mapping[Basic, Basic], box: Optional[Box] = None) -> Optional[Expr]:
    """The value of ``expr`` at a corner of a box
    (:func:`sympy_extras.assumptions.intervals.corner_value`)."""
    core_corner: native.Corner = {core_symbol(v): core_expr(c) for v, c in corner.items()}
    result = native.corner_value(core_expr(expr), core_corner, None if box is None else core_box(box))
    return None if result is None else sympy_expr(result)


def isolate_signs(f: Expr, x: Symbol, piece: Interval, max_boxes: int = 400) -> Optional[Signs]:
    """The signs of a continuous ``f`` on an interval by bisection with
    interval arithmetic (:func:`sympy_extras.assumptions.intervals.isolate_signs`)."""
    core_piece = core_set(piece)
    if not isinstance(core_piece, CoreInterval):
        raise TypeError("expected an interval, got %s" % (piece,))
    return native.isolate_signs(core_expr(f), core_symbol(x), core_piece, max_boxes)
