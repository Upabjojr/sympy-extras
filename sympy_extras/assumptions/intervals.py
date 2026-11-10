"""Rigorous interval arithmetic: branch and bound on boxes, per-variable
monotonicity, and certified root isolation in one variable.

The arithmetic is mpmath's interval arithmetic (``mpmath.iv``), which
rounds outwards, so every enclosure computed here contains the true range
of the expression on the box; a sign read from an enclosure which excludes
zero is therefore a proof. The dependency problem of interval arithmetic
(``x - x`` is not zero) only makes enclosures wider, never wrong; branch
and bound subdivides the box until the enclosures are tight enough or a
budget of boxes is spent.

Supported: rational and floating point constants, ``pi``, ``E``, sums,
products, integer and rational powers, ``exp``, ``log``, ``sqrt``,
``sin``, ``cos``, ``tan``, ``atan``, ``sinh``, ``cosh``, ``tanh``,
``Abs``. Anything else gives ``None`` (undecided).

The module works on the expressions of the core (:mod:`sympy_extras.core`,
:mod:`sympy_extras.functions`, :mod:`sympy_extras.sets`) and imports no
SymPy; :mod:`sympy_extras.assumptions.intervals_sympy` is its face on
SymPy objects.

Examples
========

>>> from sympy_extras.core import Symbol, Integer
>>> from sympy_extras.functions import exp, sin
>>> from sympy_extras.assumptions.intervals import evaluate, interval_sign, signs_on_box, range_on_box
>>> x, y = Symbol('x'), Symbol('y')
>>> interval_sign(evaluate(exp(x) + 1, {x: (Integer(-1), Integer(1))}))
1
>>> signs_on_box(sin(x) - 2 + y, {x: (Integer(0), Integer(1)), y: (Integer(0), Integer(1))})
frozenset({-1})
>>> range_on_box(x*y + 1, {x: (Integer(0), Integer(1)), y: (Integer(0), Integer(2))})
Interval(1, 3)
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import mpmath
from mpmath.ctx_iv import ivmpf

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import mpf_to_fraction
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.limits import limit
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import (Exp1, Float, Infinity, Integer, NegativeInfinity, Pi, Rational, oo,
    number_from_fraction)
from sympy_extras.core.power import Pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, abs_
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Sinh, Tanh
from sympy_extras.functions.elementary.trigonometric import Atan, Cos, Sin, Tan
from sympy_extras.sets.sets import Interval, Set, interval
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

__all__ = ['Box', 'Openness', 'Sign', 'Signs', 'Corner', 'replaced', 'evaluate', 'interval_sign', 'signs_on_box',
    'range_on_box', 'corner_signs', 'corner_value', 'monotone_range', 'isolate_signs']

#: a sign: -1, 0 or 1
Sign = int
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
#: a corner of a box: a value (or an infinity) for every variable
Corner = dict[Symbol, Expr]

_iv = mpmath.iv


def _symbols(e: Basic) -> set[Symbol]:
    """The free symbols which are plain symbols."""
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced and the rebuilt nodes
    evaluated (SymPy's ``xreplace``)."""
    return as_expr(e._xreplace_evaluated(rule))


def _greater(a: Expr, b: Expr) -> bool:
    """``a > b`` for numbers, as SymPy's ``bool(a > b)``: ``TypeError``
    when it cannot be decided."""
    positive = (a - b).is_extended_positive
    if positive is None:
        raise TypeError("cannot determine truth value of %s > %s" % (a, b))
    return positive


def _endpoint(value: Expr) -> Optional[ivmpf]:
    """A constant as an interval (a degenerate one for rationals)."""
    if isinstance(value, Infinity):
        return _iv.mpf(mpmath.inf)
    if isinstance(value, NegativeInfinity):
        return _iv.mpf(-mpmath.inf)
    return evaluate(value, {})


def _hull(lo: Expr, hi: Expr) -> Optional[ivmpf]:
    a, b = _endpoint(lo), _endpoint(hi)
    if a is None or b is None:
        return None
    return _iv.mpf([a.a, b.b])


def _float_interval(value: Float) -> ivmpf:
    """A float as an interval: its binary value rounded to the working
    precision, as ``mpmath.mpf`` rounds a raw float."""
    fraction = value.as_fraction()
    exponent = 1 - fraction.denominator.bit_length()
    return _iv.mpf(mpmath.mpf((fraction.numerator, exponent)))


def evaluate(expr: Expr, box: Mapping[Symbol, tuple[Expr, Expr]]) -> Optional[ivmpf]:
    """An enclosure of the values of ``expr`` on the box, or ``None``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.assumptions.intervals import evaluate
    >>> x = Symbol('x')
    >>> evaluate(x**2 + 1, {x: (Integer(-3), Integer(3))})
    [1.0, 10.0]
    >>> evaluate(log(x), {x: (Integer(-1), Integer(1))}) is None
    True
    """
    if isinstance(expr, Symbol):
        if expr not in box:
            return None
        lo, hi = box[expr]
        return _hull(lo, hi)
    if isinstance(expr, Integer):
        return _iv.mpf(expr.p)
    if isinstance(expr, Rational):
        return _iv.mpf(expr.p)/_iv.mpf(expr.q)
    if isinstance(expr, Float):
        return _float_interval(expr)
    if isinstance(expr, Pi):
        return _iv.pi
    if isinstance(expr, Exp1):
        return _iv.e
    if isinstance(expr, Add):
        total = _iv.mpf(0)
        for a in expr.args:
            v = evaluate(as_expr(a), box)
            if v is None:
                return None
            total = total + v
        return total
    if isinstance(expr, Mul):
        product = _iv.mpf(1)
        for a in expr.args:
            v = evaluate(as_expr(a), box)
            if v is None:
                return None
            product = product*v
        return product
    if isinstance(expr, Pow):
        return _power(expr.base, expr.exp, box)
    if isinstance(expr, (Exp, Log, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh, Abs)):
        v = evaluate(as_expr(expr.args[0]), box)
        if v is None:
            return None
        return _function(expr, v)
    return None


def _power(base: Expr, exponent: Expr, box: Mapping[Symbol, tuple[Expr, Expr]]) -> Optional[ivmpf]:
    b = evaluate(base, box)
    if b is None:
        return None
    if isinstance(exponent, Integer):
        n = exponent.p
        if n >= 0:
            return b**n
        if b.a > 0 or b.b < 0:
            return _iv.mpf(1)/(b**(-n))
        return None
    if isinstance(exponent, Rational):
        ratio = _iv.mpf(exponent.p)/_iv.mpf(exponent.q)
        if b.a > 0:
            return _iv.exp(ratio*_iv.log(b))
        if b.a == 0 and exponent.p > 0 and b.b > 0:
            upper = _iv.exp(ratio*_iv.log(_iv.mpf(b.b)))
            return _iv.mpf([0, upper.b])
        if b.a == 0 and b.b == 0 and exponent.p > 0:
            return _iv.mpf(0)
        return None
    e = evaluate(exponent, box)
    if e is None or not b.a > 0:
        return None
    return _iv.exp(e*_iv.log(b))


def _function(f: Expr, v: ivmpf) -> Optional[ivmpf]:
    try:
        if isinstance(f, Exp):
            return _iv.exp(v)
        if isinstance(f, Log):
            return _iv.log(v) if v.a > 0 else None
        if isinstance(f, Sin):
            return _iv.sin(v)
        if isinstance(f, Cos):
            return _iv.cos(v)
        if isinstance(f, Tan):
            return _iv.tan(v)
        if isinstance(f, Atan):
            return _iv.atan2(v, _iv.mpf(1))
        if isinstance(f, Sinh):
            return (_iv.exp(v) - _iv.exp(-v))/2
        if isinstance(f, Cosh):
            return (_iv.exp(v) + _iv.exp(-v))/2
        if isinstance(f, Tanh):
            w = _iv.exp(2*v)
            return (w - 1)/(w + 1)
        if isinstance(f, Abs):
            return abs(v)
    except (ValueError, ZeroDivisionError, TypeError, mpmath.libmp.libmpf.ComplexResult):
        return None
    return None


def interval_sign(v: ivmpf) -> Optional[Sign]:
    """The sign of every number in the enclosure, or ``None`` when the
    enclosure contains numbers of different signs."""
    if v.a > 0:
        return 1
    if v.b < 0:
        return -1
    if v.a == 0 and v.b == 0:
        return 0
    return None


def _to_expr(value: ivmpf) -> Expr:
    """An interval endpoint as an exact number."""
    if value == mpmath.inf:
        return oo
    if value == -mpmath.inf:
        return -oo
    return number_from_fraction(mpf_to_fraction(mpmath.mpf(value)))


def _split(box: Box) -> Optional[list[Box]]:
    """Bisect the box along its widest finite direction, or cut an infinite
    direction at a finite point."""
    widest: Optional[Symbol] = None
    width: Expr = Integer(-1)
    for v, (lo, hi) in box.items():
        if lo.is_finite and hi.is_finite:
            w = hi - lo
            if _greater(w, width):
                width, widest = w, v
    if widest is not None and _greater(width, Integer(0)):
        lo, hi = box[widest]
        mid = (lo + hi)/2
        left, right = dict(box), dict(box)
        left[widest] = (lo, mid)
        right[widest] = (mid, hi)
        return [left, right]
    for v, (lo, hi) in box.items():
        if not (lo.is_finite and hi.is_finite):
            cut: Expr
            if not lo.is_finite and not hi.is_finite:
                cut = Integer(0)
            elif not lo.is_finite:
                cut = hi - 1 - abs_(hi)
            else:
                cut = lo + 1 + abs_(lo)
            left, right = dict(box), dict(box)
            left[v] = (lo, cut)
            right[v] = (cut, hi)
            return [left, right]
    return None


def signs_on_box(expr: Expr, box: Box, max_boxes: int = 120, weak: bool = False) -> Optional[Signs]:
    """The signs of ``expr`` on the box by branch and bound, or ``None``
    when the budget of boxes is spent before every leaf has a definite
    sign (which happens, in particular, when ``expr`` has a zero on the
    box). With ``weak``, a leaf whose enclosure touches zero at one end
    is accepted as nonnegative or nonpositive instead of being split.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.assumptions.intervals import signs_on_box
    >>> x, y = Symbol('x'), Symbol('y')
    >>> box = {x: (Integer(-1), Integer(1)), y: (Integer(-1), Integer(1))}
    >>> signs_on_box(x**2 + y**2 + 1, box), signs_on_box(x**2 + y**2 - 1, box)
    (frozenset({1}), None)
    """
    if not _symbols(expr) <= set(box):
        return None
    queue: list[Box] = [box]
    signs: set[Sign] = set()
    examined = 0
    while queue:
        current = queue.pop()
        examined += 1
        if examined > max_boxes:
            return None
        value = evaluate(expr, current)
        if value is None:
            return None
        s = interval_sign(value)
        if s is not None:
            signs.add(s)
            continue
        if weak and value.a >= 0:
            signs.update((0, 1))
            continue
        if weak and value.b <= 0:
            signs.update((-1, 0))
            continue
        halves = _split(current)
        if halves is None:
            return None
        queue.extend(halves)
    return frozenset(signs)


def range_on_box(expr: Expr, box: Box) -> Optional[Set]:
    """A set containing the values of ``expr`` on the box (an interval
    with exact rational endpoints, rounded outwards; a point when the
    enclosure is one)."""
    value = evaluate(expr, box)
    if value is None:
        return None
    lo, hi = _to_expr(value.a), _to_expr(value.b)
    if lo.is_finite and hi.is_finite and _greater(lo, hi):
        return None
    return interval(lo, hi)


def _corners(expr: Expr, box: Box, openness: Openness, partial_sign: Optional[SignFinder], max_boxes: int
             ) -> Optional[tuple[Corner, Corner, bool, bool]]:
    """The corners where a function monotone in every variable takes its
    minimum and its maximum, and whether each of them is unattained
    (the function is strictly monotone in a variable whose range is open
    at that corner); ``None`` when some partial derivative has no
    definite sign (found with ``partial_sign``, by default branch and
    bound accepting weak signs)."""
    variables = sorted(box, key=lambda s: s.name)
    if not variables or not _symbols(expr) <= set(box):
        return None
    low_corner: Corner = {}
    high_corner: Corner = {}
    low_unattained = high_unattained = False
    for v in variables:
        lo, hi = box[v]
        left_open, right_open = openness.get(v, (False, False))
        derivative = expr.diff(v)
        slope = partial_sign(derivative) if partial_sign is not None else signs_on_box(derivative, box, max_boxes, weak=True)
        if slope is None or (1 in slope and -1 in slope):
            return None
        strict = 0 not in slope
        if 1 in slope:
            low_corner[v], high_corner[v] = lo, hi
            low_unattained = low_unattained or (strict and left_open)
            high_unattained = high_unattained or (strict and right_open)
        elif -1 in slope:
            low_corner[v], high_corner[v] = hi, lo
            low_unattained = low_unattained or (strict and right_open)
            high_unattained = high_unattained or (strict and left_open)
        else:
            low_corner[v], high_corner[v] = lo, lo
    return low_corner, high_corner, low_unattained, high_unattained


def monotone_range(expr: Expr, box: Box, openness: Openness, partial_sign: Optional[SignFinder] = None,
                   max_boxes: int = 40) -> Optional[Set]:
    """The exact range of ``expr`` on the box when it is monotone in every
    variable: the interval between its values at two corners, open at an
    unattained end.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.assumptions.intervals import monotone_range
    >>> x, y = Symbol('x'), Symbol('y')
    >>> monotone_range(x + y, {x: (Integer(0), Integer(1)), y: (Integer(0), Integer(2))},
    ...                {x: (True, False), y: (False, False)})
    Interval.Lopen(0, 3)
    """
    corners = _corners(expr, box, openness, partial_sign, max_boxes)
    if corners is None:
        return None
    low_corner, high_corner, low_unattained, high_unattained = corners
    low, high = corner_value(expr, low_corner, box), corner_value(expr, high_corner, box)
    if low is None or high is None or low.is_real is False or high.is_real is False:
        return None
    return interval(low, high, low_unattained or not low.is_finite, high_unattained or not high.is_finite)


def corner_signs(expr: Expr, box: Box, openness: Openness, partial_sign: Optional[SignFinder] = None,
                 max_boxes: int = 40) -> Optional[Signs]:
    """The signs of ``expr`` on the box when it is monotone in every
    variable (each partial derivative has a definite sign on the box, by
    ``partial_sign`` or branch and bound): its extreme values are taken
    at two corners (limits at infinite ones), unattained when the
    function is strictly monotone in a variable whose range is open
    there.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.assumptions.intervals import corner_signs
    >>> x, y = Symbol('x'), Symbol('y')
    >>> box = {x: (Integer(0), Integer(2)), y: (Integer(0), Integer(3))}
    >>> corner_signs(exp(x) + y - 1, box, {x: (True, True), y: (True, True)})
    frozenset({1})
    >>> corner_signs(exp(x) + y - 1, box, {x: (False, False), y: (False, False)})
    frozenset({0, 1})
    """
    corners = _corners(expr, box, openness, partial_sign, max_boxes)
    if corners is None:
        return None
    low_corner, high_corner, low_unattained, high_unattained = corners
    from .analysis import certified_sign
    low_value = corner_value(expr, low_corner, box)
    if low_value is None:
        return None
    low = certified_sign(low_value)
    if low is None:
        return None
    if low == 1 or (low == 0 and low_unattained):
        return frozenset([1])
    high_value = corner_value(expr, high_corner, box)
    if high_value is None:
        return None
    high = certified_sign(high_value)
    if high is None:
        return None
    if high == -1 or (high == 0 and high_unattained):
        return frozenset([-1])
    signs: set[Sign] = set()
    if low < 0:
        signs.add(-1)
    if high > 0:
        signs.add(1)
    if low <= 0 <= high:
        signs.add(0)
    return frozenset(signs)


def corner_value(expr: Expr, corner: Corner, box: Optional[Box] = None) -> Optional[Expr]:
    """The value of ``expr`` at a corner of a box: the finite coordinates
    are substituted and the infinite ones taken as limits, one at a
    time (valid for a function monotone in every variable). While a
    limit is taken, the other variables carry the sign their range in the
    box implies, so that the limits are not blocked by unknown signs.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.assumptions.intervals import corner_value
    >>> x, y = Symbol('x'), Symbol('y')
    >>> corner_value(exp(-x)*y + 1, {x: oo, y: Integer(2)}), corner_value(x + y, {x: Integer(1), y: Integer(2)})
    (1, 3)
    """
    finite: dict[Basic, Basic] = {v: c for v, c in corner.items() if c.is_finite}
    value = replaced(expr, finite)
    infinite = [v for v in sorted(corner, key=str) if v not in finite]
    for i, v in enumerate(infinite):
        c = corner[v]
        direction = '-' if isinstance(c, Infinity) else '+'
        signed: dict[Symbol, Dummy] = {}
        for w in infinite[i + 1:]:
            if box is not None and w in box:
                lo, hi = box[w]
                if lo.is_nonnegative:
                    signed[w] = Dummy(w.name, real=True, positive=True)
                elif hi.is_nonpositive:
                    signed[w] = Dummy(w.name, real=True, negative=True)
                else:
                    signed[w] = Dummy(w.name, real=True)
        forth: dict[Basic, Basic] = {w: d for w, d in signed.items()}
        current = replaced(value, forth)
        found = attempt(lambda: limit(current, v, c, direction), settings.timeout, native=True)
        if found is None:
            return None
        back: dict[Basic, Basic] = {d: w for w, d in signed.items()}
        value = replaced(found, back)
    return value if not _symbols(value) else None


def isolate_signs(f: Expr, x: Symbol, piece: Interval, max_boxes: int = 400) -> Optional[Signs]:
    """The signs of a continuous ``f`` on an interval by bisection with
    interval arithmetic: a leaf whose enclosure excludes zero has a
    definite sign; a leaf on which ``f'`` has a definite sign and ``f``
    has opposite signs at the ends contains exactly one simple zero.
    ``None`` when the budget is spent (a zero of even multiplicity can
    never be certified this way).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.sets import Interval
    >>> from sympy_extras.assumptions.intervals import isolate_signs
    >>> x = Symbol('x')
    >>> sorted(isolate_signs(x*exp(x) - 1, x, Interval(0, 2, True, True)))
    [-1, 0, 1]
    >>> isolate_signs(x**2 - 2*x + 1, x, Interval(0, 2)) is None
    True
    """
    from .analysis import certified_sign
    derivative = f.diff(x)
    a, b = piece.start, piece.end
    queue: list[tuple[Expr, Expr]] = [(a, b)]
    signs: set[Sign] = set()
    examined = 0
    while queue:
        lo, hi = queue.pop()
        examined += 1
        if examined > max_boxes:
            return None
        value = evaluate(f, {x: (lo, hi)})
        if value is None:
            return None
        s = interval_sign(value)
        if s is not None:
            signs.add(s)
            continue
        if lo.is_finite and hi.is_finite:
            slope = evaluate(derivative, {x: (lo, hi)})
            if slope is not None and interval_sign(slope) not in (None, 0):
                left = certified_sign(f.subs(x, lo))
                right = certified_sign(f.subs(x, hi))
                if left is None or right is None:
                    return None
                if left == right:
                    signs.add(left)
                    continue
                # exactly one zero on the leaf, at an end or inside
                if left == 0:
                    if not (lo == a and piece.left_open):
                        signs.add(0)
                    signs.add(right)
                elif right == 0:
                    if not (hi == b and piece.right_open):
                        signs.add(0)
                    signs.add(left)
                else:
                    signs.update((left, 0, right))
                continue
        halves = _split({x: (lo, hi)})
        if halves is None:
            return None
        queue.extend(h[x] for h in halves)
    return frozenset(signs)
