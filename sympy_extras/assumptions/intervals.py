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
"""
from __future__ import annotations

from typing import Callable, Optional

import mpmath
from mpmath.ctx_iv import ivmpf

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Float, Integer, Rational, pi, E
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh
from sympy.functions.elementary.trigonometric import sin, cos, tan, atan
from sympy.sets.sets import Interval

from sympy_extras._typing import Sign, as_expr, free_symbols

__all__ = ['Box', 'Openness', 'evaluate', 'interval_sign', 'signs_on_box', 'range_on_box',
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

_iv = mpmath.iv


def _endpoint(value: Expr) -> Optional[ivmpf]:
    """A constant as an interval (a degenerate one for rationals)."""
    if value is S.Infinity:
        return _iv.mpf(mpmath.inf)
    if value is S.NegativeInfinity:
        return _iv.mpf(-mpmath.inf)
    return evaluate(value, {})


def _hull(lo: Expr, hi: Expr) -> Optional[ivmpf]:
    a, b = _endpoint(lo), _endpoint(hi)
    if a is None or b is None:
        return None
    return _iv.mpf([a.a, b.b])


def evaluate(expr: Expr, box: Box) -> Optional[ivmpf]:
    """An enclosure of the values of ``expr`` on the box, or ``None``."""
    if isinstance(expr, Symbol):
        if expr not in box:
            return None
        return _hull(*box[expr])
    if isinstance(expr, Integer):
        return _iv.mpf(int(expr))
    if isinstance(expr, Rational):
        return _iv.mpf(int(expr.p))/_iv.mpf(int(expr.q))
    if isinstance(expr, Float):
        return _iv.mpf(mpmath.mpf(expr._mpf_))
    if expr is pi:
        return _iv.pi
    if expr is E:
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
        return _power(as_expr(expr.base), as_expr(expr.exp), box)
    if isinstance(expr, (exp, log, sin, cos, tan, atan, sinh, cosh, tanh, Abs)):
        v = evaluate(as_expr(expr.args[0]), box)
        if v is None:
            return None
        return _function(expr, v)
    return None


def _power(base: Expr, exponent: Expr, box: Box) -> Optional[ivmpf]:
    b = evaluate(base, box)
    if b is None:
        return None
    if isinstance(exponent, Integer):
        n = int(exponent)
        if n >= 0:
            return b**n
        if b.a > 0 or b.b < 0:
            return _iv.mpf(1)/(b**(-n))
        return None
    if isinstance(exponent, Rational):
        if b.a > 0:
            return _iv.exp(_iv.mpf(int(exponent.p))/_iv.mpf(int(exponent.q))*_iv.log(b))
        if b.a == 0 and exponent > 0 and b.b > 0:
            upper = _iv.exp(_iv.mpf(int(exponent.p))/_iv.mpf(int(exponent.q))*_iv.log(_iv.mpf(b.b)))
            return _iv.mpf([0, upper.b])
        if b.a == 0 and b.b == 0 and exponent > 0:
            return _iv.mpf(0)
        return None
    e = evaluate(exponent, box)
    if e is None or not b.a > 0:
        return None
    return _iv.exp(e*_iv.log(b))


def _function(f: Expr, v: ivmpf) -> Optional[ivmpf]:
    try:
        if isinstance(f, exp):
            return _iv.exp(v)
        if isinstance(f, log):
            return _iv.log(v) if v.a > 0 else None
        if isinstance(f, sin):
            return _iv.sin(v)
        if isinstance(f, cos):
            return _iv.cos(v)
        if isinstance(f, tan):
            return _iv.tan(v)
        if isinstance(f, atan):
            return _iv.atan2(v, _iv.mpf(1))
        if isinstance(f, sinh):
            return (_iv.exp(v) - _iv.exp(-v))/2
        if isinstance(f, cosh):
            return (_iv.exp(v) + _iv.exp(-v))/2
        if isinstance(f, tanh):
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


def _to_expr(value: object) -> Expr:
    """An interval endpoint as an exact SymPy number."""
    if value == mpmath.inf:
        return S.Infinity
    if value == -mpmath.inf:
        return S.NegativeInfinity
    return as_expr(Rational(Float(mpmath.mpf(value), 40)))


def _split(box: Box) -> Optional[list[Box]]:
    """Bisect the box along its widest finite direction, or cut an infinite
    direction at a finite point."""
    widest: Optional[Symbol] = None
    width: Expr = S.NegativeOne
    for v, (lo, hi) in box.items():
        if lo.is_finite and hi.is_finite:
            w = as_expr(hi - lo)
            if w > width:
                width, widest = w, v
    if widest is not None and width > 0:
        lo, hi = box[widest]
        mid = as_expr((lo + hi)/2)
        left, right = dict(box), dict(box)
        left[widest] = (lo, mid)
        right[widest] = (mid, hi)
        return [left, right]
    for v, (lo, hi) in box.items():
        if not (lo.is_finite and hi.is_finite):
            cut: Expr
            if not lo.is_finite and not hi.is_finite:
                cut = S.Zero
            elif not lo.is_finite:
                cut = as_expr(hi - 1 - abs(hi))
            else:
                cut = as_expr(lo + 1 + abs(lo))
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
    is accepted as nonnegative or nonpositive instead of being split."""
    if not free_symbols(expr) <= set(box):
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


def range_on_box(expr: Expr, box: Box) -> Optional[Interval]:
    """An interval containing the values of ``expr`` on the box, with
    exact rational endpoints (rounded outwards)."""
    value = evaluate(expr, box)
    if value is None:
        return None
    lo, hi = _to_expr(value.a), _to_expr(value.b)
    if lo.is_finite and hi.is_finite and lo > hi:
        return None
    return Interval(lo, hi)


def _corners(expr: Expr, box: Box, openness: Openness, partial_sign: Optional[SignFinder], max_boxes: int
             ) -> Optional[tuple[dict[Basic, Basic], dict[Basic, Basic], bool, bool]]:
    """The corners where a function monotone in every variable takes its
    minimum and its maximum, and whether each of them is unattained
    (the function is strictly monotone in a variable whose range is open
    at that corner); ``None`` when some partial derivative has no
    definite sign (found with ``partial_sign``, by default branch and
    bound accepting weak signs)."""
    variables = sorted(box, key=lambda s: s.name)
    if not variables or not free_symbols(expr) <= set(box):
        return None
    low_corner: dict[Basic, Basic] = {}
    high_corner: dict[Basic, Basic] = {}
    low_unattained = high_unattained = False
    for v in variables:
        lo, hi = box[v]
        left_open, right_open = openness.get(v, (False, False))
        derivative = as_expr(expr.diff(v))
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
                   max_boxes: int = 40) -> Optional[Interval]:
    """The exact range of ``expr`` on the box when it is monotone in every
    variable: the interval between its values at two corners, open at an
    unattained end."""
    corners = _corners(expr, box, openness, partial_sign, max_boxes)
    if corners is None:
        return None
    low_corner, high_corner, low_unattained, high_unattained = corners
    low, high = corner_value(expr, low_corner, box), corner_value(expr, high_corner, box)
    if low is None or high is None or low.is_real is False or high.is_real is False:
        return None
    return Interval(low, high, low_unattained or not low.is_finite, high_unattained or not high.is_finite)


def corner_signs(expr: Expr, box: Box, openness: Openness, partial_sign: Optional[SignFinder] = None,
                 max_boxes: int = 40) -> Optional[Signs]:
    """The signs of ``expr`` on the box when it is monotone in every
    variable (each partial derivative has a definite sign on the box, by
    ``partial_sign`` or branch and bound): its extreme values are taken
    at two corners (limits at infinite ones), unattained when the
    function is strictly monotone in a variable whose range is open
    there."""
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


def corner_value(expr: Expr, corner: dict[Basic, Basic], box: Optional[Box] = None) -> Optional[Expr]:
    """The value of ``expr`` at a corner of a box: the finite coordinates
    are substituted and the infinite ones taken as limits, one at a
    time (valid for a function monotone in every variable). While a
    limit is taken, the other variables carry the sign their range in the
    box implies, so that SymPy's limits are not blocked by unknown signs."""
    from sympy.core.symbol import Dummy
    from sympy.series.limits import limit
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    finite = {v: c for v, c in corner.items() if isinstance(c, Expr) and c.is_finite}
    value = as_expr(expr.xreplace(finite))
    infinite = [v for v in sorted(corner, key=str) if v not in finite]
    for i, v in enumerate(infinite):
        c = corner[v]
        direction = '-' if c is S.Infinity else '+'
        signed: dict[Basic, Basic] = {}
        for w in infinite[i + 1:]:
            if isinstance(w, Symbol) and box is not None and w in box:
                lo, hi = box[w]
                keys = {'real': True}
                if lo.is_nonnegative:
                    keys['positive'] = True
                elif hi.is_nonpositive:
                    keys['negative'] = True
                signed[w] = Dummy(w.name, **keys)
        current = as_expr(value.xreplace(signed))
        value_ = attempt(lambda: as_expr(limit(current, v, c, direction)), settings.timeout)
        if value_ is None:
            return None
        value = as_expr(value_.xreplace({d: w for w, d in signed.items()}))
    return value if not free_symbols(value) else None


def isolate_signs(f: Expr, x: Symbol, piece: Interval, max_boxes: int = 400) -> Optional[Signs]:
    """The signs of a continuous ``f`` on an interval by bisection with
    interval arithmetic: a leaf whose enclosure excludes zero has a
    definite sign; a leaf on which ``f'`` has a definite sign and ``f``
    has opposite signs at the ends contains exactly one simple zero.
    ``None`` when the budget is spent (a zero of even multiplicity can
    never be certified this way)."""
    from .analysis import certified_sign
    derivative = as_expr(f.diff(x))
    a, b = as_expr(piece.start), as_expr(piece.end)
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
                left = certified_sign(as_expr(f.subs(x, lo)))
                right = certified_sign(as_expr(f.subs(x, hi)))
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
