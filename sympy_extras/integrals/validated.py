"""Validated numerical integration: a value with a rigorous error bound.

When no closed form is found, a number with a proved error bound is the
next best answer. The composite Simpson rule on a panel `[c, d]` of width
`h` has the error bound

.. math::

    \\left| \\int_c^d f\\, dx - \\frac{h}{6}\\bigl(f(c) + 4 f(m) + f(d)\\bigr) \\right|
    \\le \\frac{h^5}{2880} \\max_{[c, d]} |f^{(4)}|,

and interval arithmetic (:mod:`sympy_extras.assumptions.intervals`,
mpmath's outward rounding) gives a proved bound of `|f^{(4)}|` on the
panel and proved enclosures of the three values of `f`, so the sum over
the panels is an enclosure of the integral [Moore]_. The panels are
bisected adaptively where the bound is largest [Petras]_ until the total
is below the requested tolerance; a panel on which the fourth derivative
has no finite bound (an endpoint singularity, `x^x` at `0`) is bounded
by its width times the range of `f`, which interval arithmetic still
encloses, and is bisected towards the endpoint. An algebraic endpoint
singularity `\\sqrt{x - a}` is removed by `x = a + u^2` first, and an
infinite range by `x = a + t/(1 - t)` when the transformed integrand is
bounded. The method follows the principles of verified quadrature
[Petras]_ and of Arb's ball-arithmetic integration [Johansson]_ in the
simplest form.

Examples
========

>>> from sympy import symbols, exp, sqrt, pi, erf
>>> from sympy_extras.integrals.validated import validated_integral
>>> x = symbols('x')
>>> value, error = validated_integral(exp(-x**2), x, 0, 1)
>>> round(value, 12), error < 1e-15
(0.746824132812, True)
>>> abs(value - sqrt(pi)*erf(1)/2) < error
True

References
==========

.. [Petras] K. Petras, *Principles of verified numerical integration*,
   Journal of Computational and Applied Mathematics 199 (2007) 317–328.
.. [Johansson] F. Johansson, *Numerical integration in arbitrary-precision
   ball arithmetic*, Mathematical Software – ICMS 2018, LNCS 10931.
.. [Moore] R. E. Moore, *Interval Analysis*, Prentice-Hall, 1966.
"""
from __future__ import annotations

import heapq
from typing import Optional

import mpmath
from mpmath.ctx_iv import ivmpf

from sympy.core.expr import Expr
from sympy.core.numbers import Float, Rational, oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polytools import cancel
from sympy.simplify.powsimp import powdenest
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr, free_symbols
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.intervals import Box, evaluate
from sympy_extras.settings import settings

__all__ = ['validated_integral', 'enclosure']

_iv = mpmath.iv

#: the largest number of panels before the refinement stops
_MAX_PANELS = 6000


def _enclose(expr: Expr, x: Symbol, lo: Expr, hi: Expr) -> Optional[ivmpf]:
    """An enclosure of ``expr`` on ``[lo, hi]``, with the powers of a
    nonnegative base and a nonnegative exponent (``x**x`` at 0) bounded
    by monotonicity where the interval evaluator gives up."""
    box: Box = {x: (lo, hi)}
    value = evaluate(expr, box)
    if value is not None:
        return value
    return _enclose_powers(expr, x, box)


def _enclose_powers(expr: Expr, x: Symbol, box: Box) -> Optional[ivmpf]:
    """The evaluator with the rule ``b**e`` in ``[0, max(B**e1, B**e2, 1)]``
    for ``b`` in ``[0, B]`` and ``e`` in ``[e1, e2]``, ``e1 >= 0`` (the power
    increases with the base; the value 1 covers ``0**0``)."""
    if isinstance(expr, Pow):
        base = _enclose(as_expr(expr.base), x, *box[x])
        exponent = _enclose(as_expr(expr.exp), x, *box[x])
        if base is None or exponent is None or base.a < 0 or exponent.a < 0:
            return None
        top = _iv.mpf(1) if exponent.a == 0 else _iv.mpf(0)
        if base.b > 0:
            for e in (exponent.a, exponent.b):
                candidate = _iv.exp(_iv.mpf(e) * _iv.log(_iv.mpf(base.b)))
                top = _iv.mpf([0, max(top.b, candidate.b)])
        return _iv.mpf([0, top.b])
    parts = [as_expr(a) for a in expr.args]
    if not parts or not expr.is_Add and not expr.is_Mul:
        return None
    total: Optional[ivmpf] = None
    for part in parts:
        value = _enclose(part, x, *box[x])
        if value is None:
            return None
        if total is None:
            total = value
        elif expr.is_Add:
            total = total + value
        else:
            total = total * value
    return total


class _Panel:
    """A panel of the subdivision with its enclosure and error bound."""

    def __init__(self, lo: Rational, hi: Rational, value: ivmpf, error: mpmath.mpf) -> None:
        self.lo = lo
        self.hi = hi
        self.value = value
        self.error = error
        #: what splitting the panel can gain: the Simpson bound and the
        #: width of the enclosure of the value
        self.priority: mpmath.mpf = error + mpmath.mpf(value.delta) / 2 if mpmath.isfinite(error) else error

    def __lt__(self, other: _Panel) -> bool:
        return bool(self.priority > other.priority)    # the largest first


def _panel(f: Expr, fourth: Expr, x: Symbol, lo: Rational, hi: Rational) -> _Panel:
    """Simpson's rule on the panel with its bound; the range bound when
    the fourth derivative has no finite enclosure; an infinite error when
    even the range has none (the panel is then split first: interval
    arithmetic loses its dependency problem on narrow panels)."""
    width = _iv.mpf(int((hi - lo).p)) / _iv.mpf(int((hi - lo).q))
    bound = _enclose(fourth, x, lo, hi)
    if bound is not None and mpmath.isfinite(bound.a) and mpmath.isfinite(bound.b):
        mid = as_expr((lo + hi) / 2)
        values = [_enclose(f, x, p, p) for p in (lo, mid, hi)]
        finite = [v for v in values if v is not None and mpmath.isfinite(v.a) and mpmath.isfinite(v.b)]
        if len(finite) == 3:
            f0, f1, f2 = finite
            simpson = width / 6 * (f0 + 4 * f1 + f2)
            # the bound itself in interval arithmetic, rounded outwards
            error = (width**5 / 2880 * abs(bound)).b
            return _Panel(lo, hi, simpson, error)
    values_ = _enclose(f, x, lo, hi)
    if values_ is None or not mpmath.isfinite(values_.a) or not mpmath.isfinite(values_.b):
        return _Panel(lo, hi, _iv.mpf(0), mpmath.inf)
    return _Panel(lo, hi, width * values_, mpmath.mpf(0))


def _rational(value: Expr) -> Optional[Rational]:
    if isinstance(value, Rational):
        return value
    if isinstance(value, Float):
        return Rational(value)
    return None


def _total(heap: list[_Panel]) -> tuple[ivmpf, mpmath.mpf]:
    total = _iv.mpf(0)
    spread = mpmath.mpf(0)
    for panel in heap:
        total = total + panel.value
        spread += panel.error
    return total, spread


def enclosure(f: Expr, x: Symbol, a: Rational, b: Rational, tolerance: mpmath.mpf,
              max_panels: int = _MAX_PANELS) -> Optional[ivmpf]:
    """An interval containing ``Integral(f, (x, a, b))`` for rational
    ``a < b``, refined until its width is below ``tolerance`` or the panel
    budget is spent; ``None`` when a panel narrower than the tolerance
    still cannot be bounded (a non-integrable singularity, or a function
    the interval evaluator does not know)."""
    fourth = as_expr(f.diff(x, 4))
    heap: list[_Panel] = [_panel(f, fourth, x, a, b)]
    while len(heap) < max_panels:
        total, spread = _total(heap)
        if mpmath.isfinite(spread) and mpmath.mpf(total.delta) / 2 + spread <= tolerance:
            break
        worst = heapq.heappop(heap)
        if not mpmath.isfinite(worst.error) and float(worst.hi - worst.lo) < float(tolerance) * float(b - a):
            return None
        mid_ = _rational(as_expr((worst.lo + worst.hi) / 2))
        if mid_ is None:
            return None
        heapq.heappush(heap, _panel(f, fourth, x, worst.lo, mid_))
        heapq.heappush(heap, _panel(f, fourth, x, mid_, worst.hi))
    total, spread = _total(heap)
    if not mpmath.isfinite(spread):
        return None
    return _iv.mpf([total.a - spread, total.b + spread])


def _remove_root_singularity(f: Expr, x: Symbol, a: Rational, b: Rational,
                             side: str) -> Optional[tuple[Expr, Symbol, Rational]]:
    """``f`` with ``x = a + u**2`` (``dx = 2 u du``; at the right end
    ``x = b - u**2``) when it has a square root vanishing at that end, on
    ``u`` in ``[0, sqrt(b - a)]``; ``None`` when the new upper bound is
    not rational."""
    u = Dummy('u', positive=True)
    end = a if side == 'left' else b
    for node in f.atoms(Pow):
        exponent = as_expr(node.exp)
        if isinstance(exponent, Rational) and exponent.q == 2 and as_expr(node.base).subs(x, end) == 0:
            image = a + u**2 if side == 'left' else b - u**2
            g = as_expr(powdenest(f.subs(x, image) * 2 * u, force=True))
            upper = _rational(as_expr((b - a)**S.Half))
            if upper is None:
                return None
            return g, u, upper
    return None


def validated_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, digits: int = 15,
                       assumptions: Assumptions = None) -> Optional[tuple[Expr, Expr]]:
    """``(value, error)`` with ``|Integral(f, (x, a, b)) - value| <= error``
    proved by interval arithmetic, the error aimed at ``10**(-digits)``
    (it may be larger when the panel budget or the time limit is spent
    first); ``None`` when the integrand has parameters, uses functions
    the interval evaluator does not know, or cannot be bounded on some
    panel (a non-integrable singularity, an infinite range on which the
    transformed integrand is unbounded). Irrational bounds are enclosed
    and the strips between the enclosures bounded by the range of ``f``.

    Examples
    ========

    >>> from sympy import symbols, sqrt, pi, oo, Rational
    >>> from sympy_extras.integrals.validated import validated_integral
    >>> x = symbols('x')
    >>> value, error = validated_integral(sqrt(x), x, 0, 1)
    >>> abs(value - Rational(2, 3)) < error, error < 1e-15
    (True, True)
    >>> value, error = validated_integral(1/(1 + x**2), x, 0, oo)
    >>> abs(value - pi/2) < error
    True
    >>> validated_integral(1/x, x, 0, 1) is None
    True
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    if free_symbols(f_) - {x} or free_symbols(a_) or free_symbols(b_):
        return None
    if a_ == b_:
        return (Float(0, digits), Float(0, digits))
    if a_ in (oo, -oo) or b_ in (oo, -oo):
        return _infinite(f_, x, a_, b_, digits)
    # the enclosure is refined to a quarter of the target, so that the
    # reported bound, with its margins, stays below 10**(-digits)
    tolerance = mpmath.mpf(10)**(-digits) / 4
    previous = _iv.dps
    try:
        _iv.dps = digits + 10
        bounds = _bounds(a_, b_)
        if bounds is None:
            return None
        lo, hi, strips = bounds
        if lo > hi:
            found = validated_integral(f_, x, b_, a_, digits)
            return None if found is None else (as_expr(-found[0]), found[1])
        result: Optional[ivmpf] = None
        for transformed in (_remove_root_singularity(f_, x, lo, hi, 'left'),
                            _remove_root_singularity(f_, x, lo, hi, 'right')):
            if transformed is not None:
                g, u, upper = transformed
                result = enclosure(g, u, Rational(0), upper, tolerance)
                if result is not None:
                    break
        if result is None:
            result = enclosure(f_, x, lo, hi, tolerance)
        if result is None:
            return None
        for strip_lo, strip_hi in strips:
            values = _enclose(f_, x, strip_lo, strip_hi)
            if values is None or not mpmath.isfinite(values.a) or not mpmath.isfinite(values.b):
                return None
            width = _iv.mpf(int((strip_hi - strip_lo).p)) / _iv.mpf(int((strip_hi - strip_lo).q))
            result = result + width * values
        # converted while the working precision is still in force
        return _pair(result, digits)
    finally:
        _iv.dps = previous


def _bounds(a: Expr, b: Expr) -> Optional[tuple[Rational, Rational, list[tuple[Rational, Rational]]]]:
    """Rational bounds inside ``[a, b]`` and the strips left over at
    irrational endpoints (``pi``), each a rational interval containing
    the endpoint."""
    strips: list[tuple[Rational, Rational]] = []
    lo, hi = _rational(a), _rational(b)
    if lo is None:
        value = _enclose(a, Dummy('x'), Rational(0), Rational(0))
        if value is None:
            return None
        lo = Rational(Float(mpmath.mpf(value.b), 40))
        strips.append((Rational(Float(mpmath.mpf(value.a), 40)), lo))
    if hi is None:
        value = _enclose(b, Dummy('x'), Rational(0), Rational(0))
        if value is None:
            return None
        hi = Rational(Float(mpmath.mpf(value.a), 40))
        strips.append((hi, Rational(Float(mpmath.mpf(value.b), 40))))
    return lo, hi, strips


def _pair(result: ivmpf, digits: int) -> tuple[Expr, Expr]:
    """The midpoint and the radius of the enclosure as floats, converted
    at the working precision (the default 53 bits would round the
    midpoint outside the enclosure)."""
    with mpmath.mp.workdps(digits + 10):
        value = Float(mpmath.nstr(mpmath.mpf(result.mid.a), digits + 3), digits + 3)
        radius = _iv.mpf(result.delta) / 2
        # the rounding of the value to digits + 3 places, and a margin for
        # the five places of the error itself, are added to the radius
        rounding = mpmath.mpf(abs(result.mid.b)) * mpmath.mpf(10)**(-(digits + 2))
        total = (mpmath.mpf(radius.b) + rounding) * (1 + mpmath.mpf('2e-3'))
        error = Float(mpmath.nstr(total, 5, min_fixed=-100), 5)
    return (value, error)


def _infinite(f: Expr, x: Symbol, a: Expr, b: Expr, digits: int) -> Optional[tuple[Expr, Expr]]:
    """An infinite range mapped onto ``[0, 1]`` by ``x = a + t/(1 - t)``
    (or ``x = b - t/(1 - t)``), the whole line cut at 0; the transformed
    integrand must have a finite fourth derivative near ``t = 1``, which
    ``cancel`` and ``simplify`` are given a chance to expose."""
    if a == -oo and b == oo:
        left = _infinite(f, x, -oo, S.Zero, digits)
        right = _infinite(f, x, S.Zero, oo, digits)
        if left is None or right is None:
            return None
        return (as_expr(left[0] + right[0]), as_expr(left[1] + right[1]))
    t = Dummy('t', positive=True)
    if b == oo and a not in (oo, -oo):
        g = as_expr(f.subs(x, a + t / (1 - t)) / (1 - t)**2)
    elif a == -oo and b not in (oo, -oo):
        g = as_expr(f.subs(x, b - t / (1 - t)) / (1 - t)**2)
    else:
        return None
    simpler = attempt(lambda: as_expr(simplify(cancel(g))), settings.timeout)
    if simpler is not None:
        g = simpler
    found = validated_integral(g, t, S.Zero, S.One, digits)
    return found
