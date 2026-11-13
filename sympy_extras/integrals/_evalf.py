"""The numerical value of a definite ``Integral``: SymPy's ``evalf_integral``.

Ported from SymPy's ``sympy.core.evalf`` (``evalf_integral``,
``do_integral``): the tanh-sinh quadrature of mpmath (``quadts``) of the
integrand evaluated by the numerical rules of the core's nodes, at
increasing working precision until the accuracy estimated from the
largest value of the integrand and the error of the quadrature reaches
the precision asked for (or the largest precision, twice the asked one
and at least 333 bits, SymPy's ``maxprec`` of ``evalf``).

References
==========

.. [1] F. Johansson, mpmath: a Python library for arbitrary-precision
       floating-point arithmetic, http://mpmath.org (``quad``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/evalf.py (evalf_integral, do_integral) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; mpmath numbers instead of SymPy's mpf tuples;
# the integrand compiled by the numerical rules of the core's nodes; no
# oscillatory quadrature option (``quad='osc'``); None when not evaluable.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

import mpmath

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Infinity, NegativeInfinity
from sympy_extras.core.symbol import Symbol

__all__ = ['evalf_integral', 'do_integral']

#: a value of the quadrature with its estimated accuracy in bits
Quadrature = tuple[mpmath.mpc, int]

_MINUS_INF = -(10**9)


def _fastlog(x: mpmath.mpf) -> int:
    """SymPy's ``fastlog``: an estimate of ``log2(|x|)``."""
    if x == 0 or not mpmath.isfinite(x):
        return _MINUS_INF
    sign, man, exp, bc = mpmath.mpf(x)._mpf_
    return int(exp) + int(bc)


def _bound(e: Expr, prec: int) -> Optional[mpmath.mpf]:
    """An end of the range as an mpmath number (``inf`` for ``oo``)."""
    from sympy_extras.core.evalf import numeric_value
    if isinstance(e, Infinity):
        return mpmath.mpf('inf')
    if isinstance(e, NegativeInfinity):
        return mpmath.mpf('-inf')
    value = numeric_value(e, prec)
    if value is None or mpmath.im(value) != 0:
        return None
    return mpmath.mpf(mpmath.re(value))


def _integrand(f: Expr, x: Symbol) -> Optional[Callable[[mpmath.mpc], mpmath.mpc]]:
    """``f`` as a function of ``x`` evaluated by mpmath at its working
    precision; ``None`` for a node without a numerical rule."""
    from sympy_extras.integrals.definite import _numeric_function
    try:
        return _numeric_function(f, x)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _subdivision(lo: mpmath.mpf, hi: mpmath.mpf) -> list[mpmath.mpf]:
    """The nodes of the subdivision of the range: a long finite range in
    pieces of length about one (at most 100), an infinite one with nodes
    at distances 1, 10, 100 and 1000 of its finite end (a single
    tanh-sinh rule over ``(-326, 500)`` misses the peak of ``exp(-x**2)``
    and estimates no error; SymPy's ``quadts`` over the whole range
    does)."""
    inf = mpmath.mpf('inf')
    if lo == -inf and hi == inf:
        return [lo, mpmath.mpf(-100), mpmath.mpf(-10), mpmath.mpf(-1), mpmath.mpf(0), mpmath.mpf(1),
                mpmath.mpf(10), mpmath.mpf(100), hi]
    if hi == inf or hi == -inf:
        sign = 1 if hi == inf else -1
        return [lo] + [lo + sign * mpmath.mpf(d) for d in (1, 10, 100, 1000)] + [hi]
    if lo == inf or lo == -inf:
        return list(reversed(_subdivision(hi, lo)))
    pieces = min(int(mpmath.ceil(abs(hi - lo))), 100)
    if pieces <= 2:
        return [lo, hi]
    return [lo + (hi - lo) * k / pieces for k in range(pieces)] + [hi]


# PORTED FROM SYMPY: sympy/core/evalf.py:do_integral
def do_integral(f: Expr, x: Symbol, lower: Expr, upper: Expr, prec: int) -> Optional[Quadrature]:
    """The quadrature of ``f`` over ``(lower, upper)`` at ``prec`` bits with
    its estimated accuracy (SymPy's ``re_acc``/``im_acc``, the smaller);
    ``None`` when the integrand or the ends are not evaluable."""
    if lower == upper:
        return mpmath.mpc(0), prec
    with mpmath.workprec(prec + 5):
        lo, hi = _bound(lower, prec + 15), _bound(upper, prec + 15)
        if lo is None or hi is None:
            return None
        g = _integrand(f, x)
        if g is None:
            return None
        largest = [_MINUS_INF, _MINUS_INF]

        def h(t: mpmath.mpf) -> mpmath.mpc:
            value = g(mpmath.mpc(t))
            largest[0] = max(largest[0], _fastlog(mpmath.re(value)))
            largest[1] = max(largest[1], _fastlog(mpmath.im(value)))
            return value
        try:
            result, error = mpmath.quad(h, _subdivision(lo, hi), method='tanh-sinh', error=True)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, NotImplementedError,
                mpmath.libmp.NoConvergence):
            return None
        value = mpmath.mpc(result)
        if not (mpmath.isfinite(mpmath.re(value)) and mpmath.isfinite(mpmath.im(value))):
            return None
        quadrature_error = _fastlog(mpmath.mpf(abs(error)))
    accuracies: list[int] = []
    for part, big in ((mpmath.re(value), largest[0]), (mpmath.im(value), largest[1])):
        if big == _MINUS_INF:
            continue  # the part is zero all along: exact
        if part == 0:
            accuracies.append(-1)
        else:
            accuracies.append(int(-max(big - _fastlog(part) - prec, quadrature_error)))
    return value, min(accuracies) if accuracies else prec


# PORTED FROM SYMPY: sympy/core/evalf.py:evalf_integral
def evalf_integral(f: Expr, x: Symbol, lower: Expr, upper: Expr, prec: int) -> Optional[mpmath.mpc]:
    """The value of ``Integral(f, (x, lower, upper))`` at ``prec`` bits,
    the working precision raised until the estimated accuracy reaches
    ``prec`` and the value is confirmed at a higher precision; ``None``
    when not evaluable or not accurate (SymPy returns such a value with
    the digits it trusts; a value of the core has no accuracy, so a value
    not accurate is none: the quadrature of ``sin(x)/x`` over ``(0, oo)``
    is not decided by tanh-sinh).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer, oo
    >>> from sympy_extras.functions import exp, sin
    >>> x = Symbol('x')
    >>> print(evalf_integral(exp(-x**2), x, Integer(0), oo, 53).real)
    0.886226925452758
    >>> print(evalf_integral(sin(x)/x, x, Integer(1), Integer(2), 53).real)
    0.659329906435512
    >>> evalf_integral(sin(x)/x, x, Integer(0), oo, 53) is None
    True
    """
    workprec = prec
    maxprec = max(2 * prec + 40, 333)
    i = 0
    previous: Optional[mpmath.mpc] = None
    while True:
        found = do_integral(f, x, lower, upper, workprec)
        if found is None:
            return None
        result, accuracy = found
        if accuracy >= prec:
            # confirmed by the value at a higher precision (mpmath's error
            # estimate is optimistic for an algebraic decay: x**(-3/2)
            # over (161, oo) at 90 bits estimates 98 bits of accuracy and
            # has 53)
            if previous is not None and _agree(previous, result, prec):
                return result
            previous = result
            if workprec >= maxprec:
                return None
            workprec = min(workprec + max(prec // 2, 20), maxprec)
            continue
        previous = None
        if workprec >= maxprec:
            return None
        if accuracy == -1:
            # maybe the answer is zero: the precision doubled
            workprec *= 2
        else:
            workprec += max(prec, 2**i)
        workprec = min(workprec, maxprec)
        i += 1


def _agree(a: mpmath.mpc, b: mpmath.mpc, prec: int) -> bool:
    with mpmath.workprec(prec + 20):
        return bool(abs(a - b) <= mpmath.mpf(2) ** (-prec) * max(abs(b), mpmath.mpf(2) ** (-prec)))
