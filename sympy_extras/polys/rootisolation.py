r"""Real root isolation and counting for dense polynomials with rational
coefficients.

**Descartes' rule with bisection** (Collins and Akritas; Rouillier and
Zimmermann): for a square-free $f$, the positive roots are brought into
$(0, 1)$ by the Cauchy bound, and an interval $(c/2^k, (c + 1)/2^k)$
carries the polynomial $p$ whose roots in $(0, 1)$ are the roots of $f$
in the interval. The number $v$ of sign variations of $(x + 1)^n p(1/(x
+ 1))$ bounds the number of roots of $p$ in $(0, 1)$ and has the same
parity: $v = 0$ means no root, $v = 1$ exactly one; otherwise the
interval is bisected ($p(x/2)$ and $p((x + 1)/2)$ up to scaling), the
midpoint being checked for an exact root. Each interval returned contains
one root, as an open interval with rational endpoints or as a point
(``a == b``) for a rational root. **Sturm sequences** count the roots in
an interval by the difference of the sign variations at its ends, and
the isolating intervals are refined by bisection.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.rootisolation import dup_isolate_real_roots, dup_count_real_roots, dup_refine_real_root
>>> f = [Fraction(1), Fraction(0), Fraction(-2)]
>>> dup_isolate_real_roots(f)
[((Fraction(-3, 1), Fraction(0, 1)), 1), ((Fraction(0, 1), Fraction(3, 1)), 1)]
>>> dup_count_real_roots(f), dup_count_real_roots(f, Fraction(0), Fraction(3))
(2, 1)
>>> dup_refine_real_root(f, Fraction(0), Fraction(3), Fraction(1, 100))
(Fraction(723, 512), Fraction(363, 256))
>>> dup_isolate_real_roots([Fraction(1), Fraction(-3), Fraction(3), Fraction(-1)])
[((Fraction(0, 1), Fraction(2, 1)), 3)]

References
==========

.. [1] G. E. Collins, A. G. Akritas, *Polynomial real root isolation
       using Descartes' rule of signs*, SYMSAC '76, 272-275.
.. [2] F. Rouillier, P. Zimmermann, *Efficient isolation of polynomial's
       real roots*, Journal of Computational and Applied Mathematics 162
       (2004), 33-50.
.. [3] S. Basu, R. Pollack, M.-F. Roy, *Algorithms in Real Algebraic
       Geometry*, 2nd ed., Springer, 2006, Section 2.2 (Sturm sequences)
       and Chapter 10 (real root isolation).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .densearith import dup_div, dup_mul_ground, dup_neg
from .densebasic import dup_degree, dup_reverse, dup_strip
from .densetools import dup_diff, dup_eval, dup_mirror, dup_monic, dup_scale, dup_shift, dup_sign_variations
from .domains.rationalfield import QQ
from .sqfreetools import dup_sqf_list, dup_sqf_part

__all__ = ['dup_cauchy_bound', 'dup_sturm', 'dup_count_real_roots', 'dup_isolate_real_roots_sqf',
           'dup_isolate_real_roots', 'dup_refine_real_root', 'dup_sign_at', 'Interval']

#: an isolating interval: ``(a, b)`` with the root in the open interval
#: when ``a < b``, the root itself when ``a == b``
Interval = tuple[Fraction, Fraction]


def dup_cauchy_bound(f: list[Fraction]) -> Fraction:
    """``1 + max |a_i/a_n|``: every root has absolute value below it."""
    if dup_degree(f) < 1:
        return Fraction(1)
    lc = f[0]
    return 1 + max(abs(c/lc) for c in f[1:])


def dup_sign_at(f: list[Fraction], a: Fraction) -> int:
    """The sign of ``f(a)``."""
    return QQ.sign(dup_eval(f, a, QQ))


def dup_sturm(f: list[Fraction]) -> list[list[Fraction]]:
    """The Sturm sequence ``f, f', -rem(f, f'), ...`` (each polynomial
    made monic up to its sign, which the counting allows)."""
    sequence = [dup_monic_signed(f), dup_monic_signed(dup_diff(f, 1, QQ))]
    while sequence[-1]:
        remainder = dup_div(sequence[-2], sequence[-1], QQ)[1]
        sequence.append(dup_monic_signed(dup_neg(remainder, QQ)))
    sequence.pop()
    return sequence


def dup_monic_signed(f: list[Fraction]) -> list[Fraction]:
    """``f`` divided by the absolute value of its leading coefficient."""
    if not f:
        return []
    return dup_mul_ground(f, 1/abs(f[0]), QQ)


def _variations_at(sequence: list[list[Fraction]], a: Optional[Fraction], at_infinity: int) -> int:
    """The sign variations of the sequence at ``a``, or at ``-oo``
    (``at_infinity = -1``) or ``+oo`` (``+1``) when ``a`` is ``None``."""
    signs: list[int] = []
    for g in sequence:
        if a is None:
            s = QQ.sign(g[0]) if g else 0
            if at_infinity < 0 and dup_degree(g) % 2:
                s = -s
        else:
            s = dup_sign_at(g, a)
        if s:
            signs.append(s)
    return sum(1 for s, t in zip(signs, signs[1:]) if s != t)


def dup_count_real_roots(f: list[Fraction], inf: Optional[Fraction] = None, sup: Optional[Fraction] = None) -> int:
    """The number of distinct real roots in ``(inf, sup]`` (the whole line
    when the bounds are ``None``), by Sturm's theorem on the square-free
    part."""
    if dup_degree(f) < 1:
        return 0
    g = dup_sqf_part(f, QQ)
    sequence = dup_sturm(g)
    return _variations_at(sequence, inf, -1) - _variations_at(sequence, sup, 1)


def _descartes_bound(p: list[Fraction]) -> int:
    """The sign variations of ``(x + 1)**n p(1/(x + 1))``: a bound on the
    roots of ``p`` in ``(0, 1)`` with the same parity."""
    return dup_sign_variations(dup_shift(dup_reverse(p, QQ), Fraction(1), QQ), QQ)


def _isolate_in_unit_interval(p: list[Fraction], c: int, k: int, found: list[tuple[int, int, int]]) -> None:
    """The roots of ``p`` in ``(0, 1)`` as the nodes ``(c, k, exact)`` of
    the bisection tree: the interval ``(c/2**k, (c + 1)/2**k)`` of the
    original unit interval, ``exact = 1`` when the root is its left end
    (``c/2**k``), ``0`` for a root inside."""
    v = _descartes_bound(p)
    if v == 0:
        return
    if v == 1:
        found.append((c, k, 0))
        return
    # the left half: the roots of p(x/2) in (0, 1)
    left = dup_scale(p, Fraction(1, 2), QQ)
    _isolate_in_unit_interval(left, 2*c, k + 1, found)
    right = dup_shift(left, Fraction(1), QQ)
    if right and right[-1] == 0:
        # the midpoint is a root: strip it and go on with the right half
        found.append((2*c + 1, k + 1, 1))
        right = dup_strip(right[:-1], QQ)
    _isolate_in_unit_interval(right, 2*c + 1, k + 1, found)


def dup_isolate_real_roots_sqf(f: list[Fraction], eps: Optional[Fraction] = None) -> list[Interval]:
    """Isolating intervals of the real roots of a square-free ``f``, in
    increasing order, each open interval (rational endpoints) containing
    one root or a point for a rational root; refined below the width
    ``eps`` when given."""
    if dup_degree(f) < 1:
        return []
    f = dup_monic(f, QQ)
    intervals: list[Interval] = []
    whole = f
    # the root 0
    if f[-1] == 0:
        intervals.append((Fraction(0), Fraction(0)))
        f = dup_strip(f[:-1], QQ)
    bound = dup_cauchy_bound(f)
    for sign in (-1, 1):
        g = dup_mirror(f, QQ) if sign < 0 else f
        # the positive roots of g into (0, 1): g(bound x)
        scaled = dup_scale(g, bound, QQ)
        found: list[tuple[int, int, int]] = []
        _isolate_in_unit_interval(scaled, 0, 0, found)
        for c, k, exact in found:
            a = Fraction(c, 2**k)*bound
            b = Fraction(c + 1, 2**k)*bound
            if exact:
                interval = (a, a)
            else:
                interval = (a, b)
            if sign < 0:
                interval = (-interval[1], -interval[0])
            intervals.append(interval)
    # an endpoint of an open interval may be another root (found exactly
    # at a midpoint, or 0): shrink such intervals
    sequence: list[list[Fraction]] = []
    for i, (a, b) in enumerate(intervals):
        if a != b and (dup_sign_at(whole, a) == 0 or dup_sign_at(whole, b) == 0):
            if not sequence:
                sequence = dup_sturm(whole)
            intervals[i] = _endpoints_off_roots(whole, sequence, a, b)
    intervals.sort()
    if eps is not None:
        intervals = [dup_refine_real_root(f, a, b, eps) for a, b in intervals]
    return intervals


def _endpoints_off_roots(f: list[Fraction], sequence: list[list[Fraction]], a: Fraction, b: Fraction) -> Interval:
    """The isolating interval ``(a, b)`` of a root of the square-free ``f``
    bisected (with Sturm counts) until neither endpoint is a root of
    ``f`` (an endpoint may be another root found exactly, or ``0``)."""
    while dup_sign_at(f, a) == 0 or dup_sign_at(f, b) == 0:
        m = (a + b)/2
        if dup_sign_at(f, m) == 0:
            return m, m
        # the number of roots in (a, m]
        if _variations_at(sequence, a, 0) - _variations_at(sequence, m, 0) == 1:
            b = m
        else:
            a = m
    return a, b


def dup_refine_real_root(f: list[Fraction], a: Fraction, b: Fraction, eps: Fraction) -> Interval:
    """The isolating interval ``(a, b)`` of a root of ``f`` bisected until
    its width is below ``eps`` (a point when the root is found exactly)."""
    if a == b:
        return a, b
    sa = dup_sign_at(f, a)
    if sa == 0:
        return a, a
    while b - a >= eps:
        m = (a + b)/2
        sm = dup_sign_at(f, m)
        if sm == 0:
            return m, m
        if sm == sa:
            a = m
        else:
            b = m
    return a, b


def dup_isolate_real_roots(f: list[Fraction], eps: Optional[Fraction] = None) -> list[tuple[Interval, int]]:
    """Isolating intervals of the real roots of ``f`` with their
    multiplicities, in increasing order."""
    if dup_degree(f) < 1:
        return []
    _, factors = dup_sqf_list(f, QQ)
    result: list[tuple[Interval, int]] = []
    for g, k in factors:
        for interval in dup_isolate_real_roots_sqf(g, eps):
            result.append((interval, k))
    result.sort(key=lambda item: item[0])
    return result
