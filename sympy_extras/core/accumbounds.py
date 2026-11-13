"""Accumulation bounds: ``AccumBounds(a, b)`` and ``accum_bounds``.

``AccumBounds(a, b)`` is the set of the values a quantity approaches
without settling (``sin(x)`` as ``x`` goes to ``oo`` accumulates on
``[-1, 1]``): the limits of oscillating functions are such bounds. The
arithmetic is interval arithmetic: ``AccumBounds(-1, 1) + 4`` is
``AccumBounds(3, 5)``, ``2*AccumBounds(-1, 1)`` is ``AccumBounds(-2, 2)``,
``AccumBounds(1, 2)**2`` is ``AccumBounds(1, 4)``; in sums and products a
bound is a coefficient (it combines with the numbers). ``accum_bounds``
evaluates (``accum_bounds(1, 1)`` is ``1``).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: AccumulationBounds*,
       https://docs.sympy.org/latest/modules/calculus/index.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/calculus/accumulationbounds.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the class is unevaluated and accum_bounds evaluates; the
# operations are module functions the operators and the evaluating
# functions of the core call; the comparisons of the relational module
# are not ported.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional, Union

from .basic import Basic, NodePrinter
from .expr import Expr, ExprLike, as_expr, to_expr
from .kind import Kind, NumberKind
from .numbers import ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity, nan, oo, zoo
from .sorting import ClassKey

__all__ = ['AccumBounds', 'AccumulationBounds', 'accum_bounds']


def _is_oo(e: Expr) -> bool:
    return isinstance(e, Infinity)


def _is_moo(e: Expr) -> bool:
    return isinstance(e, NegativeInfinity)


def _min(*args: Expr) -> Expr:
    from sympy_extras.functions.elementary.miscellaneous import min_
    return min_(*args)


def _max(*args: Expr) -> Expr:
    from sympy_extras.functions.elementary.miscellaneous import max_
    return max_(*args)


def _lt(a: Expr, b: Expr) -> Optional[bool]:
    """``a < b`` (``None`` when undecided)."""
    return (a - b).is_extended_negative


class AccumBounds(Expr):
    """The unevaluated accumulation bounds ``<min, max>``.

    Examples
    ========

    >>> from sympy_extras.core import oo
    >>> from sympy_extras.core.accumbounds import AccumBounds, accum_bounds
    >>> b = AccumBounds(-1, 1)
    >>> b, b + 4, 2*b, -b, b**2, 1/AccumBounds(1, 2), accum_bounds(3, 3)
    (AccumBounds(-1, 1), AccumBounds(3, 5), AccumBounds(-2, 2), AccumBounds(-1, 1), AccumBounds(0, 1), AccumBounds(1/2, 1), 3)
    >>> b.min, b.max, b.delta, b.mid, AccumBounds(0, oo) + AccumBounds(-oo, 0)
    (-1, 1, 2, 0, AccumBounds(-oo, oo))
    """

    __slots__ = ('_min', '_max')

    _op_priority = 11.0

    def __init__(self, min: ExprLike, max: ExprLike) -> None:
        self._min: Final[Expr] = to_expr(min)
        self._max: Final[Expr] = to_expr(max)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._min, self._max)

    @property
    def min(self) -> Expr:
        return self._min

    @property
    def max(self) -> Expr:
        return self._max

    @property
    def delta(self) -> Expr:
        return self._max - self._min

    @property
    def mid(self) -> Expr:
        return (self._min + self._max) / 2

    @property
    def kind(self) -> Kind:
        return NumberKind

    @property
    def is_number(self) -> bool:
        return False

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'AccumulationBounds')

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        a, b = args
        return AccumBounds(as_expr(a), as_expr(b))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        a, b = args
        return accum_bounds(as_expr(a), as_expr(b))

    def _eval_facts(self) -> dict[str, bool]:
        # PORTED FROM SYMPY: AccumulationBounds.is_extended_real, _eval_is_real
        facts = {'extended_real': True, 'commutative': True}
        if self._min.is_real is True and self._max.is_real is True:
            facts['real'] = True
        return facts

    def _sympystr(self, printer: NodePrinter) -> str:
        return 'AccumBounds(%s, %s)' % (printer.doprint(self._min), printer.doprint(self._max))

    # the operators: the interval arithmetic
    def __add__(self, other: ExprLike) -> Expr:
        return ab_add(self, to_expr(other))

    def __radd__(self, other: ExprLike) -> Expr:
        return ab_add(self, to_expr(other))

    def __sub__(self, other: ExprLike) -> Expr:
        return ab_sub(self, to_expr(other))

    def __rsub__(self, other: ExprLike) -> Expr:
        return ab_neg(self) + to_expr(other)

    def __neg__(self) -> Expr:
        return ab_neg(self)

    def __mul__(self, other: ExprLike) -> Expr:
        return ab_mul(self, to_expr(other))

    def __rmul__(self, other: ExprLike) -> Expr:
        return ab_mul(self, to_expr(other))

    def __truediv__(self, other: ExprLike) -> Expr:
        return ab_truediv(self, to_expr(other))

    def __rtruediv__(self, other: ExprLike) -> Expr:
        return ab_rtruediv(self, to_expr(other))

    def __pow__(self, other: ExprLike) -> Expr:
        return ab_pow(self, to_expr(other))

    def __rpow__(self, other: ExprLike) -> Expr:
        return ab_rpow(self, to_expr(other))

    def __abs__(self) -> Expr:
        return ab_abs(self)

    def contains(self, other: ExprLike) -> bool:
        """Whether ``other`` is in the bounds (``TypeError`` when undecided)."""
        # PORTED FROM SYMPY: AccumulationBounds.__contains__
        o = to_expr(other)
        if _is_oo(o) or _is_moo(o):
            return _is_moo(self._min) or _is_oo(self._max)
        low = (self._min - o).is_extended_nonpositive
        high = (self._max - o).is_extended_nonnegative
        if low is None or high is None:
            raise TypeError("input failed to evaluate")
        return low and high

    # PORTED FROM SYMPY: AccumulationBounds.intersection (with bounds)
    def intersection(self, other: AccumBounds) -> Optional[Expr]:
        """The common part of two bounds, ``None`` when empty."""
        if _lt(self._max, other._min) or _lt(other._max, self._min):
            return None
        if (self._min - other._min).is_extended_nonpositive:
            if (self._max - other._max).is_extended_nonpositive:
                return accum_bounds(other._min, self._max)
            if _lt(other._max, self._max):
                return other
        if (other._min - self._min).is_extended_nonpositive:
            if _lt(other._max, self._max):
                return accum_bounds(self._min, other._max)
            if _lt(self._max, other._max):
                return self
        return None

    def meets_points(self, points: list[Expr]) -> bool:
        """Whether one of ``points`` is in the bounds (SymPy's
        ``intersection`` with a finite set, not empty)."""
        return any(self.contains(p) for p in points)

    # PORTED FROM SYMPY: AccumulationBounds.union
    def union(self, other: AccumBounds) -> Optional[Expr]:
        if (self._min - other._min).is_extended_nonpositive and (self._max - other._min).is_extended_nonnegative:
            return accum_bounds(self._min, _max(self._max, other._max))
        if (other._min - self._min).is_extended_nonpositive and (other._max - self._min).is_extended_nonnegative:
            return accum_bounds(other._min, _max(self._max, other._max))
        return None


AccumulationBounds = AccumBounds


# PORTED FROM SYMPY: sympy/calculus/accumulationbounds.py:AccumulationBounds.__new__
def accum_bounds(min: ExprLike, max: ExprLike) -> Expr:
    """The evaluated bounds: a single value when ``min == max``; the
    bounds must be real and ordered."""
    lo, hi = to_expr(min), to_expr(max)
    if lo.is_extended_real is not True or hi.is_extended_real is not True:
        raise ValueError("Only real AccumulationBounds are supported")
    if hi == lo:
        return hi
    if hi.is_number and lo.is_number:
        bad = hi.is_comparable and lo.is_comparable and _lt(hi, lo) is True
    else:
        bad = (hi - lo).is_extended_negative is True
    if bad:
        raise ValueError("Lower limit should be smaller than upper limit")
    return AccumBounds(lo, hi)


# PORTED FROM SYMPY: AccumulationBounds.__add__
def ab_add(self: AccumBounds, other: Expr) -> Expr:
    from .add import Add
    if isinstance(other, AccumBounds):
        return accum_bounds(self.min + other.min, self.max + other.max)
    if (_is_oo(other) and _is_moo(self.min)) or (_is_moo(other) and _is_oo(self.max)):
        return accum_bounds(-oo, oo)
    if other.is_extended_real is True:
        if _is_moo(self.min) and _is_oo(self.max):
            return accum_bounds(-oo, oo)
        if _is_moo(self.min):
            return accum_bounds(-oo, self.max + other)
        if _is_oo(self.max):
            return accum_bounds(self.min + other, oo)
        return accum_bounds(self.min + other, self.max + other)
    return Add(self, other)


# PORTED FROM SYMPY: AccumulationBounds.__neg__
def ab_neg(self: AccumBounds) -> Expr:
    return accum_bounds(-self.max, -self.min)


# PORTED FROM SYMPY: AccumulationBounds.__sub__
def ab_sub(self: AccumBounds, other: Expr) -> Expr:
    from .add import Add
    if isinstance(other, AccumBounds):
        return accum_bounds(self.min - other.max, self.max - other.min)
    if (_is_moo(other) and _is_moo(self.min)) or (_is_oo(other) and _is_oo(self.max)):
        return accum_bounds(-oo, oo)
    if other.is_extended_real is True:
        if _is_moo(self.min) and _is_oo(self.max):
            return accum_bounds(-oo, oo)
        if _is_moo(self.min):
            return accum_bounds(-oo, self.max - other)
        if _is_oo(self.max):
            return accum_bounds(self.min - other, oo)
        return accum_bounds(self.min - other, self.max - other)
    return Add(self, -other)


def _is_zero(e: Expr) -> bool:
    return e.is_zero is True


# PORTED FROM SYMPY: AccumulationBounds.__mul__
def ab_mul(self: AccumBounds, other: Expr) -> Expr:
    from .mul import Mul
    from .order import Order
    if _is_moo(self.min) and _is_oo(self.max):
        return self
    if isinstance(other, AccumBounds):
        if _is_moo(other.min) and _is_oo(other.max):
            return other
        v: list[Expr] = []
        for a in (self.min, self.max):
            vi = other * a
            for t in (vi.args if isinstance(vi, AccumBounds) else (vi,)):
                if isinstance(t, Expr) and t not in v:
                    v.append(t)
        return accum_bounds(_min(*v), _max(*v))
    if _is_oo(other):
        if _is_zero(self.min):
            return accum_bounds(Integer(0), oo)
        if _is_zero(self.max):
            return accum_bounds(-oo, Integer(0))
    if _is_moo(other):
        if _is_zero(self.min):
            return accum_bounds(-oo, Integer(0))
        if _is_zero(self.max):
            return accum_bounds(Integer(0), oo)
    if other.is_extended_real is True:
        if _is_zero(other):
            if _is_oo(self.max):
                return accum_bounds(Integer(0), oo)
            if _is_moo(self.min):
                return accum_bounds(-oo, Integer(0))
            return Integer(0)
        if other.is_extended_positive is True:
            return accum_bounds(self.min * other, self.max * other)
        if other.is_extended_negative is True:
            return accum_bounds(self.max * other, self.min * other)
    if isinstance(other, Order):
        return other
    return Mul(self, other)


# PORTED FROM SYMPY: AccumulationBounds.__truediv__
def ab_truediv(self: AccumBounds, other: Expr) -> Expr:
    from .mul import Mul
    if isinstance(other, AccumBounds):
        if other.min.is_positive is True or other.max.is_negative is True:
            return ab_mul(self, _as_bounds(accum_bounds(1 / other.max, 1 / other.min)))
        if (self.min.is_extended_nonpositive is True and self.max.is_extended_nonnegative is True and
                other.min.is_extended_nonpositive is True and other.max.is_extended_nonnegative is True):
            if _is_zero(self.min) and _is_zero(other.min):
                return accum_bounds(Integer(0), oo)
            if _is_zero(self.max) and _is_zero(other.min):
                return accum_bounds(-oo, Integer(0))
            return accum_bounds(-oo, oo)
        if self.max.is_extended_negative is True:
            if other.min.is_extended_negative is True:
                if _is_zero(other.max):
                    return accum_bounds(self.max / other.min, oo)
                if other.max.is_extended_positive is True:
                    return accum_bounds(-oo, oo)
            if _is_zero(other.min) and other.max.is_extended_positive is True:
                return accum_bounds(-oo, self.max / other.max)
        if self.min.is_extended_positive is True:
            if other.min.is_extended_negative is True:
                if _is_zero(other.max):
                    return accum_bounds(-oo, self.min / other.min)
                if other.max.is_extended_positive is True:
                    return accum_bounds(-oo, oo)
            if _is_zero(other.min) and other.max.is_extended_positive is True:
                return accum_bounds(self.min / other.max, oo)
    elif other.is_extended_real is True:
        if _is_oo(other) or _is_moo(other):
            if _is_moo(self.min) and _is_oo(self.max):
                return accum_bounds(-oo, oo)
            if _is_oo(self.max):
                return accum_bounds(_min(Integer(0), other), _max(Integer(0), other))
            if _is_moo(self.min):
                return accum_bounds(_min(Integer(0), -other), _max(Integer(0), -other))
        if other.is_extended_positive is True:
            return accum_bounds(self.min / other, self.max / other)
        if other.is_extended_negative is True:
            return accum_bounds(self.max / other, self.min / other)
    inverse = 1 / other
    if isinstance(inverse, ComplexInfinity):
        return Mul(self, inverse)
    return self * inverse


# PORTED FROM SYMPY: AccumulationBounds.__rtruediv__
def ab_rtruediv(self: AccumBounds, other: Expr) -> Expr:
    from .mul import Mul
    if other.is_extended_real is True:
        if _is_zero(other):
            return Integer(0)
        if self.min.is_extended_nonpositive is True and self.max.is_extended_nonnegative is True:
            if _is_zero(self.min):
                if other.is_extended_positive is True:
                    return accum_bounds(other * (1 / self.max), oo)
                if other.is_extended_negative is True:
                    return accum_bounds(-oo, other * (1 / self.max))
            if _is_zero(self.max):
                if other.is_extended_positive is True:
                    return accum_bounds(-oo, other * (1 / self.min))
                if other.is_extended_negative is True:
                    return accum_bounds(other * (1 / self.min), oo)
            return accum_bounds(-oo, oo)
        return accum_bounds(_min(other / self.min, other / self.max), _max(other / self.min, other / self.max))
    return Mul(other, ab_pow(self, Integer(-1)))


def _as_bounds(e: Expr) -> AccumBounds:
    if not isinstance(e, AccumBounds):
        raise TypeError("bounds are expected, got %s" % (e,))
    return e


# PORTED FROM SYMPY: AccumulationBounds.__pow__
def ab_pow(self: AccumBounds, other: Expr) -> Expr:
    from .power import Pow
    if isinstance(other, Infinity):
        if self.min.is_extended_nonnegative is True:
            if _lt(self.max, Integer(1)):
                return Integer(0)
            if _lt(Integer(1), self.min):
                return oo
            return accum_bounds(Integer(0), oo)
        if self.max.is_extended_negative is True:
            if _lt(Integer(-1), self.min):
                return Integer(0)
            if _lt(self.max, Integer(-1)):
                return zoo
            return nan
        if _lt(Integer(-1), self.min):
            if _lt(self.max, Integer(1)):
                return Integer(0)
            return accum_bounds(Integer(0), oo)
        return accum_bounds(-oo, oo)
    if isinstance(other, NegativeInfinity):
        return (1 / self)**oo
    if (self.max - self.min).is_nonnegative is True:
        if self.min.is_nonnegative is True:
            if other.is_nonnegative is True:
                return accum_bounds(self.min**other, self.max**other)
    if _is_zero(other):
        return Integer(1)  # x**0 = 1
    if other.is_integer is True and isinstance(other, Integer):
        if self.min.is_extended_positive is True:
            return accum_bounds(_min(self.min**other, self.max**other), _max(self.min**other, self.max**other))
        if self.max.is_extended_negative is True:
            return accum_bounds(_min(self.max**other, self.min**other), _max(self.max**other, self.min**other))
        if other.p % 2 == 0:
            if other.p < 0:
                if _is_zero(self.min):
                    return accum_bounds(self.max**other, oo)
                if _is_zero(self.max):
                    return accum_bounds(self.min**other, oo)
                return (1 / self)**(-other)
            return accum_bounds(Integer(0), _max(self.min**other, self.max**other))
        if other.p < 0:
            if _is_zero(self.min):
                return accum_bounds(self.max**other, oo)
            if _is_zero(self.max):
                return accum_bounds(-oo, self.min**other)
            return (1 / self)**(-other)
        return accum_bounds(self.min**other, self.max**other)
    # (SymPy's condition: a nonnegative minimum, its second alternative
    # implies the first)
    if (other.is_number or other.is_rational is True) and self.min.is_extended_nonnegative is True:
        num, den = other.as_numer_denom()
        if num == Integer(1):
            return accum_bounds(self.min**(1 / den), self.max**(1 / den))
        if den != Integer(1):
            return (self**num)**(1 / den)  # a nonnegative base
    if isinstance(other, AccumBounds):
        if (self.min.is_extended_positive is True or
                (self.min.is_extended_nonnegative is True and other.min.is_extended_nonnegative is True)):
            p = [self**i for i in (other.min, other.max)]
            if not any(isinstance(i, Pow) for i in p):
                a = [j for i in p for j in (i.args if isinstance(i, AccumBounds) else (i,)) if isinstance(j, Expr)]
                least, most = a[0], a[0]
                for v in a[1:]:
                    lt_least, gt_most = _lt(v, least), _lt(most, v)
                    if lt_least is None or gt_most is None:
                        return Pow(self, other)  # cannot sort
                    if lt_least:
                        least = v
                    if gt_most:
                        most = v
                return accum_bounds(least, most)
    return Pow(self, other)


# PORTED FROM SYMPY: AccumulationBounds.__rpow__
def ab_rpow(self: AccumBounds, other: Expr) -> Expr:
    from .power import Pow
    if other.is_real is True and other.is_extended_nonnegative is True and (
            self.max - self.min).is_extended_positive is True:
        if other == Integer(1):
            return Integer(1)
        if other.is_extended_positive is True:
            a, b = other**self.min, other**self.max
            if _lt(b, a):
                a, b = b, a
            return accum_bounds(a, b)
        if _is_zero(other):
            if _is_zero(self.min):
                return accum_bounds(Integer(0), Integer(1))
            if self.min.is_extended_positive is True:
                return Integer(0)
    return Pow(other, self)


# PORTED FROM SYMPY: AccumulationBounds.__abs__
def ab_abs(self: AccumBounds) -> Expr:
    from sympy_extras.functions.elementary.complexes import abs_
    if self.max.is_extended_negative is True:
        return ab_neg(self)
    if self.min.is_extended_negative is True:
        return accum_bounds(Integer(0), _max(abs_(self.min), self.max))
    return self

