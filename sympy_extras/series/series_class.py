"""The base class of the series on the typed core (SymPy's
``sympy.series.series_class``): :class:`SeriesBase`, an expression whose
terms are indexed by the integers of an interval, with ``term``, the
iteration over the terms and their slicing.

This module is a typed port of SymPy's ``sympy/series/series_class.py``.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/series_class.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; the terms not cached; no interval property (start and stop).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Iterator, Union, overload

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Infinity, Integer, NegativeInfinity
from sympy_extras.core.symbol import Symbol

__all__ = ['SeriesBase']

#: a point of a series: an integer or an expression
Point = Union[int, Expr]


def _less(a: Expr, b: Expr) -> bool:
    return (b - a).is_extended_positive is True


class SeriesBase(Expr):
    """The base class of the series (SymPy's ``SeriesBase``): the terms
    at the points ``start, start + 1, ..., stop``."""

    __slots__ = ()

    @property
    def start(self) -> Expr:
        """The first point of the series (included)."""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self) -> Expr:
        """The last point of the series (included)."""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self) -> Expr:
        """The number of terms (``oo`` for an infinite series)."""
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self) -> tuple[Symbol, ...]:
        """The bound variables."""
        return ()

    @property
    def free_symbols(self) -> set[Symbol]:
        result: set[Symbol] = set()
        for a in self.args:
            result |= a.free_symbols
        return result - set(self.variables)

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}

    # PORTED FROM SYMPY: sympy/series/series_class.py:SeriesBase.term
    def term(self, pt: Point) -> Expr:
        """The term at the point ``pt``; ``IndexError`` outside the range."""
        p = Integer(pt) if isinstance(pt, int) else pt
        if _less(p, self.start) or _less(self.stop, p):
            raise IndexError("Index %s out of bounds (%s, %s)" % (p, self.start, self.stop))
        return self._eval_term(p)

    def _eval_term(self, pt: Expr) -> Expr:
        raise NotImplementedError("The _eval_term method should be added to %s to return series term so it is "
                                  "available when 'term' calls it." % type(self).__name__)

    # PORTED FROM SYMPY: sympy/series/series_class.py:SeriesBase._ith_point
    def _ith_point(self, i: int) -> Expr:
        if isinstance(self.start, NegativeInfinity):
            return self.stop - i
        return self.start + i

    def _length_bound(self) -> Union[int, None]:
        length = self.length
        if isinstance(length, Infinity):
            return None
        if isinstance(length, Integer):
            return int(length)
        raise ValueError("the length of %s is not known" % (self,))

    def __iter__(self) -> Iterator[Expr]:
        bound = self._length_bound()
        i = 0
        while bound is None or i < bound:
            yield self.term(self._ith_point(i))
            i += 1

    @overload
    def __getitem__(self, index: int) -> Expr: ...

    @overload
    def __getitem__(self, index: slice) -> list[Expr]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Expr, list[Expr]]:
        if isinstance(index, int):
            return self.term(self._ith_point(index))
        start = 0 if index.start is None else int(index.start)
        if index.stop is None:
            bound = self._length_bound()
            if bound is None:
                raise ValueError("an infinite series needs the end of the slice")
            stop = bound
        else:
            stop = int(index.stop)
        step = 1 if index.step is None else int(index.step)
        return [self.term(self._ith_point(i)) for i in range(start, stop, step)]
