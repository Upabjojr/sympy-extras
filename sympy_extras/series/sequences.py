"""Sequences on the typed core (SymPy's ``sympy.series.sequences``): the
sequences given by a formula (:class:`SeqFormula`) or by a period
(:class:`SeqPer`), their termwise sums and products (:class:`SeqAdd`,
:class:`SeqMul`) and the function :func:`sequence` which builds them.

This module is a typed port of the parts of SymPy's
``sympy/series/sequences.py`` which the formal power series and the
Fourier series use. The classes never evaluate: ``SeqAdd(a, b)`` is the
node, :func:`seq_add` (what ``a + b`` calls) combines two formulas into
one formula on the intersection of their intervals as SymPy's ``SeqAdd``
does, and :func:`sequence` returns :class:`EmptySequence` for an empty
range. The sequences are not expressions (``Basic`` nodes, as in SymPy).

Examples
========

>>> from sympy_extras.core import Symbol, oo
>>> from sympy_extras.series.sequences import sequence
>>> n = Symbol('n')
>>> s = sequence(n**2, (n, 0, oo))
>>> s, s[:4], s.coeff(5)
(SeqFormula(n**2, (n, 0, oo)), [0, 1, 4, 9], 25)
>>> (s + sequence(n, (n, 2, oo)))[:3]
[6, 12, 20]
>>> p = sequence((1, 2), (n, 0, oo))
>>> (p*s)[:4], (-p)[:3]
([0, 2, 4, 18], [-1, -2, -1])
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/sequences.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; unevaluating classes with the evaluating functions
# sequence, seq_add, seq_mul; only SeqFormula, SeqPer, SeqAdd, SeqMul, EmptySequence.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Final, Iterator, Optional, Sequence, Union, overload

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Infinity, Integer, NegativeInfinity, oo
from sympy_extras.core.symbol import Dummy, Symbol

__all__ = ['SeqBase', 'SeqFormula', 'SeqPer', 'SeqAdd', 'SeqMul', 'EmptySequence', 'sequence',
           'seq_add', 'seq_mul']

#: a point of a sequence: an integer or an expression
Point = Union[int, Expr]
#: the limits of a sequence: ``(start, stop)`` or ``(variable, start, stop)``
SeqLimits = Sequence[ExprLike]


def _less(a: Expr, b: Expr) -> bool:
    """Whether ``a < b`` is known to hold."""
    return (b - a).is_extended_positive is True


def _max(a: Expr, b: Expr) -> Expr:
    from sympy_extras.functions.elementary.miscellaneous import max_
    return max_(a, b)


def _min(a: Expr, b: Expr) -> Expr:
    from sympy_extras.functions.elementary.miscellaneous import min_
    return min_(a, b)


def _empty_range(start: Expr, stop: Expr) -> bool:
    return _less(stop, start)


class SeqBase(Basic):
    """The base class of the sequences (SymPy's ``SeqBase``)."""

    __slots__ = ()

    @property
    def start(self) -> Expr:
        """The first point of the sequence (included)."""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self) -> Expr:
        """The last point of the sequence (included)."""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self) -> Expr:
        """The number of points of the sequence (``oo`` for an infinite one)."""
        return self.stop - self.start + 1

    @property
    def variables(self) -> tuple[Symbol, ...]:
        """The bound variables."""
        return ()

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return self.variables

    @property
    def free_symbols(self) -> set[Symbol]:
        result: set[Symbol] = set()
        for a in self.args:
            result |= a.free_symbols
        return result - set(self.variables)

    def _intersect_interval(self, other: SeqBase) -> tuple[Expr, Expr]:
        return _max(self.start, other.start), _min(self.stop, other.stop)

    # PORTED FROM SYMPY: sympy/series/sequences.py:SeqBase.coeff
    def coeff(self, pt: Point) -> Expr:
        """The term at the point ``pt``; ``IndexError`` outside the range."""
        p = to_expr(pt)
        if _less(p, self.start) or _less(self.stop, p):
            raise IndexError("Index %s out of bounds (%s, %s)" % (p, self.start, self.stop))
        return self._eval_coeff(p)

    def _eval_coeff(self, pt: Expr) -> Expr:
        raise NotImplementedError("(%s)._eval_coeff" % self)

    # PORTED FROM SYMPY: sympy/series/sequences.py:SeqBase._ith_point
    def _ith_point(self, i: int) -> Expr:
        if isinstance(self.start, NegativeInfinity):
            return self.stop - i
        return self.start + i

    def _add(self, other: SeqBase) -> Optional[SeqBase]:
        """The termwise sum when ``self`` knows how to add ``other``."""
        return None

    def _mul(self, other: SeqBase) -> Optional[SeqBase]:
        """The termwise product when ``self`` knows how to multiply ``other``."""
        return None

    def coeff_mul(self, c: ExprLike) -> SeqBase:
        """The sequence multiplied termwise by the expression ``c``."""
        raise NotImplementedError("(%s).coeff_mul" % self)

    def __add__(self, other: SeqBase) -> SeqBase:
        if not isinstance(other, SeqBase):
            raise TypeError('cannot add sequence and %s' % type(other))
        return seq_add(self, other)

    def __sub__(self, other: SeqBase) -> SeqBase:
        if not isinstance(other, SeqBase):
            raise TypeError('cannot subtract sequence and %s' % type(other))
        return seq_add(self, -other)

    def __neg__(self) -> SeqBase:
        return self.coeff_mul(-1)

    def __mul__(self, other: SeqBase) -> SeqBase:
        if not isinstance(other, SeqBase):
            raise TypeError('cannot multiply sequence and %s' % type(other))
        return seq_mul(self, other)

    def __iter__(self) -> Iterator[Expr]:
        if isinstance(self, EmptySequence):
            return
        length = self.length
        if not isinstance(length, (Integer, Infinity)):
            raise ValueError("the length of %s is not known" % (self,))
        i = 0
        while isinstance(length, Infinity) or i < int(length):
            yield self.coeff(self._ith_point(i))
            i += 1

    @overload
    def __getitem__(self, index: int) -> Expr: ...

    @overload
    def __getitem__(self, index: slice) -> list[Expr]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Expr, list[Expr]]:
        if isinstance(index, int):
            return self.coeff(self._ith_point(index))
        start = 0 if index.start is None else int(index.start)
        if index.stop is None:
            length = self.length
            if not isinstance(length, Integer):
                raise ValueError("an infinite sequence needs the end of the slice")
            stop = int(length)
        else:
            stop = int(index.stop)
        step = 1 if index.step is None else int(index.step)
        return [self.coeff(self._ith_point(i)) for i in range(start, stop, step)]


# PORTED FROM SYMPY: sympy/series/sequences.py:EmptySequence
class EmptySequence(SeqBase):
    """The sequence without terms."""

    __slots__ = ()

    def __init__(self) -> None:
        Basic.__init__(self)

    @property
    def length(self) -> Expr:
        return Integer(0)

    @property
    def start(self) -> Expr:
        return oo

    @property
    def stop(self) -> Expr:
        return -oo

    def coeff_mul(self, c: ExprLike) -> SeqBase:
        return self

    def _sympystr(self, printer: object) -> str:
        return 'EmptySequence'


class _SeqExpr(SeqBase):
    """A sequence on a range of its own variable (SymPy's ``SeqExpr``)."""

    __slots__ = ('_variable', '_start', '_stop')

    def __init__(self, variable: Symbol, start: ExprLike, stop: ExprLike) -> None:
        if not isinstance(variable, Symbol):
            raise ValueError('Invalid limits given: %s' % ((variable, start, stop),))
        s, e = to_expr(start), to_expr(stop)
        if isinstance(s, NegativeInfinity) and isinstance(e, Infinity):
            raise ValueError("Both the start and end value cannot be unbounded")
        self._variable: Final[Symbol] = variable
        self._start: Final[Expr] = s
        self._stop: Final[Expr] = e

    @property
    def limits(self) -> Tuple:
        """``(variable, start, stop)``."""
        return Tuple(self._variable, self._start, self._stop)

    @property
    def start(self) -> Expr:
        return self._start

    @property
    def stop(self) -> Expr:
        return self._stop

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return (self._variable,)

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # the variable is bound
        if old == self._variable:
            return self
        return SeqBase._subs(self, old, new)


def _parse_limits(free: set[Symbol], limits: Optional[SeqLimits], what: str) -> tuple[Symbol, Expr, Expr]:
    if limits is None:
        return _find_x(free, what), Integer(0), oo
    if len(limits) == 3:
        v = limits[0]
        if not isinstance(v, Symbol):
            raise ValueError('Invalid limits given: %s' % (tuple(limits),))
        return v, to_expr(limits[1]), to_expr(limits[2])
    if len(limits) == 2:
        return _find_x(free, what), to_expr(limits[0]), to_expr(limits[1])
    raise ValueError('Invalid limits given: %s' % (tuple(limits),))


def _find_x(free: set[Symbol], what: str) -> Symbol:
    if len(free) == 1:
        return next(iter(free))
    if not free or what == 'periodical':
        return Dummy('k')
    raise ValueError("specify dummy variables for %s. If the formula contains more than one free symbol, "
                     "a dummy variable should be supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))" % what)


# PORTED FROM SYMPY: sympy/series/sequences.py:SeqFormula
class SeqFormula(_SeqExpr):
    """The sequence of the values of ``formula`` at the integers of a range.

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.series.sequences import SeqFormula
    >>> n = Symbol('n')
    >>> s = SeqFormula(n**2, (n, 0, 5))
    >>> s, s.formula, s.length, list(s)
    (SeqFormula(n**2, (n, 0, 5)), n**2, 6, [0, 1, 4, 9, 16, 25])
    >>> SeqFormula(n**2, (n, 0, oo)).coeff_mul(2)[:3]
    [0, 2, 8]
    """

    __slots__ = ('_formula',)

    def __init__(self, formula: ExprLike, limits: Optional[Union[SeqLimits, Tuple]] = None) -> None:
        f = to_expr(formula)
        lims: Optional[SeqLimits] = None
        if isinstance(limits, Tuple):
            lims = [as_expr(a) for a in limits.args]
        else:
            lims = limits
        v, start, stop = _parse_limits(f.free_symbols, lims, str(f))
        self._formula: Final[Expr] = f
        _SeqExpr.__init__(self, v, start, stop)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._formula, self.limits)

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, limits = args
        if not isinstance(limits, Tuple):
            raise TypeError("the limits of a sequence are a Tuple")
        return SeqFormula(as_expr(f), limits)

    @property
    def formula(self) -> Expr:
        """The formula of the terms."""
        return self._formula

    @property
    def gen(self) -> Expr:
        return self._formula

    def _eval_coeff(self, pt: Expr) -> Expr:
        return self._formula.subs(self._variable, pt)

    def _add(self, other: SeqBase) -> Optional[SeqBase]:
        if isinstance(other, SeqFormula):
            formula = self._formula + other.formula.subs(other.variables[0], self._variable)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (self._variable, start, stop))
        return None

    def _mul(self, other: SeqBase) -> Optional[SeqBase]:
        if isinstance(other, SeqFormula):
            formula = self._formula * other.formula.subs(other.variables[0], self._variable)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (self._variable, start, stop))
        return None

    def coeff_mul(self, c: ExprLike) -> SeqFormula:
        return SeqFormula(self._formula * to_expr(c), self.limits)

    def expand(self) -> SeqFormula:
        """The sequence of the expanded formula."""
        from sympy_extras.core.function import expand
        return SeqFormula(expand(self._formula), self.limits)


# PORTED FROM SYMPY: sympy/series/sequences.py:SeqPer
class SeqPer(_SeqExpr):
    """The periodic sequence repeating ``periodical``.

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.series.sequences import SeqPer
    >>> n = Symbol('n')
    >>> s = SeqPer((1, n, 3), (n, 0, oo))
    >>> s.period, s[:6]
    (3, [1, 1, 3, 1, 4, 3])
    """

    __slots__ = ('_periodical',)

    def __init__(self, periodical: Union[Sequence[ExprLike], Tuple],
                 limits: Optional[Union[SeqLimits, Tuple]] = None) -> None:
        if isinstance(periodical, Tuple):
            elements = tuple(as_expr(a) for a in periodical.args)
        else:
            elements = tuple(to_expr(a) for a in periodical)
        if not elements:
            raise ValueError("invalid period %s should be something like e.g (1, 2)" % (periodical,))
        free: set[Symbol] = set()
        for e in elements:
            free |= e.free_symbols
        lims: Optional[SeqLimits]
        if isinstance(limits, Tuple):
            lims = [as_expr(a) for a in limits.args]
        else:
            lims = limits
        v, start, stop = _parse_limits(free, lims, 'periodical')
        self._periodical: Final[tuple[Expr, ...]] = elements
        _SeqExpr.__init__(self, v, start, stop)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (Tuple(*self._periodical), self.limits)

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        per, limits = args
        if not isinstance(per, Tuple) or not isinstance(limits, Tuple):
            raise TypeError("invalid arguments of a SeqPer: %s" % (args,))
        return SeqPer(per, limits)

    @property
    def period(self) -> int:
        return len(self._periodical)

    @property
    def periodical(self) -> tuple[Expr, ...]:
        return self._periodical

    def _eval_coeff(self, pt: Expr) -> Expr:
        if isinstance(self._start, NegativeInfinity):
            idx = self._stop - pt
        else:
            idx = pt - self._start
        if not isinstance(idx, Integer):
            raise ValueError("a periodic sequence takes integer points, not %s" % (pt,))
        return self._periodical[int(idx) % self.period].subs(self._variable, pt)

    def _combine(self, other: SeqPer, product: bool) -> SeqPer:
        n = lcm(self.period, other.period)
        new: list[Expr] = []
        for i in range(n):
            a, b = self._periodical[i % self.period], other.periodical[i % other.period]
            new.append(mul(a, b) if product else add(a, b))
        start, stop = self._intersect_interval(other)
        return SeqPer(new, (self._variable, start, stop))

    def _add(self, other: SeqBase) -> Optional[SeqBase]:
        return self._combine(other, False) if isinstance(other, SeqPer) else None

    def _mul(self, other: SeqBase) -> Optional[SeqBase]:
        return self._combine(other, True) if isinstance(other, SeqPer) else None

    def coeff_mul(self, c: ExprLike) -> SeqPer:
        ce = to_expr(c)
        return SeqPer([e * ce for e in self._periodical], self.limits)


class _SeqExprOp(SeqBase):
    """A termwise operation on sequences (SymPy's ``SeqExprOp``)."""

    __slots__ = ('_sequences',)

    def __init__(self, *sequences: SeqBase) -> None:
        self._sequences: Final[tuple[SeqBase, ...]] = sequences
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return self._sequences

    @property
    def sequences(self) -> tuple[SeqBase, ...]:
        return self._sequences

    @property
    def start(self) -> Expr:
        result = self._sequences[0].start
        for s in self._sequences[1:]:
            result = _max(result, s.start)
        return result

    @property
    def stop(self) -> Expr:
        result = self._sequences[0].stop
        for s in self._sequences[1:]:
            result = _min(result, s.stop)
        return result

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return tuple(v for s in self._sequences for v in s.variables)

    @staticmethod
    def _checked(args: tuple[Basic, ...]) -> list[SeqBase]:
        result: list[SeqBase] = []
        for a in args:
            if not isinstance(a, SeqBase):
                raise TypeError("Input must be Sequences")
            result.append(a)
        return result


# PORTED FROM SYMPY: sympy/series/sequences.py:SeqAdd
class SeqAdd(_SeqExprOp):
    """The termwise sum of sequences, unevaluated (:func:`seq_add` evaluates)."""

    __slots__ = ()

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        return SeqAdd(*self._checked(args))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return seq_add(*self._checked(args))

    def _eval_coeff(self, pt: Expr) -> Expr:
        return add(*[s.coeff(pt) for s in self._sequences])

    def coeff_mul(self, c: ExprLike) -> SeqBase:
        return seq_add(*[s.coeff_mul(c) for s in self._sequences])


# PORTED FROM SYMPY: sympy/series/sequences.py:SeqMul
class SeqMul(_SeqExprOp):
    """The termwise product of sequences, unevaluated (:func:`seq_mul` evaluates)."""

    __slots__ = ()

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        return SeqMul(*self._checked(args))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return seq_mul(*self._checked(args))

    def _eval_coeff(self, pt: Expr) -> Expr:
        return mul(*[s.coeff(pt) for s in self._sequences])

    def coeff_mul(self, c: ExprLike) -> SeqBase:
        first = self._sequences[0].coeff_mul(c)
        return seq_mul(first, *self._sequences[1:])


def _flatten(sequences: Sequence[SeqBase], cls: type[_SeqExprOp]) -> list[SeqBase]:
    result: list[SeqBase] = []
    for s in sequences:
        if isinstance(s, cls):
            result.extend(_flatten(s.sequences, cls))
        else:
            result.append(s)
    return result


def _reduce(args: list[SeqBase], product: bool) -> list[SeqBase]:
    # adapted from Union.reduce: combine the pairs which know how to
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(args):
            for j, t in enumerate(args):
                if i == j:
                    continue
                combined = s._mul(t) if product else s._add(t)
                if combined is not None:
                    args = [a for k, a in enumerate(args) if k not in (i, j)] + [combined]
                    changed = True
                    break
            if changed:
                break
    return args


def _start_key(s: SeqBase) -> tuple[int, str]:
    return (0, str(s.start))


def _empty(args: list[SeqBase]) -> bool:
    start = args[0].start
    stop = args[0].stop
    for a in args[1:]:
        start, stop = _max(start, a.start), _min(stop, a.stop)
    return _empty_range(start, stop)


# PORTED FROM SYMPY: sympy/series/sequences.py:SeqAdd.__new__
def seq_add(*sequences: SeqBase) -> SeqBase:
    """The termwise sum of the sequences on the intersection of their
    ranges: two formulas give one formula (SymPy's ``SeqAdd``).

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.series.sequences import SeqFormula, SeqPer, seq_add
    >>> n = Symbol('n')
    >>> seq_add(SeqFormula(n**2, (n, 0, oo)), SeqFormula(n**3, (n, 0, oo)))
    SeqFormula(n**3 + n**2, (n, 0, oo))
    >>> seq_add(SeqFormula(n**2, (n, 0, 5)), SeqFormula(n**2, (n, 6, 10)))
    EmptySequence
    """
    args = [a for a in _flatten(sequences, SeqAdd) if not isinstance(a, EmptySequence)]
    if not args or _empty(args):
        return EmptySequence()
    reduced = _reduce(args, False)
    if len(reduced) == 1:
        return reduced[0]
    return SeqAdd(*sorted(reduced, key=_start_key))


# PORTED FROM SYMPY: sympy/series/sequences.py:SeqMul.__new__
def seq_mul(*sequences: SeqBase) -> SeqBase:
    """The termwise product of the sequences on the intersection of their
    ranges: two formulas give one formula (SymPy's ``SeqMul``).

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.series.sequences import SeqFormula, SeqPer, seq_mul
    >>> n = Symbol('n')
    >>> seq_mul(SeqFormula(n**2, (n, 0, oo)), SeqFormula(n**3, (n, 0, oo)))
    SeqFormula(n**5, (n, 0, oo))
    >>> seq_mul(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))[:4]
    [0, 2, 4, 18]
    """
    args = _flatten(sequences, SeqMul)
    if not args or any(isinstance(a, EmptySequence) for a in args) or _empty(args):
        return EmptySequence()
    reduced = _reduce(args, True)
    if len(reduced) == 1:
        return reduced[0]
    return SeqMul(*sorted(reduced, key=_start_key))


# PORTED FROM SYMPY: sympy/series/sequences.py:sequence
def sequence(seq: Union[ExprLike, Sequence[ExprLike], Tuple], limits: Optional[SeqLimits] = None) -> SeqBase:
    """The sequence of a formula (:class:`SeqFormula`) or, for a tuple,
    the periodic sequence (:class:`SeqPer`); :class:`EmptySequence` for an
    empty range.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.series.sequences import sequence
    >>> n = Symbol('n')
    >>> sequence(n**2, (n, 0, 5)), sequence((1, 2, 3), (n, 0, 5)), sequence(n, (n, 3, 1))
    (SeqFormula(n**2, (n, 0, 5)), SeqPer((1, 2, 3), (n, 0, 5)), EmptySequence)
    """
    result: _SeqExpr
    if isinstance(seq, (Expr, int, float, Fraction)):
        result = SeqFormula(seq, limits)
    else:
        result = SeqPer(seq, limits)
    if _empty_range(result.start, result.stop):
        return EmptySequence()
    return result
