"""Sets of numbers on the typed core: the empty and the universal set,
intervals, the real line, finite sets, unions, intersections,
complements and the unevaluated membership ``Contains``.

Ported from SymPy's ``sympy.sets.sets`` with the handlers of
``sympy.sets.handlers`` (union, intersection, subset) for these classes.
The classes are immutable and unevaluating; the functions ``interval``,
``finite_set``, ``union``, ``intersection``, ``complement`` and
``contains`` (and the methods of :class:`Set`) evaluate as SymPy's
constructors do.

Examples
========

>>> from sympy_extras.core import Symbol, oo
>>> from sympy_extras.sets import Reals, finite_set, interval, union
>>> x = Symbol('x')
>>> union(interval(0, 1), interval(2, 3)), union(interval(0, 2), interval(1, 3))
(Union(Interval(0, 1), Interval(2, 3)), Interval(0, 3))
>>> Reals - finite_set(1), interval(0, 5) - interval(1, 2)
(Union(Interval.open(-oo, 1), Interval.open(1, oo)), Union(Interval.Ropen(0, 1), Interval.Lopen(2, 5)))
>>> interval(0, 1).contains(x), finite_set(x, 2, 1), interval(0, 1, True) & interval(0.5, 3)
((0 <= x) & (x <= 1), {1, 2, x}, Interval(0.500000000000000, 1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/sets/sets.py, sympy/sets/contains.py,
# sympy/sets/fancysets.py (Reals), sympy/sets/handlers/union.py,
# sympy/sets/handlers/intersection.py, sympy/sets/handlers/issubset.py
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the dispatchers are functions over the classes ported
# (ProductSet, ImageSet, the number sets other than Reals, ConditionSet
# are not ported); SymPy's Python sets of arguments are lists without
# duplicates (deterministic order); Reals is an Interval subclass equal
# to Interval(-oo, oo), as in SymPy.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Iterator, Optional, Sequence, Union as TUnion

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.fuzzy import fuzzy_and, fuzzy_not, fuzzy_or
from sympy_extras.core.logic import Boolean, BooleanLike, and_, false, not_, or_, to_boolean, true
from sympy_extras.core.numbers import ComplexInfinity, Float, Infinity, NaN, NegativeInfinity, oo
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.sorting import ClassKey, default_sort_key, ordered

__all__ = ['Set', 'EmptySetClass', 'UniversalSetClass', 'Interval', 'RealsClass', 'FiniteSet', 'Union',
           'Intersection', 'Complement', 'Contains', 'EmptySet', 'UniversalSet', 'Reals', 'interval', 'finite_set',
           'union', 'intersection', 'complement', 'contains']

Truth = Optional[bool]


def _fuzzy_bool(b: Boolean) -> Truth:
    """SymPy's ``fuzzy_bool`` of a condition."""
    if b is true:
        return True
    if b is false:
        return False
    return None


# PORTED FROM SYMPY: sympy/sets/sets.py:Set
class Set(Basic):
    """The base class of the sets."""

    __slots__ = ()

    is_Interval = False
    is_FiniteSet = False
    is_Union = False
    is_Intersection = False
    is_Complement = False
    is_EmptySet = False
    is_UniversalSet = False

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, cls.__name__)

    # the operations

    def union(self, other: Set) -> Set:
        return union(self, other)

    def intersect(self, other: Set) -> Set:
        return intersection(self, other)

    def complement(self, universe: Set) -> Set:
        """``universe - self``."""
        return complement(universe, self)

    def __or__(self, other: Set) -> Set:
        return union(self, other)

    def __and__(self, other: Set) -> Set:
        return intersection(self, other)

    def __sub__(self, other: Set) -> Set:
        return complement(self, other)

    # PORTED FROM SYMPY: sympy/sets/sets.py:Set.contains
    def contains(self, other: ExprLike) -> Boolean:
        """Whether ``other`` is an element: ``true``, ``false``, a condition
        or an unevaluated :class:`Contains`."""
        element = to_expr(other)
        c = self._contains(element)
        if c is None:
            return Contains(element, self)
        return c

    def _contains(self, other: Expr) -> Optional[Boolean]:
        raise NotImplementedError('%s._contains' % type(self).__name__)

    # PORTED FROM SYMPY: sympy/sets/sets.py:Set._complement
    def _complement(self, other: Set) -> Optional[Set]:
        """``other - self``, ``None`` when there is no rule."""
        if isinstance(other, Interval):
            if isinstance(self, (Interval, FiniteSet)):
                return intersection(other, self.complement(Reals))
            return None
        if isinstance(other, Union):
            return union(*[complement(o, self) for o in other.sets])
        if isinstance(other, Complement):
            return Complement(other.args_sets[0], union(other.args_sets[1], self))
        if other is EmptySet:
            return EmptySet
        if isinstance(other, FiniteSet):
            false_part: list[Expr] = []
            unknown: list[Expr] = []
            for x in other.elements:
                c = _fuzzy_bool(self.contains(x))
                if c is False:
                    false_part.append(x)
                elif c is None:
                    unknown.append(x)
            return union(finite_set(*false_part), Complement(finite_set(*unknown), self) if unknown else EmptySet)
        return None

    # PORTED FROM SYMPY: sympy/sets/sets.py:Set.is_subset
    def is_subset(self, other: Set) -> Truth:
        """Whether ``self`` is a subset of ``other`` (``None`` when it is
        not decided)."""
        if self == other:
            return True
        is_empty = self.is_empty
        if is_empty is True:
            return True
        if fuzzy_not(is_empty) and other.is_empty:
            return False
        if self.is_finite_set is False and other.is_finite_set:
            return False
        ret = self._eval_is_subset(other)
        if ret is not None:
            return ret
        ret = _is_subset_sets(self, other)
        if ret is not None:
            return ret
        if intersection(self, other) == self:
            return True
        return None

    def is_superset(self, other: Set) -> Truth:
        return other.is_subset(self)

    def is_proper_subset(self, other: Set) -> Truth:
        if self == other:
            return False
        return self.is_subset(other)

    def _eval_is_subset(self, other: Set) -> Truth:
        return None

    @property
    def is_empty(self) -> Truth:
        return None

    @property
    def is_finite_set(self) -> Truth:
        return None

    @property
    def inf(self) -> Expr:
        """The infimum."""
        return self._inf()

    @property
    def sup(self) -> Expr:
        """The supremum."""
        return self._sup()

    def _inf(self) -> Expr:
        raise NotImplementedError('(%s)._inf' % self)

    def _sup(self) -> Expr:
        raise NotImplementedError('(%s)._sup' % self)

    def as_relational(self, x: Expr) -> Boolean:
        """The condition that ``x`` is an element."""
        raise NotImplementedError('%s.as_relational' % type(self).__name__)

    @property
    def sets(self) -> tuple[Set, ...]:
        return tuple(a for a in self.args if isinstance(a, Set))


def _infimum_key(s: Basic) -> tuple[float, object]:
    """SymPy's ``Set._infimum_key`` (the infimum evaluated, ``oo`` when it
    is not a comparable number), with the canonical key breaking ties."""
    value = float('inf')
    if isinstance(s, Set):
        try:
            infimum = s.inf
        except NotImplementedError:
            infimum = oo
        if infimum.is_comparable:
            value = float(infimum.evalf())
    return (value, default_sort_key(s))


# PORTED FROM SYMPY: sympy/sets/sets.py:EmptySet
class EmptySetClass(Set):
    """The empty set (the singleton :data:`EmptySet`)."""

    __slots__ = ()

    is_EmptySet = True
    is_FiniteSet = True

    @property
    def args(self) -> tuple[Basic, ...]:
        return ()

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'EmptySet'

    @property
    def is_empty(self) -> Truth:
        return True

    @property
    def is_finite_set(self) -> Truth:
        return True

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return false

    def _complement(self, other: Set) -> Optional[Set]:
        return other

    def as_relational(self, x: Expr) -> Boolean:
        return false

    def __iter__(self) -> Iterator[Expr]:
        return iter(())

    def __len__(self) -> int:
        return 0


# PORTED FROM SYMPY: sympy/sets/sets.py:UniversalSet
class UniversalSetClass(Set):
    """The universal set (the singleton :data:`UniversalSet`)."""

    __slots__ = ()

    is_UniversalSet = True

    @property
    def args(self) -> tuple[Basic, ...]:
        return ()

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'UniversalSet'

    @property
    def is_empty(self) -> Truth:
        return False

    @property
    def is_finite_set(self) -> Truth:
        return False

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return true

    def _complement(self, other: Set) -> Optional[Set]:
        return EmptySet

    def as_relational(self, x: Expr) -> Boolean:
        return true


EmptySet: Final[EmptySetClass] = EmptySetClass()
UniversalSet: Final[UniversalSetClass] = UniversalSetClass()


def _bool_arg(b: BooleanLike) -> Boolean:
    r = to_boolean(b)
    if r is not true and r is not false:
        raise NotImplementedError('left_open and right_open can have only true/false values, got %s' % (b,))
    return r


# PORTED FROM SYMPY: sympy/sets/sets.py:Interval
class Interval(Set):
    """The interval between ``start`` and ``end``, each end open or closed
    (``interval`` evaluates: empty and degenerate intervals, infinite ends
    open)."""

    __slots__ = ('_start', '_end', '_left_open', '_right_open')

    is_Interval = True

    def __init__(self, start: ExprLike, end: ExprLike, left_open: BooleanLike = False,
                 right_open: BooleanLike = False) -> None:
        self._start: Final[Expr] = to_expr(start)
        self._end: Final[Expr] = to_expr(end)
        self._left_open: Final[Boolean] = _bool_arg(left_open)
        self._right_open: Final[Boolean] = _bool_arg(right_open)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._start, self._end, self._left_open, self._right_open)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        start, end, lo, ro = args
        if not isinstance(start, Expr) or not isinstance(end, Expr) or not isinstance(lo, Boolean) \
                or not isinstance(ro, Boolean):
            raise TypeError('Interval takes two expressions and two truth values')
        return interval(start, end, lo is true, ro is true)

    @property
    def start(self) -> Expr:
        return self._start

    @property
    def end(self) -> Expr:
        return self._end

    @property
    def left(self) -> Expr:
        return self._start

    @property
    def right(self) -> Expr:
        return self._end

    @property
    def left_open(self) -> bool:
        return self._left_open is true

    @property
    def right_open(self) -> bool:
        return self._right_open is true

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RealsClass) and not isinstance(self, RealsClass):
            return self._is_whole_line()
        return Basic.__eq__(self, other)

    def __hash__(self) -> int:
        return Basic.__hash__(self)

    def _is_whole_line(self) -> bool:
        return isinstance(self._start, NegativeInfinity) and isinstance(self._end, Infinity) \
            and self.left_open and self.right_open

    # PORTED FROM SYMPY: sympy/printing/str.py:StrPrinter._print_Interval
    def _sympystr(self, printer: StrPrinter) -> str:
        a, b, lo, ro = self._start, self._end, self.left_open, self.right_open
        if a.is_infinite and b.is_infinite:
            m = ''
        elif a.is_infinite and not ro:
            m = ''
        elif b.is_infinite and not lo:
            m = ''
        elif not lo and not ro:
            m = ''
        elif lo and ro:
            m = '.open'
        elif lo:
            m = '.Lopen'
        else:
            m = '.Ropen'
        # SymPy formats the ends with str(): a printing of their own (a
        # Float in full)
        from sympy_extras.core.printing import sstr
        return 'Interval%s(%s, %s)' % (m, sstr(a), sstr(b))

    def _inf(self) -> Expr:
        return self._start

    def _sup(self) -> Expr:
        return self._end

    @property
    def measure(self) -> Expr:
        return self._end - self._start

    @property
    def is_empty(self) -> Truth:
        if self.left_open or self.right_open:
            return _fuzzy_bool(self._start >= self._end)
        return _fuzzy_bool(self._start > self._end)

    @property
    def is_finite_set(self) -> Truth:
        return self.measure.is_zero

    def _is_comparable(self, other: Interval) -> bool:
        return all(e.is_comparable for e in (self._start, self._end, other._start, other._end))

    # PORTED FROM SYMPY: sympy/sets/sets.py:Interval.as_relational
    def as_relational(self, x: Expr) -> Boolean:
        right = x < self._end if self.right_open else x <= self._end
        left = self._start < x if self.left_open else self._start <= x
        return and_(left, right)

    # PORTED FROM SYMPY: sympy/sets/sets.py:Interval._contains
    def _contains(self, other: Expr) -> Optional[Boolean]:
        if isinstance(other, NaN) or other.is_real is False or other.has(ComplexInfinity):
            return false
        if isinstance(self._start, NegativeInfinity) and isinstance(self._end, Infinity):
            if other.is_real is not None:
                return true if other.is_real else false
        return self.as_relational(other)

    # PORTED FROM SYMPY: sympy/sets/sets.py:Interval._complement
    def _complement(self, other: Set) -> Optional[Set]:
        if other == Reals:
            a = interval(NegativeInfinity(), self._start, True, not self.left_open)
            b = interval(self._end, oo, not self.right_open, True)
            return union(a, b)
        if isinstance(other, FiniteSet):
            if not [m for m in other.elements if m.is_number]:
                return None
        return Set._complement(self, other)


# PORTED FROM SYMPY: sympy/sets/fancysets.py:Reals
class RealsClass(Interval):
    """The real line (the singleton :data:`Reals`), equal to
    ``Interval(-oo, oo)``."""

    __slots__ = ()

    def __init__(self) -> None:
        Interval.__init__(self, NegativeInfinity(), oo, True, True)

    @property
    def args(self) -> tuple[Basic, ...]:
        return ()

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Reals'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Interval):
            return other._is_whole_line()
        return False

    def __hash__(self) -> int:
        return hash(Interval(NegativeInfinity(), oo, True, True))


Reals: Final[RealsClass] = RealsClass()


# PORTED FROM SYMPY: sympy/sets/sets.py:Interval.__new__
def interval(start: ExprLike, end: ExprLike, left_open: BooleanLike = False, right_open: BooleanLike = False) -> Set:
    """The interval, evaluated: empty when ``end < start``, ``{start}``
    when degenerate and closed, infinite ends open."""
    a, b = to_expr(start), to_expr(end)
    lo, ro = _bool_arg(left_open) is true, _bool_arg(right_open) is true
    if fuzzy_not(fuzzy_and([a.is_extended_real, b.is_extended_real, (b - a).is_extended_real])):
        raise ValueError('Non-real intervals are not supported')
    if _fuzzy_bool(b < a) is True:
        return EmptySet
    if (b - a).is_negative is True:
        return EmptySet
    if b == a and (lo or ro):
        return EmptySet
    if b == a and not (lo or ro):
        if isinstance(a, (Infinity, NegativeInfinity)):
            return EmptySet
        return finite_set(b)
    if isinstance(a, NegativeInfinity):
        lo = True
    if isinstance(b, Infinity):
        ro = True
    if isinstance(a, Infinity) or isinstance(b, NegativeInfinity):
        return EmptySet
    return Interval(a, b, lo, ro)


# PORTED FROM SYMPY: sympy/sets/sets.py:FiniteSet
class FiniteSet(Set):
    """A finite set of elements, in SymPy's order (``finite_set``
    evaluates: no duplicates, empty is :data:`EmptySet`)."""

    __slots__ = ('_elements',)

    is_FiniteSet = True

    def __init__(self, *elements: ExprLike) -> None:
        self._elements: Final[tuple[Expr, ...]] = tuple(to_expr(e) for e in elements)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return self._elements

    @property
    def elements(self) -> tuple[Expr, ...]:
        return self._elements

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        if not all(isinstance(a, Expr) for a in args):
            raise TypeError('FiniteSet takes expressions')
        return finite_set(*[a for a in args if isinstance(a, Expr)])

    # PORTED FROM SYMPY: sympy/printing/str.py:StrPrinter._print_FiniteSet
    def _sympystr(self, printer: StrPrinter) -> str:
        items = sorted(self._elements, key=default_sort_key)
        return '{' + ', '.join(printer.doprint(e) for e in items) + '}'

    def __iter__(self) -> Iterator[Expr]:
        return iter(self._elements)

    def __len__(self) -> int:
        return len(self._elements)

    @property
    def is_empty(self) -> Truth:
        return len(self._elements) == 0

    @property
    def is_finite_set(self) -> Truth:
        return True

    @property
    def measure(self) -> Expr:
        from sympy_extras.core.numbers import Integer
        return Integer(0)

    def _inf(self) -> Expr:
        from sympy_extras.functions.elementary.miscellaneous import min_
        return min_(*self._elements)

    def _sup(self) -> Expr:
        from sympy_extras.functions.elementary.miscellaneous import max_
        return max_(*self._elements)

    # PORTED FROM SYMPY: sympy/sets/sets.py:FiniteSet._contains
    def _contains(self, other: Expr) -> Optional[Boolean]:
        from sympy_extras.core.relational import eq
        if other in self._elements:
            return true
        return or_(*[eq(e, other) for e in self._elements])

    def _eval_is_subset(self, other: Set) -> Truth:
        results: list[Truth] = []
        for e in self._elements:
            c = other._contains(e)
            results.append(None if c is None else _fuzzy_bool(c))
        return fuzzy_and(results)

    # PORTED FROM SYMPY: sympy/sets/sets.py:FiniteSet.as_relational
    def as_relational(self, x: Expr) -> Boolean:
        from sympy_extras.core.relational import Eq
        return or_(*[Eq(x, e) for e in self._elements])

    # PORTED FROM SYMPY: sympy/sets/sets.py:FiniteSet._complement
    def _complement(self, other: Set) -> Optional[Set]:
        if isinstance(other, Interval):
            nums: list[Expr] = []
            syms: list[Expr] = []
            for m in self._elements:
                if m.is_number and m.is_real is True:
                    nums.append(m)
                elif m.is_real is False:
                    pass
                else:
                    syms.append(m)
            if other == Reals and nums:
                nums.sort(key=_NumberKey)
                intervals: list[Set] = [interval(NegativeInfinity(), nums[0], True, True)]
                for a, b in zip(nums[:-1], nums[1:]):
                    intervals.append(interval(a, b, True, True))
                intervals.append(interval(nums[-1], oo, True, True))
                if syms:
                    return Complement(Union(*intervals), finite_set(*syms))
                return Union(*intervals)
            if not nums:
                if syms:
                    return Complement(other, finite_set(*syms))
                return other
        elif isinstance(other, FiniteSet):
            unknown = [i for i in self._elements if other.contains(i) not in (true, false)]
            unk = finite_set(*unknown)
            if unk == self:
                return None
            not_true = [i for i in other.elements if self.contains(i) is not true]
            return complement(finite_set(*not_true), unk)
        return Set._complement(self, other)


class _NumberKey:
    """The order of real numbers (SymPy's ``nums.sort()``)."""

    __slots__ = ('e',)

    def __init__(self, e: Expr) -> None:
        self.e = e

    def __lt__(self, other: _NumberKey) -> bool:
        r = self.e < other.e
        if r is true:
            return True
        if r is false:
            return False
        raise TypeError('cannot determine truth value of %s' % (r,))


def _dedupe(items: Sequence[Basic]) -> list[Basic]:
    out: list[Basic] = []
    for i in items:
        if i not in out:
            out.append(i)
    return out


# PORTED FROM SYMPY: sympy/sets/sets.py:FiniteSet.__new__
def finite_set(*elements: ExprLike) -> Set:
    """The finite set, evaluated: duplicates removed, empty is
    :data:`EmptySet`."""
    items = [to_expr(e) for e in elements]
    if not items:
        return EmptySet
    unique = _dedupe(items)
    ordered_items = sorted(unique, key=_infimum_key)
    return FiniteSet(*[e for e in ordered_items if isinstance(e, Expr)])


def _sorted_sets(sets: Sequence[Set]) -> list[Set]:
    """SymPy's ``ordered(args, Set._infimum_key)``."""
    return sorted(sets, key=_infimum_key)


# PORTED FROM SYMPY: sympy/sets/sets.py:Union
class Union(Set):
    """The union of sets, unevaluated (``union`` evaluates); the sets in
    SymPy's order (by infimum)."""

    __slots__ = ('_sets',)

    is_Union = True

    def __init__(self, *sets: Set) -> None:
        self._sets: Final[tuple[Set, ...]] = tuple(_sorted_sets(_dedupe_sets(sets)))
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return self._sets

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return union(*[a for a in args if isinstance(a, Set)])

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Union(%s)' % ', '.join(printer.doprint(s) for s in self._sets)

    @property
    def is_empty(self) -> Truth:
        return fuzzy_and(s.is_empty for s in self._sets)

    @property
    def is_finite_set(self) -> Truth:
        return fuzzy_and(s.is_finite_set for s in self._sets)

    def _inf(self) -> Expr:
        from sympy_extras.functions.elementary.miscellaneous import min_
        return min_(*[s.inf for s in self._sets])

    def _sup(self) -> Expr:
        from sympy_extras.functions.elementary.miscellaneous import max_
        return max_(*[s.sup for s in self._sets])

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return or_(*[s.contains(other) for s in self._sets])

    # PORTED FROM SYMPY: sympy/sets/sets.py:Union.is_subset
    def is_subset(self, other: Set) -> Truth:
        return fuzzy_and(s.is_subset(other) for s in self._sets)

    def _complement(self, universe: Set) -> Optional[Set]:
        return intersection(*[s.complement(universe) for s in self._sets])

    # PORTED FROM SYMPY: sympy/sets/sets.py:Union.as_relational
    def as_relational(self, x: Expr) -> Boolean:
        from sympy_extras.core.relational import Ne
        if len(self._sets) == 2 and all(isinstance(s, Interval) for s in self._sets):
            a, b = self._sets
            assert isinstance(a, Interval) and isinstance(b, Interval)
            if a.sup == b.inf and a.right_open and b.left_open:
                mincond = x > a.inf if a.left_open else x >= a.inf
                maxcond = x < b.sup if b.right_open else x <= b.sup
                return and_(Ne(x, a.sup), mincond, maxcond)
        return or_(*[s.as_relational(x) for s in self._sets])


def _dedupe_sets(sets: Sequence[Set]) -> list[Set]:
    out: list[Set] = []
    for s in sets:
        if s not in out:
            out.append(s)
    return out


# PORTED FROM SYMPY: sympy/sets/sets.py:Intersection
class Intersection(Set):
    """The intersection of sets, unevaluated (``intersection``
    evaluates)."""

    __slots__ = ('_sets',)

    is_Intersection = True

    def __init__(self, *sets: Set) -> None:
        self._sets: Final[tuple[Set, ...]] = tuple(_sorted_sets(_dedupe_sets(sets)))
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return self._sets

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return intersection(*[a for a in args if isinstance(a, Set)])

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Intersection(%s)' % ', '.join(printer.doprint(s) for s in self._sets)

    @property
    def is_finite_set(self) -> Truth:
        if fuzzy_or(s.is_finite_set for s in self._sets):
            return True
        return None

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return and_(*[s.contains(other) for s in self._sets])

    def as_relational(self, x: Expr) -> Boolean:
        return and_(*[s.as_relational(x) for s in self._sets])


# PORTED FROM SYMPY: sympy/sets/sets.py:Complement
class Complement(Set):
    """The elements of ``a`` not in ``b``, unevaluated (``complement``
    evaluates)."""

    __slots__ = ('_a', '_b')

    is_Complement = True

    def __init__(self, a: Set, b: Set) -> None:
        self._a: Final[Set] = a
        self._b: Final[Set] = b
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._a, self._b)

    @property
    def args_sets(self) -> tuple[Set, Set]:
        return (self._a, self._b)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        a, b = args
        if not isinstance(a, Set) or not isinstance(b, Set):
            raise TypeError('Complement takes two sets')
        return complement(a, b)

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Complement(%s, %s)' % (printer.doprint(self._a), printer.doprint(self._b))

    @property
    def is_finite_set(self) -> Truth:
        a_finite = self._a.is_finite_set
        if a_finite is True:
            return True
        if a_finite is False and self._b.is_finite_set:
            return False
        return None

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return and_(self._a.contains(other), not_(self._b.contains(other)))

    def as_relational(self, x: Expr) -> Boolean:
        return and_(self._a.as_relational(x), not_(self._b.as_relational(x)))


# PORTED FROM SYMPY: sympy/sets/contains.py:Contains
class Contains(Boolean):
    """The unevaluated membership ``x`` in ``s``."""

    __slots__ = ('_x', '_s')

    def __init__(self, x: ExprLike, s: Set) -> None:
        self._x: Final[Expr] = to_expr(x)
        self._s: Final[Set] = s
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._x, self._s)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Contains')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        x, s = args
        if not isinstance(x, Expr) or not isinstance(s, Set):
            raise TypeError('Contains takes an expression and a set')
        return contains(x, s)

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Contains(%s, %s)' % (printer.doprint(self._x), printer.doprint(self._s))


def contains(x: ExprLike, s: Set) -> Boolean:
    """Whether ``x`` is an element of ``s``, evaluated."""
    return s.contains(x)


# ---------------------------------------------------------------------
# the handlers (SymPy's dispatchers, for the classes ported)


def _contains_true(s: Set, x: Expr) -> bool:
    return s.contains(x) is true


# PORTED FROM SYMPY: sympy/sets/handlers/union.py:union_sets
def _union_sets(a: Set, b: Set) -> TUnion[None, Set, list[Set]]:
    """The union of two sets when a rule applies (a list of two sets for
    the rules which only change ``a``), ``None`` otherwise."""
    from sympy_extras.functions.elementary.miscellaneous import max_, min_
    if isinstance(a, EmptySetClass):
        return b
    if isinstance(a, UniversalSetClass):
        return a
    if isinstance(a, Interval) and isinstance(b, Interval):
        if not a._is_comparable(b):
            return None
        end = min_(a.end, b.end)
        start = max_(a.start, b.start)
        if _fuzzy_bool(end < start) is True or (end == start and not _contains_true(a, end)
                                                 and not _contains_true(b, end)):
            return None
        start = min_(a.start, b.start)
        end = max_(a.end, b.end)
        left_open = (a.start != start or a.left_open) and (b.start != start or b.left_open)
        right_open = (a.end != end or a.right_open) and (b.end != end or b.right_open)
        return interval(start, end, left_open, right_open)
    if isinstance(a, Interval) and isinstance(b, UniversalSetClass):
        return UniversalSet
    if isinstance(a, Interval):
        open_left = a.left_open and b.contains(a.start) is true and a.start.is_finite is True
        open_right = a.right_open and b.contains(a.end) is true and a.end.is_finite is True
        if open_left or open_right:
            new_a = interval(a.start, a.end, a.left_open and not _contains_true(b, a.start),
                             a.right_open and not _contains_true(b, a.end))
            return [new_a, b]
        return None
    if isinstance(a, FiniteSet) and isinstance(b, FiniteSet):
        return finite_set(*(list(a.elements) + [e for e in b.elements if e not in a.elements]))
    if isinstance(a, FiniteSet):
        if any(b.contains(x) is true for x in a.elements):
            return [finite_set(*[x for x in a.elements if b.contains(x) is not true]), b]
        return None
    return None


# PORTED FROM SYMPY: sympy/sets/sets.py:simplify_union
def union(*sets: Set) -> Set:
    """The union, evaluated (SymPy's ``Union``)."""
    args: list[Set] = []
    for s in sets:
        if not isinstance(s, Set):
            raise TypeError('Input args to Union must be Sets')
        # SymPy's _new_args_filter: nested unions flattened
        args.extend(s.sets if isinstance(s, Union) else [s])
    if not args:
        return EmptySet
    finite = [x for x in args if isinstance(x, FiniteSet)]
    if len(finite) > 1:
        elements = [e for f in finite for e in f.elements]
        args = [finite_set(*elements)] + [x for x in args if not isinstance(x, FiniteSet)]
    work = _dedupe_sets(_sorted_sets(args))
    changed = True
    while changed:
        changed = False
        for s in work:
            for t in work:
                if t is s:
                    continue
                new = _union_sets(s, t)
                if new is None:
                    continue
                rest = [w for w in work if w is not s and w is not t]
                work = _dedupe_sets(rest + (new if isinstance(new, list) else [new]))
                changed = True
                break
            if changed:
                break
    if len(work) == 1:
        return work[0]
    return Union(*work)


# PORTED FROM SYMPY: sympy/sets/handlers/intersection.py:intersection_sets
def _intersection_sets(a: Set, b: Set) -> Optional[Set]:
    """The intersection of two sets when a rule applies."""
    if isinstance(a, Interval) and isinstance(b, Interval):
        if a == Interval(NegativeInfinity(), oo, True, True):
            l_, r_ = a.left, a.right
            if l_.is_real or isinstance(l_, (NegativeInfinity, Infinity)) or r_.is_real \
                    or isinstance(r_, (NegativeInfinity, Infinity)):
                return b
        if not a._is_comparable(b):
            return None
        if _fuzzy_bool(a.start <= b.end) is True and _fuzzy_bool(b.start <= a.end) is True:
            if _fuzzy_bool(a.start < b.start) is True:
                start, left_open = b.start, b.left_open
            elif _fuzzy_bool(a.start > b.start) is True:
                start, left_open = a.start, a.left_open
            else:
                start = a.start
                if a.start != b.start:   # SymPy's not _aresame
                    if b.start.has(Float) and not a.start.has(Float):
                        start = b.start
                    elif a.start.has(Float) and not b.start.has(Float):
                        start = a.start
                    else:
                        first = ordered([a, b])[0]
                        assert isinstance(first, Interval)
                        start = first.start
                left_open = a.left_open or b.left_open
            if _fuzzy_bool(a.end < b.end) is True:
                end, right_open = a.end, a.right_open
            elif _fuzzy_bool(a.end > b.end) is True:
                end, right_open = b.end, b.right_open
            else:
                end = a.end
                if a.end != b.end:
                    if b.end.has(Float) and not a.end.has(Float):
                        end = b.end
                    elif a.end.has(Float) and not b.end.has(Float):
                        end = a.end
                    else:
                        first = ordered([a, b])[0]
                        assert isinstance(first, Interval)
                        end = first.end
                right_open = a.right_open or b.right_open
            if (end - start).is_zero is True and (left_open or right_open):
                return EmptySet
            return interval(start, end, left_open, right_open)
        return EmptySet
    if isinstance(a, EmptySetClass):
        return EmptySet
    if isinstance(a, UniversalSetClass):
        return b
    if isinstance(a, FiniteSet) and isinstance(b, FiniteSet):
        return finite_set(*[e for e in a.elements if e in b.elements])
    if isinstance(a, FiniteSet):
        kept: list[Expr] = []
        for el in a.elements:
            c = b.contains(el)
            if c is true:
                kept.append(el)
            elif c is not false:
                return None   # SymPy's TypeError on `el in b`
        return finite_set(*kept)
    return None


def _handle_finite_sets(args: list[Set]) -> Optional[Set]:
    """SymPy's ``Intersection._handle_finite_sets``: the elements of the
    finite sets in all the sets, with the undecided ones kept."""
    fs_args = [s for s in args if isinstance(s, FiniteSet)]
    others = [s for s in args if not isinstance(s, FiniteSet)]
    if not fs_args:
        return None
    fs_sets: list[list[Expr]] = [list(fs.elements) for fs in fs_args]
    all_elements: list[Expr] = []
    for fs in fs_sets:
        for e in fs:
            if e not in all_elements:
                all_elements.append(e)
    definite: list[Expr] = []
    for e in all_elements:
        inall = fuzzy_and(_fuzzy_bool(s.contains(e)) for s in args)
        if inall is True and e not in definite:
            definite.append(e)
        if inall is not None:
            for fs in fs_sets:
                if e in fs:
                    fs.remove(e)
    fs_elements: list[Expr] = []
    for fs in fs_sets:
        for e in fs:
            if e not in fs_elements:
                fs_elements.append(e)
    fs_symsets: list[Set] = [finite_set(*fs) for fs in fs_sets]
    while fs_elements:
        for e in fs_elements:
            infs = fuzzy_and(_fuzzy_bool(s.contains(e)) for s in fs_symsets)
            if infs is True and e not in definite:
                definite.append(e)
            if infs is not None:
                for n, fs in enumerate(fs_sets):
                    if e in fs:
                        fs.remove(e)
                        fs_symsets[n] = finite_set(*fs)
                fs_elements.remove(e)
                break
        else:
            break
    if not all(fs_sets):
        fs_sets = [[]]
    if definite:
        fs_sets = [fs + [d for d in definite if d not in fs] for fs in fs_sets]
    if fs_sets == [[]]:
        return EmptySet
    sets: list[Set] = [finite_set(*fs) for fs in fs_sets]
    every: list[Expr] = [e for fs in fs_sets for e in fs]
    others = [o for o in others if not all(_fuzzy_bool(o.contains(e)) is True for e in every)]
    if others:
        rest = intersection(*others)
        if rest is EmptySet:
            return EmptySet
        if isinstance(rest, Intersection):
            sets.extend(rest.sets)
        else:
            sets.append(rest)
    if len(sets) == 1:
        return sets[0]
    return Intersection(*sets)


# PORTED FROM SYMPY: sympy/sets/sets.py:simplify_intersection
def intersection(*sets: Set) -> Set:
    """The intersection, evaluated (SymPy's ``Intersection``)."""
    args: list[Set] = []
    for s in sets:
        if not isinstance(s, Set):
            raise TypeError('Input args to Intersection must be Sets')
        args.extend(s.sets if isinstance(s, Intersection) else [s])
    args = _dedupe_sets(args)
    ordered_args = ordered(args)
    args = [s for s in ordered_args if isinstance(s, Set)]
    if not args:
        return UniversalSet
    if EmptySet in args:
        return EmptySet
    rv = _handle_finite_sets(args)
    if rv is not None:
        return rv
    for s in args:
        if isinstance(s, Union):
            other_sets = [o for o in args if o is not s]
            if other_sets:
                other = intersection(*other_sets)
                return union(*[intersection(arg, other) for arg in s.sets])
            return union(*s.sets)
    for s in args:
        if isinstance(s, Complement):
            rest = [o for o in args if o is not s] + [s.args_sets[0]]
            return complement(intersection(*rest), s.args_sets[1])
    work = list(args)
    changed = True
    while changed:
        changed = False
        for s in work:
            for t in work:
                if t is s:
                    continue
                new = _intersection_sets(s, t)
                if new is None:
                    continue
                work = _dedupe_sets([w for w in work if w is not s and w is not t] + [new])
                changed = True
                break
            if changed:
                break
    if len(work) == 1:
        return work[0]
    return Intersection(*work)


# PORTED FROM SYMPY: sympy/sets/sets.py:Complement.reduce
def complement(a: Set, b: Set) -> Set:
    """``a - b``, evaluated (SymPy's ``Complement``)."""
    if b == UniversalSet or a.is_subset(b) is True:
        return EmptySet
    if isinstance(b, Union):
        return intersection(*[s.complement(a) for s in b.sets])
    result = b._complement(a)
    if result is not None:
        return result
    return Complement(a, b)


# PORTED FROM SYMPY: sympy/sets/handlers/issubset.py:is_subset_sets
def _is_subset_sets(a: Set, b: Set) -> Truth:
    """The subset rules between intervals and unions of intervals."""
    if isinstance(a, Interval) and isinstance(b, Interval):
        if _fuzzy_bool(a.start < b.start) is True:
            return False
        if _fuzzy_bool(a.end > b.end) is True:
            return False
        if b.left_open and not a.left_open and _fuzzy_bool(_eq(a.start, b.start)) is True:
            return False
        if b.right_open and not a.right_open and _fuzzy_bool(_eq(a.end, b.end)) is True:
            return False
        return None
    if isinstance(a, Interval) and isinstance(b, FiniteSet):
        if fuzzy_not(a.measure.is_zero):
            return False
        return None
    if isinstance(a, Interval) and isinstance(b, Union):
        if all(isinstance(s, (Interval, FiniteSet)) for s in b.sets):
            intervals = [s for s in b.sets if isinstance(s, Interval)]
            if all(_fuzzy_bool(a.start < s.start) is True for s in intervals):
                return False
            if all(_fuzzy_bool(a.end > s.end) is True for s in intervals):
                return False
            if a.measure.is_nonzero:
                def no_overlap(s1: Interval, s2: Interval) -> Truth:
                    return fuzzy_or([_fuzzy_bool(s1.end <= s2.start), _fuzzy_bool(s1.start >= s2.end)])
                if all(no_overlap(s, a) for s in intervals):
                    return False
        return None
    return None


def _eq(a: Expr, b: Expr) -> Boolean:
    from sympy_extras.core.relational import eq
    return eq(a, b)

