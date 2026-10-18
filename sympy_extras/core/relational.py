"""Relations: ``Eq``, ``Ne``, ``Lt``, ``Le``, ``Gt``, ``Ge`` and the
evaluating functions ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``.

The classes build unevaluated relations (``Eq(x, x)`` is a relation with
two equal sides); the functions decide what can be decided: two
structurally equal sides, two numbers, or a difference which is a
nonzero number give ``true`` or ``false``, anything else is returned as
the relation. The comparison operators of ``Expr`` call the functions.
SymPy's class names ``Equality``, ``Unequality``, ``StrictLessThan``,
``LessThan``, ``StrictGreaterThan`` and ``GreaterThan`` are aliases.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .add import sub, to_expr
from .basic import Basic
from .expr import Expr, ExprLike
from .logic import Boolean, false, true
from .numbers import ComplexInfinity, Float, Infinity, NaN, NegativeInfinity, Number, Rational
from .sorting import ClassKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Relational', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'Unequality',
    'StrictLessThan', 'LessThan', 'StrictGreaterThan', 'GreaterThan',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge']


class Relational(Boolean):
    """A relation between two expressions.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq, Lt
    >>> x, y = symbols('x y')
    >>> r = Eq(x, y)
    >>> r.lhs, r.rhs, r.rel_op, r.reversed
    (x, y, '==', Eq(y, x))
    >>> Lt(x, 1), Lt(x, 1).args
    (x < 1, (x, 1))
    """

    __slots__ = ('_lhs', '_rhs')

    is_Relational = True
    rel_op: str = ''

    def __init__(self, lhs: ExprLike, rhs: ExprLike) -> None:
        self._lhs: Expr = to_expr(lhs)
        self._rhs: Expr = to_expr(rhs)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Expr, ...]:
        return (self._lhs, self._rhs)

    @property
    def lhs(self) -> Expr:
        return self._lhs

    @property
    def rhs(self) -> Expr:
        return self._rhs

    @property
    def reversed(self) -> Relational:
        """The relation with the sides exchanged."""
        return _REVERSED[type(self)](self._rhs, self._lhs)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, cls.__name__)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        lhs, rhs = args
        if not (isinstance(lhs, Expr) and isinstance(rhs, Expr)):
            raise TypeError("a relation between expressions is expected")
        return _FUNCTIONS[type(self)](lhs, rhs)

    def _sympystr(self, printer: StrPrinter) -> str:
        from .printing import PREC_RELATIONAL
        if self.rel_op in ('==', '!='):
            return '%s(%s, %s)' % (type(self).__name__, printer.doprint(self._lhs), printer.doprint(self._rhs))
        return '%s %s %s' % (printer.parenthesize(self._lhs, PREC_RELATIONAL), self.rel_op,
                             printer.parenthesize(self._rhs, PREC_RELATIONAL))


class Eq(Relational):
    """``lhs == rhs``, unevaluated."""

    __slots__ = ()
    is_Equality = True
    rel_op = '=='


class Ne(Relational):
    """``lhs != rhs``, unevaluated."""

    __slots__ = ()
    rel_op = '!='


class Lt(Relational):
    """``lhs < rhs``, unevaluated."""

    __slots__ = ()
    rel_op = '<'


class Le(Relational):
    """``lhs <= rhs``, unevaluated."""

    __slots__ = ()
    rel_op = '<='


class Gt(Relational):
    """``lhs > rhs``, unevaluated."""

    __slots__ = ()
    rel_op = '>'


class Ge(Relational):
    """``lhs >= rhs``, unevaluated."""

    __slots__ = ()
    rel_op = '>='


Equality = Eq
Unequality = Ne
StrictLessThan = Lt
LessThan = Le
StrictGreaterThan = Gt
GreaterThan = Ge

_REVERSED: dict[type[Relational], type[Relational]] = {Eq: Eq, Ne: Ne, Lt: Gt, Gt: Lt, Le: Ge, Ge: Le}


def _decide_equal(a: Expr, b: Expr) -> Basic:
    """``true``, ``false`` or ``None`` as a relation, for ``a == b``."""
    if a == b:
        return true
    if isinstance(a, Number) and isinstance(b, Number):
        if isinstance(a, NaN) or isinstance(b, NaN):
            return false
        va, vb = a.exact_value(), b.exact_value()
        if va is not None and vb is not None:
            return true if va == vb else false
        return false
    difference = sub(a, b)
    if isinstance(difference, Number):
        value = difference.exact_value()
        if value is not None:
            return true if value == 0 else false
        if not isinstance(difference, NaN):
            return false
    zero = difference.is_zero
    if zero is not None:
        return true if zero else false
    return Eq(a, b)


def eq(a: ExprLike, b: ExprLike) -> Basic:
    """The evaluated equation ``a == b``: ``true``, ``false`` or ``Eq``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, eq
    >>> x, y = symbols('x y')
    >>> eq(x, x), eq(x + 1, x), eq(x, y), eq(2, 2.0)
    (True, False, Eq(x, y), True)
    """
    return _decide_equal(to_expr(a), to_expr(b))


def ne(a: ExprLike, b: ExprLike) -> Basic:
    """The evaluated inequation ``a != b``: ``true``, ``false`` or ``Ne``."""
    result = _decide_equal(to_expr(a), to_expr(b))
    if result == true:
        return false
    if result == false:
        return true
    return Ne(to_expr(a), to_expr(b))


def _order(a: Number, b: Number) -> int:
    """``-1``, ``0`` or ``1`` for two real numbers; ``2`` when undecidable."""
    if isinstance(a, (NaN, ComplexInfinity)) or isinstance(b, (NaN, ComplexInfinity)):
        return 2
    if isinstance(a, Infinity):
        return 0 if isinstance(b, Infinity) else 1
    if isinstance(a, NegativeInfinity):
        return 0 if isinstance(b, NegativeInfinity) else -1
    if isinstance(b, Infinity):
        return -1
    if isinstance(b, NegativeInfinity):
        return 1
    va, vb = a.exact_value(), b.exact_value()
    if va is None or vb is None:
        return 2
    return (va > vb) - (va < vb)


def _compare(a: ExprLike, b: ExprLike, accepted: tuple[int, ...], cls: type[Relational]) -> Basic:
    lhs, rhs = to_expr(a), to_expr(b)
    for side in (lhs, rhs):
        if side.is_real is False and side.is_extended_real is not True:
            raise TypeError("invalid comparison of non-real %s" % (side,))
    order = 2
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        order = _order(lhs, rhs)
    else:
        difference = sub(lhs, rhs)
        if isinstance(difference, Number) and not isinstance(difference, (Float,)):
            order = _order(difference, Rational(0))
        elif isinstance(difference, Float):
            order = _order(difference, Rational(0))
    if order == 2:
        return cls(lhs, rhs)
    return true if order in accepted else false


def lt(a: ExprLike, b: ExprLike) -> Basic:
    """The evaluated ``a < b``: ``true``, ``false`` or ``Lt``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, lt, oo
    >>> x = symbols('x')
    >>> lt(1, 2), lt(x, x), lt(x, x + 1), lt(x, 1), lt(-oo, 3)
    (True, False, True, x < 1, True)
    """
    return _compare(a, b, (-1,), Lt)


def le(a: ExprLike, b: ExprLike) -> Basic:
    """The evaluated ``a <= b``: ``true``, ``false`` or ``Le``."""
    return _compare(a, b, (-1, 0), Le)


def gt(a: ExprLike, b: ExprLike) -> Basic:
    """The evaluated ``a > b``: ``true``, ``false`` or ``Gt``."""
    return _compare(a, b, (1,), Gt)


def ge(a: ExprLike, b: ExprLike) -> Basic:
    """The evaluated ``a >= b``: ``true``, ``false`` or ``Ge``."""
    return _compare(a, b, (0, 1), Ge)


_FUNCTIONS = {Eq: eq, Ne: ne, Lt: lt, Le: le, Gt: gt, Ge: ge}
