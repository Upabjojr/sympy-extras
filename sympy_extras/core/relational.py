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

from typing import Final, TYPE_CHECKING

from .add import sub
from .basic import Basic
from .expr import Expr, ExprLike, to_expr
from .kind import BooleanKind, Kind
from .logic import Boolean, false, true
from .numbers import ComplexInfinity, Float, Infinity, NaN, NegativeInfinity, Number, Rational
from .parameters import global_parameters
from .sorting import ClassKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Relational', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'Unequality',
    'StrictLessThan', 'LessThan', 'StrictGreaterThan', 'GreaterThan', 'Rel',
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
        self._lhs: Final[Expr] = to_expr(lhs)
        self._rhs: Final[Expr] = to_expr(rhs)
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

    @property
    def kind(self) -> Kind:
        return BooleanKind

    @property
    def binary_symbols(self) -> set[Basic]:
        return set()

    @property
    def reversedsign(self) -> Relational:
        """The relation with both sides negated (``-x > -1`` for ``x < 1``)."""
        from .mul import neg
        return _REVERSED[type(self)](neg(self._lhs), neg(self._rhs)) if not isinstance(self, (Eq, Ne)) \
            else type(self)(neg(self._lhs), neg(self._rhs))

    # PORTED FROM SYMPY: sympy/core/relational.py:Relational.canonical
    @property
    def canonical(self) -> Relational:
        """A canonical form (SymPy's): a number on the right, otherwise
        the sides in canonical order, and no minus sign in front of the
        left side (``-x < 1`` is ``x > -1``)."""
        from .sorting import default_sort_key
        r: Relational = self
        if r.rhs.is_number:
            if r.rhs.is_Number and r.lhs.is_Number and _ge_numbers(r.lhs, r.rhs):
                r = r.reversed
        elif r.lhs.is_number:
            r = r.reversed
        elif sorted(r.args, key=default_sort_key) != list(r.args):
            r = r.reversed
        if r.lhs.could_extract_minus_sign():
            return r.reversedsign
        if not r.rhs.is_number and r.rhs.could_extract_minus_sign():
            from .mul import neg
            first = sorted([r.lhs, neg(r.rhs)], key=default_sort_key)[0]
            if first != r.lhs:
                return r.reversed.reversedsign
        return r

    @property
    def negated(self) -> Relational:
        """The relation with the opposite meaning (``Lt`` for ``Ge``)."""
        return _NEGATED[type(self)](self._lhs, self._rhs)

    @classmethod
    def class_key(cls) -> ClassKey:
        # SymPy's class names (LessThan, StrictGreaterThan, ...): the order
        # of the arguments of And and Or
        return (5, 0, _SYMPY_NAMES.get(cls, cls.__name__))

    def _compare_name(self) -> str:
        return _SYMPY_NAMES[type(self)]

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
_NEGATED: dict[type[Relational], type[Relational]] = {Eq: Ne, Ne: Eq, Lt: Ge, Ge: Lt, Le: Gt, Gt: Le}
_NAMES: dict[str, type[Relational]] = {
    '==': Eq, 'eq': Eq, '!=': Ne, '<>': Ne, 'ne': Ne, '<': Lt, 'lt': Lt, '<=': Le, 'le': Le,
    '>': Gt, 'gt': Gt, '>=': Ge, 'ge': Ge}


#: the names of the classes in SymPy, for the ``compare`` order
_SYMPY_NAMES: dict[type[Relational], str] = {Eq: 'Equality', Ne: 'Unequality', Lt: 'StrictLessThan',
    Le: 'LessThan', Gt: 'StrictGreaterThan', Ge: 'GreaterThan'}


def Rel(lhs: ExprLike, rhs: ExprLike, rop: str) -> Relational:
    """The unevaluated relation named by ``rop`` (``'<'``, ``'lt'``, ``'=='``, ...).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Rel
    >>> x = Symbol('x')
    >>> Rel(x, 1, '<'), Rel(x, 1, 'ge'), Rel(x, 1, '==')
    (x < 1, x >= 1, Eq(x, 1))
    """
    if rop not in _NAMES:
        raise ValueError("unknown relation %r" % (rop,))
    return _NAMES[rop](lhs, rhs)


def _decide_equal(a: Expr, b: Expr) -> Boolean:
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


def eq(a: ExprLike, b: ExprLike) -> Boolean:
    """The evaluated equation ``a == b``: ``true``, ``false`` or ``Eq``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, eq
    >>> x, y = symbols('x y')
    >>> eq(x, x), eq(x + 1, x), eq(x, y), eq(2, 2.0)
    (True, False, Eq(x, y), True)
    """
    if not global_parameters.evaluate:
        return Eq(a, b)
    return _decide_equal(to_expr(a), to_expr(b))


def ne(a: ExprLike, b: ExprLike) -> Boolean:
    """The evaluated inequation ``a != b``: ``true``, ``false`` or ``Ne``."""
    if not global_parameters.evaluate:
        return Ne(a, b)
    result = _decide_equal(to_expr(a), to_expr(b))
    if result == true:
        return false
    if result == false:
        return true
    return Ne(to_expr(a), to_expr(b))


def _ge_numbers(a: Expr, b: Expr) -> bool:
    """``a > b`` for two numbers (SymPy's ``r.lhs > r.rhs`` in
    ``canonical``)."""
    return isinstance(a, Number) and isinstance(b, Number) and _order(a, b) == 1


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


def _compare(a: ExprLike, b: ExprLike, accepted: tuple[int, ...], cls: type[Relational]) -> Boolean:
    lhs, rhs = to_expr(a), to_expr(b)
    if not global_parameters.evaluate:
        return cls(lhs, rhs)
    for side in (lhs, rhs):
        if side.is_real is False and side.is_extended_real is not True:
            raise TypeError("invalid comparison of non-real %s" % (side,))
    order = 2
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        order = _order(lhs, rhs)
    elif lhs.is_extended_real and rhs.is_extended_real and \
            (isinstance(rhs, (Infinity, NegativeInfinity)) or isinstance(lhs, (Infinity, NegativeInfinity))):
        # SymPy's is_ge: every extended real is between -oo and oo, a
        # finite one strictly (p - oo is -oo)
        other = lhs if isinstance(rhs, (Infinity, NegativeInfinity)) else rhs
        if isinstance(rhs, Infinity) or isinstance(lhs, NegativeInfinity):
            order = -1 if other.is_finite is True else {(-1, 0): -1, (1,): -1}.get(accepted, 2)
        else:
            order = 1 if other.is_finite is True else {(0, 1): 1, (-1,): 1}.get(accepted, 2)
    elif (lhs.is_comparable and rhs.is_comparable) or \
            (lhs.is_extended_real is True and rhs.is_extended_real is True):
        # SymPy's is_ge: a numerical difference of comparable numbers, or
        # the sign of the difference of two extended reals (x + 1 > x is
        # not decided for an x not known real)
        difference = sub(lhs, rhs)
        if isinstance(difference, Number) and not isinstance(difference, (Float,)):
            order = _order(difference, Rational(0))
        elif isinstance(difference, Float):
            order = _order(difference, Rational(0))
        elif not isinstance(difference, NaN):
            # SymPy's is_gt/is_ge: the sign of the difference from its facts
            if difference.is_extended_positive:
                order = 1
            elif difference.is_extended_negative:
                order = -1
            elif difference.is_zero:
                order = 0
            elif difference.is_extended_nonnegative and accepted == (0, 1):
                order = 1
            elif difference.is_extended_nonpositive and accepted == (-1, 0):
                order = -1
            elif difference.is_extended_nonnegative and accepted == (-1,):
                order = 1
            elif difference.is_extended_nonpositive and accepted == (1,):
                order = -1
    if order == 2:
        return cls(lhs, rhs)
    return true if order in accepted else false


def lt(a: ExprLike, b: ExprLike) -> Boolean:
    """The evaluated ``a < b``: ``true``, ``false`` or ``Lt``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, symbols, lt, oo
    >>> x, r = symbols('x'), Symbol('r', real=True)
    >>> lt(1, 2), lt(r, r), lt(r, r + 1), lt(x, 1), lt(-oo, 3), lt(x, x + 1)
    (True, False, True, x < 1, True, x < x + 1)
    """
    return _compare(a, b, (-1,), Lt)


def le(a: ExprLike, b: ExprLike) -> Boolean:
    """The evaluated ``a <= b``: ``true``, ``false`` or ``Le``."""
    return _compare(a, b, (-1, 0), Le)


def gt(a: ExprLike, b: ExprLike) -> Boolean:
    """The evaluated ``a > b``: ``true``, ``false`` or ``Gt``."""
    return _compare(a, b, (1,), Gt)


def ge(a: ExprLike, b: ExprLike) -> Boolean:
    """The evaluated ``a >= b``: ``true``, ``false`` or ``Ge``."""
    return _compare(a, b, (0, 1), Ge)


_FUNCTIONS = {Eq: eq, Ne: ne, Lt: lt, Le: le, Gt: gt, Ge: ge}
