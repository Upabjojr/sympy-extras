"""Boolean values and the logical connectives.

``true`` and ``false`` are the results of the evaluating relational
functions (``eq(x, x)`` is ``true``); they are atoms of the tree, convert
to Python booleans and compare equal to ``True`` and ``False``. The
connectives ``And``, ``Or``, ``Not``, ``Xor``, ``Implies``,
``Equivalent`` and ``ITE`` are unevaluating classes, as everything here,
with the evaluating functions ``and_``, ``or_``, ``not_``, ``xor``,
``implies``, ``equivalent`` and ``ite`` (SymPy's constructors evaluate):
``and_`` flattens, drops ``true``, gives ``false`` for a ``false``
argument or for an argument and its negation, removes the duplicates and
sorts the arguments; ``not_`` negates a relation (``not_(x > 0)`` is
``x <= 0``) and a truth value and cancels a double negation. The
operators ``&``, ``|``, ``~``, ``^``, ``>>`` and ``<<`` of
:class:`Boolean` (the base of the symbols, the relations and the
predicates) call the functions, as in SymPy [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Boolean algebra*,
       https://docs.sympy.org/latest/modules/logic.html
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Union

from .basic import Atom, Basic
from .kind import BooleanKind, Kind
from .operations import LatticeOp
from .sorting import ClassKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Boolean', 'BooleanAtom', 'BooleanTrue', 'BooleanFalse', 'true', 'false', 'BooleanLike',
    'to_boolean', 'BooleanFunction', 'And', 'Or', 'Not', 'Xor', 'Implies', 'Equivalent', 'ITE',
    'and_', 'or_', 'not_', 'xor', 'implies', 'equivalent', 'ite', 'Nand', 'Nor', 'Xnor']

#: what the logical functions accept: a boolean node or a Python bool
BooleanLike = Union['Boolean', bool]


class Boolean(Basic):
    """The base class of the objects with a truth value: the truth values,
    the relations, the symbols, the predicates and the connectives. Its
    operators build the connectives with the evaluating functions.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> x, y = symbols('x y')
    >>> x & y, x | y, ~x, x ^ y, x >> y, x << y, (x > 0) & (x < 1)
    (x & y, x | y, ~x, x ^ y, Implies(x, y), Implies(y, x), (x > 0) & (x < 1))
    """

    __slots__ = ()

    is_Boolean = True

    @property
    def kind(self) -> Kind:
        return BooleanKind

    @property
    def binary_symbols(self) -> set[Basic]:
        """The symbols of the tree used as truth values."""
        result: set[Basic] = set()
        for a in self.args:
            if isinstance(a, Boolean):
                result |= a.binary_symbols
        return result

    def __and__(self, other: BooleanLike) -> Boolean:
        return and_(self, other)

    def __rand__(self, other: BooleanLike) -> Boolean:
        return and_(other, self)

    def __or__(self, other: BooleanLike) -> Boolean:
        return or_(self, other)

    def __ror__(self, other: BooleanLike) -> Boolean:
        return or_(other, self)

    def __invert__(self) -> Boolean:
        return not_(self)

    def __xor__(self, other: BooleanLike) -> Boolean:
        return xor(self, other)

    def __rxor__(self, other: BooleanLike) -> Boolean:
        return xor(other, self)

    def __rshift__(self, other: BooleanLike) -> Boolean:
        return implies(self, other)

    def __lshift__(self, other: BooleanLike) -> Boolean:
        return implies(other, self)

    def __rrshift__(self, other: BooleanLike) -> Boolean:
        return implies(other, self)

    def __rlshift__(self, other: BooleanLike) -> Boolean:
        return implies(self, other)


def to_boolean(x: object) -> Boolean:
    """``x`` as a boolean node: ``True`` and ``False`` become ``true`` and
    ``false``; anything which is not a :class:`Boolean` is refused.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.core.logic import to_boolean
    >>> to_boolean(True), to_boolean(Symbol('x') > 0)
    (True, x > 0)
    >>> to_boolean(Integer(2))
    Traceback (most recent call last):
    ...
    TypeError: expecting bool or Boolean, not `2`.
    """
    if isinstance(x, Boolean):
        return x
    if isinstance(x, bool):
        return true if x else false
    raise TypeError("expecting bool or Boolean, not `%s`." % (x,))


class BooleanAtom(Boolean, Atom):
    """``true`` or ``false``."""

    __slots__ = ('_value',)

    def __init__(self, value: bool) -> None:
        self._value: bool = value
        Basic.__init__(self)

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BooleanAtom):
            return self._value == other._value
        if isinstance(other, bool):
            return self._value == other
        if isinstance(other, Basic):
            return False
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __bool__(self) -> bool:
        return self._value

    def __str__(self) -> str:
        return 'True' if self._value else 'False'

    def _sympystr(self, printer: StrPrinter) -> str:
        return str(self)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, cls.__name__)


class BooleanTrue(BooleanAtom):
    """The value ``true``."""

    __slots__ = ()

    def __init__(self) -> None:
        BooleanAtom.__init__(self, True)


class BooleanFalse(BooleanAtom):
    """The value ``false``."""

    __slots__ = ()

    def __init__(self) -> None:
        BooleanAtom.__init__(self, False)


true = BooleanTrue()
false = BooleanFalse()


class BooleanFunction(Boolean):
    """The base class of the connectives: a boolean node with boolean
    arguments."""

    __slots__ = ()

    @property
    def args(self) -> tuple[Boolean, ...]:
        raise NotImplementedError

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, cls.__name__)

    def _sympystr(self, printer: StrPrinter) -> str:
        return '%s(%s)' % (type(self).__name__, ', '.join(printer.doprint(a) for a in self.args))


class _Connective(BooleanFunction):
    """A connective with its arguments in a tuple."""

    __slots__ = ('_args',)

    def __init__(self, *args: BooleanLike) -> None:
        self._args: tuple[Boolean, ...] = tuple(to_boolean(a) for a in args)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Boolean, ...]:
        return self._args


class And(LatticeOp, BooleanFunction):
    """The conjunction, unevaluated; ``and_`` evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, And, and_
    >>> x, y = symbols('x y')
    >>> And(x, x), And(x, x).doit(), and_(x, True, y), and_(x, False), and_(x > 0, x <= 0), and_()
    (x & x, x, x & y, False, False, True)
    """

    __slots__ = ()

    identity = true
    zero = false

    def __init__(self, *args: BooleanLike) -> None:
        LatticeOp.__init__(self, *[to_boolean(a) for a in args])

    @property
    def args(self) -> tuple[Boolean, ...]:
        return tuple(to_boolean(a) for a in self._args)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return and_(*[to_boolean(a) for a in args])

    def _sympystr(self, printer: StrPrinter) -> str:
        from .printing import PREC_AND_ARGS
        return ' & '.join(printer.parenthesize(a, PREC_AND_ARGS) for a in self.args)


class Or(LatticeOp, BooleanFunction):
    """The disjunction, unevaluated; ``or_`` evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Or, or_, Eq, Ne
    >>> x, y = symbols('x y')
    >>> Or(x, x), or_(x, False, y), or_(x, True), or_(Eq(x, 1), Ne(x, 1)), or_()
    (x | x, x | y, True, True, False)
    """

    __slots__ = ()

    identity = false
    zero = true

    def __init__(self, *args: BooleanLike) -> None:
        LatticeOp.__init__(self, *[to_boolean(a) for a in args])

    @property
    def args(self) -> tuple[Boolean, ...]:
        return tuple(to_boolean(a) for a in self._args)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return or_(*[to_boolean(a) for a in args])

    def _sympystr(self, printer: StrPrinter) -> str:
        from .printing import PREC_OR_ARGS
        return ' | '.join(printer.parenthesize(a, PREC_OR_ARGS) for a in self.args)


class Not(_Connective):
    """The negation, unevaluated; ``not_`` evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Not, not_
    >>> x = symbols('x')
    >>> Not(Not(x)), not_(Not(x)), not_(x > 0), not_(True), ~(x & Not(x))
    (~~x, x, x <= 0, False, ~(x & ~x))
    """

    __slots__ = ()

    is_Not = True

    def __init__(self, arg: BooleanLike) -> None:
        _Connective.__init__(self, arg)

    @property
    def arg(self) -> Boolean:
        return self._args[0]

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return not_(to_boolean(args[0]))

    def _sympystr(self, printer: StrPrinter) -> str:
        from .printing import PREC_FUNC
        return '~%s' % printer.parenthesize(self._args[0], PREC_FUNC)


class Xor(_Connective):
    """The exclusive disjunction, unevaluated; ``xor`` evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, xor
    >>> x, y, z = symbols('x y z')
    >>> xor(x, y), xor(x, x), xor(x, True), xor(x, y, x), xor(x, xor(y, z)), xor()
    (x ^ y, False, ~x, y, x ^ y ^ z, False)
    """

    __slots__ = ()

    def _hashable_content(self) -> tuple[Basic, ...]:
        from .sorting import ordered
        return tuple(ordered(self._args))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return xor(*[to_boolean(a) for a in args])

    def _sympystr(self, printer: StrPrinter) -> str:
        from .printing import PREC_XOR_ARGS
        return ' ^ '.join(printer.parenthesize(a, PREC_XOR_ARGS) for a in self._args)


class Implies(_Connective):
    """The implication, unevaluated; ``implies`` evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, implies
    >>> x, y = symbols('x y')
    >>> implies(x, y), implies(True, y), implies(False, y), implies(x, True), implies(x, False), implies(x, x)
    (Implies(x, y), y, True, True, ~x, True)
    """

    __slots__ = ()

    def __init__(self, antecedent: BooleanLike, consequent: BooleanLike) -> None:
        _Connective.__init__(self, antecedent, consequent)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return implies(to_boolean(args[0]), to_boolean(args[1]))


class Equivalent(_Connective):
    """The equivalence of its arguments, unevaluated; ``equivalent``
    evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, equivalent
    >>> x, y, z = symbols('x y z')
    >>> equivalent(x, y), equivalent(x, x), equivalent(x, True), equivalent(x, False), equivalent(z, y, x)
    (Equivalent(x, y), True, x, ~x, Equivalent(x, y, z))
    """

    __slots__ = ()

    def _hashable_content(self) -> tuple[Basic, ...]:
        from .sorting import ordered
        return tuple(ordered(set(self._args)))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return equivalent(*[to_boolean(a) for a in args])


class ITE(_Connective):
    """``ITE(c, a, b)``: ``a`` if ``c`` else ``b``, unevaluated; ``ite``
    evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, ite
    >>> x, y, z = symbols('x y z')
    >>> ite(x, y, z), ite(True, y, z), ite(False, y, z), ite(x, y, z).subs(x, True)
    (ITE(x, y, z), y, z, y)
    """

    __slots__ = ()

    def __init__(self, condition: BooleanLike, then: BooleanLike, otherwise: BooleanLike) -> None:
        _Connective.__init__(self, condition, then, otherwise)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return ite(to_boolean(args[0]), to_boolean(args[1]), to_boolean(args[2]))


# ----------------------------------------------------------------------
# the evaluating functions

def _flatten(cls: type[LatticeOp], args: Iterable[BooleanLike]) -> list[Boolean]:
    """The arguments with the nested nodes of ``cls`` opened."""
    result: list[Boolean] = []
    for a in args:
        b = to_boolean(a)
        if isinstance(b, cls):
            result.extend(_flatten(cls, [to_boolean(a) for a in b.args]))
        else:
            result.append(b)
    return result


def _lattice(cls: type[LatticeOp], identity: Boolean, zero: Boolean, args: Iterable[BooleanLike]) -> Boolean:
    from .sorting import ordered
    kept: list[Boolean] = []
    seen: set[Boolean] = set()
    for a in _flatten(cls, args):
        if a == identity or a in seen:
            continue
        if a == zero:
            return zero
        seen.add(a)
        kept.append(a)
    from .relational import Relational
    for a in kept:
        # a relation and its negation give the zero (x > 0 and x <= 0);
        # SymPy keeps x & ~x and Q.positive(x) & ~Q.positive(x)
        if isinstance(a, Relational) and a.negated in seen:
            return zero
    if not kept:
        return identity
    if len(kept) == 1:
        return kept[0]
    return to_boolean(cls(*ordered(kept)))


def and_(*args: BooleanLike) -> Boolean:
    """The evaluated conjunction: flattened, without ``true`` and the
    duplicates, ``false`` when an argument is ``false`` or a relation
    whose negation is another argument, the arguments sorted (as in
    SymPy, ``x & ~x`` stays).

    Examples
    ========

    >>> from sympy_extras.core import symbols, and_
    >>> x, y, z = symbols('x y z')
    >>> and_(y, x), and_(x, and_(y, z)), and_(x > 0, 1 > 0), and_(x > 0, x <= 0), and_(x), and_(x, ~x)
    (x & y, x & y & z, x > 0, False, x, x & ~x)
    """
    return _lattice(And, true, false, args)


def or_(*args: BooleanLike) -> Boolean:
    """The evaluated disjunction, dual to :func:`and_`.

    Examples
    ========

    >>> from sympy_extras.core import symbols, or_, Eq, Ne
    >>> x, y = symbols('x y')
    >>> or_(y, x), or_(x, or_(y, x)), or_(Eq(x, 1), Ne(x, 1)), or_(x > 0, 1 < 0)
    (x | y, x | y, True, x > 0)
    """
    return _lattice(Or, false, true, args)


def not_(arg: BooleanLike) -> Boolean:
    """The evaluated negation: a truth value is flipped, a relation
    reversed (``x >= 0`` for ``x < 0``, ``Ne`` for ``Eq``), a double
    negation cancelled.

    Examples
    ========

    >>> from sympy_extras.core import symbols, not_, Eq, And
    >>> x = symbols('x')
    >>> not_(x), not_(not_(x)), not_(x < 0), not_(Eq(x, 1)), not_(False), not_(And(x, x))
    (~x, x, x >= 0, Ne(x, 1), True, ~(x & x))
    """
    from .relational import Relational
    a = to_boolean(arg)
    if isinstance(a, BooleanAtom):
        return false if a else true
    if isinstance(a, Not):
        return a.arg
    if isinstance(a, Relational):
        return a.negated
    return Not(a)


def xor(*args: BooleanLike) -> Boolean:
    """The evaluated exclusive disjunction: flattened, ``true`` arguments
    negate the result, ``false`` ones are dropped, a pair of equal
    arguments cancels.

    Examples
    ========

    >>> from sympy_extras.core import symbols, xor
    >>> x, y = symbols('x y')
    >>> xor(x, y, True), xor(x, ~x), xor(x, False), xor(True, True), xor(True)
    (~(x ^ y), x ^ ~x, x, False, True)
    """
    from .sorting import default_sort_key
    negate = False
    kept: list[Boolean] = []
    for a in _flatten_xor(args):
        if isinstance(a, BooleanAtom):
            negate ^= bool(a)
            continue
        if a in kept:
            kept.remove(a)
            continue
        kept.append(a)
    if not kept:
        return true if negate else false
    result: Boolean = kept[0] if len(kept) == 1 else Xor(*sorted(kept, key=default_sort_key))
    return not_(result) if negate else result


def _flatten_xor(args: Iterable[BooleanLike]) -> list[Boolean]:
    result: list[Boolean] = []
    for a in args:
        b = to_boolean(a)
        if isinstance(b, Xor):
            result.extend(_flatten_xor(b.args))
        else:
            result.append(b)
    return result


def implies(antecedent: BooleanLike, consequent: BooleanLike) -> Boolean:
    """The evaluated implication: ``true`` when the antecedent is
    ``false``, the consequent is ``true`` or both are equal; the
    consequent for a ``true`` antecedent; the negated antecedent for a
    ``false`` consequent.

    Examples
    ========

    >>> from sympy_extras.core import symbols, implies
    >>> x, y = symbols('x y')
    >>> implies(x, y), implies(True, y), implies(x, False), implies(x, ~x)
    (Implies(x, y), y, ~x, Implies(x, ~x))
    """
    a, b = to_boolean(antecedent), to_boolean(consequent)
    if isinstance(a, BooleanAtom):
        return b if a else true
    if isinstance(b, BooleanAtom):
        return true if b else not_(a)
    if a == b:
        return true
    return Implies(a, b)


def equivalent(*args: BooleanLike) -> Boolean:
    """The evaluated equivalence: the duplicates removed, ``true`` for at
    most one distinct argument, the conjunction of the others (or of
    their negations) when ``true`` (or ``false``) is among them.

    Examples
    ========

    >>> from sympy_extras.core import symbols, equivalent
    >>> x, y, z = symbols('x y z')
    >>> equivalent(x, y, y), equivalent(x, True, y), equivalent(x, True, False), equivalent(True, True)
    (Equivalent(x, y), x & y, False, True)
    """
    from .sorting import default_sort_key
    items: list[Boolean] = []
    for a in args:
        b = to_boolean(a)
        if b not in items:
            items.append(b)
    if true in items:
        return and_(*[a for a in items if a != true])
    if false in items:
        return and_(*[not_(a) for a in items if a != false])
    if len(items) <= 1:
        return true
    return Equivalent(*sorted(items, key=default_sort_key))


def ite(condition: BooleanLike, then: BooleanLike, otherwise: BooleanLike) -> Boolean:
    """The evaluated ``ITE``: the branch chosen by a known condition.

    Examples
    ========

    >>> from sympy_extras.core import symbols, ite
    >>> x, y, z = symbols('x y z')
    >>> ite(x, y, z), ite(x > 0, y, z), ite(1 > 0, y, z)
    (ITE(x, y, z), ITE(x > 0, y, z), y)
    """
    c, a, b = to_boolean(condition), to_boolean(then), to_boolean(otherwise)
    if isinstance(c, BooleanAtom):
        return a if c else b
    return ITE(c, a, b)


def Nand(*args: BooleanLike) -> Boolean:
    """``~(a & b & ...)``, evaluated.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Nand
    >>> x, y = symbols('x y')
    >>> Nand(x, y), Nand(x, True), Nand(x, False)
    (~(x & y), ~x, True)
    """
    return not_(and_(*args))


def Nor(*args: BooleanLike) -> Boolean:
    """``~(a | b | ...)``, evaluated.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Nor
    >>> x, y = symbols('x y')
    >>> Nor(x, y), Nor(x, False), Nor(x, True)
    (~(x | y), ~x, False)
    """
    return not_(or_(*args))


def Xnor(*args: BooleanLike) -> Boolean:
    """``~(a ^ b ^ ...)``, evaluated.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Xnor
    >>> x, y = symbols('x y')
    >>> Xnor(x, y), Xnor(x, x)
    (~(x ^ y), True)
    """
    return not_(xor(*args))
