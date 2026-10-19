"""Predicates and contexts of assumptions: ``Q.positive(x)``,
``global_assumptions``, ``assuming``.

A :class:`Predicate` is a boolean atom (``Q.positive``) which applied to
arguments gives an :class:`AppliedPredicate` (``Q.positive(x)``), a
boolean node that never evaluates: ``Q.positive(1)`` stays as it is, and
:func:`~sympy_extras.core.ask.ask` answers it. Handlers can be
registered on a predicate for a class of expressions, as on SymPy's
predicates [1]_. An :class:`AssumptionsContext` is a set of boolean
assumptions; ``global_assumptions`` is the one ``ask`` and ``refine``
consult, and :func:`assuming` adds to it for the duration of a block.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Predicate*,
       https://docs.sympy.org/latest/modules/assumptions/assume.html
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Optional, Union

from .basic import Basic
from .logic import Boolean, BooleanLike, to_boolean
from .sorting import ClassKey, SortKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Predicate', 'AppliedPredicate', 'Handler', 'AssumptionsContext', 'global_assumptions', 'assuming']

#: a handler registered on a predicate: ``(expression, assumptions) -> True/False/None``
Handler = Callable[[Basic, Boolean], Optional[bool]]
#: what a predicate accepts as argument
PredicateArg = Union[Basic, int, float, Fraction, bool, str]


class Predicate(Boolean):
    """A predicate: a boolean atom with a name, applied to arguments to
    make a proposition. ``Predicate(name)`` is a predicate of the user
    (SymPy's ``UndefinedPredicate``), distinct from the predicate of the
    same name in ``Q``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Predicate, Q, ask
    >>> x = Symbol('x')
    >>> Q.positive, Q.positive.name, Q.positive(x), Q.positive(x).args, Q.gt(x, 0)
    (Q.positive, 'positive', Q.positive(x), (Q.positive, x), Q.gt(x, 0))
    >>> P = Predicate('mypred')
    >>> P(x), P == Q.positive, Predicate('positive') == Q.positive, Predicate('mypred') == P
    (Q.mypred(x), False, False, True)
    >>> @P.register(Symbol)
    ... def _(expr, assumptions):
    ...     return True
    >>> ask(P(x)), ask(P(x + 1))
    (True, None)
    """

    __slots__ = ('name', '_arity', '_defined', '_handlers')

    def __init__(self, name: str, arity: int = 1) -> None:
        if not isinstance(name, str):
            raise TypeError("the name of a predicate is a string, got %r" % (name,))
        self.name: str = name
        self._arity: int = arity
        self._defined: bool = False
        self._handlers: list[tuple[type, Handler]] = []
        Basic.__init__(self)

    @classmethod
    def _known(cls, name: str, arity: int = 1) -> Predicate:
        """A predicate of ``Q``."""
        p = cls(name, arity)
        p._defined = True
        return p

    @property
    def args(self) -> tuple[Basic, ...]:
        return ()

    @property
    def arity(self) -> int:
        """The number of arguments the predicate takes."""
        return self._arity

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    def __hash__(self) -> int:
        return hash((self.name, self._defined))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Predicate):
            if isinstance(other, Basic):
                return False
            return NotImplemented
        return self.name == other.name and self._defined == other._defined

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __call__(self, *args: PredicateArg) -> AppliedPredicate:
        from .sympify import sympify
        if len(args) != self._arity:
            raise TypeError("%s takes %d argument(s), %d given" % (self, self._arity, len(args)))
        return AppliedPredicate(self, *[sympify(a) for a in args])

    def register(self, *types: type) -> Callable[[Handler], Handler]:
        """A decorator registering a handler for the expressions of the
        given classes: ``handler(expr, assumptions)`` answers ``True``,
        ``False`` or ``None``."""
        def decorator(handler: Handler) -> Handler:
            for t in types:
                self._handlers.append((t, handler))
            return handler
        return decorator

    def register_many(self, *types: type) -> Callable[[Handler], Handler]:
        """:meth:`register` for several classes (SymPy's name)."""
        return self.register(*types)

    def eval(self, expr: Basic, assumptions: BooleanLike = True) -> Optional[bool]:
        """The answer of the registered handlers for ``expr``, the first
        which is not ``None``."""
        context = to_boolean(assumptions)
        for t, handler in self._handlers:
            if isinstance(expr, t):
                result = handler(expr, context)
                if result is not None:
                    return result
        return None

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Predicate')

    def sort_key(self) -> SortKey:
        return ((5, 0, _class_name(self.name)), (0, ()), '', (), Fraction(1))

    def _compare_name(self) -> str:
        return _class_name(self.name)

    def __str__(self) -> str:
        return 'Q.' + self.name

    def _sympystr(self, printer: StrPrinter) -> str:
        return str(self)


def _class_name(name: str) -> str:
    """SymPy's class name of the predicate (``PositivePredicate``)."""
    return ''.join(part.capitalize() for part in name.split('_')) + 'Predicate'


class AppliedPredicate(Boolean):
    """A predicate applied to arguments, a proposition which never
    evaluates by itself.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Q, srepr
    >>> x = Symbol('x')
    >>> p = Q.positive(x)
    >>> p.function, p.arguments, p.arg, p.subs(x, 1), p & Q.real(x)
    (Q.positive, (x,), x, Q.positive(1), Q.positive(x) & Q.real(x))
    >>> srepr(p)
    AppliedPredicate(Q.positive, Symbol('x'))
    """

    __slots__ = ('_function', '_arguments')

    def __init__(self, predicate: Predicate, *args: Basic) -> None:
        if not isinstance(predicate, Predicate):
            raise TypeError("a predicate is expected, got %s" % (predicate,))
        self._function: Predicate = predicate
        self._arguments: tuple[Basic, ...] = args
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function,) + self._arguments

    @property
    def function(self) -> Predicate:
        """The predicate applied."""
        return self._function

    @property
    def arguments(self) -> tuple[Basic, ...]:
        """The arguments the predicate is applied to."""
        return self._arguments

    @property
    def arg(self) -> Basic:
        """The argument of a unary predicate."""
        if len(self._arguments) != 1:
            raise TypeError("%s has %d arguments" % (self, len(self._arguments)))
        return self._arguments[0]

    @property
    def binary_symbols(self) -> set[Basic]:
        return set()

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'AppliedPredicate')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        predicate = args[0]
        if not isinstance(predicate, Predicate):
            raise TypeError("a predicate is expected, got %s" % (predicate,))
        return AppliedPredicate(predicate, *args[1:])

    def _sympystr(self, printer: StrPrinter) -> str:
        return '%s(%s)' % (self._function, ', '.join(printer.doprint(a) for a in self._arguments))


class AssumptionsContext:
    """A set of assumptions (boolean nodes); ``global_assumptions`` is
    the one consulted by ``ask`` and ``refine``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Q, ask, global_assumptions
    >>> x = Symbol('x')
    >>> global_assumptions.add(Q.positive(x))
    >>> global_assumptions, ask(Q.positive(x)), Q.positive(x) in global_assumptions, len(global_assumptions)
    (AssumptionsContext({Q.positive(x)}), True, True, 1)
    >>> global_assumptions.remove(Q.positive(x))
    >>> global_assumptions, ask(Q.positive(x))
    (AssumptionsContext(), None)
    """

    __slots__ = ('_items',)

    def __init__(self, assumptions: Iterable[BooleanLike] = ()) -> None:
        self._items: set[Boolean] = {to_boolean(a) for a in assumptions}

    def add(self, *assumptions: BooleanLike) -> None:
        """Add the assumptions."""
        for a in assumptions:
            self._items.add(to_boolean(a))

    def remove(self, *assumptions: BooleanLike) -> None:
        """Remove the assumptions (``KeyError`` when one is absent)."""
        for a in assumptions:
            self._items.remove(to_boolean(a))

    def clear(self) -> None:
        """Remove every assumption."""
        self._items.clear()

    def __contains__(self, assumption: object) -> bool:
        return assumption in self._items

    def __iter__(self) -> Iterator[Boolean]:
        return iter(sorted(self._items, key=str))

    def __len__(self) -> int:
        return len(self._items)

    def __str__(self) -> str:
        if not self._items:
            return 'AssumptionsContext()'
        return 'AssumptionsContext({%s})' % ', '.join(str(a) for a in self)

    def __repr__(self) -> str:
        return str(self)


#: the assumptions in force everywhere
global_assumptions = AssumptionsContext()


@contextmanager
def assuming(*assumptions: BooleanLike) -> Iterator[None]:
    """A context manager adding the assumptions to ``global_assumptions``
    for the duration of the block.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Q, ask, assuming, sqrt
    >>> x = Symbol('x')
    >>> with assuming(Q.positive(x)):
    ...     print(ask(Q.positive(x + 1)), sqrt(x**2).refine())
    True x
    >>> ask(Q.positive(x + 1))
    """
    added = [to_boolean(a) for a in assumptions if to_boolean(a) not in global_assumptions]
    global_assumptions.add(*added)
    try:
        yield
    finally:
        global_assumptions.remove(*added)
