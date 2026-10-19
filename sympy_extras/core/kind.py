"""Kinds: a coarse classification of objects, finer than the class.

``NumberKind`` is the kind of scalar expressions, ``BooleanKind`` of
relations and truth values, ``UndefinedKind`` of everything which is
not known to be one of those (a noncommutative symbol, an undefined
function applied to arguments), and ``TupleKind(*kinds)`` of a tuple.
The ``kind`` property of every node gives its kind, as in SymPy [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Kind*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.kind
"""
from __future__ import annotations

__all__ = ['Kind', 'NumberKind', 'BooleanKind', 'UndefinedKind', 'TupleKind']


class Kind:
    """A kind, identified by its name and its parameters.

    Examples
    ========

    >>> from sympy_extras.core.kind import NumberKind, BooleanKind, TupleKind
    >>> NumberKind, TupleKind(NumberKind, BooleanKind)
    (NumberKind, TupleKind(NumberKind, BooleanKind))
    >>> NumberKind == NumberKind, NumberKind == BooleanKind
    (True, False)
    """

    __slots__ = ('name', 'parameters')

    def __init__(self, name: str, *parameters: Kind) -> None:
        self.name: str = name
        self.parameters: tuple[Kind, ...] = parameters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Kind):
            return NotImplemented
        return self.name == other.name and self.parameters == other.parameters

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        return hash((self.name, self.parameters))

    def __str__(self) -> str:
        if not self.parameters:
            return self.name
        return '%s(%s)' % (self.name, ', '.join(str(p) for p in self.parameters))

    def __repr__(self) -> str:
        return str(self)


#: the kind of scalar expressions
NumberKind = Kind('NumberKind')
#: the kind of relations and truth values
BooleanKind = Kind('BooleanKind')
#: the kind of what is not known to be a number or a boolean
UndefinedKind = Kind('UndefinedKind')


def TupleKind(*kinds: Kind) -> Kind:
    """The kind of a tuple whose elements have the given kinds."""
    return Kind('TupleKind', *kinds)
