"""Boolean values.

``true`` and ``false`` are the results of the evaluating relational
functions (``eq(x, x)`` is ``true``); they are atoms of the tree, convert
to Python booleans and compare equal to ``True`` and ``False``. The
connectives (``And``, ``Or``, ``Not``) belong to ``sympy.logic`` and are
not part of the core.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .basic import Atom, Basic
from .kind import BooleanKind, Kind
from .sorting import ClassKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Boolean', 'BooleanAtom', 'BooleanTrue', 'BooleanFalse', 'true', 'false']


class Boolean(Basic):
    """The base class of the objects with a truth value."""

    __slots__ = ()

    is_Boolean = True

    @property
    def kind(self) -> Kind:
        return BooleanKind


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
