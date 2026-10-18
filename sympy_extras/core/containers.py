"""Tuples of expressions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, overload

from .basic import Basic
from .sorting import ClassKey
from .sympify import SympifyLike

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Tuple']


class Tuple(Basic):
    """An immutable sequence of expressions which is itself a node.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Tuple
    >>> x, y = symbols('x y')
    >>> t = Tuple(x, y, 1)
    >>> t
    (x, y, 1)
    >>> t[0], len(t), x in t
    (x, 3, True)
    """

    __slots__ = ('_args',)

    is_Tuple = True

    def __init__(self, *args: SympifyLike) -> None:
        from .sympify import sympify
        self._args: tuple[Basic, ...] = tuple(sympify(a) for a in args)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return self._args

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Tuple')

    @overload
    def __getitem__(self, i: int) -> Basic: ...

    @overload
    def __getitem__(self, i: slice) -> Tuple: ...

    def __getitem__(self, i: int | slice) -> Basic:
        if isinstance(i, slice):
            return Tuple(*self._args[i])
        return self._args[i]

    def __len__(self) -> int:
        return len(self._args)

    def __iter__(self) -> Iterator[Basic]:
        return iter(self._args)

    def __contains__(self, item: object) -> bool:
        return item in self._args

    def _sympystr(self, printer: StrPrinter) -> str:
        if len(self._args) == 1:
            return '(%s,)' % printer.doprint(self._args[0])
        return '(%s)' % ', '.join(printer.doprint(a) for a in self._args)
