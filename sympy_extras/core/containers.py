"""Tuples and dictionaries of expressions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Union, overload

from .basic import Basic
from .kind import Kind, TupleKind
from .sorting import ClassKey
from .sympify import SympifyLike

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Tuple', 'Dict']


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

    @property
    def kind(self) -> Kind:
        return TupleKind(*[a.kind for a in self._args])

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

    def index(self, value: Basic) -> int:
        """The position of ``value``."""
        return self._args.index(value)

    def _sympystr(self, printer: StrPrinter) -> str:
        if len(self._args) == 1:
            return '(%s,)' % printer.doprint(self._args[0])
        return '(%s)' % ', '.join(printer.doprint(a) for a in self._args)


#: what ``Dict`` accepts: a mapping or pairs
DictLike = Union[Mapping[SympifyLike, SympifyLike], tuple[SympifyLike, SympifyLike], 'Dict']


class Dict(Basic):
    """An immutable mapping of expressions which is itself a node; its
    arguments are the pairs as :class:`Tuple` nodes.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Dict
    >>> x, y = symbols('x y')
    >>> d = Dict({x: 1, y: 2})
    >>> d, d.args, d[x], d.get(y), len(d), x in d, list(d)
    ({x: 1, y: 2}, ((x, 1), (y, 2)), 1, 2, 2, True, [x, y])
    >>> Dict((x, 1), (y, 2)) == d, d.subs(x, y)
    (True, {y: 2})
    """

    __slots__ = ('_pairs',)

    def __init__(self, *args: DictLike) -> None:
        from .sympify import sympify
        pairs: list[tuple[Basic, Basic]] = []
        for a in args:
            if isinstance(a, Dict):
                pairs.extend(a.items())
            elif isinstance(a, Mapping):
                pairs.extend((sympify(k), sympify(v)) for k, v in a.items())
            else:
                k, v = a
                pairs.append((sympify(k), sympify(v)))
        self._pairs: dict[Basic, Basic] = dict(pairs)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return tuple(Tuple(k, v) for k, v in self._pairs.items())

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Dict')

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        pairs: list[tuple[Basic, Basic]] = []
        for a in args:
            if not (isinstance(a, Tuple) and len(a) == 2):
                raise TypeError("a pair is expected, got %s" % (a,))
            pairs.append((a[0], a[1]))
        return Dict(*pairs)

    def __getitem__(self, key: Basic) -> Basic:
        return self._pairs[key]

    def get(self, key: Basic, default: Optional[Basic] = None) -> Optional[Basic]:
        """The value of ``key``, or ``default``."""
        return self._pairs.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[Basic]:
        return iter(self._pairs)

    def keys(self) -> list[Basic]:
        return list(self._pairs.keys())

    def values(self) -> list[Basic]:
        return list(self._pairs.values())

    def items(self) -> list[tuple[Basic, Basic]]:
        return list(self._pairs.items())

    def _sympystr(self, printer: StrPrinter) -> str:
        return '{%s}' % ', '.join('%s: %s' % (printer.doprint(k), printer.doprint(v)) for k, v in self._pairs.items())
