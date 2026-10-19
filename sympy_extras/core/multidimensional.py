"""``vectorize``: a decorator applying a function elementwise to lists
given in chosen argument positions (``sympy.core.multidimensional``)."""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar, Union

__all__ = ['vectorize']

T = TypeVar('T')
#: an argument which may be a value or a list of values
Vectorized = Union[T, Sequence[T]]


class vectorize:
    """Apply a function to each element of a list given in one of the
    positions ``argnums`` (nested lists are mapped recursively, several
    vectorised arguments give a nested result).

    Examples
    ========

    >>> from sympy_extras.core.multidimensional import vectorize
    >>> @vectorize(0)
    ... def square(a: int) -> int:
    ...     return a*a
    >>> square([1, 2, 3]), square(4)
    ([1, 4, 9], 16)
    >>> @vectorize(0, 1)
    ... def add(a: int, b: int) -> int:
    ...     return a + b
    >>> add([1, 2], [10, 20])
    [[11, 21], [12, 22]]
    """

    def __init__(self, *argnums: int) -> None:
        self.argnums: tuple[int, ...] = argnums

    def __call__(self, f: Callable[..., object]) -> Callable[..., object]:
        def wrapped(*args: object) -> object:
            return self._apply(f, list(args), 0)
        wrapped.__name__ = f.__name__
        wrapped.__doc__ = f.__doc__
        return wrapped

    def _apply(self, f: Callable[..., object], args: list[object], position: int) -> object:
        for i in self.argnums[position:]:
            value = args[i] if i < len(args) else None
            if isinstance(value, (list, tuple)):
                results: list[object] = []
                for v in value:
                    new_args = list(args)
                    new_args[i] = v
                    results.append(self._apply(f, new_args, position))
                return results
            position += 1
        return f(*args)
