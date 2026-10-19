"""A memoising decorator, ``cacheit``, and ``clear_cache``.

The nodes of the tree are immutable and hashable, so a function of nodes
can be memoised safely; ``cacheit`` keeps the signature of the function
it wraps (a ``ParamSpec``), and ``clear_cache`` empties every cache made
with it, as SymPy's ``cacheit`` and ``clear_cache`` do [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: cache*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.cache
"""
from __future__ import annotations

from typing import Callable, Generic, Hashable, ParamSpec, Protocol, TypeVar

__all__ = ['cacheit', 'clear_cache', 'Cached', 'register_clearable']

P = ParamSpec('P')
R = TypeVar('R')



class _Clearable(Protocol):
    def clear(self) -> None: ...


_caches: list[_Clearable] = []


class Cached(Generic[P, R]):
    """A memoised callable; the key is the tuple of the arguments."""

    def __init__(self, function: Callable[P, R]) -> None:
        self.function: Callable[P, R] = function
        self.cache: dict[Hashable, R] = {}
        self.__doc__ = function.__doc__
        self.__name__ = function.__name__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key: Hashable = (args, tuple(sorted(kwargs.items())))
        try:
            return self.cache[key]
        except KeyError:
            value = self.function(*args, **kwargs)
            self.cache[key] = value
            return value
        except TypeError:  # an unhashable argument
            return self.function(*args, **kwargs)

    def clear(self) -> None:
        """Empty the cache."""
        self.cache.clear()


def cacheit(function: Callable[P, R]) -> Cached[P, R]:
    """Memoise ``function`` (its arguments must be hashable).

    Examples
    ========

    >>> from sympy_extras.core.cache import cacheit, clear_cache
    >>> calls = []
    >>> @cacheit
    ... def square(n: int) -> int:
    ...     calls.append(n)
    ...     return n*n
    >>> square(3), square(3), calls
    (9, 9, [3])
    >>> clear_cache()
    >>> square(3), calls
    (9, [3, 3])
    """
    cached = Cached(function)
    _caches.append(cached)
    return cached


def register_clearable(cache: _Clearable) -> None:
    """Register an object with a ``clear`` method, emptied by :func:`clear_cache`."""
    _caches.append(cache)


def clear_cache() -> None:
    """Empty every cache made by :func:`cacheit` (and the caches of the
    conversion to and from SymPy)."""
    for c in _caches:
        c.clear()
