"""Canonical ordering of expressions.

Sums and products are stored with their arguments in a canonical order, so
that ``x + y`` and ``y + x`` are the same tree; the order is given by
:meth:`Basic.sort_key`, a tuple of plain Python values which compares
without evaluating anything. The scheme is the one documented for
SymPy's ``default_sort_key`` [1]_: the key of an expression is made of
the key of its class, the keys of its arguments, the name of an atom, the
key of its exponent and its numerical coefficient, so that ``x``,
``2*x`` and ``x**2`` sort next to each other.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: default_sort_key*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.sorting.default_sort_key
"""
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Iterable, Union

if TYPE_CHECKING:
    from .basic import Basic

__all__ = ['ClassKey', 'ArgsKey', 'SortKey', 'default_sort_key', 'ordered']

#: (group, subgroup, class name): groups order numbers before atoms before
#: compound expressions, as SymPy's ``class_key``
ClassKey = tuple[int, int, str]
#: (number of arguments, their keys)
ArgsKey = tuple[int, tuple['SortKey', ...]]
#: the key of an exponent: empty for the exponent 1 of a leaf
ExpKey = Union[tuple[()], 'SortKey']
#: (class key, arguments key, name, exponent key, coefficient)
SortKey = tuple[ClassKey, ArgsKey, str, ExpKey, Fraction]


def default_sort_key(item: Basic) -> SortKey:
    """The canonical key of an expression (``item.sort_key()``).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Integer, default_sort_key
    >>> x, y = symbols('x y')
    >>> sorted([y, x, Integer(2)], key=default_sort_key)
    [2, x, y]
    """
    return item.sort_key()


def ordered(items: Iterable[Basic]) -> list[Basic]:
    """The expressions sorted by their canonical key."""
    return sorted(items, key=default_sort_key)
