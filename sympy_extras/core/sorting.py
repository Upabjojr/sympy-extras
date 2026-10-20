"""Canonical ordering of expressions.

Two orderings, as in SymPy. Sums and products store their arguments in
the order of :meth:`Basic.compare` [1]_, which SymPy uses for the
arguments of ``Add`` and ``Mul``: the classes in a fixed order (the
numbers, the constants, the symbols, the powers, the products, the sums,
then alphabetically), then the length of the content of the node, then
its content; :func:`compare_key` is that order as a tuple of plain Python
values, cached on the node. :meth:`Basic.sort_key` is the order documented
for SymPy's ``default_sort_key`` [2]_ (``ordered``, the printing, the
arguments of ``And``): the key of the class, the keys of the arguments,
the name of an atom, the key of the exponent and the numerical
coefficient, so that ``x``, ``2*x`` and ``x**2`` sort next to each other.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Basic.compare*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.compare
.. [2] SymPy Development Team, *SymPy documentation: default_sort_key*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.sorting.default_sort_key
"""
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Iterable, Union

if TYPE_CHECKING:
    from .basic import Basic

__all__ = ['ClassKey', 'ArgsKey', 'SortKey', 'CompareKey', 'CompareItem', 'default_sort_key', 'compare_key',
    'class_rank', 'ordered']

#: (group, subgroup, class name): groups order numbers before atoms before
#: compound expressions, as SymPy's ``class_key``
ClassKey = tuple[int, int, str]
#: (number of arguments, their keys), or (1, (name,)) for an atom, (2, (name, index)) for a dummy
ArgsKey = tuple[int, Union[tuple['SortKey', ...], tuple[str], tuple[str, int]]]
#: the key of an exponent: empty for a number
ExpKey = Union[tuple[()], 'SortKey']
#: (class key, arguments key, exponent key, coefficient), as SymPy's
SortKey = tuple[ClassKey, ArgsKey, ExpKey, Fraction]
#: an element of the content of a node in the ``compare`` order: the key
#: of an argument, the name of a symbol, one of its assumptions, the
#: numerator and denominator of a rational, ...
CompareItem = Union['CompareKey', str, int, Fraction, tuple[str, bool]]
#: ((rank of the class, class name), length of the content, the content)
CompareKey = tuple[tuple[int, str], int, tuple[CompareItem, ...]]

#: the classes which SymPy's ``compare`` orders first, in this order (the
#: names are SymPy's); every other class follows, by name
_CLASS_ORDER: dict[str, int] = {name: i for i, name in enumerate((
    'Zero', 'One', 'Half', 'Infinity', 'NaN', 'NegativeOne', 'NegativeInfinity', 'Integer', 'Rational',
    'Float', 'Exp1', 'Pi', 'ImaginaryUnit', 'Symbol', 'Wild', 'Pow', 'Mul', 'Add', 'Derivative', 'Integral',
    'Abs', 'factorial', 'Lambda', 'Order', 'Equality', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',
    'GreaterThan', 'LessThan'))}


def class_rank(name: str) -> tuple[int, str]:
    """The place of a class (by its SymPy name) in the ``compare`` order.

    Examples
    ========

    >>> from sympy_extras.core.sorting import class_rank
    >>> class_rank('Symbol') < class_rank('Pow') < class_rank('Add') < class_rank('Dummy') < class_rank('sin')
    True
    """
    return (_CLASS_ORDER.get(name, len(_CLASS_ORDER)), name)


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


def compare_key(item: Basic) -> CompareKey:
    """The key of the ``compare`` order (``item.compare_key()``), the
    order of the arguments of sums and products.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Integer, Symbol
    >>> from sympy_extras.core.sorting import compare_key
    >>> x, y = symbols('x y')
    >>> n = Symbol('n', integer=True)
    >>> sorted([n, y, x**2, Integer(2), x*y, x + 1, x], key=compare_key)
    [2, x, y, n, x**2, x*y, x + 1]
    """
    return item.compare_key()


def ordered(items: Iterable[Basic]) -> list[Basic]:
    """The expressions sorted as SymPy's ``ordered`` sorts them: by their
    number of nodes, then by their canonical key.

    Examples
    ========

    >>> from sympy_extras.core import symbols, ordered
    >>> x, y = symbols('x y')
    >>> ordered([x + y, x*y, x**2, y, x])
    [x, y, x**2, x*y, x + y]
    """
    return sorted(items, key=lambda e: (_nodes(e), e.sort_key()))


def _nodes(e: Basic) -> int:
    """The number of nodes of the tree."""
    return 1 + sum(_nodes(a) for a in e.args)
