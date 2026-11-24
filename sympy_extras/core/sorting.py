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

The counts of nodes of ``ordered`` (``_nodes``, ``_node_count``) are
ported from SymPy's ``sympy.core.sorting``.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/sorting.py (_nodes, _node_count) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the nodes are those of the typed core (a Protocol), the
# multiplicities of a derivative are integers.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Protocol, Sequence, TypeVar, Union

__all__ = ['ClassKey', 'ArgsKey', 'SortKey', 'CompareKey', 'CompareItem', 'Sortable', 'default_sort_key', 'compare_key',
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


class Sortable(Protocol):
    """What the keys are taken from: a node of the tree (satisfied by
    :class:`~sympy_extras.core.basic.Basic`, which imports this module)."""

    is_Float: bool

    @property
    def args(self) -> Sequence[Sortable]: ...

    def sort_key(self) -> SortKey: ...

    def compare_key(self) -> CompareKey: ...


_Node = TypeVar('_Node', bound=Sortable)

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


def default_sort_key(item: Sortable) -> SortKey:
    """The canonical key of an expression (``item.sort_key()``).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Integer, default_sort_key
    >>> x, y = symbols('x y')
    >>> sorted([y, x, Integer(2)], key=default_sort_key)
    [2, x, y]
    """
    return item.sort_key()


def compare_key(item: Sortable) -> CompareKey:
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


def ordered(items: Iterable[_Node]) -> list[_Node]:
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


# PORTED FROM SYMPY: sympy/core/sorting.py:_node_count
def _node_count(e: Sortable) -> float:
    """The number of nodes of the tree, a float counting for a half (as
    in SymPy: ``exp(2.5*x)`` sorts before ``exp(2*x)``)."""
    if e.is_Float:
        return 0.5
    return 1 + sum(_node_count(a) for a in e.args)


# PORTED FROM SYMPY: sympy/core/sorting.py:_nodes
def _nodes(e: Sortable) -> float:
    """The number of nodes by which SymPy's ``ordered`` sorts: the nodes
    of the tree (:func:`_node_count`), and for a derivative those of the
    function plus the number of differentiations (``Derivative(f(y), y)``
    counts 3, less than ``exp(y/2)``).

    Both rules were missing (every node counted 1), so that ``ordered``
    differed from SymPy's on expressions with floats and on derivatives.
    """
    from .function import Derivative
    from .numbers import Integer
    if isinstance(e, Derivative):
        # (a symbolic order counts as its nodes)
        return _nodes(e.expr) + sum(count.p if isinstance(count, Integer) else _nodes(count)
                                    for _, count in e.variable_orders)
    return _node_count(e)
