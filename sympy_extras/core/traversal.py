"""Traversals of the tree and rebuilding helpers, as ``sympy.core.traversal``."""
from __future__ import annotations

from typing import Callable, Iterator

from .basic import Basic, preorder_traversal

__all__ = ['preorder_traversal', 'postorder_traversal', 'iterargs', 'iterfreeargs', 'bottom_up', 'use']


def postorder_traversal(node: Basic) -> Iterator[Basic]:
    """The nodes of the tree, children before parents.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.core.traversal import postorder_traversal
    >>> x, y = symbols('x y')
    >>> list(postorder_traversal(x + 2*y))
    [x, 2, y, 2*y, x + 2*y]
    """
    for a in node.args:
        yield from postorder_traversal(a)
    yield node


def iterargs(node: Basic) -> Iterator[Basic]:
    """The nodes of the tree in breadth first order."""
    queue = [node]
    while queue:
        current = queue.pop(0)
        yield current
        queue.extend(current.args)


def iterfreeargs(node: Basic) -> Iterator[Basic]:
    """The nodes of the tree in breadth first order, skipping the
    arguments which contain a symbol bound by their parent.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Lambda, Tuple
    >>> from sympy_extras.core.traversal import iterfreeargs
    >>> x, y = symbols('x y')
    >>> list(iterfreeargs(Tuple(x, Lambda(y, y + x))))
    [(x, Lambda(y, x + y)), x, Lambda(y, x + y)]
    """
    queue = [node]
    while queue:
        current = queue.pop(0)
        yield current
        bound = set(current.bound_symbols)
        for a in current.args:
            if bound and (a.free_symbols & bound or any(s in a.atoms() for s in bound)):
                continue
            queue.append(a)


def bottom_up(rv: Basic, F: Callable[[Basic], Basic], atoms: bool = False, nonbasic: bool = False) -> Basic:
    """``F`` applied to every node of the tree, children first (the atoms
    only when ``atoms`` is set); the rebuilt nodes are evaluated.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Symbol
    >>> from sympy_extras.core.traversal import bottom_up
    >>> x, y = symbols('x y')
    >>> bottom_up(x + 2*y, lambda e: e**2 if e.is_Symbol else e, atoms=True)
    x**2 + 2*y**2
    """
    if not rv.args:
        return F(rv) if atoms else rv
    args = tuple(bottom_up(a, F, atoms, nonbasic) for a in rv.args)
    if not all(a is b for a, b in zip(args, rv.args)):
        rv = rv._evaluated(args)
    return F(rv)


def use(expr: Basic, func: Callable[[Basic], Basic], level: int = 0) -> Basic:
    """``func`` applied to the nodes at depth ``level`` of the tree (the
    root at level 0); the rebuilt nodes are evaluated.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.core.traversal import use
    >>> x, y = symbols('x y')
    >>> use(x + 2*y, lambda e: e*2, level=1)
    2*x + 4*y
    """
    if level == 0:
        return func(expr)
    if not expr.args:
        return expr
    args = tuple(use(a, func, level - 1) for a in expr.args)
    if all(a is b for a, b in zip(args, expr.args)):
        return expr
    return expr._evaluated(args)
