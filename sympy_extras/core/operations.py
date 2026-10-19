"""The common base of the associative operations ``Add`` and ``Mul``.

:class:`AssocOp` carries the identity of the operation and the helpers
``make_args`` (the arguments of a node of the class, ``(expr,)`` for
anything else) and ``_from_args``, as SymPy's ``AssocOp`` [1]_.
``LatticeOp`` (the base of ``And`` and ``Or``) belongs to the logic
module of SymPy and is not part of this core.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: AssocOp*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.operations
"""
from __future__ import annotations

from typing import ClassVar, Iterable

from .expr import Expr, ExprLike, to_expr

__all__ = ['AssocOp']


class AssocOp(Expr):
    """An associative operation with an identity.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Add, Mul
    >>> x, y = symbols('x y')
    >>> Add.identity, Mul.identity, Add.make_args(x + y), Add.make_args(x), Mul.make_args(x*y)
    (0, 1, (x, y), (x,), (x, y))
    """

    __slots__ = ('_args',)

    #: the identity of the operation (``0`` for ``Add``, ``1`` for ``Mul``)
    identity: ClassVar[Expr]

    def __init__(self, *args: ExprLike) -> None:
        from .basic import Basic
        self._args: tuple[Expr, ...] = tuple(to_expr(a) for a in args)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Expr, ...]:
        return self._args

    @classmethod
    def make_args(cls, expr: ExprLike) -> tuple[Expr, ...]:
        """The arguments of ``expr`` when it is a node of this class,
        ``(expr,)`` otherwise."""
        e = to_expr(expr)
        if isinstance(e, cls):
            return e.args
        return (e,)

    @classmethod
    def _from_args(cls, args: Iterable[ExprLike]) -> Expr:
        """The node with the given arguments, one argument as itself."""
        items = [to_expr(a) for a in args]
        if not items:
            return cls.identity
        if len(items) == 1:
            return items[0]
        return cls(*items)
