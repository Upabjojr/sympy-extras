"""The common bases of the associative operations: ``AssocOp`` for
``Add`` and ``Mul``, ``LatticeOp`` for ``And`` and ``Or``.

:class:`AssocOp` carries the identity of the operation and the helpers
``make_args`` (the arguments of a node of the class, ``(expr,)`` for
anything else) and ``_from_args``, as SymPy's ``AssocOp`` [1]_;
:class:`LatticeOp` adds the zero of the operation and the equality of
nodes up to the order of their arguments.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: AssocOp*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.operations
"""
from __future__ import annotations

from typing import ClassVar, Iterable

from .basic import Basic
from .expr import Expr, ExprLike, to_expr

__all__ = ['AssocOp', 'LatticeOp']


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


class LatticeOp(Basic):
    """A join or meet operation (``And``, ``Or``): associative,
    commutative and idempotent, with an identity and a zero, as SymPy's
    ``LatticeOp``. Two nodes with the same arguments in any order are
    equal (the arguments are a set), while ``args`` keeps the order given
    to the constructor.

    Examples
    ========

    >>> from sympy_extras.core import symbols, And, Or
    >>> x, y, z = symbols('x y z')
    >>> And.identity, And.zero, Or.identity, Or.zero
    (True, False, False, True)
    >>> And(y, x) == And(x, y), And(y, x).args, And.make_args(x & y), And.make_args(x)
    (True, (y, x), (x, y), (x,))
    """

    __slots__ = ('_args',)

    #: the argument which leaves the others unchanged (``true`` for ``And``)
    identity: ClassVar[Basic]
    #: the argument which decides the result (``false`` for ``And``)
    zero: ClassVar[Basic]

    def __init__(self, *args: Basic) -> None:
        self._args: tuple[Basic, ...] = args
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return self._args

    def _hashable_content(self) -> tuple[Basic, ...]:
        from .sorting import ordered
        return tuple(ordered(set(self._args)))

    @classmethod
    def make_args(cls, expr: Basic) -> tuple[Basic, ...]:
        """The arguments of ``expr`` when it is a node of this class,
        ``(expr,)`` otherwise."""
        if isinstance(expr, cls):
            return expr.args
        return (expr,)

    @classmethod
    def _from_args(cls, args: Iterable[Basic]) -> Basic:
        """The node with the given arguments, one argument as itself, the
        identity for none."""
        items = list(args)
        if not items:
            return cls.identity
        if len(items) == 1:
            return items[0]
        return cls(*items)
