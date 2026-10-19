"""Decorators of the operators: ``call_highest_priority``.

An operator of ``Expr`` defers to the reflected operator of its other
operand when that operand has a higher ``_op_priority`` (the mechanism
by which SymPy's matrices and other containers take over the arithmetic
with scalars); the decorator applies it to a method of a subclass.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from .expr import Expr

__all__ = ['call_highest_priority']

E = TypeVar('E', bound=Expr)


def call_highest_priority(method_name: str) -> Callable[[Callable[[E, Expr], Expr]], Callable[[E, Expr], Expr]]:
    """Decorate a binary operator so that it defers to ``method_name`` of
    an operand of higher priority.

    Examples
    ========

    >>> from sympy_extras.core import Expr, Symbol
    >>> from sympy_extras.core.decorators import call_highest_priority
    >>> class Loud(Expr):
    ...     __slots__ = ()
    ...     _op_priority = 20.0
    ...     def __init__(self) -> None:
    ...         Expr.__init__(self)
    ...     @call_highest_priority('__radd__')
    ...     def __add__(self, other: Expr) -> Expr:
    ...         return Symbol('loud')
    ...     @call_highest_priority('__add__')
    ...     def __radd__(self, other: Expr) -> Expr:
    ...         return Symbol('loud_r')
    >>> x = Symbol('x')
    >>> x + Loud(), Loud() + x
    (loud_r, loud)
    """
    def decorator(method: Callable[[E, Expr], Expr]) -> Callable[[E, Expr], Expr]:
        def wrapped(self: E, other: Expr) -> Expr:
            if isinstance(other, Expr) and other._op_priority > self._op_priority:
                reflected: Callable[[Expr], Expr] = _bound_operator(other, method_name)
                return reflected(self)
            return method(self, other)
        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        return wrapped
    return decorator


_OPERATORS: dict[str, Callable[[Expr, Expr], Expr]] = {
    '__add__': lambda a, b: a.__add__(b), '__radd__': lambda a, b: a.__radd__(b),
    '__sub__': lambda a, b: a.__sub__(b), '__rsub__': lambda a, b: a.__rsub__(b),
    '__mul__': lambda a, b: a.__mul__(b), '__rmul__': lambda a, b: a.__rmul__(b),
    '__truediv__': lambda a, b: a.__truediv__(b), '__rtruediv__': lambda a, b: a.__rtruediv__(b),
    '__pow__': lambda a, b: a.__pow__(b), '__rpow__': lambda a, b: a.__rpow__(b),
    '__mod__': lambda a, b: a.__mod__(b), '__rmod__': lambda a, b: a.__rmod__(b),
}


def _bound_operator(e: Expr, name: str) -> Callable[[Expr], Expr]:
    if name not in _OPERATORS:
        raise ValueError("unknown operator %r" % (name,))
    operator = _OPERATORS[name]
    return lambda other: operator(e, other)
