"""The unevaluated indefinite integral of the core: ``Integral(f, x)``.

:func:`sympy_extras.integrals.indefinite.indefinite_integral` returns it
when no method gives an antiderivative which checks, as SymPy's
``integrate`` returns its ``Integral``. The node never evaluates (its
``doit`` runs :func:`~sympy_extras.integrals.indefinite.indefinite_integral`),
its derivative in its variable is its integrand, and
:mod:`sympy_extras.interop` converts it to SymPy's ``Integral(f, x)``.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import exp
>>> from sympy_extras.integrals.integral import Integral
>>> x, y = symbols('x y')
>>> F = Integral(exp(x**2)*y, x)
>>> F, F.diff(x), F.diff(y), F.free_symbols == {x, y}
(Integral(y*exp(x**2), x), y*exp(x**2), Integral(exp(x**2), x), True)
"""
from __future__ import annotations

from typing import Final, Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Integer
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.symbol import Symbol

__all__ = ['Integral', 'is_unevaluated']


class Integral(Expr):
    """``Integral(function, variable)``: an indefinite integral left
    unevaluated. The constructor only checks and stores its arguments."""

    __slots__ = ('_function', '_variable')

    def __init__(self, function: Expr, variable: Symbol) -> None:
        if not isinstance(function, Expr):
            raise TypeError("an expression is expected, got %s" % (function,))
        if not isinstance(variable, Symbol):
            raise TypeError("a symbol is expected, got %s" % (variable,))
        self._function: Final[Expr] = function
        self._variable: Final[Symbol] = variable
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function, self._variable)

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def variable(self) -> Symbol:
        return self._variable

    @property
    def free_symbols(self) -> set[Symbol]:
        # an indefinite integral depends on its variable (SymPy's)
        return self._function.free_symbols | {self._variable}

    @staticmethod
    def _checked(args: tuple[Basic, ...]) -> tuple[Expr, Symbol]:
        f, v = args
        if not (isinstance(f, Expr) and isinstance(v, Symbol)):
            raise TypeError("invalid arguments of an Integral: %s" % (args,))
        return f, v

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, v = self._checked(args)
        return Integral(f, v)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self._rebuild(args)

    def _eval_facts(self) -> dict[str, bool]:
        commutative = self._function.is_commutative
        return {} if commutative is None else {'commutative': commutative}

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # the variable is the one of the integration (SymPy's Integral(f,
        # x).subs(x, y) is Integral(f, (x, y)), which is not kept here)
        if self == old:
            return new
        if old == self._variable:
            return self
        f = self._function._subs(old, new)
        if f is self._function:
            return self
        if not isinstance(f, Expr):
            raise TypeError("the substitution made %s a non-expression" % (self,))
        return Integral(f, self._variable)

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        if s == self._variable:
            return self._function
        if s not in self._function.free_symbols:
            return Integer(0)
        # under the integral sign, as SymPy's Integral._eval_derivative
        return Integral(self._function.diff(s), self._variable)

    def doit(self) -> Expr:
        """The integral by
        :func:`~sympy_extras.integrals.indefinite.indefinite_integral`
        (itself again when no antiderivative checks)."""
        from .indefinite import indefinite_integral
        return indefinite_integral(self._function, self._variable)

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Integral(%s, %s)' % (printer.doprint(self._function), printer.doprint(self._variable))


def is_unevaluated(e: Basic) -> bool:
    """Whether ``e`` has an unevaluated integral: an :class:`Integral`, a
    :class:`~sympy_extras.integrals.risch.risch.NonElementaryIntegral` or
    SymPy's ``Integral`` wrapped in a node of the core.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.integral import Integral, is_unevaluated
    >>> x = symbols('x')
    >>> is_unevaluated(x + Integral(x**x, x)), is_unevaluated(x**2/2)
    (True, False)
    """
    from sympy_extras.core.basic import preorder_traversal
    from sympy_extras.integrals.risch.risch import NonElementaryIntegral
    for node in preorder_traversal(e):
        if isinstance(node, (Integral, NonElementaryIntegral)):
            return True
        if node._is_foreign_node() and node._compare_name() in ('Integral', 'NonElementaryIntegral'):
            return True
    return False
