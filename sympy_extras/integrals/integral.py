"""The unevaluated integral of the core: ``Integral(f, x)`` and
``Integral(f, (x, a, b))``.

:func:`sympy_extras.integrals.definite.definite_integral` returns the
definite one when no method gives a value, as SymPy's ``integrate``
returns its ``Integral``; the indefinite one is the integral which no
antiderivative checks. The node never evaluates (its ``doit`` runs the
integrator), the derivative of an indefinite integral in its variable is
its integrand, and :mod:`sympy_extras.interop` converts it to SymPy's
``Integral(f, x)`` or ``Integral(f, (x, a, b))``.

Examples
========

>>> from sympy_extras.core import symbols, Integer, oo
>>> from sympy_extras.functions import exp
>>> from sympy_extras.integrals.integral import Integral
>>> x, y = symbols('x y')
>>> F = Integral(exp(x**2)*y, x)
>>> F, F.diff(x), F.diff(y), F.free_symbols == {x, y}
(Integral(y*exp(x**2), x), y*exp(x**2), Integral(exp(x**2), x), True)
>>> G = Integral(exp(-y*x**2), x, Integer(0), oo)
>>> G, G.free_symbols == {y}, G.subs(y, 2)
(Integral(exp(-x**2*y), (x, 0, oo)), True, Integral(exp(-2*x**2), (x, 0, oo)))
>>> G.subs(y, 1).evalf(20)
0.88622692545275801365

A definite integral without free symbols is a number, evaluated by
mpmath's tanh-sinh quadrature (:mod:`sympy_extras.integrals._evalf`, a
port of SymPy's ``evalf_integral``).
"""
from __future__ import annotations

from typing import Final, Optional

import mpmath

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Integer
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.symbol import Symbol

__all__ = ['Integral', 'is_unevaluated']


class Integral(Expr):
    """``Integral(function, variable)``, an indefinite integral, or
    ``Integral(function, variable, lower, upper)``, the definite integral
    over ``(lower, upper)``, left unevaluated. The constructor only checks
    and stores its arguments."""

    __slots__ = ('_function', '_variable', '_bounds')

    def __init__(self, function: Expr, variable: Symbol, lower: Optional[Expr] = None,
                 upper: Optional[Expr] = None) -> None:
        if not isinstance(function, Expr):
            raise TypeError("an expression is expected, got %s" % (function,))
        if not isinstance(variable, Symbol):
            raise TypeError("a symbol is expected, got %s" % (variable,))
        if (lower is None) != (upper is None):
            raise TypeError("both bounds or neither are expected")
        self._function: Final[Expr] = function
        self._variable: Final[Symbol] = variable
        self._bounds: Final[Optional[tuple[Expr, Expr]]] = \
            None if lower is None or upper is None else (lower, upper)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        if self._bounds is None:
            return (self._function, self._variable)
        return (self._function, self._variable, self._bounds[0], self._bounds[1])

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def variable(self) -> Symbol:
        return self._variable

    @property
    def bounds(self) -> Optional[tuple[Expr, Expr]]:
        """``(lower, upper)`` for a definite integral, ``None`` otherwise."""
        return self._bounds

    @property
    def free_symbols(self) -> set[Symbol]:
        if self._bounds is None:
            # an indefinite integral depends on its variable (SymPy's)
            return self._function.free_symbols | {self._variable}
        lower, upper = self._bounds
        return (self._function.free_symbols - {self._variable}) | lower.free_symbols | upper.free_symbols

    @staticmethod
    def _checked(args: tuple[Basic, ...]) -> tuple[Expr, Symbol, Optional[Expr], Optional[Expr]]:
        if len(args) == 2:
            f, v = args
            if isinstance(f, Expr) and isinstance(v, Symbol):
                return f, v, None, None
        elif len(args) == 4:
            f, v, a, b = args
            if isinstance(f, Expr) and isinstance(v, Symbol) and isinstance(a, Expr) and isinstance(b, Expr):
                return f, v, a, b
        raise TypeError("invalid arguments of an Integral: %s" % (args,))

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, v, a, b = self._checked(args)
        return Integral(f, v, a, b)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self._rebuild(args)

    def _is_number_node(self) -> Optional[bool]:
        # SymPy's Integral.is_number: a definite integral without free
        # symbols is a number (its variable is bound)
        return not self.free_symbols if self._bounds is not None else False

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        # PORTED FROM SYMPY: sympy/core/evalf.py:evalf_integral (sympy_extras.integrals._evalf)
        if self._bounds is None or self.free_symbols:
            return None
        from ._evalf import evalf_integral
        return evalf_integral(self._function, self._variable, self._bounds[0], self._bounds[1], prec)

    def _eval_facts(self) -> dict[str, bool]:
        commutative = self._function.is_commutative
        return {} if commutative is None else {'commutative': commutative}

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # the variable is the one of the integration (SymPy's Integral(f,
        # x).subs(x, y) is Integral(f, (x, y)), which is not kept here)
        if self == old:
            return new
        f: Basic = self._function if old == self._variable else self._function._subs(old, new)
        if self._bounds is None:
            if old == self._variable or f is self._function:
                return self
            if not isinstance(f, Expr):
                raise TypeError("the substitution made %s a non-expression" % (self,))
            return Integral(f, self._variable)
        a, b = self._bounds[0]._subs(old, new), self._bounds[1]._subs(old, new)
        if f is self._function and a is self._bounds[0] and b is self._bounds[1]:
            return self
        if not (isinstance(f, Expr) and isinstance(a, Expr) and isinstance(b, Expr)):
            raise TypeError("the substitution made %s a non-expression" % (self,))
        return Integral(f, self._variable, a, b)

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        if self._bounds is None:
            if s == self._variable:
                return self._function
            if s not in self._function.free_symbols:
                return Integer(0)
            # under the integral sign, as SymPy's Integral._eval_derivative
            return Integral(self._function.diff(s), self._variable)
        if s not in self.free_symbols:
            return Integer(0)
        # Leibniz's rule
        lower, upper = self._bounds
        result: Expr = Integer(0)
        if s in self._function.free_symbols and s != self._variable:
            result = Integral(self._function.diff(s), self._variable, lower, upper)
        if s in upper.free_symbols:
            result = result + self._function.subs(self._variable, upper) * upper.diff(s)
        if s in lower.free_symbols:
            result = result - self._function.subs(self._variable, lower) * lower.diff(s)
        return result

    def doit(self) -> Expr:
        """The integral by
        :func:`~sympy_extras.integrals.indefinite.indefinite_integral` or
        :func:`~sympy_extras.integrals.definite.definite_integral` (itself
        again when it is not found)."""
        if self._bounds is None:
            from .indefinite import indefinite_integral
            return indefinite_integral(self._function, self._variable)
        from .definite import definite_integral
        return definite_integral(self._function, (self._variable, self._bounds[0], self._bounds[1]))

    def _sympystr(self, printer: StrPrinter) -> str:
        if self._bounds is None:
            return 'Integral(%s, %s)' % (printer.doprint(self._function), printer.doprint(self._variable))
        return 'Integral(%s, (%s, %s, %s))' % (printer.doprint(self._function), printer.doprint(self._variable),
                                              printer.doprint(self._bounds[0]), printer.doprint(self._bounds[1]))


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
