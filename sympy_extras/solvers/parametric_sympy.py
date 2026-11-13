"""The SymPy face of the cases of polynomial systems with parameters
(:mod:`sympy_extras.solvers.parametric`).

:func:`parametric_cases` takes SymPy expressions and symbols and returns
:class:`ParametricCase` objects of SymPy objects (the condition a SymPy
Boolean, the solutions a SymPy ``FiniteSet`` of ``Tuple``), as before the
native port; the work is done by the native function on the objects
converted by :mod:`sympy_extras.interop`. Nothing stays on SymPy.

Examples
========

>>> from sympy.abc import a, b, c, x, y
>>> from sympy_extras.solvers.parametric_sympy import parametric_cases
>>> parametric_cases([a*x - b], [x])
[ParametricCase(Ne(a, 0), {(b/a,)}), ParametricCase(Eq(a, 0) & Eq(b, 0), {(x,)})]
>>> for case in parametric_cases([a*x**2 + b*x + c], [x]):
...     print(case)
ParametricCase(Ne(a, 0), {(-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a),), (-b/(2*a) + sqrt(-4*a*c + b**2)/(2*a),)})
ParametricCase(Eq(a, 0) & Ne(b, 0), {(-c/b,)})
ParametricCase(Eq(a, 0) & Eq(b, 0) & Eq(c, 0), {(x,)})
>>> parametric_cases([a*x + y - 1, x + a*y - 1], [x, y])
[ParametricCase(Eq(a, 1), {(1 - y, y)}), ParametricCase(Ne(a + 1, 0), {(1/(a + 1), 1/(a + 1))})]
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import And, Boolean
from sympy.sets.sets import FiniteSet

from sympy_extras._typing_sympy import as_boolean, as_expr, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import parametric as native

__all__ = ['ParametricCase', 'parametric_cases', 'sympy_case']


def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(as_expr(e))
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(as_symbol(s))
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_expr(e: CoreBasic) -> Expr:
    return as_expr(to_sympy(e))


class ParametricCase:
    """A case of a system with parameters, on SymPy objects
    (:class:`sympy_extras.solvers.parametric.ParametricCase`).

    Attributes
    ==========

    condition : Boolean
        The equations and inequations of the case.
    equations : list of Expr
        The polynomials of the chain in the unknowns, which vanish.
    solutions : FiniteSet or None
        Their solutions, tuples of the unknowns in terms of the parameters;
        ``None`` when an unknown is a root of a polynomial of degree three
        or more.
    """

    __slots__ = ('condition', 'solutions', 'equations')

    def __init__(self, condition: Boolean, solutions: Optional[FiniteSet], equations: list[Expr]) -> None:
        self.condition = condition
        self.solutions = solutions
        self.equations = equations

    def __repr__(self) -> str:
        shown = self.solutions if self.solutions is not None else [Eq(e, 0) for e in self.equations]
        return "ParametricCase(%s, %s)" % (self.condition, shown)

    def as_formula(self) -> Boolean:
        """The case as a formula in polynomials of the unknowns and the
        parameters."""
        return as_boolean(And(self.condition, *[Eq(e, 0) for e in self.equations]))


def sympy_case(case: native.ParametricCase) -> ParametricCase:
    """The case of the face of a native case."""
    solutions: Optional[FiniteSet] = None
    if case.solutions is not None:
        converted = to_sympy(case.solutions)
        if not isinstance(converted, FiniteSet):
            raise TypeError("expected a finite set, got %s" % (converted,))
        solutions = converted
    return ParametricCase(as_boolean(to_sympy(case.condition)), solutions,
                          [_sympy_expr(e) for e in case.equations])


def parametric_cases(equations: Iterable[Expr], unknowns: Sequence[Symbol],
                     parameters: Optional[Sequence[Symbol]] = None,
                     inequations: Iterable[Expr] = ()) -> list[ParametricCase]:
    """The complex solutions of polynomial ``equations`` with rational
    coefficients in the ``unknowns``, for every value of the ``parameters``
    (:func:`sympy_extras.solvers.parametric.parametric_cases` on SymPy
    objects)."""
    found = native.parametric_cases([_core_expr(e) for e in equations], [_core_symbol(s) for s in unknowns],
                                    None if parameters is None else [_core_symbol(s) for s in parameters],
                                    [_core_expr(h) for h in inequations])
    return [sympy_case(case) for case in found]
