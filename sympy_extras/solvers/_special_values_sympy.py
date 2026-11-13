"""The SymPy face of the isolated values of the parameters of the ODE
solutions (:mod:`sympy_extras.solvers._special_values`), and the step
which is not ported.

:func:`general_solution_cases`, :func:`basis_cases`,
:func:`solution_jacobian`, :func:`explicit_solution`,
:func:`solution_cases` and :func:`as_zero` take and return SymPy objects
(the callbacks too: ``solve_at`` and ``solve`` receive the equation at a
point as a SymPy expression and return SymPy objects), converted by
:mod:`sympy_extras.interop` around the native functions (the ODE solvers
are native and call :mod:`~sympy_extras.solvers._special_values`
directly). Not to be confused with
:mod:`sympy_extras._special_values_sympy`, the face of the candidate points
and their bookkeeping shared with the integrals and the sums.

Examples
========

>>> from sympy import Function, symbols, exp
>>> from sympy_extras.solvers._special_values_sympy import general_solution_cases
>>> x, k, C1 = symbols('x k C1')
>>> y = Function('y')(x)
>>> equation = y.diff(x) - k*y - exp(x)
>>> general_solution_cases(C1*exp(k*x) - exp(x)/(k - 1), equation, y,
...                        lambda e: C1*exp(x) + x*exp(x))
Piecewise((C1*exp(x) + x*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True))
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol

from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.interop import from_sympy, to_sympy

from . import _special_values as native

__all__ = ['general_solution_cases', 'basis_cases', 'solution_jacobian', 'explicit_solution', 'solution_cases',
           'as_zero']


# ---------------------------------------------------------------------------
# the boundary with the core

def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def _core_basic(e: Basic) -> CoreBasic:
    converted = from_sympy(e)
    if not isinstance(converted, CoreBasic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _optional(value: Optional[Expr]) -> Optional[CoreExpr]:
    return None if value is None else core_expr(value)


# ---------------------------------------------------------------------------
# the face

def solution_jacobian(solution: Expr, constants: Sequence[Symbol], x: Symbol) -> Expr:
    """:func:`sympy_extras.solvers._special_values.solution_jacobian` on
    SymPy objects.

    >>> from sympy import symbols, exp, factor
    >>> from sympy_extras.solvers._special_values_sympy import solution_jacobian
    >>> x, a, b, C1, C2 = symbols('x a b C1 C2')
    >>> factor(solution_jacobian(C1*exp(a*x) + C2*exp(b*x), [C1, C2], x))
    -(a - b)*exp(a*x)*exp(b*x)
    """
    return sympy_expr(native.solution_jacobian(core_expr(as_expr(solution)), [_core_symbol(c) for c in constants],
                                               _core_symbol(x)))


def general_solution_cases(general: Expr, equation: Expr, f: AppliedUndef,
                           solve_at: Callable[[Expr], Optional[Expr]]) -> Expr:
    """:func:`sympy_extras.solvers._special_values.general_solution_cases`
    on SymPy objects, ``solve_at`` taking and returning SymPy expressions.

    >>> from sympy import Function, symbols, exp
    >>> from sympy_extras.solvers._special_values_sympy import general_solution_cases
    >>> x, a, b, C1, C2 = symbols('x a b C1 C2')
    >>> y = Function('y')(x)
    >>> equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    >>> general_solution_cases(C1*exp(a*x) + C2*exp(b*x), equation, y,
    ...                        lambda e: (C1 + C2*x)*exp(b*x))
    Piecewise(((C1 + C2*x)*exp(b*x), Eq(a, b)), (C1*exp(a*x) + C2*exp(b*x), True))
    """
    def solve_core(at: CoreExpr) -> Optional[CoreExpr]:
        return _optional(solve_at(sympy_expr(at)))

    return sympy_expr(native.general_solution_cases(core_expr(as_expr(general)), core_expr(as_expr(equation)),
                                                    _core_function(f), solve_core))


def basis_cases(basis: list[Expr], equation: Expr, f: AppliedUndef,
                solve_at: Callable[[Expr], Optional[list[Expr]]]) -> list[Expr]:
    """:func:`sympy_extras.solvers._special_values.basis_cases` on SymPy
    objects, ``solve_at`` taking a SymPy expression and returning SymPy
    expressions.

    >>> from sympy import Function, symbols, exp
    >>> from sympy_extras.solvers._special_values_sympy import basis_cases
    >>> x, a, b = symbols('x a b')
    >>> y = Function('y')(x)
    >>> equation = y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y
    >>> basis_cases([exp(a*x), exp(b*x)], equation, y, lambda e: [exp(b*x), x*exp(b*x)])
    [Piecewise((exp(b*x), Eq(a, b)), (exp(a*x), True)), Piecewise((x*exp(b*x), Eq(a, b)), (exp(b*x), True))]
    """
    if not basis:
        return basis

    def solve_core(at: CoreExpr) -> Optional[list[CoreExpr]]:
        found = solve_at(sympy_expr(at))
        return None if found is None else [core_expr(e) for e in found]

    cases = native.basis_cases([core_expr(as_expr(e)) for e in basis], core_expr(as_expr(equation)),
                               _core_function(f), solve_core)
    return [sympy_expr(e) for e in cases]


def as_zero(equation: Basic) -> Expr:
    """The expression which is zero: ``lhs - rhs`` for an ``Eq``
    (:func:`sympy_extras.solvers._special_values.as_zero`, on SymPy
    objects: SymPy's own subtraction).

    >>> from sympy import Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers._special_values_sympy import as_zero
    >>> as_zero(Eq(x**2, 1))
    x**2 - 1
    """
    if isinstance(equation, Eq):
        return as_expr(equation.lhs - equation.rhs)
    return as_expr(equation)


def explicit_solution(solutions: Sequence[Basic], f: AppliedUndef) -> Optional[Expr]:
    """:func:`sympy_extras.solvers._special_values.explicit_solution` on
    SymPy objects.

    >>> from sympy import Function, Eq, symbols, exp
    >>> from sympy_extras.solvers._special_values_sympy import explicit_solution
    >>> x, C1 = symbols('x C1')
    >>> y = Function('y')(x)
    >>> explicit_solution([Eq(y, C1*exp(x))], y), explicit_solution([Eq(y**2, C1 + x)], y)
    (C1*exp(x), None)
    """
    if len(solutions) != 1:
        return None
    solution = solutions[0]
    if isinstance(solution, Eq) and solution.lhs == f and not solution.rhs.has(f):
        return as_expr(solution.rhs)
    return None


def solution_cases(solutions: Sequence[Basic], equation: Basic, f: AppliedUndef,
                   solve: Callable[[Expr], Sequence[Basic]]) -> Optional[Eq]:
    """:func:`sympy_extras.solvers._special_values.solution_cases` on SymPy
    objects, ``solve`` taking a SymPy expression and returning SymPy
    solutions.

    >>> from sympy import Function, Eq, symbols, exp
    >>> from sympy_extras.solvers._special_values_sympy import solution_cases
    >>> x, k, C1 = symbols('x k C1')
    >>> y = Function('y')(x)
    >>> solution_cases([Eq(y, C1*exp(k*x) - exp(x)/(k - 1))], y.diff(x) - k*y - exp(x), y,
    ...                lambda e: [Eq(y, (C1 + x)*exp(x))])
    Eq(y(x), Piecewise(((C1 + x)*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True)))
    """
    if explicit_solution(solutions, f) is None:
        return None

    def solve_core(at: CoreExpr) -> Sequence[CoreBasic]:
        return [_core_basic(s) for s in solve(sympy_expr(at))]

    found = native.solution_cases([_core_basic(s) for s in solutions], _core_basic(equation), _core_function(f),
                                  solve_core)
    if found is None:
        return None
    converted = to_sympy(found)
    if not isinstance(converted, Eq):
        raise TypeError("%s is not an equation" % (converted,))
    return converted
