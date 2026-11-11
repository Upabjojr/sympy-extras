"""The SymPy face of Charpit's method
(:mod:`sympy_extras.solvers.charpit`), and the two steps of it which stay
on SymPy.

:func:`complete_integral` and :func:`check_complete_integral` take SymPy
equations and functions, converted by :mod:`sympy_extras.interop` around
the native functions. The native module calls two functions of this
module at run time (they take and return objects of the core):

* :func:`solve_of_core`, SymPy's ``solve`` for the equations which are
  not polynomial in the unknown (radicals, exponentials and logarithms of
  it), which the native polynomial roots do not cover;
* :func:`antiderivative_of_core`, SymPy's ``integrate`` for the
  integrands which are not sums of powers of the variable.

Examples
========

>>> from sympy import Function, symbols, Eq
>>> from sympy_extras.solvers.charpit_sympy import complete_integral
>>> x, y = symbols('x y')
>>> u = Function('u')(x, y)
>>> p, q = u.diff(x), u.diff(y)
>>> complete_integral(p*q - 1, u)
Eq(u(x, y), a*x + b + y/a)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral, integrate
from sympy.solvers.solvers import solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr
from sympy_extras.assumptions.analysis_sympy import as_symbol_of
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.core.relational import Eq as CoreEq
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.settings import settings

from . import charpit as native

__all__ = ['complete_integral', 'check_complete_integral', 'solve_of_core', 'antiderivative_of_core']


# ---------------------------------------------------------------------------
# the steps of the native method which stay on SymPy

def solve_of_core(expression: CoreExpr, variable: CoreSymbol) -> list[CoreExpr]:
    """The solutions of ``expression = 0`` for ``variable`` by SymPy's
    ``solve`` (under the time limit; none when it gives up)."""
    sympy_expression, sympy_variable = sympy_expr(expression), as_symbol_of(variable)
    found = attempt(lambda: solve(sympy_expression, sympy_variable), settings.timeout)
    if not found:
        return []
    return [core_expr(v) for v in found if isinstance(v, Expr)]


def antiderivative_of_core(e: CoreExpr, x: CoreSymbol) -> Optional[CoreExpr]:
    """An antiderivative of ``e`` in ``x`` by SymPy's ``integrate`` (under
    the time limit), ``None`` when it is not found."""
    sympy_e, sympy_x = sympy_expr(e), as_symbol_of(x)
    integral = attempt(lambda: as_expr(integrate(sympy_e, sympy_x)), settings.timeout)
    if integral is None or integral.has(Integral):
        return None
    return core_expr(integral)


# ---------------------------------------------------------------------------
# the face

def _core_basic(e: Basic) -> CoreBasic:
    return from_sympy(e)


def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("an applied undefined function is expected, got %s" % (f,))
    return converted


def complete_integral(equation: Basic, f: AppliedUndef, constants: tuple[Symbol, Symbol] = (Symbol('a'), Symbol('b'))
                      ) -> Optional[Basic]:
    """A complete integral ``Eq(f, phi(x, y, a, b))`` of a first order
    partial differential equation in ``f(x, y)``, or ``None``
    (:func:`sympy_extras.solvers.charpit.complete_integral` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols, Eq
    >>> from sympy_extras.solvers.charpit_sympy import complete_integral
    >>> x, y = symbols('x y')
    >>> u = Function('u')(x, y)
    >>> p, q = u.diff(x), u.diff(y)
    >>> complete_integral(p*q - 1, u)
    Eq(u(x, y), a*x + b + y/a)
    >>> complete_integral(Eq(u, p*x + q*y + p*q), u)
    Eq(u(x, y), a*b + a*x + b*y)
    >>> complete_integral(p**2 + q**2 - u, u)
    Eq(u(x, y), (a*y + b + x)**2/(4*(a**2 + 1)))
    >>> complete_integral(p**2 + x - q - y**2, u)
    Eq(u(x, y), a*y + b - y**3/3 + 2*(a - x)**(3/2)/3)
    """
    a, b = constants
    solution = native.complete_integral(_core_basic(equation), _core_function(f), (core_symbol(a), core_symbol(b)))
    if solution is None:
        return None
    converted = to_sympy(solution)
    if not isinstance(converted, Eq):
        raise TypeError("expected an equation, got %s" % (converted,))
    return converted


def check_complete_integral(equation: Basic, f: AppliedUndef, solution: Basic) -> bool:
    """Whether ``solution`` (an equation ``f == expression`` or the
    expression) satisfies the partial differential equation
    (:func:`sympy_extras.solvers.charpit.check_complete_integral` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols, Eq
    >>> from sympy_extras.solvers.charpit_sympy import check_complete_integral
    >>> x, y, a, b = symbols('x y a b')
    >>> u = Function('u')(x, y)
    >>> check_complete_integral(u.diff(x)*u.diff(y) - 1, u, Eq(u, a*x + y/a + b))
    True
    """
    converted = _core_basic(solution)
    if not isinstance(converted, (CoreEq, CoreExpr)):
        raise TypeError("an equation or an expression is expected, got %s" % (solution,))
    return native.check_complete_integral(_core_basic(equation), _core_function(f), converted)
