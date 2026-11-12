"""The SymPy face of Kovacic's algorithm
(:mod:`sympy_extras.solvers.kovacic`), and the steps which are not
ported.

:func:`liouvillian_solution`, :func:`dsolve_kovacic` and
:func:`normal_form` take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native functions; the
:class:`KovacicSolution` of :func:`liouvillian_solution` carries SymPy
expressions.

The step of the native module which stays written on SymPy is called
from it at run time (a function ending in ``_of_core``, which takes and
returns objects of the core): :func:`riccati_of_core`, SymPy's rational
Riccati solver (``sympy.solvers.ode.riccati.solve_riccati``), the
fallback of case 1. The antiderivatives are native
(:func:`sympy_extras.integrals.indefinite.closed_antiderivative`).

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x
>>> from sympy_extras.solvers.kovacic_sympy import dsolve_kovacic
>>> y = Function('y')(x)
>>> dsolve_kovacic(y.diff(x, 2) + y.diff(x)/x + (1 - 1/(4*x**2))*y, y)
[exp(I*x)/sqrt(x), exp(-I*x)/sqrt(x)]
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Symbol

from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol, _sympy_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy
from sympy_extras.settings import settings

from . import kovacic as native

__all__ = ['liouvillian_solution', 'dsolve_kovacic', 'normal_form', 'KovacicSolution', 'riccati_of_core']


class KovacicSolution:
    """:class:`sympy_extras.solvers.kovacic.KovacicSolution` with SymPy
    expressions: a Liouvillian solution ``exp(Integral(omega))`` of
    ``z'' = r z``.

    Attributes
    ==========

    case : int
        The case of Kovacic's algorithm (1, 2 or 3).
    omega : Expr or None
        ``omega``, when it can be written explicitly (always in cases 1
        and 2).
    omega_polynomial : Expr
        The polynomial in ``w`` (the symbol ``w``) with rational function
        coefficients whose root ``omega`` is.
    solution : Expr
        ``exp(Integral(omega))`` with the integral computed when
        possible.
    """

    def __init__(self, case: int, omega: Optional[Expr], omega_polynomial: Expr, solution: Expr) -> None:
        self.case = case
        self.omega = omega
        self.omega_polynomial = omega_polynomial
        self.solution = solution

    def __repr__(self) -> str:
        return "KovacicSolution(case=%d, %s)" % (self.case, self.solution)

    @staticmethod
    def of_core(found: native.KovacicSolution) -> KovacicSolution:
        """The solution of the native module with SymPy expressions."""
        return KovacicSolution(found.case, None if found.omega is None else sympy_expr(found.omega),
                               sympy_expr(found.omega_polynomial), sympy_expr(found.solution))


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy





def riccati_of_core(r: CoreExpr, x: CoreSymbol) -> Optional[list[CoreExpr]]:
    """The rational solutions ``omega`` of ``omega' = r - omega**2`` which
    SymPy's ``solve_riccati`` finds under the time limit of the settings
    (``None`` when it does not finish)."""
    from sympy.core.function import Function
    from sympy.core.relational import Eq
    from sympy.core.singleton import S
    from sympy.solvers.ode.riccati import solve_riccati
    variable = _sympy_symbol(x)
    f = Function('f')(variable)
    rhs = sympy_expr(r)
    solutions = attempt(lambda: solve_riccati(f, variable, rhs, S.Zero, S.NegativeOne), settings.timeout)
    if solutions is None:
        return None
    return [core_expr(as_expr(s.rhs if isinstance(s, Eq) else s)) for s in solutions]


# ---------------------------------------------------------------------------
# the face

def _core_basic(e: Basic) -> CoreBasic:
    converted = from_sympy(e)
    if not isinstance(converted, CoreBasic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def normal_form(p: Expr, q: Expr, x: Symbol) -> Expr:
    """``r`` with ``z'' = r z`` the normal form of ``y'' + p y' + q y = 0``
    (:func:`sympy_extras.solvers.kovacic.normal_form` on SymPy objects).

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.kovacic_sympy import normal_form
    >>> normal_form(1/x, 1 - 1/(4*x**2), x)
    -1
    """
    return sympy_expr(native.normal_form(core_expr(as_expr(p)), core_expr(as_expr(q)), _core_symbol(x)))


def liouvillian_solution(r: Expr, x: Symbol) -> Optional[KovacicSolution]:
    """A Liouvillian solution of ``z'' = r z`` (``r`` a rational function
    of ``x``) by Kovacic's algorithm, or ``None`` when there is none
    (:func:`sympy_extras.solvers.kovacic.liouvillian_solution` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Rational
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.kovacic_sympy import liouvillian_solution
    >>> liouvillian_solution(1 + 2/x**2, x)
    KovacicSolution(case=1, (x - 1)*exp(x)/x)
    >>> liouvillian_solution(-3/(16*x**2) + 1/x, x).case
    2
    >>> liouvillian_solution(x, x) is None
    True
    """
    found = native.liouvillian_solution(core_expr(as_expr(r)), _core_symbol(x))
    return None if found is None else KovacicSolution.of_core(found)


def dsolve_kovacic(equation: Basic, f: AppliedUndef, special_values: bool = True) -> Optional[list[Expr]]:
    """The Liouvillian solutions of a second order linear homogeneous
    equation ``a y'' + b y' + c y = 0`` with rational coefficients: a list
    with one or two independent solutions, or ``None``
    (:func:`sympy_extras.solvers.kovacic.dsolve_kovacic` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.kovacic_sympy import dsolve_kovacic
    >>> y = Function('y')(x)
    >>> dsolve_kovacic(y.diff(x, 2) + y.diff(x)/x + (1 - 1/(4*x**2))*y, y)
    [exp(I*x)/sqrt(x), exp(-I*x)/sqrt(x)]
    >>> dsolve_kovacic(y.diff(x, 2) - x*y, y) is None
    True
    """
    found = native.dsolve_kovacic(_core_basic(equation), _core_function(f), special_values)
    return None if found is None else [sympy_expr(e) for e in found]
