"""The SymPy face of the symmetry method for ordinary differential
equations (:mod:`sympy_extras.solvers.ode`).

:func:`ode_symmetries`, :func:`canonical_coordinates`, :func:`reduce_order`,
:func:`dsolve_lie`, :func:`solve_ode` and :func:`check_ode_solution` take
and return SymPy objects (the symmetries of
:mod:`sympy_extras.solvers.lie_sympy`), converted by
:mod:`sympy_extras.interop` around the native functions. The steps of the
native module which stay on SymPy are the ``*_of_core`` functions of the
other faces (:mod:`~sympy_extras.solvers.pde_sympy` for ``dsolve`` and
``solve``, :mod:`~sympy_extras.solvers.kovacic_sympy` for ``integrate``).

Examples
========

>>> from sympy import Function, symbols
>>> from sympy_extras.solvers.ode_sympy import dsolve_lie
>>> x = symbols('x')
>>> y = Function('y')(x)
>>> dsolve_lie(y.diff(x, 2) - y.diff(x)**2/y, y)
[Eq(y(x), exp(C1*x + C2))]
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol

from sympy_extras._typing import as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.relational import Eq as CoreEq
from sympy_extras.interop import from_sympy, to_sympy

from . import lie, ode as native
from .lie_sympy import (Equation, JetSpace, Symmetry, _core_equation, _core_expr, _core_function, _sympy_expr,
                        _sympy_function)

__all__ = ['ode_symmetries', 'canonical_coordinates', 'reduce_order', 'dsolve_lie', 'solve_ode',
           'check_ode_solution', 'ReducedODE']

#: the default ``timeout`` argument: the value of
#: :data:`sympy_extras.settings.settings.timeout` at call time
DEFAULT_TIMEOUT = native.DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# conversions

def _sympy_eq(e: CoreEq) -> Eq:
    converted = to_sympy(e)
    if not isinstance(converted, Eq):
        raise TypeError("expected an equation, got %s" % (converted,))
    return converted


def _sympy_eqs(found: Sequence[CoreEq]) -> list[Eq]:
    return [_sympy_eq(e) for e in found]


def _core_basic(e: Basic) -> CoreBasic:
    return from_sympy(e)


def _native_symmetries(found: Optional[Sequence[Symmetry]]) -> Optional[list[lie.Symmetry]]:
    return None if found is None else [s.native for s in found]


# ---------------------------------------------------------------------------
# the face

def ode_symmetries(ode: Equation, y: AppliedUndef, degree: int = 2,
                   basis: Sequence[Expr] = ()) -> list[Symmetry]:
    """The point symmetries of an ordinary differential equation with
    polynomial infinitesimals of total degree at most ``degree`` (times
    the functions of ``basis``)
    (:func:`sympy_extras.solvers.ode.ode_symmetries` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.ode_sympy import ode_symmetries
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> for X in ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y):
    ...     print(X.generator())
    x*d/dx - y/2*d/dy
    """
    jets = JetSpace([y])
    found = native.ode_symmetries(_core_equation(ode), jets.native.functions[0], degree,
                                  [_core_expr(b) for b in basis])
    return [Symmetry._of_native(jets, s) for s in found]


def canonical_coordinates(symmetry: Symmetry, timeout: Optional[float] = DEFAULT_TIMEOUT
                          ) -> Optional[tuple[Expr, Expr]]:
    """Canonical coordinates `(r, s)` of a symmetry of an ordinary
    differential equation, or ``None``
    (:func:`sympy_extras.solvers.ode.canonical_coordinates` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.ode_sympy import canonical_coordinates, ode_symmetries
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> [X] = ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y)
    >>> canonical_coordinates(X)
    (sqrt(x)*y, log(x))
    """
    found = native.canonical_coordinates(symmetry.native, timeout)
    return None if found is None else (_sympy_expr(found[0]), _sympy_expr(found[1]))


class ReducedODE:
    """The reduction of an ordinary differential equation of order `n`
    by a symmetry (:class:`sympy_extras.solvers.ode.ReducedODE` with SymPy
    attributes).

    Attributes
    ==========

    r, s : Expr
        The canonical coordinates, expressions in the independent and
        dependent variables.
    variable : Symbol
        The symbol standing for ``r``.
    v : applied undefined function
        The unknown ``v(r) = ds/dr`` of the reduced equation.
    ode : Expr
        The reduced equation of order `n - 1` for ``v`` (for `n = 1` the
        expression ``v - G(r)``), equal to zero.
    """

    def __init__(self, r: Expr, s: Expr, variable: Symbol, v: AppliedUndef, ode: Expr) -> None:
        self.r: Final[Expr] = r
        self.s: Final[Expr] = s
        self.variable: Final[Symbol] = variable
        self.v: Final[AppliedUndef] = v
        self.ode: Final[Expr] = ode

    @classmethod
    def _of_native(cls, r: native.ReducedODE) -> ReducedODE:
        return cls(_sympy_expr(r.r), _sympy_expr(r.s), as_symbol(to_sympy(r.variable)), _sympy_function(r.v),
                   _sympy_expr(r.ode))

    def __repr__(self) -> str:
        return "ReducedODE(r=%s, s=%s, ode=%s)" % (self.r, self.s, self.ode)


def reduce_order(ode: Equation, y: AppliedUndef, symmetry: Symmetry,
                 timeout: Optional[float] = DEFAULT_TIMEOUT) -> Optional[ReducedODE]:
    """Reduce the order of an ordinary differential equation with a point
    symmetry, through its canonical coordinates
    (:func:`sympy_extras.solvers.ode.reduce_order` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.ode_sympy import reduce_order
    >>> from sympy_extras.solvers.lie_sympy import JetSpace, Symmetry
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> eq = y.diff(x, 2) - y.diff(x)**2/y
    >>> jets = JetSpace([y])
    >>> reduce_order(eq, y, Symmetry(jets, [0], [x*jets.u[0]]))
    ReducedODE(r=x, s=log(y)/x, ode=Derivative(v(r), r) + 2*v(r)/r)
    """
    found = native.reduce_order(_core_equation(ode), _core_function(y), symmetry.native, timeout)
    return None if found is None else ReducedODE._of_native(found)


def check_ode_solution(ode: Equation, solution: Basic, f: AppliedUndef,
                       timeout: Optional[float] = DEFAULT_TIMEOUT) -> bool:
    """Whether ``solution`` solves the ordinary differential equation, as
    SymPy's ``checkodesol`` decides it
    (:func:`sympy_extras.solvers.ode.check_ode_solution` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Eq, Function, exp, symbols
    >>> from sympy_extras.solvers.ode_sympy import check_ode_solution
    >>> x, C1 = symbols('x C1')
    >>> y = Function('y')(x)
    >>> check_ode_solution(y.diff(x) - y, Eq(y, C1*exp(x)), y)
    True
    """
    return native.check_ode_solution(_core_equation(ode), _core_basic(solution), _core_function(f), timeout)


def dsolve_lie(ode: Equation, y: AppliedUndef, degree: int = 2, basis: Sequence[Expr] = (),
               symmetries_: Optional[Sequence[Symmetry]] = None, check: bool = True,
               timeout: Optional[float] = DEFAULT_TIMEOUT, special_values: bool = True) -> list[Eq]:
    """Solve an ordinary differential equation through its point
    symmetries (:func:`sympy_extras.solvers.ode.dsolve_lie` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.ode_sympy import dsolve_lie
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> dsolve_lie(y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x, y)
    [Eq(y(x), exp(C1*x**2/2 + C2))]
    """
    return _sympy_eqs(native.dsolve_lie(_core_equation(ode), _core_function(y), degree,
                                        [_core_expr(b) for b in basis], _native_symmetries(symmetries_), check,
                                        timeout, special_values))


def solve_ode(ode: Equation, y: AppliedUndef, degree: int = 2, check: bool = True,
              timeout: Optional[float] = DEFAULT_TIMEOUT, special_values: bool = True) -> list[Eq]:
    """Solve an ordinary differential equation with SymPy's ``dsolve``,
    the first order solvers of the package and the point symmetries
    (:func:`sympy_extras.solvers.ode.solve_ode` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, exp, symbols
    >>> from sympy_extras.solvers.ode_sympy import solve_ode
    >>> x, k = symbols('x k')
    >>> y = Function('y')(x)
    >>> solve_ode(y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x, y)
    [Eq(y(x), exp(C1*x**2/2 + C2))]
    >>> solve_ode(y.diff(x) - k*y - exp(x), y)
    [Eq(y(x), Piecewise(((C1 + x)*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True)))]
    """
    return _sympy_eqs(native.solve_ode(_core_equation(ode), _core_function(y), degree, check, timeout,
                                       special_values))


def _first_order(ode: Equation, y: AppliedUndef, check: bool, timeout: Optional[float]) -> Optional[Eq]:
    """The solution of a first order equation by the first order solvers,
    verified when ``check`` (the private step of
    :func:`sympy_extras.solvers.ode.solve_ode` on SymPy objects)."""
    found = native._first_order(_core_equation(ode), _core_function(y), check, timeout)
    return None if found is None else _sympy_eq(found)
