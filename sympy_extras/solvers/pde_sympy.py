"""The SymPy face of the similarity reductions of partial differential
equations (:mod:`sympy_extras.solvers.pde`).

:func:`pde_symmetries`, :func:`similarity_reduction`, :func:`pdsolve_lie`
and :func:`check_pde_solution` take SymPy equations, functions and the
symmetries of :mod:`sympy_extras.solvers.lie_sympy`, converted by
:mod:`sympy_extras.interop` around the native functions. The native
module calls nothing of this module at run time.

Examples
========

>>> from sympy import Function, symbols
>>> from sympy_extras.solvers.pde_sympy import pdsolve_lie
>>> x, t = symbols('x t')
>>> u = Function('u')(x, t)
>>> for s in pdsolve_lie(u.diff(t) - u.diff(x, 2), u):
...     print(s)
Eq(u(x, t), C1 + C2*x)
Eq(u(x, t), C1)
Eq(u(x, t), C1 + C2*erf(x/(2*sqrt(t))))
Eq(u(x, t), C1*exp(-x**2/(4*t))/sqrt(t))
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol

from sympy_extras._typing_sympy import as_expr, as_symbol
from sympy_extras.interop import to_sympy

from . import pde as native
from .lie_sympy import Equation, JetSpace, Symmetry, _core_equation, _core_expr, _core_function, _sympy_expr

__all__ = ['pde_symmetries', 'similarity_reduction', 'pdsolve_lie', 'check_pde_solution', 'Reduction']

#: the default ``timeout`` argument: the value of
#: :data:`sympy_extras.settings.settings.timeout` at call time
DEFAULT_TIMEOUT = native.DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# the face

def _limit(timeout: Optional[float]) -> Optional[float]:
    return native._limit(timeout)


def pde_symmetries(pde: Equation, u: AppliedUndef, degree: int = 2,
                   basis: Sequence[Expr] = (), superposition: bool = False) -> list[Symmetry]:
    """The point symmetries of a partial differential equation with
    polynomial infinitesimals of total degree at most ``degree``
    (:func:`sympy_extras.solvers.pde.pde_symmetries` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.pde_sympy import pde_symmetries
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> for X in pde_symmetries(heat, u, degree=3):
    ...     print(X.generator())
    u*d/du
    d/dt
    d/dx
    x*d/dx + 2*t*d/dt
    t*d/dx - u*x/2*d/du
    t*x*d/dx + t**2*d/dt + (-t*u/2 - u*x**2/4)*d/du
    """
    jets = JetSpace([u])
    found = native.pde_symmetries(_core_equation(pde), jets.native.functions[0], degree,
                                  [_core_expr(b) for b in basis], superposition)
    return [Symmetry._of_native(jets, s) for s in found]


class Reduction:
    """A similarity reduction of a partial differential equation
    (:class:`sympy_extras.solvers.pde.Reduction` with SymPy attributes).

    Attributes
    ==========

    z : Expr
        The similarity variable, an expression in the independent
        variables.
    variable : Symbol
        The symbol standing for ``z`` in the reduced equation.
    F : applied undefined function
        The unknown ``F(z)`` of the reduced equation.
    solution : Expr
        The invariant solution ``u = phi(x, F(z))``.
    ode : Expr
        The reduced ordinary differential equation for ``F``.
    """

    def __init__(self, z: Expr, variable: Symbol, F: AppliedUndef, solution: Expr, ode: Expr) -> None:
        self.z: Final[Expr] = z
        self.variable: Final[Symbol] = variable
        self.F: Final[AppliedUndef] = F
        self.solution: Final[Expr] = solution
        self.ode: Final[Expr] = ode

    @classmethod
    def _of_native(cls, r: native.Reduction) -> Reduction:
        F = to_sympy(r.F)
        if not isinstance(F, AppliedUndef):
            raise TypeError("expected an applied function, got %s" % (F,))
        return cls(_sympy_expr(r.z), as_symbol(to_sympy(r.variable)), F, _sympy_expr(r.solution),
                   _sympy_expr(r.ode))

    def __repr__(self) -> str:
        return "Reduction(z=%s, solution=%s, ode=%s)" % (self.z, self.solution, self.ode)


def similarity_reduction(pde: Equation, u: AppliedUndef, symmetry: Symmetry,
                         timeout: Optional[float] = DEFAULT_TIMEOUT) -> Reduction:
    """Reduce a partial differential equation in two independent
    variables to an ordinary differential equation with a point symmetry
    (:func:`sympy_extras.solvers.pde.similarity_reduction` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.pde_sympy import pde_symmetries, similarity_reduction
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> scaling = [X for X in pde_symmetries(heat, u) if X.xi == (x, 2*t)][0]
    >>> r = similarity_reduction(heat, u, scaling)
    >>> r.z, r.solution
    (x/sqrt(t), F(x/sqrt(t)))
    >>> r.ode
    z*Derivative(F(z), z) + 2*Derivative(F(z), (z, 2))
    """
    return Reduction._of_native(native.similarity_reduction(_core_equation(pde), _core_function(u),
                                                            symmetry.native, timeout))


def check_pde_solution(pde: Equation, u: AppliedUndef, value: Basic,
                       timeout: Optional[float] = DEFAULT_TIMEOUT) -> bool:
    """Whether ``u = value`` (an expression, or ``Eq(u, value)``) solves the
    partial differential equation
    (:func:`sympy_extras.solvers.pde.check_pde_solution` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols, exp, sin, Eq
    >>> from sympy_extras.solvers.pde_sympy import check_pde_solution
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> check_pde_solution(u.diff(t) - u.diff(x, 2), u, Eq(u, exp(-t)*sin(x)))
    True
    """
    expression = as_expr(value.rhs) if isinstance(value, Eq) else as_expr(value)
    return native.check_pde_solution(_core_equation(pde), _core_function(u), _core_expr(expression), timeout)


def pdsolve_lie(pde: Equation, u: AppliedUndef, symmetries_: Optional[Sequence[Symmetry]] = None,
                degree: int = 2, check: bool = True, timeout: Optional[float] = DEFAULT_TIMEOUT) -> list[Eq]:
    """Group invariant solutions of a partial differential equation
    (:func:`sympy_extras.solvers.pde.pdsolve_lie` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.pde_sympy import pdsolve_lie
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> for s in pdsolve_lie(u.diff(t) - u.diff(x, 2), u):
    ...     print(s)
    Eq(u(x, t), C1 + C2*x)
    Eq(u(x, t), C1)
    Eq(u(x, t), C1 + C2*erf(x/(2*sqrt(t))))
    Eq(u(x, t), C1*exp(-x**2/(4*t))/sqrt(t))
    """
    given = None if symmetries_ is None else [s.native for s in symmetries_]
    found = native.pdsolve_lie(_core_equation(pde), _core_function(u), given, degree, check, timeout)
    solutions: list[Eq] = []
    for s in found:
        converted = to_sympy(s)
        if not isinstance(converted, Eq):
            raise TypeError("expected an equation, got %s" % (converted,))
        solutions.append(converted)
    return solutions
