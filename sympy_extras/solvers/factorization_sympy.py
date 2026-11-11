"""The SymPy face of the factorisation of linear differential operators
(:mod:`sympy_extras.solvers.factorization`), and the step which is not
ported.

:func:`right_factor`, :func:`left_factor`, :func:`factor_operator`,
:func:`variation_of_parameters` and :func:`solve_factored` take and
return the SymPy :class:`~sympy_extras.solvers.linear_ode_sympy.LinearOperator`
and SymPy expressions (the solver given to :func:`solve_factored` too),
converted by :mod:`sympy_extras.interop` around the native functions;
:class:`OperatorFactorization` carries SymPy operators.

The step of the native module which stays written on SymPy is called from
it at run time (it takes and returns objects of the core):

- :func:`solve_of_core`: SymPy's ``solve`` on the quadratic
  Grassmann–Plücker relations of Beke's method, when several
  hyperexponential solutions of an associated equation differ by rational
  factors.

Examples
========

>>> from sympy.abc import x
>>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
>>> from sympy_extras.solvers.factorization_sympy import factor_operator
>>> F = factor_operator(LinearOperator([0, -1, 0, 1], x))
>>> F
OperatorFactorization(1, [LinearOperator([-1, 1], x), LinearOperator([1, 1], x), LinearOperator([0, 1], x)], [True, True, True])
>>> F.expand()
LinearOperator([0, -1, 0, 1], x)
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import Truth, as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol, _sympy_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy
from sympy_extras.settings import settings

from . import factorization as native
from . import linear_ode as native_ode
from .linear_ode_sympy import LinearOperator

__all__ = ['OperatorFactorization', 'right_factor', 'left_factor', 'factor_operator',
           'variation_of_parameters', 'solve_factored', 'solve_of_core']

#: a solver of homogeneous equations: a list of independent solutions
Solver = Callable[[LinearOperator], list[Expr]]


# ---------------------------------------------------------------------------
# the step of the native module which stays on SymPy

def solve_of_core(equations: Sequence[CoreExpr], unknowns: Sequence[CoreSymbol]
                  ) -> Optional[list[dict[CoreSymbol, CoreExpr]]]:
    """The solutions of a polynomial system which SymPy's ``solve`` finds
    (``dict=True``) under the time limit of the settings, ``None`` when it
    does not finish."""
    from sympy.solvers.solvers import solve
    system = [sympy_expr(e) for e in equations]
    symbols = [_sympy_symbol(u) for u in unknowns]
    found = attempt(lambda: [{k: as_expr(v) for k, v in d.items() if isinstance(k, Basic)}
                             for d in solve(system, symbols, dict=True)], settings.timeout)
    if found is None:
        return None
    result: list[dict[CoreSymbol, CoreExpr]] = []
    for d in found:
        converted: dict[CoreSymbol, CoreExpr] = {}
        for k, v in d.items():
            key = from_sympy(k)
            if isinstance(key, CoreSymbol):
                converted[key] = core_expr(v)
        result.append(converted)
    return result


# ---------------------------------------------------------------------------
# the face

class OperatorFactorization:
    """``L = unit * factors[0] * ... * factors[-1]`` with monic factors
    (SymPy operators); ``irreducible[i]`` is ``True`` when the ``i``-th
    factor is proven irreducible and ``None`` when a search was not
    exhaustive."""

    def __init__(self, unit: Expr, factors: list[LinearOperator], irreducible: list[Truth], x: Symbol) -> None:
        self.unit = unit
        self.factors = factors
        self.irreducible = irreducible
        self.x = x

    @staticmethod
    def of_core(F: native.OperatorFactorization) -> OperatorFactorization:
        """The factorisation of the native module with SymPy operators."""
        return OperatorFactorization(sympy_expr(F.unit), [LinearOperator.of_core(f) for f in F.factors],
                                     list(F.irreducible), _sympy_symbol(F.x))

    def expand(self) -> LinearOperator:
        """The product ``unit * factors[0] * ... * factors[-1]``."""
        product = LinearOperator([self.unit], self.x)
        for factor in self.factors:
            product = product*factor
        return product

    @property
    def is_complete(self) -> bool:
        """Whether every factor is proven irreducible."""
        return all(flag is True for flag in self.irreducible)

    def __repr__(self) -> str:
        return "OperatorFactorization(%s, %s, %s)" % (self.unit, self.factors, self.irreducible)


def _optional(L: Optional[native_ode.LinearOperator]) -> Optional[LinearOperator]:
    return None if L is None else LinearOperator.of_core(L)


def right_factor(L: LinearOperator, order: int) -> Optional[LinearOperator]:
    """A monic right factor of ``L`` of the given order, or ``None``
    (:func:`sympy_extras.solvers.factorization.right_factor`).

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> from sympy_extras.solvers.factorization_sympy import right_factor
    >>> D = LinearOperator([0, 1], x)
    >>> A = D*D - LinearOperator([2/x**2], x)
    >>> L = (D - LinearOperator([1/x], x))*A
    >>> right_factor(L, 2) == A
    True
    """
    return _optional(native.right_factor(L.core(), order))


def left_factor(L: LinearOperator, order: int) -> Optional[LinearOperator]:
    """A monic left factor of ``L`` of the given order, or ``None``
    (:func:`sympy_extras.solvers.factorization.left_factor`).

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> from sympy_extras.solvers.factorization_sympy import left_factor
    >>> D = LinearOperator([0, 1], x)
    >>> L = (D - LinearOperator([1/x], x))*(D*D + LinearOperator([1], x))
    >>> left_factor(L, 1)
    LinearOperator([-1/x, 1], x)
    """
    return _optional(native.left_factor(L.core(), order))


def factor_operator(L: LinearOperator) -> OperatorFactorization:
    """A factorisation of ``L`` into monic factors
    (:func:`sympy_extras.solvers.factorization.factor_operator`).

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> from sympy_extras.solvers.factorization_sympy import factor_operator
    >>> D = LinearOperator([0, 1], x)
    >>> A = LinearOperator([1, 1/x, 1], x)
    >>> F = factor_operator(A*(D - LinearOperator([1/x], x)))
    >>> F.factors
    [LinearOperator([1, 1/x, 1], x), LinearOperator([-1/x, 1], x)]
    >>> F.irreducible
    [True, True]
    """
    return OperatorFactorization.of_core(native.factor_operator(L.core()))


def variation_of_parameters(L: LinearOperator, basis: list[Expr], rhs: Expr) -> Expr:
    """A solution of ``L(y) = rhs`` from a basis of the solutions of
    ``L(y) = 0`` (:func:`sympy_extras.solvers.factorization.variation_of_parameters`).

    >>> from sympy import exp, simplify
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> from sympy_extras.solvers.factorization_sympy import variation_of_parameters
    >>> L = LinearOperator([-1, 0, 1], x)
    >>> y = variation_of_parameters(L, [exp(x), exp(-x)], x)
    >>> simplify(L(y) - x)
    0
    """
    return sympy_expr(native.variation_of_parameters(L.core(), [core_expr(as_expr(b)) for b in basis],
                                                     core_expr(as_expr(rhs))))


def solve_factored(factors: list[LinearOperator], solver: Solver) -> list[Expr]:
    """Independent solutions of ``factors[0] * ... * factors[-1]``
    (:func:`sympy_extras.solvers.factorization.solve_factored`; ``solver``
    takes and returns SymPy objects).

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, hyperexponential_solutions
    >>> from sympy_extras.solvers.factorization_sympy import solve_factored
    >>> D = LinearOperator([0, 1], x)
    >>> solve_factored([D, D - LinearOperator([1/x], x)], hyperexponential_solutions)
    [x, x*log(x)]
    """
    def native_solver(M: native_ode.LinearOperator) -> list[CoreExpr]:
        return [core_expr(as_expr(e)) for e in solver(LinearOperator.of_core(M))]

    return [sympy_expr(e) for e in native.solve_factored([f.core() for f in factors], native_solver)]


# ---------------------------------------------------------------------------
# the private helpers on SymPy objects (the old linear_systems used them;
# the tests)

def _quadrature(integrand: Expr, x: Symbol) -> Expr:
    """``Integral(integrand, x)``, evaluated when SymPy can within the time
    limit (:func:`sympy_extras.solvers.factorization._quadrature`)."""
    return sympy_expr(native._quadrature(core_expr(as_expr(integrand)), _core_symbol(x)))


def _rational_ratio(d: Expr, x: Symbol) -> tuple[Optional[Expr], bool]:
    f, decided = native._rational_ratio(core_expr(as_expr(d)), _core_symbol(x))
    return (None if f is None else sympy_expr(f)), decided

