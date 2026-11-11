"""The SymPy face of the linear systems ``Y' = A(x) Y + b(x)``
(:mod:`sympy_extras.solvers.linear_systems`), and the steps which are not
ported.

The functions take and return SymPy matrices (``Matrix``), converted
entry by entry by :mod:`sympy_extras.interop` to the
:class:`~sympy_extras.matrices.dense.ExprMatrix` of the native module and
back; :class:`CyclicReduction` and :class:`LinearSystemSolution` hold
SymPy matrices and the SymPy
:class:`~sympy_extras.solvers.linear_ode_sympy.LinearOperator`.

The steps of the native module which stay written on SymPy are called
from it at run time (they take and return objects of the core):

- :func:`rational_integral_of_core`: the antiderivative of a rational
  function by SymPy's ``ratint_ratpart`` (Hermite's reduction) and
  ``ratint_logpart`` (Rothstein–Trager), with a ``RootSum`` for the
  factors of degree three and more;
- :func:`linodesolve_of_core`: SymPy's ``linodesolve``, the last resort
  for a fundamental matrix (systems with a commuting antiderivative).

The antiderivatives of Liouville's formula and of variation of constants
go through :func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core`
and :func:`~sympy_extras.solvers.kovacic_sympy.integral_of_core`.

Examples
========

>>> from sympy import Matrix
>>> from sympy.abc import x
>>> from sympy_extras.solvers.linear_systems_sympy import dsolve_system
>>> S = dsolve_system(Matrix([[1/x, 1], [0, 1/x]]), x, Matrix([1, 0]))
>>> S.fundamental
Matrix([[x, x**2], [0, x]])
>>> S.particular.T
Matrix([[x*log(x), 0]])
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.matrices.matrixbase import MatrixBase

from sympy_extras._timeout import attempt
from sympy_extras._typing import Truth, as_expr
from sympy_extras.assumptions.analysis_sympy import as_symbol_of
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.settings import settings

from . import linear_systems as native
from .dae_sympy import core_matrix, sympy_matrix
from .linear_ode_sympy import LinearOperator

__all__ = ['CyclicReduction', 'LinearSystemSolution', 'cyclic_vector', 'cyclic_reduction', 'system_to_scalar',
           'rational_system_solutions', 'hyperexponential_system_solutions', 'dsolve_system',
           'dsolve_linear_system', 'rational_integral_of_core', 'linodesolve_of_core']


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy

def rational_integral_of_core(f: CoreExpr, x: CoreSymbol) -> CoreExpr:
    """An antiderivative of a rational function: the polynomial and
    rational parts (Hermite's reduction) and the logarithmic part of
    Rothstein and Trager, with the roots of its factors of degree at most
    two written out and a ``RootSum`` for the others (SymPy's
    ``integrate`` writes them in radicals, nested ones for quartics)."""
    from sympy.core.function import Lambda
    from sympy.core.symbol import Dummy
    from sympy.functions.elementary.exponential import log
    from sympy.integrals.rationaltools import ratint_logpart, ratint_ratpart
    from sympy.polys.polyroots import roots
    from sympy.polys.polytools import Poly, cancel, factor_list
    from sympy.polys.rootoftools import RootSum
    variable = as_symbol_of(x)
    numerator, denominator = as_expr(cancel(sympy_expr(f))).as_numer_denom()
    P = Poly(numerator, variable)
    Q = Poly(denominator, variable)
    quotient, remainder = P.div(Q)
    result = as_expr(quotient.integrate().as_expr())
    if remainder.is_zero:
        return core_expr(result)
    rational, logarithmic = ratint_ratpart(remainder, Q, variable)
    result = as_expr(result + rational)
    top, bottom = as_expr(cancel(as_expr(logarithmic))).as_numer_denom()
    if top == 0:
        return core_expr(result)
    t = Dummy('t')
    for argument, resultant_poly in ratint_logpart(Poly(top, variable), Poly(bottom, variable), variable, t):
        S_ = as_expr(argument.as_expr())
        _, factors = factor_list(as_expr(resultant_poly.as_expr()), t)
        for factor, _ in factors:
            F = Poly(factor, t)
            if F.degree() == 0:
                continue
            if F.degree() <= 2:
                for a in roots(F):
                    result = as_expr(result + a*log(as_expr(S_.subs(t, a))))
            else:
                result = as_expr(result + RootSum(F, Lambda(t, t*log(S_))))
    return core_expr(result)


def linodesolve_of_core(A: ExprMatrix, x: CoreSymbol) -> Optional[list[CoreExpr]]:
    """The general solution vector of ``Y' = A Y`` given by SymPy's
    ``linodesolve`` under the time limit of the settings (in its constants
    ``C1, C2, ...``), ``None`` when it finds none."""
    from sympy.solvers.ode.systems import linodesolve
    matrix = sympy_matrix(A)
    variable = as_symbol_of(x)
    result = attempt(lambda: linodesolve(matrix, variable), settings.timeout)
    if result is None:
        return None
    return [core_expr(as_expr(e)) for e in result]


# ---------------------------------------------------------------------------
# the face

class CyclicReduction:
    """The scalar form of ``Y' = A Y`` given by a cyclic row vector
    ``vector`` (:class:`sympy_extras.solvers.linear_systems.CyclicReduction`
    with SymPy matrices): ``u = vector * Y`` satisfies ``operator(u) = 0``,
    and ``(u, u', ..., u^(n-1))^T = matrix * Y``, so that :meth:`to_system`
    maps the solutions of the operator to those of the system.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import cyclic_reduction
    >>> R = cyclic_reduction(Matrix([[0, 1], [2/x**2, 0]]), x)
    >>> R.operator
    LinearOperator([-2, 0, x**2], x)
    >>> R.to_system(x**2).T
    Matrix([[x**2, 2*x]])
    """

    def __init__(self, reduction: native.CyclicReduction) -> None:
        self.core = reduction
        self.vector: MutableDenseMatrix = sympy_matrix(reduction.vector)
        self.matrix: MutableDenseMatrix = sympy_matrix(reduction.matrix)
        self.operator = LinearOperator.of_core(reduction.operator)
        self.x: Symbol = as_symbol_of(reduction.x)

    def to_system(self, u: Expr) -> MutableDenseMatrix:
        """The solution ``Y = matrix**-1 (u, ..., u^(n-1))^T`` of the
        system given by a solution ``u`` of the operator."""
        return sympy_matrix(self.core.to_system(core_expr(as_expr(u))))

    def __repr__(self) -> str:
        return "CyclicReduction(%s, %s)" % (self.vector.tolist(), self.operator)


def _core_system(A: MutableDenseMatrix) -> ExprMatrix:
    if not isinstance(A, MatrixBase):
        raise TypeError("the matrix of the system must be a Matrix, not %r" % (A,))
    return core_matrix(A)


def cyclic_reduction(A: MutableDenseMatrix, x: Symbol) -> CyclicReduction:
    """The scalar equation of a cyclic vector of ``Y' = A Y`` with the
    matrices mapping its solutions to those of the system
    (:func:`sympy_extras.solvers.linear_systems.cyclic_reduction` on SymPy
    matrices).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import cyclic_reduction
    >>> R = cyclic_reduction(Matrix([[1, 0], [0, 2]]), x)
    >>> R.vector, R.operator
    (Matrix([[2, -3]]), LinearOperator([2, -3, 1], x))
    """
    return CyclicReduction(native.cyclic_reduction(_core_system(A), core_symbol(x)))


def cyclic_vector(A: MutableDenseMatrix, x: Symbol) -> tuple[MutableDenseMatrix, LinearOperator]:
    """A cyclic row vector ``c_0`` for ``Y' = A Y`` and the scalar operator
    satisfied by ``u = c_0 Y``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import cyclic_vector
    >>> c, L = cyclic_vector(Matrix([[0, 1], [-1, 0]]), x)
    >>> c, L.coefficients
    (Matrix([[1, 0]]), [1, 0, 1])
    """
    R = cyclic_reduction(A, x)
    return R.vector, R.operator


def system_to_scalar(A: MutableDenseMatrix, x: Symbol) -> tuple[LinearOperator, MutableDenseMatrix]:
    """The scalar operator of a cyclic vector and the matrix ``M`` with
    ``(u, u', ..., u**(n-1))**T = M Y``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import system_to_scalar
    >>> system_to_scalar(Matrix([[1/x, 1], [0, 1/x]]), x)
    (LinearOperator([2, -2*x, x**2], x), Matrix([
    [  1, 0],
    [1/x, 1]]))
    """
    R = cyclic_reduction(A, x)
    return R.operator, R.matrix


def rational_system_solutions(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """A basis of the rational solutions of ``Y' = A Y`` (Barkatou's
    method, :func:`sympy_extras.solvers.linear_systems.rational_system_solutions`
    on SymPy matrices).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import rational_system_solutions
    >>> rational_system_solutions(Matrix([[1/x, 1], [0, 1/x]]), x)
    [Matrix([
    [x],
    [0]]), Matrix([
    [x**2],
    [   x]])]
    >>> rational_system_solutions(Matrix([[-1/x, 0], [1, 0]]), x)
    [Matrix([
    [0],
    [1]])]
    """
    return [sympy_matrix(Y) for Y in native.rational_system_solutions(_core_system(A), core_symbol(x))]


def hyperexponential_system_solutions(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """Independent solutions ``h R`` of ``Y' = A Y`` with ``h'/h`` and the
    vector ``R`` rational
    (:func:`sympy_extras.solvers.linear_systems.hyperexponential_system_solutions`
    on SymPy matrices).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import hyperexponential_system_solutions
    >>> hyperexponential_system_solutions(Matrix([[1, 1/x], [0, 0]]), x)
    [Matrix([
    [exp(x)],
    [     0]])]
    """
    return [sympy_matrix(Y) for Y in native.hyperexponential_system_solutions(_core_system(A), core_symbol(x))]


class LinearSystemSolution:
    """The solutions of ``Y' = A Y + b`` found by :func:`dsolve_system`
    (:class:`sympy_extras.solvers.linear_systems.LinearSystemSolution` with
    SymPy matrices).

    ``fundamental`` is an ``n x k`` matrix whose columns are independent
    solutions of the homogeneous system, ``particular`` a solution of the
    inhomogeneous one (the zero vector for a homogeneous system, ``None``
    when none was found), and ``complete`` is ``True`` when ``k = n`` and
    the particular solution was found, so that every solution is
    ``fundamental * C + particular``, and ``None`` otherwise. ``reduction``
    is the cyclic vector reduction used.

    Examples
    ========

    >>> from sympy import Matrix, symbols
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import dsolve_system
    >>> S = dsolve_system(Matrix([[0, 1], [2/x**2, 0]]), x)
    >>> S.complete
    True
    >>> S.general_solution(symbols('C1 C2'))
    Matrix([[C1/x + C2*x**2], [-C1/x**2 + 2*C2*x]])
    """

    def __init__(self, fundamental: MutableDenseMatrix, particular: Optional[MutableDenseMatrix],
                 complete: Truth, reduction: CyclicReduction) -> None:
        self.fundamental = fundamental
        self.particular = particular
        self.complete = complete
        self.reduction = reduction

    @staticmethod
    def of_core(S: native.LinearSystemSolution) -> LinearSystemSolution:
        """The solution of the native module with SymPy matrices."""
        return LinearSystemSolution(sympy_matrix(S.fundamental),
                                    None if S.particular is None else sympy_matrix(S.particular), S.complete,
                                    CyclicReduction(S.reduction))

    @property
    def rank(self) -> int:
        """The number of independent homogeneous solutions found."""
        return int(self.fundamental.cols)

    def columns(self) -> list[MutableDenseMatrix]:
        """The independent homogeneous solutions."""
        return [Matrix(self.fundamental[:, j]) for j in range(self.rank)]

    def general_solution(self, constants: Sequence[Symbol]) -> MutableDenseMatrix:
        """``fundamental * C + particular`` for the given constants (one per
        column); the particular part is left out when it was not found."""
        if len(constants) != self.rank:
            raise ValueError("%d constants are needed" % self.rank)
        result = Matrix.zeros(self.fundamental.rows, 1)
        for j, c in enumerate(constants):
            result += c*Matrix(self.fundamental[:, j])
        if self.particular is not None:
            result += self.particular
        return result

    def __repr__(self) -> str:
        return "LinearSystemSolution(%s, %s, %s)" % (self.fundamental.tolist(),
            None if self.particular is None else self.particular.tolist(), self.complete)


def dsolve_system(A: MutableDenseMatrix, x: Symbol, b: Optional[MutableDenseMatrix] = None
                  ) -> LinearSystemSolution:
    """The solutions of ``Y' = A Y + b`` with ``A`` rational in ``x``
    (:func:`sympy_extras.solvers.linear_systems.dsolve_system` on SymPy
    matrices; ``b`` is a column of any functions of ``x``).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import dsolve_system
    >>> S = dsolve_system(Matrix([[1/x, 1], [0, 1/x]]), x, Matrix([1, 0]))
    >>> S.fundamental
    Matrix([[x, x**2], [0, x]])
    >>> S.particular.T
    Matrix([[x*log(x), 0]])
    >>> S.complete
    True
    """
    system = _core_system(A)
    column: Optional[ExprMatrix] = None
    if b is not None:
        if not isinstance(b, MatrixBase):
            raise TypeError("the inhomogeneous term must be a Matrix, not %r" % (b,))
        column = core_matrix(b)
    return LinearSystemSolution.of_core(native.dsolve_system(system, core_symbol(x), column))


def dsolve_linear_system(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """Independent solution vectors of ``Y' = A Y`` (the columns of the
    fundamental matrix of :func:`dsolve_system`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems_sympy import dsolve_linear_system
    >>> sorted((list(Y) for Y in dsolve_linear_system(Matrix([[0, 1], [-1, 0]]), x)), key=str)
    [[exp(-I*x), -I*exp(-I*x)], [exp(I*x), I*exp(I*x)]]
    >>> dsolve_linear_system(Matrix([[1/x, 1], [0, 1/x]]), x)
    [Matrix([
    [x],
    [0]]), Matrix([
    [x**2],
    [   x]])]
    """
    return dsolve_system(A, x).columns()
