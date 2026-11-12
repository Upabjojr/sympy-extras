"""The SymPy face of the linear systems ``Y' = A(x) Y + b(x)``
(:mod:`sympy_extras.solvers.linear_systems`), and the steps which are not
ported.

The functions take and return SymPy matrices (``Matrix``), converted
entry by entry by :mod:`sympy_extras.interop` to the
:class:`~sympy_extras.matrices.dense.ExprMatrix` of the native module and
back; :class:`CyclicReduction` and :class:`LinearSystemSolution` hold
SymPy matrices and the SymPy
:class:`~sympy_extras.solvers.linear_ode_sympy.LinearOperator`.

The antiderivatives of Liouville's formula and of variation of constants
are native (:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`).

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

from sympy_extras._typing_sympy import Truth, as_expr
from sympy_extras.assumptions.analysis_sympy import as_symbol_of
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol
from sympy_extras.matrices.dense import ExprMatrix

from . import linear_systems as native
from .dae_sympy import core_matrix, sympy_matrix
from .linear_ode_sympy import LinearOperator

__all__ = ['CyclicReduction', 'LinearSystemSolution', 'cyclic_vector', 'cyclic_reduction', 'system_to_scalar',
           'rational_system_solutions', 'hyperexponential_system_solutions', 'dsolve_system',
           'dsolve_linear_system']


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy

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
