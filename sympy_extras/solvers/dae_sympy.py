"""The SymPy face of the differential-algebraic solver
(:mod:`sympy_extras.solvers.dae`).

The functions take and return SymPy matrices (``Matrix``), converted
entry by entry by :mod:`sympy_extras.interop` to the
:class:`~sympy_extras.matrices.dense.ExprMatrix` of the native module and
back; :class:`DAESolution` holds a SymPy matrix. The native module
integrates by the native indefinite integration
(:func:`sympy_extras.integrals.indefinite.closed_antiderivative`).

Examples
========

>>> from sympy import Matrix, sin, Symbol
>>> from sympy_extras.solvers.dae_sympy import dsolve_dae
>>> x = Symbol('x')
>>> dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, -1], [1, 0]]), Matrix([0, sin(x)]), x).solution.T
Matrix([[sin(x), cos(x)]])
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Symbol
from sympy.matrices.dense import Matrix, MutableDenseMatrix

from sympy_extras.assumptions.analysis_sympy import as_symbol_of
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.interop import from_sympy
from sympy_extras.matrices.dense import ExprMatrix

from . import dae as native

__all__ = ['core_nilpotent_decomposition', 'dae_index', 'dsolve_dae', 'dae_matrices', 'check_dae', 'DAESolution',
           'core_matrix', 'sympy_matrix']


# ---------------------------------------------------------------------------
# the step of the native solver which stays on SymPy



# ---------------------------------------------------------------------------
# the face

def core_matrix(M: MutableDenseMatrix) -> ExprMatrix:
    """A SymPy matrix as a matrix of core expressions."""
    return ExprMatrix(M.rows, M.cols, [core_expr(e) for e in M.flat()])


def sympy_matrix(M: ExprMatrix) -> MutableDenseMatrix:
    """A matrix of core expressions as a SymPy matrix."""
    return Matrix(M.rows, M.cols, [sympy_expr(e) for e in M.entries])


def _optional_matrix(f: Optional[MutableDenseMatrix]) -> Optional[ExprMatrix]:
    return core_matrix(f) if f is not None else None


def core_nilpotent_decomposition(M: MutableDenseMatrix) -> tuple[MutableDenseMatrix, MutableDenseMatrix,
                                                                  MutableDenseMatrix, int]:
    """The core-nilpotent (Drazin) decomposition ``T``, ``C``, ``N``, ``k``
    of a square matrix
    (:func:`sympy_extras.solvers.dae.core_nilpotent_decomposition` on SymPy
    matrices).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy_extras.solvers.dae_sympy import core_nilpotent_decomposition
    >>> T, C, N, k = core_nilpotent_decomposition(Matrix([[1, 1], [0, 0]]))
    >>> C, N, k
    (Matrix([[1]]), Matrix([[0]]), 1)
    >>> T.inv()*Matrix([[1, 1], [0, 0]])*T == Matrix([[1, 0], [0, 0]])
    True
    """
    T, C, N, k = native.core_nilpotent_decomposition(core_matrix(M))
    return sympy_matrix(T), sympy_matrix(C), sympy_matrix(N), k


def dae_index(A: MutableDenseMatrix, B: MutableDenseMatrix) -> int:
    """The index of the regular pencil ``lambda*A + B``
    (:func:`sympy_extras.solvers.dae.dae_index` on SymPy matrices).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy_extras.solvers.dae_sympy import dae_index
    >>> dae_index(Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]]))
    2
    >>> dae_index(Matrix([[1, 0], [0, 1]]), Matrix([[0, 1], [1, 0]]))
    0
    """
    return native.dae_index(core_matrix(A), core_matrix(B))


class DAESolution:
    """The general solution of ``A y' + B y = f``.

    Attributes
    ==========

    solution : Matrix
        The column of the components of ``y``, in the free constants.
    constants : list of Symbols
        The free constants (as many as the dimension of the regular part).
    index : int
        The index of the pencil.
    """

    def __init__(self, solution: MutableDenseMatrix, constants: list[Symbol], index: int) -> None:
        self.solution = solution
        self.constants = constants
        self.index = index

    def __repr__(self) -> str:
        return "DAESolution(%s, constants=%s, index=%s)" % (self.solution, self.constants, self.index)


def dsolve_dae(A: MutableDenseMatrix, B: MutableDenseMatrix, f: Optional[MutableDenseMatrix], x: Symbol
               ) -> DAESolution:
    """The general solution of ``A y' + B y = f(x)`` with constant square
    matrices ``A`` and ``B`` (:func:`sympy_extras.solvers.dae.dsolve_dae`
    on SymPy matrices).

    Examples
    ========

    >>> from sympy import Matrix, sin, Symbol
    >>> from sympy_extras.solvers.dae_sympy import dsolve_dae
    >>> x = Symbol('x')
    >>> A = Matrix([[1, 0], [0, 0]])
    >>> B = Matrix([[0, -1], [1, 0]])          # y1' = y2, y1 = sin(x)
    >>> result = dsolve_dae(A, B, Matrix([0, sin(x)]), x)
    >>> result.solution.T, result.constants, result.index
    (Matrix([[sin(x), cos(x)]]), [], 2)
    >>> result = dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, 0], [1, -1]]), None, x)
    >>> result.solution.T                      # y1' = 0, y1 = y2
    Matrix([[C1, C1]])
    """
    result = native.dsolve_dae(core_matrix(A), core_matrix(B), _optional_matrix(f), core_symbol(x))
    return DAESolution(sympy_matrix(result.solution), [as_symbol_of(c) for c in result.constants], result.index)


def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("an applied undefined function is expected, got %s" % (f,))
    return converted


def dae_matrices(equations: Sequence[Basic], functions: Sequence[AppliedUndef]
                 ) -> tuple[MutableDenseMatrix, MutableDenseMatrix, MutableDenseMatrix, Symbol]:
    """The matrices ``A``, ``B`` and the column ``f`` of a system of linear
    equations with constant coefficients in the functions and their first
    derivatives, ``A y' + B y = f``
    (:func:`sympy_extras.solvers.dae.dae_matrices` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, Symbol, Eq, sin
    >>> from sympy_extras.solvers.dae_sympy import dae_matrices
    >>> x = Symbol('x')
    >>> y1, y2 = Function('y1')(x), Function('y2')(x)
    >>> A, B, f, _ = dae_matrices([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])
    >>> A, B, f.T
    (Matrix([
    [1, 0],
    [0, 0]]), Matrix([
    [0, -1],
    [1,  0]]), Matrix([[0, sin(x)]]))
    """
    converted: list[CoreBasic] = [from_sympy(e) for e in equations]
    A, B, f, x = native.dae_matrices(converted, [_core_function(fn) for fn in functions])
    return sympy_matrix(A), sympy_matrix(B), sympy_matrix(f), as_symbol_of(x)


def check_dae(A: MutableDenseMatrix, B: MutableDenseMatrix, f: Optional[MutableDenseMatrix],
              y: MutableDenseMatrix, x: Symbol) -> bool:
    """Whether ``y`` solves ``A y' + B y = f``
    (:func:`sympy_extras.solvers.dae.check_dae` on SymPy matrices).

    Examples
    ========

    >>> from sympy import Matrix, Symbol, sin, cos, eye
    >>> from sympy_extras.solvers.dae_sympy import check_dae
    >>> x = Symbol('x')
    >>> check_dae(eye(2), Matrix([[0, -1], [1, 0]]), None, Matrix([cos(x), -sin(x)]), x)
    True
    """
    return native.check_dae(core_matrix(A), core_matrix(B), _optional_matrix(f), core_matrix(y), core_symbol(x))
