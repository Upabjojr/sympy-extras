r"""Linear matrix equations: Sylvester and Lyapunov equations, the
matrices commuting with a matrix, and Kronecker products.

A linear equation in an unknown matrix $X$,

.. math:: \sum_i A_i\, X\, B_i = C,

becomes an ordinary linear system by stacking the columns of $X$ into
the vector $\operatorname{vec}(X)$: since $\operatorname{vec}(A X B) =
(B^T \otimes A) \operatorname{vec}(X)$ with $\otimes$ the Kronecker
product [1]_,

.. math:: \Bigl(\sum_i B_i^T \otimes A_i\Bigr) \operatorname{vec}(X)
          = \operatorname{vec}(C).

The system is solved exactly by the fraction-free elimination of
:mod:`sympy_extras.matrices.linalg` (entries: rational functions of
symbols, with ``I`` and square roots of rationals), which also gives the
solutions of the homogeneous equation when the solution is not unique.

* :func:`sylvester_solve`: $A X + X B = C$, which has one solution
  exactly when $A$ and $-B$ have no common eigenvalue [2]_;
* :func:`lyapunov_solve`: $A X + X A^* + Q = 0$, the equation of the
  stability of $x' = A x$ (when $A$ is stable and $Q$ positive definite
  the solution is positive definite) [3]_; :func:`discrete_lyapunov_solve`:
  $A X A^* - X + Q = 0$ (Stein's equation), for $x_{k+1} = A x_k$;
* :func:`linear_matrix_equation`: the general case;
* :func:`commuting_matrices`: a basis of the matrices $X$ with
  $A X = X A$.

With ``I`` in the entries the symbols stand for real numbers and $A^*$
is the conjugate transpose; otherwise it is the transpose.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.equations import sylvester_solve, lyapunov_solve, commuting_matrices, kronecker_product
>>> a = Symbol('a')
>>> A = ExprMatrix.from_rows([[1, 2], [0, 3]])
>>> sylvester_solve(A, ExprMatrix.from_rows([[a]]), ExprMatrix.column([1, 1])).particular
ExprMatrix([[1/(a + 3)], [1/(a + 3)]])
>>> lyapunov_solve(ExprMatrix.from_rows([[-1, 2], [0, -3]]), ExprMatrix.eye(2)).particular
ExprMatrix([[2/3, 1/12], [1/12, 1/6]])
>>> commuting_matrices(ExprMatrix.from_rows([[0, 1], [0, 0]]))
[ExprMatrix([[0, 1], [0, 0]]), ExprMatrix([[1, 0], [0, 1]])]
>>> kronecker_product(ExprMatrix.from_rows([[1, a]]), ExprMatrix.eye(2))
ExprMatrix([[1, 0, a, 0], [0, 1, 0, a]])

References
==========

.. [1] R. A. Horn, C. R. Johnson, *Topics in Matrix Analysis*, Cambridge
       University Press, 1991, Chapter 4 (Kronecker products, the vec
       operator, linear matrix equations).
.. [2] J. J. Sylvester, *Sur l'équation en matrices px = xq*, Comptes
       Rendus de l'Académie des Sciences 99 (1884), 67-71, 115-116.
.. [3] T. Kailath, *Linear Systems*, Prentice-Hall, 1980, Section 2.6
       (Lyapunov's equation and stability).
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import I, ImaginaryUnit

from .dense import ExprMatrix
from .linalg import _square, linear_solve, normal

__all__ = ['MatrixSolution', 'kronecker_product', 'vec', 'unvec', 'adjoint', 'linear_matrix_equation', 'sylvester_solve',
           'lyapunov_solve', 'discrete_lyapunov_solve', 'commuting_matrices']


class MatrixSolution:
    """The solutions of a linear matrix equation: ``particular`` plus any
    combination of the matrices of ``homogeneous`` (solutions of the
    equation with right-hand side zero)."""

    __slots__ = ('particular', 'homogeneous')

    def __init__(self, particular: ExprMatrix, homogeneous: Sequence[ExprMatrix]) -> None:
        self.particular: Final[ExprMatrix] = particular
        self.homogeneous: Final[tuple[ExprMatrix, ...]] = tuple(homogeneous)

    @property
    def is_unique(self) -> bool:
        return not self.homogeneous

    def __repr__(self) -> str:
        return 'MatrixSolution(%r, %r)' % (self.particular, list(self.homogeneous))


def kronecker_product(A: ExprMatrix, B: ExprMatrix) -> ExprMatrix:
    """The Kronecker product: the blocks ``A[i, j] * B``.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import kronecker_product
    >>> kronecker_product(ExprMatrix.from_rows([[1, 2], [3, 4]]), ExprMatrix.from_rows([[0, 1], [1, 0]]))
    ExprMatrix([[0, 1, 0, 2], [1, 0, 2, 0], [0, 3, 0, 4], [3, 0, 4, 0]])
    """
    entries: list[Expr] = []
    for i in range(A.rows):
        for k in range(B.rows):
            for j in range(A.cols):
                for l in range(B.cols):
                    entries.append(A[i, j]*B[k, l])
    return ExprMatrix(A.rows*B.rows, A.cols*B.cols, entries)


def vec(M: ExprMatrix) -> ExprMatrix:
    """The columns of the matrix stacked into one column.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import vec
    >>> vec(ExprMatrix.from_rows([[1, 2], [3, 4]]))
    ExprMatrix([[1], [3], [2], [4]])
    """
    return ExprMatrix.column([M[i, j] for j in range(M.cols) for i in range(M.rows)])


def unvec(v: ExprMatrix, rows: int, cols: int) -> ExprMatrix:
    """The inverse of :func:`vec`: the ``rows`` by ``cols`` matrix whose
    stacked columns are the column ``v``.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import unvec
    >>> unvec(ExprMatrix.column([1, 3, 2, 4]), 2, 2)
    ExprMatrix([[1, 2], [3, 4]])
    """
    if v.cols != 1 or v.rows != rows*cols:
        raise ValueError('a column of %d entries is expected' % (rows*cols))
    return ExprMatrix(rows, cols, [v[j*rows + i, 0] for i in range(rows) for j in range(cols)])


def adjoint(M: ExprMatrix) -> ExprMatrix:
    """The conjugate transpose, the symbols being real: the transpose,
    for a matrix without ``I``.

    >>> from sympy_extras.core import I
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import adjoint
    >>> adjoint(ExprMatrix.from_rows([[1, 2 + I], [3*I, 4]]))
    ExprMatrix([[1, -3*I], [2 - I, 4]])
    """
    transposed = M.transpose()
    if not transposed.has(ImaginaryUnit):
        return transposed
    return transposed.xreplace({I: -I}).applyfunc(normal)


def linear_matrix_equation(terms: Sequence[tuple[ExprMatrix, ExprMatrix]], C: ExprMatrix) -> Optional[MatrixSolution]:
    """The solutions ``X`` of ``sum(A * X * B for A, B in terms) = C``, or
    ``None`` when there is none.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import linear_matrix_equation
    >>> A, B = ExprMatrix.from_rows([[1, 1], [0, 1]]), ExprMatrix.from_rows([[2, 0], [1, 1]])
    >>> X = ExprMatrix.from_rows([[1, 2], [3, 4]])
    >>> linear_matrix_equation([(A, B), (B, A)], A*X*B + B*X*A)
    MatrixSolution(ExprMatrix([[1, 2], [3, 4]]), [])
    """
    if not terms:
        raise ValueError('an equation without terms')
    rows, cols = terms[0][0].cols, terms[0][1].rows
    for A, B in terms:
        if A.cols != rows or B.rows != cols or A.rows != C.rows or B.cols != C.cols:
            raise ValueError('the shapes of the terms do not agree')
    system = kronecker_product(terms[0][1].transpose(), terms[0][0])
    for A, B in terms[1:]:
        system = system + kronecker_product(B.transpose(), A)
    solution = linear_solve(system, vec(C))
    if solution is None:
        return None
    return MatrixSolution(unvec(solution.particular, rows, cols), [unvec(v, rows, cols) for v in solution.nullspace])


def sylvester_solve(A: ExprMatrix, B: ExprMatrix, C: ExprMatrix) -> Optional[MatrixSolution]:
    """The solutions ``X`` of Sylvester's equation ``A X + X B = C``
    (``A`` and ``B`` square, ``C`` with the rows of ``A`` and the columns
    of ``B``), or ``None`` when there is none. The solution is unique
    exactly when ``A`` and ``-B`` have no common eigenvalue.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import sylvester_solve
    >>> A, B = ExprMatrix.from_rows([[1, 2], [3, 4]]), ExprMatrix.from_rows([[0, 1], [1, 1]])
    >>> sylvester_solve(A, B, ExprMatrix.from_rows([[1, 0], [0, 1]]))
    MatrixSolution(ExprMatrix([[-26/41, 23/41], [22/41, -10/41]]), [])
    >>> solution = sylvester_solve(A, -A, ExprMatrix.zeros(2, 2))
    >>> solution.homogeneous
    (ExprMatrix([[-3/2, 1], [3/2, 0]]), ExprMatrix([[1, 0], [0, 1]]))
    """
    m, n = _square(A), _square(B)
    return linear_matrix_equation([(A, ExprMatrix.eye(n)), (ExprMatrix.eye(m), B)], C)


def lyapunov_solve(A: ExprMatrix, Q: ExprMatrix) -> Optional[MatrixSolution]:
    """The solutions ``X`` of the continuous Lyapunov equation
    ``A X + X A^* + Q = 0``, or ``None``. The solution is unique exactly
    when no two eigenvalues of ``A`` have a sum zero (their conjugates
    included); for a stable ``A`` (eigenvalues of negative real part) and
    a positive definite ``Q`` it is positive definite.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import lyapunov_solve
    >>> k = Symbol('k')
    >>> lyapunov_solve(ExprMatrix.from_rows([[0, 1], [-1, -k]]), ExprMatrix.eye(2)).particular
    ExprMatrix([[(k**2 + 2)/(2*k), -1/2], [-1/2, 1/k]])
    """
    n = _square(A)
    identity = ExprMatrix.eye(n)
    return linear_matrix_equation([(A, identity), (identity, adjoint(A))], -Q)


def discrete_lyapunov_solve(A: ExprMatrix, Q: ExprMatrix) -> Optional[MatrixSolution]:
    """The solutions ``X`` of the discrete Lyapunov (Stein) equation
    ``A X A^* - X + Q = 0``, or ``None``. The solution is unique exactly
    when no two eigenvalues of ``A`` have a product one (their conjugates
    included).

    >>> from sympy_extras.core import Rational
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import discrete_lyapunov_solve
    >>> discrete_lyapunov_solve(ExprMatrix.from_rows([[Rational(1, 2), 1], [0, Rational(1, 3)]]), ExprMatrix.eye(2)).particular
    ExprMatrix([[103/30, 9/20], [9/20, 9/8]])
    """
    n = _square(A)
    identity = ExprMatrix.eye(n)
    return linear_matrix_equation([(A, adjoint(A)), (-identity, identity)], -Q)


def commuting_matrices(A: ExprMatrix) -> list[ExprMatrix]:
    """A basis of the matrices ``X`` with ``A X = X A`` (the centraliser
    of ``A``). Its dimension is at least the size of ``A``, with equality
    exactly when the minimal polynomial is the characteristic one; the
    matrices are then the polynomials in ``A``.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.equations import commuting_matrices
    >>> commuting_matrices(ExprMatrix.from_rows([[1, 0], [0, 2]]))
    [ExprMatrix([[1, 0], [0, 0]]), ExprMatrix([[0, 0], [0, 1]])]
    >>> len(commuting_matrices(ExprMatrix.eye(2)))
    4
    """
    n = _square(A)
    identity = ExprMatrix.eye(n)
    solution = linear_matrix_equation([(A, identity), (-identity, A)], ExprMatrix.zeros(n, n))
    if solution is None:
        raise ArithmeticError('a homogeneous system without solution')
    return list(solution.homogeneous)
