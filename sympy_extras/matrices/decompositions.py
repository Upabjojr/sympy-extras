r"""Exact matrix decompositions: LU, LDL, Cholesky, QR (Gram–Schmidt),
rank factorisation, diagonalisation by congruence and the inertia of
symmetric and Hermitian matrices.

The entries are rational functions of symbols, possibly with the
imaginary unit and square roots of rational numbers; the arithmetic is
that of the field they generate (:mod:`sympy_extras.matrices.linalg`), so
that every test for zero made by the eliminations is exact and the
factors come out in lowest terms. With ``I`` in the entries the symbols
stand for real numbers, the transposes below are conjugate transposes
and "symmetric" reads "Hermitian". As everywhere in the linear algebra
of this package, the answers are the generic ones: they hold for the
values of the symbols at which no pivot vanishes.

* :func:`lu`: $P M = L U$ for a matrix of any shape, with $L$ unit lower
  triangular, $U$ in row echelon form and $P$ a permutation which only
  moves a row when the entry in the pivot position is zero (Doolittle's
  elimination [1]_).
* :func:`ldl`: $M = L D L^*$ for a symmetric matrix whose leading
  principal minors do not vanish, $D$ diagonal [1]_; :func:`cholesky`:
  $M = C C^*$ with $C = L \sqrt{D}$ for a positive definite one.
* :func:`gram_schmidt` and :func:`qr`: $M = Q R$ with the columns of $Q$
  orthonormal and $R$ upper triangular (in row echelon form, with $Q$ of
  as many columns as the rank, when the columns of $M$ are dependent).
  The orthogonalisation is done without square roots, the norms being
  taken at the end [2]_.
* :func:`congruence_diagonalize`: $P^* M P = D$ diagonal for any
  symmetric matrix, by simultaneous row and column operations (Lagrange's
  reduction of the quadratic form [3]_); when the diagonal of the part
  which remains is zero a nonzero entry $a_{ij}$ off the diagonal is
  first brought onto it, by adding $\bar a_{ij}$ times column $j$ to
  column $i$. By Sylvester's law of inertia the signs of $D$ do not
  depend on $P$: :func:`inertia`, :func:`is_positive_definite`,
  :func:`is_positive_semidefinite` for matrices of numbers.
  With symbols, :func:`leading_principal_minors` gives the quantities of
  Sylvester's criterion.

Kernels which are not symbols (``sin(t)``) are taken as independent
indeterminates and every pivot is submitted to the zero test of
:mod:`sympy_extras.simplify.structure`; ``NotImplementedError`` is raised
unless all are proved nonzero.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.decompositions import lu, ldl, cholesky, qr, inertia
>>> a, b, c, d = (Symbol(s) for s in 'abcd')
>>> F = lu(ExprMatrix.from_rows([[a, b], [c, d]]))
>>> F.L, F.U, F.permutation
(ExprMatrix([[1, 0], [c/a, 1]]), ExprMatrix([[a, b], [0, (a*d - b*c)/a]]), (0, 1))
>>> S = ExprMatrix.from_rows([[4, 2, 2], [2, 5, 3], [2, 3, 6]])
>>> ldl(S)
(ExprMatrix([[1, 0, 0], [1/2, 1, 0], [1/2, 1/2, 1]]), ExprMatrix([[4, 0, 0], [0, 4, 0], [0, 0, 4]]))
>>> cholesky(S)
ExprMatrix([[2, 0, 0], [1, 2, 0], [1, 1, 2]])
>>> qr(ExprMatrix.from_rows([[3, 1], [4, 2]]))
(ExprMatrix([[3/5, -4/5], [4/5, 3/5]]), ExprMatrix([[5, 11/5], [0, 2/5]]))
>>> inertia(ExprMatrix.from_rows([[0, 1], [1, 0]]))
(1, 1, 0)

References
==========

.. [1] G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
       Hopkins University Press, 2013, Sections 3.2 (the LU
       factorisation), 4.1 (LDL) and 4.2 (Cholesky).
.. [2] U. Erlingsson, E. Kaltofen, D. Musser, *Generic Gram-Schmidt
       orthogonalization by exact division*, Proceedings of ISSAC 1996,
       275-282.
.. [3] R. A. Horn, C. R. Johnson, *Matrix Analysis*, 2nd ed., Cambridge
       University Press, 2013, Sections 4.5 (congruence, Sylvester's law
       of inertia) and 7.2 (positive definite matrices, Sylvester's
       criterion).
"""
from __future__ import annotations

from typing import Final, Generic, Sequence, TypeVar

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import sqrt

from .dense import ExprMatrix
from .eigen import _Field, _FunctionField, _GaussianField, _RationalField
from .linalg import _PolynomialLift, _complex_lift, _kernels, _square, det, rref

__all__ = ['LUDecomposition', 'lu', 'ldl', 'cholesky', 'gram_schmidt', 'qr', 'rank_factorization', 'is_hermitian',
           'congruence_diagonalize', 'inertia', 'is_positive_definite', 'is_positive_semidefinite',
           'leading_principal_minors']

T = TypeVar('T')


class LUDecomposition:
    """``P M = L U``: ``L`` unit lower triangular (square), ``U`` in row
    echelon form (of the shape of ``M``), and row ``i`` of ``P M`` is row
    ``permutation[i]`` of ``M``."""

    __slots__ = ('L', 'U', 'permutation', 'rank')

    def __init__(self, L: ExprMatrix, U: ExprMatrix, permutation: Sequence[int], rank: int) -> None:
        self.L: Final[ExprMatrix] = L
        self.U: Final[ExprMatrix] = U
        self.permutation: Final[tuple[int, ...]] = tuple(permutation)
        self.rank: Final[int] = rank

    @property
    def P(self) -> ExprMatrix:
        """The permutation matrix."""
        n = len(self.permutation)
        return ExprMatrix(n, n, [Integer(1) if self.permutation[i] == j else Integer(0) for i in range(n) for j in range(n)])

    def __repr__(self) -> str:
        return 'LUDecomposition(%r, %r, %r)' % (self.L, self.U, self.permutation)


class _Orthogonalisation(Generic[T]):
    """``M = U C``: the columns of ``U`` (``vectors``) orthogonal with the
    squared norms ``norms``, ``C`` (``coefficients``, one row per vector)
    in row echelon form with ones at the columns ``pivots``."""

    def __init__(self, vectors: list[list[T]], norms: list[T], coefficients: list[list[T]], pivots: list[int]) -> None:
        self.vectors: Final[list[list[T]]] = vectors
        self.norms: Final[list[T]] = norms
        self.coefficients: Final[list[list[T]]] = coefficients
        self.pivots: Final[list[int]] = pivots


class _Algorithms:
    """The decompositions over the field of the entries (the interface
    which does not depend on the type of its elements)."""

    def lu(self, M: ExprMatrix) -> LUDecomposition:
        raise NotImplementedError

    def ldl(self, M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
        raise NotImplementedError

    def qr(self, M: ExprMatrix, normalize: bool) -> tuple[ExprMatrix, ExprMatrix, list[int]]:
        raise NotImplementedError

    def is_hermitian(self, M: ExprMatrix) -> bool:
        raise NotImplementedError

    def congruence(self, M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
        raise NotImplementedError


class _Worker(_Algorithms, Generic[T]):
    def __init__(self, field: _Field[T], exact: bool) -> None:
        self.field: Final[_Field[T]] = field
        self.exact: Final[bool] = exact

    # -- conversions

    def _rows(self, M: ExprMatrix) -> list[list[T]]:
        return [[self.field.element(e) for e in M.row(i)] for i in range(M.rows)]

    def _matrix(self, rows: Sequence[Sequence[T]], ncols: int) -> ExprMatrix:
        return ExprMatrix(len(rows), ncols, [self.field.expr(a) for row in rows for a in row])

    def _pivot(self, a: T) -> T:
        """The nonzero element ``a``, proved nonzero as a quantity."""
        if not self.exact:
            from sympy_extras.interop import to_sympy
            from sympy_extras.simplify.structure import is_zero
            e = self.field.expr(a)
            if not (isinstance(e, Rational) and e.p != 0) and is_zero(to_sympy(e)) is not False:
                raise NotImplementedError('the pivot %s is not proved nonzero: simplify the entries of the matrix first'
                                          % (e,))
        return a

    def _inner(self, u: Sequence[T], v: Sequence[T]) -> T:
        """``u^* v``."""
        K = self.field.domain
        total = K.zero
        for a, b in zip(u, v):
            if not K.is_zero(a) and not K.is_zero(b):
                total = K.add(total, K.mul(self.field.conjugate(a), b))
        return total

    def _hermitian(self, rows: Sequence[Sequence[T]]) -> bool:
        n = len(rows)
        return all(rows[i][j] == self.field.conjugate(rows[j][i]) for i in range(n) for j in range(i, n))

    # -- LU

    def lu(self, M: ExprMatrix) -> LUDecomposition:
        K = self.field.domain
        m, n = M.shape
        rows = self._rows(M)
        lower = [[K.zero] * m for _ in range(m)]
        permutation = list(range(m))
        r = 0
        for column in range(n):
            if r == m:
                break
            chosen = next((i for i in range(r, m) if not K.is_zero(rows[i][column])), None)
            if chosen is None:
                continue
            if chosen != r:
                rows[r], rows[chosen] = rows[chosen], rows[r]
                lower[r], lower[chosen] = lower[chosen], lower[r]
                permutation[r], permutation[chosen] = permutation[chosen], permutation[r]
            inverse = K.invert(self._pivot(rows[r][column]))
            pivot_row = rows[r]
            for i in range(r + 1, m):
                if K.is_zero(rows[i][column]):
                    continue
                factor = K.mul(rows[i][column], inverse)
                lower[i][r] = factor
                rows[i] = [a if K.is_zero(b) else K.sub(a, K.mul(factor, b)) for a, b in zip(rows[i], pivot_row)]
            r += 1
        for i in range(m):
            lower[i][i] = K.one
        return LUDecomposition(self._matrix(lower, m), self._matrix(rows, n), permutation, r)

    # -- LDL

    def is_hermitian(self, M: ExprMatrix) -> bool:
        return M.rows == M.cols and self._hermitian(self._rows(M))

    def ldl(self, M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
        K = self.field.domain
        n = M.rows
        rows = self._rows(M)
        if not self._hermitian(rows):
            raise ValueError('the matrix is not symmetric (Hermitian)')
        lower = [[K.one if i == j else K.zero for j in range(n)] for i in range(n)]
        diagonal: list[T] = []
        for k in range(n):
            pivot = rows[k][k]
            if K.is_zero(pivot):
                raise ValueError('the leading principal minor of order %d is zero: no LDL decomposition without '
                                 'pivoting (see congruence_diagonalize)' % (k + 1))
            inverse = K.invert(self._pivot(pivot))
            diagonal.append(pivot)
            pivot_row = rows[k]
            for i in range(k + 1, n):
                if K.is_zero(rows[i][k]):
                    continue
                factor = K.mul(rows[i][k], inverse)
                lower[i][k] = factor
                rows[i] = [a if K.is_zero(b) else K.sub(a, K.mul(factor, b)) for a, b in zip(rows[i], pivot_row)]
        D = [[diagonal[i] if i == j else K.zero for j in range(n)] for i in range(n)]
        return self._matrix(lower, n), self._matrix(D, n)

    # -- Gram-Schmidt

    def _orthogonalise(self, columns: Sequence[list[T]]) -> _Orthogonalisation[T]:
        K = self.field.domain
        n = len(columns)
        vectors: list[list[T]] = []
        norms: list[T] = []
        inverses: list[T] = []
        coefficients: list[list[T]] = []
        pivots: list[int] = []
        for k, column in enumerate(columns):
            residual = list(column)
            for j, u in enumerate(vectors):
                product = self._inner(u, column)
                if K.is_zero(product):
                    continue
                c = K.mul(product, inverses[j])
                coefficients[j][k] = c
                residual = [a if K.is_zero(b) else K.sub(a, K.mul(c, b)) for a, b in zip(residual, u)]
            if all(K.is_zero(a) for a in residual):
                continue
            norm = self._pivot(self._inner(residual, residual))
            vectors.append(residual)
            norms.append(norm)
            inverses.append(K.invert(norm))
            line = [K.zero] * n
            line[k] = K.one
            coefficients.append(line)
            pivots.append(k)
        return _Orthogonalisation(vectors, norms, coefficients, pivots)

    def qr(self, M: ExprMatrix, normalize: bool) -> tuple[ExprMatrix, ExprMatrix, list[int]]:
        m, n = M.shape
        rows = self._rows(M)
        result = self._orthogonalise([[rows[i][j] for i in range(m)] for j in range(n)])
        r = len(result.vectors)
        lengths: list[Expr] = [sqrt(self.field.expr(norm)) if normalize else Integer(1) for norm in result.norms]
        Q = ExprMatrix(m, r, [self.field.expr(result.vectors[j][i])/lengths[j] for i in range(m) for j in range(r)])
        R = ExprMatrix(r, n, [self.field.expr(result.coefficients[j][k])*lengths[j] for j in range(r) for k in range(n)])
        return Q, R, result.pivots

    # -- congruence

    def congruence(self, M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
        K = self.field.domain
        conjugate = self.field.conjugate
        n = M.rows
        A = self._rows(M)
        if not self._hermitian(A):
            raise ValueError('the matrix is not symmetric (Hermitian)')
        P = [[K.one if i == j else K.zero for j in range(n)] for i in range(n)]

        def add_column(target: int, source: int, c: T) -> None:
            # column target += c * column source, row target += conjugate(c) * row source
            for row in A:
                if not K.is_zero(row[source]):
                    row[target] = K.add(row[target], K.mul(c, row[source]))
            for row in P:
                if not K.is_zero(row[source]):
                    row[target] = K.add(row[target], K.mul(c, row[source]))
            cc = conjugate(c)
            A[target] = [a if K.is_zero(b) else K.add(a, K.mul(cc, b)) for a, b in zip(A[target], A[source])]

        def swap(i: int, j: int) -> None:
            for row in A:
                row[i], row[j] = row[j], row[i]
            for row in P:
                row[i], row[j] = row[j], row[i]
            A[i], A[j] = A[j], A[i]

        for k in range(n):
            if K.is_zero(A[k][k]):
                chosen = next((j for j in range(k + 1, n) if not K.is_zero(A[j][j])), None)
                if chosen is None:
                    found = next(((i, j) for i in range(k, n) for j in range(i + 1, n) if not K.is_zero(A[i][j])), None)
                    if found is None:
                        break
                    i, j = found
                    # the new diagonal entry is c*a_ij + conjugate(c*a_ij): twice a_ij, or twice its squared modulus
                    entry = A[i][j]
                    add_column(i, j, K.one if conjugate(entry) == entry else conjugate(entry))
                    chosen = i
                if chosen != k:
                    swap(k, chosen)
            inverse = K.invert(self._pivot(A[k][k]))
            for i in range(k + 1, n):
                if not K.is_zero(A[i][k]):
                    add_column(i, k, K.neg(conjugate(K.mul(A[i][k], inverse))))
        return self._matrix(P, n), self._matrix(A, n)


def _algorithms(matrices: Sequence[ExprMatrix]) -> _Algorithms:
    kernels = _kernels(matrices)
    gaussian = _complex_lift(kernels)
    if gaussian is not None:
        return _Worker(_GaussianField(gaussian), True)
    if kernels:
        lift = _PolynomialLift(kernels)
        return _Worker(_FunctionField(lift), lift.exact)
    return _Worker(_RationalField(), True)


# ---------------------------------------------------------------------
# the public functions

def lu(M: ExprMatrix) -> LUDecomposition:
    """The LU decomposition ``P M = L U`` of a matrix of any shape and
    rank (see :class:`LUDecomposition`). Rows are exchanged only when the
    entry in the pivot position is zero.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import lu
    >>> x = Symbol('x')
    >>> F = lu(ExprMatrix.from_rows([[0, 1, 2], [1, x, 0], [2, 2*x, 1]]))
    >>> F.L, F.U, F.permutation
    (ExprMatrix([[1, 0, 0], [0, 1, 0], [2, 0, 1]]), ExprMatrix([[1, x, 0], [0, 1, 2], [0, 0, 1]]), (1, 0, 2))
    >>> F.P * ExprMatrix.from_rows([[0, 1, 2], [1, x, 0], [2, 2*x, 1]]) == F.L * F.U
    True
    >>> lu(ExprMatrix.from_rows([[1, 2], [2, 4], [3, 7]])).U
    ExprMatrix([[1, 2], [0, 1], [0, 0]])
    """
    return _algorithms([M]).lu(M)


def is_hermitian(M: ExprMatrix) -> bool:
    """Whether the matrix equals its conjugate transpose, the symbols
    being real: symmetric, for a matrix without ``I``.

    >>> from sympy_extras.core import I
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import is_hermitian
    >>> is_hermitian(ExprMatrix.from_rows([[1, I], [-I, 2]])), is_hermitian(ExprMatrix.from_rows([[1, I], [I, 2]]))
    (True, False)
    """
    return _algorithms([M]).is_hermitian(M)


def ldl(M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
    """``(L, D)`` with ``M = L D L^*``, ``L`` unit lower triangular and
    ``D`` diagonal, for a symmetric (Hermitian) matrix; ``ValueError``
    when the matrix is not symmetric or a leading principal minor is zero.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import ldl
    >>> a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    >>> ldl(ExprMatrix.from_rows([[a, b], [b, c]]))
    (ExprMatrix([[1, 0], [b/a, 1]]), ExprMatrix([[a, 0], [0, (a*c - b**2)/a]]))
    """
    _square(M)
    return _algorithms([M]).ldl(M)


def cholesky(M: ExprMatrix) -> ExprMatrix:
    """The lower triangular ``C`` with ``M = C C^*`` and a positive
    diagonal, for a positive definite symmetric (Hermitian) matrix.
    ``ValueError`` when a pivot is a rational number which is not
    positive; the pivots which are not rational numbers are taken to be
    positive.

    >>> from sympy_extras.core import I, Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import cholesky
    >>> cholesky(ExprMatrix.from_rows([[2, 1], [1, 2]]))
    ExprMatrix([[sqrt(2), 0], [sqrt(2)/2, sqrt(6)/2]])
    >>> cholesky(ExprMatrix.from_rows([[2, I], [-I, 1]]))
    ExprMatrix([[sqrt(2), 0], [-sqrt(2)*I/2, sqrt(2)/2]])
    """
    L, D = ldl(M)
    n = M.rows
    roots: list[Expr] = []
    for k in range(n):
        pivot = D[k, k]
        if isinstance(pivot, Rational) and pivot.p <= 0:
            raise ValueError('the matrix is not positive definite: the pivot %s' % (pivot,))
        roots.append(sqrt(pivot))
    return ExprMatrix(n, n, [L[i, j]*roots[j] if j <= i else Integer(0) for i in range(n) for j in range(n)])


def gram_schmidt(vectors: Sequence[ExprMatrix], normalize: bool = False) -> list[ExprMatrix]:
    """The vectors (columns) orthogonalised in order: vector ``k`` of the
    result is vector ``k`` minus its projections on the previous ones.
    The vectors which depend on the previous ones are dropped. With
    ``normalize`` the results have norm one.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import gram_schmidt
    >>> v = [ExprMatrix.column([1, 1, 0]), ExprMatrix.column([1, 0, 1]), ExprMatrix.column([2, 1, 1])]
    >>> gram_schmidt(v)
    [ExprMatrix([[1], [1], [0]]), ExprMatrix([[1/2], [-1/2], [1]])]
    >>> gram_schmidt(v, normalize=True)
    [ExprMatrix([[sqrt(2)/2], [sqrt(2)/2], [0]]), ExprMatrix([[sqrt(6)/6], [-sqrt(6)/6], [sqrt(6)/3]])]
    """
    if not vectors:
        return []
    size = vectors[0].rows
    if any(v.cols != 1 or v.rows != size for v in vectors):
        raise ValueError('column vectors of the same length are expected')
    M = ExprMatrix(size, len(vectors), [v[i, 0] for i in range(size) for v in vectors])
    Q, _, _ = _algorithms([M]).qr(M, normalize)
    return [ExprMatrix.column([Q[i, j] for i in range(size)]) for j in range(Q.cols)]


def qr(M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
    """``(Q, R)`` with ``M = Q R``, the columns of ``Q`` orthonormal
    (``Q^* Q = I``) and ``R`` upper triangular with a positive diagonal.
    When the columns of ``M`` are dependent ``Q`` has as many columns as
    the rank and ``R`` is in row echelon form.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import qr
    >>> x = Symbol('x')
    >>> qr(ExprMatrix.from_rows([[1, 1], [1, 0], [0, 1]]))
    (ExprMatrix([[sqrt(2)/2, sqrt(6)/6], [sqrt(2)/2, -sqrt(6)/6], [0, sqrt(6)/3]]), ExprMatrix([[sqrt(2), sqrt(2)/2], [0, sqrt(6)/2]]))
    >>> qr(ExprMatrix.from_rows([[1, x], [x, 1]]))[1]
    ExprMatrix([[sqrt(x**2 + 1), 2*x/sqrt(x**2 + 1)], [0, sqrt((x**4 - 2*x**2 + 1)/(x**2 + 1))]])
    >>> qr(ExprMatrix.from_rows([[1, 2], [2, 4]]))
    (ExprMatrix([[sqrt(5)/5], [2*sqrt(5)/5]]), ExprMatrix([[sqrt(5), 2*sqrt(5)]]))
    """
    Q, R, _ = _algorithms([M]).qr(M, True)
    return Q, R


def rank_factorization(M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
    """``(B, C)`` with ``M = B C``, ``B`` of full column rank (the pivot
    columns of ``M``) and ``C`` of full row rank (the nonzero rows of the
    reduced row echelon form).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import rank_factorization
    >>> rank_factorization(ExprMatrix.from_rows([[1, 2, 3], [2, 4, 7], [3, 6, 10]]))
    (ExprMatrix([[1, 3], [2, 7], [3, 10]]), ExprMatrix([[1, 2, 0], [0, 0, 1]]))
    """
    reduced, pivots = rref(M)
    r = len(pivots)
    B = ExprMatrix(M.rows, r, [M[i, j] for i in range(M.rows) for j in pivots])
    C = ExprMatrix(r, M.cols, [reduced[i, j] for i in range(r) for j in range(M.cols)])
    return B, C


def congruence_diagonalize(M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
    """``(P, D)`` with ``P`` invertible and ``P^* M P = D`` diagonal, for
    any symmetric (Hermitian) matrix: the quadratic form ``x^* M x`` as a
    combination of squares. The zeros of ``D`` come last.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import congruence_diagonalize
    >>> a = Symbol('a')
    >>> congruence_diagonalize(ExprMatrix.from_rows([[0, 1], [1, 0]]))
    (ExprMatrix([[1, -1/2], [1, 1/2]]), ExprMatrix([[2, 0], [0, -1/2]]))
    >>> congruence_diagonalize(ExprMatrix.from_rows([[1, a], [a, 1]]))
    (ExprMatrix([[1, -a], [0, 1]]), ExprMatrix([[1, 0], [0, 1 - a**2]]))
    """
    _square(M)
    return _algorithms([M]).congruence(M)


def inertia(M: ExprMatrix) -> tuple[int, int, int]:
    """The numbers of positive, negative and zero eigenvalues of a
    symmetric (Hermitian) matrix, counted with multiplicity, by
    Sylvester's law of inertia: no eigenvalue is computed. ``ValueError``
    when the sign of an entry of the diagonal of
    :func:`congruence_diagonalize` is not known: the signs are those of
    rational numbers, and otherwise the ones the assumptions of the core
    decide (``is_positive``, ``is_negative``: numbers with square roots,
    symbols declared positive).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sqrt
    >>> p = Symbol('p', positive=True)
    >>> inertia(ExprMatrix.from_rows([[sqrt(2), 1], [1, sqrt(2)]])), inertia(ExprMatrix.from_rows([[p, 0], [0, -p - 1]]))
    ((2, 0, 0), (1, 1, 0))

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import inertia
    >>> inertia(ExprMatrix.from_rows([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])), inertia(ExprMatrix.from_rows([[1, 2], [2, 4]]))
    ((3, 0, 0), (1, 0, 1))
    """
    _, D = congruence_diagonalize(M)
    positive = negative = zero = 0
    for k in range(D.rows):
        entry = D[k, k]
        if entry.is_positive is True:
            positive += 1
        elif entry.is_negative is True:
            negative += 1
        elif entry.is_zero is True:
            zero += 1
        else:
            raise ValueError('the sign of %s is not known' % (entry,))
    return positive, negative, zero


def is_positive_definite(M: ExprMatrix) -> bool:
    """Whether ``x^* M x > 0`` for every nonzero ``x``, for a symmetric
    (Hermitian) matrix of numbers (see :func:`inertia`).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import is_positive_definite
    >>> is_positive_definite(ExprMatrix.from_rows([[2, 1], [1, 2]])), is_positive_definite(ExprMatrix.from_rows([[1, 2], [2, 1]]))
    (True, False)
    """
    positive, _, _ = inertia(M)
    return positive == M.rows


def is_positive_semidefinite(M: ExprMatrix) -> bool:
    """Whether ``x^* M x >= 0`` for every ``x`` (see :func:`inertia`).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import is_positive_semidefinite
    >>> is_positive_semidefinite(ExprMatrix.from_rows([[1, 1], [1, 1]]))
    True
    """
    _, negative, _ = inertia(M)
    return negative == 0


def leading_principal_minors(M: ExprMatrix) -> list[Expr]:
    """The determinants of the upper left blocks of orders ``1, ..., n``.
    A symmetric (Hermitian) matrix is positive definite exactly when all
    are positive (Sylvester's criterion): with symbols, these are the
    conditions.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.decompositions import leading_principal_minors
    >>> a = Symbol('a')
    >>> leading_principal_minors(ExprMatrix.from_rows([[2, a, 0], [a, 2, a], [0, a, 2]]))
    [2, 4 - a**2, 8 - 4*a**2]
    """
    n = _square(M)
    return [det(ExprMatrix(k, k, [M[i, j] for i in range(k) for j in range(k)])) for k in range(1, n + 1)]
