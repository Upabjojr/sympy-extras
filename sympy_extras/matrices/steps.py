r"""Step-by-step Gaussian elimination: the elementary row operations which
bring a matrix to its (reduced) row echelon form, with the matrix after
each of them, for the row reduction itself, the inverse, the determinant
and linear systems.

The other modules of this package compute these results by fraction-free
methods whose intermediate matrices mean nothing to a reader. Here the
elimination is the one of the textbooks [1]_, done by hand in the same
order:

* in each column, the first row at or below the current one with a
  nonzero entry gives the pivot (a row exchange $R_i \leftrightarrow
  R_j$ when it is not the current one);
* for the reduced form the pivot row is divided by the pivot ($R_i
  \leftarrow c\, R_i$);
* multiples of the pivot row are subtracted from the other rows ($R_i
  \leftarrow R_i + c\, R_j$): from the rows below for the echelon form,
  from all of them for the reduced form.

Each operation is a :class:`RowOperation` (it prints as above, with
rows numbered from one, and :meth:`RowOperation.matrix` is its elementary
matrix); a :class:`Step` pairs it with the matrix it produces. The
arithmetic is exact, in the field of the entries (rational functions of
symbols, ``I``, square roots of rationals), and a pivot is an entry
which is not identically zero: the steps are those for generic values
of the symbols.

Examples
========

>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.steps import rref_steps, determinant_steps, format_steps
>>> M = ExprMatrix.from_rows([[0, 2, 4], [1, 1, 1]])
>>> for step in rref_steps(M):
...     print(step.operation, ' ', step.matrix)
R1 <-> R2   ExprMatrix([[1, 1, 1], [0, 2, 4]])
R2 <- (1/2)*R2   ExprMatrix([[1, 1, 1], [0, 1, 2]])
R1 <- R1 - R2   ExprMatrix([[1, 0, -1], [0, 1, 2]])
>>> D = determinant_steps(ExprMatrix.from_rows([[0, 2], [3, 4]]))
>>> [str(step.operation) for step in D.steps], D.exchanges, D.diagonal, D.value
(['R1 <-> R2'], 1, (3, 2), -6)
>>> print(format_steps(M, rref_steps(M)))
[0, 2, 4]
[1, 1, 1]
R1 <-> R2
[1, 1, 1]
[0, 2, 4]
R2 <- (1/2)*R2
[1, 1, 1]
[0, 1, 2]
R1 <- R1 - R2
[1, 0, -1]
[0, 1,  2]

References
==========

.. [1] G. Strang, *Introduction to Linear Algebra*, 5th ed.,
       Wellesley-Cambridge Press, 2016, Chapter 2 (elimination,
       elimination matrices, inverses by Gauss-Jordan) and Section 5.1
       (the determinant from the pivots).
"""
from __future__ import annotations

from typing import Final, Generic, Sequence, TypeVar

from sympy_extras.core.add import Add
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Rational

from .dense import ExprMatrix
from .eigen import _Field, _FunctionField, _GaussianField, _RationalField
from .linalg import _PolynomialLift, _complex_lift, _kernels, _square, normal

__all__ = ['RowOperation', 'Step', 'DeterminantSteps', 'echelon_steps', 'rref_steps', 'inverse_steps', 'solve_steps',
           'determinant_steps', 'format_steps']

T = TypeVar('T')


class RowOperation:
    """An elementary row operation, rows counted from zero: ``kind`` is
    ``'swap'`` (rows ``target`` and ``source`` exchanged), ``'scale'``
    (row ``target`` multiplied by ``factor``) or ``'add'`` (``factor``
    times row ``source`` added to row ``target``)."""

    __slots__ = ('kind', 'target', 'source', 'factor')

    def __init__(self, kind: str, target: int, source: int, factor: Expr) -> None:
        if kind not in ('swap', 'scale', 'add'):
            raise ValueError('unknown row operation %r' % (kind,))
        self.kind: Final[str] = kind
        self.target: Final[int] = target
        self.source: Final[int] = source
        self.factor: Final[Expr] = factor

    def __repr__(self) -> str:
        target, source = 'R%d' % (self.target + 1), 'R%d' % (self.source + 1)
        if self.kind == 'swap':
            return '%s <-> %s' % (target, source)
        if self.kind == 'scale':
            return '%s <- (%s)*%s' % (target, self.factor, target)
        factor = self.factor
        if factor == 1:
            return '%s <- %s + %s' % (target, target, source)
        if factor == -1:
            return '%s <- %s - %s' % (target, target, source)
        negative = (isinstance(factor, Rational) and factor.p < 0) or (not isinstance(factor, Add) and str(factor).startswith('-'))
        if negative:
            return '%s <- %s - (%s)*%s' % (target, target, normal(-factor), source)
        return '%s <- %s + (%s)*%s' % (target, target, factor, source)

    __str__ = __repr__

    def matrix(self, n: int) -> ExprMatrix:
        """The elementary matrix of order ``n``: multiplying on the left
        by it performs the operation."""
        E = ExprMatrix.eye(n)
        if self.kind == 'swap':
            E = E.with_entry(self.target, self.target, 0).with_entry(self.source, self.source, 0)
            return E.with_entry(self.target, self.source, 1).with_entry(self.source, self.target, 1)
        if self.kind == 'scale':
            return E.with_entry(self.target, self.target, self.factor)
        return E.with_entry(self.target, self.source, self.factor)

    def apply(self, M: ExprMatrix) -> ExprMatrix:
        """The operation performed on ``M``."""
        return (self.matrix(M.rows)*M).applyfunc(normal)


class Step:
    """A row operation and the matrix it produces."""

    __slots__ = ('operation', 'matrix')

    def __init__(self, operation: RowOperation, matrix: ExprMatrix) -> None:
        self.operation: Final[RowOperation] = operation
        self.matrix: Final[ExprMatrix] = matrix

    def __repr__(self) -> str:
        return 'Step(%s, %r)' % (self.operation, self.matrix)


class DeterminantSteps:
    """The determinant by elimination: after ``steps`` (row exchanges and
    additions of multiples of rows only) the matrix is triangular with
    the diagonal ``diagonal``, and ``value`` is the product of the
    diagonal with the sign changed once for each of the ``exchanges``."""

    __slots__ = ('steps', 'exchanges', 'diagonal', 'value')

    def __init__(self, steps: Sequence[Step], exchanges: int, diagonal: Sequence[Expr], value: Expr) -> None:
        self.steps: Final[tuple[Step, ...]] = tuple(steps)
        self.exchanges: Final[int] = exchanges
        self.diagonal: Final[tuple[Expr, ...]] = tuple(diagonal)
        self.value: Final[Expr] = value

    def __repr__(self) -> str:
        return 'DeterminantSteps(%d steps, %d exchanges, diagonal %s, value %s)' % (
            len(self.steps), self.exchanges, list(self.diagonal), self.value)


class _Elimination:
    """The interface which does not depend on the type of the elements
    of the field."""

    def steps(self, M: ExprMatrix, reduced: bool, columns: int) -> list[Step]:
        raise NotImplementedError


class _Worker(_Elimination, Generic[T]):
    def __init__(self, field: _Field[T]) -> None:
        self.field: Final[_Field[T]] = field

    def steps(self, M: ExprMatrix, reduced: bool, columns: int) -> list[Step]:
        """The elimination with pivots in the first ``columns`` columns."""
        K = self.field.domain
        m, n = M.shape
        rows = [[self.field.element(e) for e in M.row(i)] for i in range(m)]
        found: list[Step] = []

        def record(kind: str, target: int, source: int, factor: T) -> None:
            matrix = ExprMatrix(m, n, [self.field.expr(a) for row in rows for a in row])
            found.append(Step(RowOperation(kind, target, source, self.field.expr(factor)), matrix))

        r = 0
        for column in range(min(n, columns)):
            if r == m:
                break
            chosen = next((i for i in range(r, m) if not K.is_zero(rows[i][column])), None)
            if chosen is None:
                continue
            if chosen != r:
                rows[r], rows[chosen] = rows[chosen], rows[r]
                record('swap', r, chosen, K.one)
            pivot = rows[r][column]
            if reduced and not K.is_one(pivot):
                inverse = K.invert(pivot)
                rows[r] = [K.mul(a, inverse) for a in rows[r]]
                record('scale', r, r, inverse)
                pivot = K.one
            for i in (range(m) if reduced else range(r + 1, m)):
                if i == r or K.is_zero(rows[i][column]):
                    continue
                factor = K.neg(K.exquo(rows[i][column], pivot))
                rows[i] = [a if K.is_zero(b) else K.add(a, K.mul(factor, b)) for a, b in zip(rows[i], rows[r])]
                record('add', i, r, factor)
            r += 1
        return found


def _elimination(M: ExprMatrix) -> _Elimination:
    kernels = _kernels([M])
    gaussian = _complex_lift(kernels)
    if gaussian is not None:
        return _Worker(_GaussianField(gaussian))
    if kernels:
        lift = _PolynomialLift(kernels)
        if not lift.exact:
            raise NotImplementedError('elimination steps for a matrix whose entries are not rational functions of symbols, '
                                      'I and square roots of rationals')
        return _Worker(_FunctionField(lift))
    return _Worker(_RationalField())


def echelon_steps(M: ExprMatrix) -> list[Step]:
    """The steps of the forward elimination to a row echelon form (row
    exchanges and additions of multiples of the pivot rows to the rows
    below; the pivots are not scaled).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.steps import echelon_steps
    >>> a = Symbol('a')
    >>> echelon_steps(ExprMatrix.from_rows([[2, 1], [4, a]]))
    [Step(R2 <- R2 - (2)*R1, ExprMatrix([[2, 1], [0, a - 2]]))]
    """
    return _elimination(M).steps(M, False, M.cols)


def rref_steps(M: ExprMatrix) -> list[Step]:
    """The steps of the Gauss–Jordan elimination to the reduced row
    echelon form.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.steps import rref_steps
    >>> for step in rref_steps(ExprMatrix.from_rows([[2, 4], [1, 3]])):
    ...     print(step)
    Step(R1 <- (1/2)*R1, ExprMatrix([[1, 2], [1, 3]]))
    Step(R2 <- R2 - R1, ExprMatrix([[1, 2], [0, 1]]))
    Step(R1 <- R1 - (2)*R2, ExprMatrix([[1, 0], [0, 1]]))
    """
    return _elimination(M).steps(M, True, M.cols)


def inverse_steps(M: ExprMatrix) -> list[Step]:
    """The steps of the inversion by Gauss–Jordan elimination of
    ``[M | I]``: the last matrix is ``[I | M**-1]``. ``ValueError`` for a
    singular matrix.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.steps import inverse_steps
    >>> for step in inverse_steps(ExprMatrix.from_rows([[1, 2], [3, 7]])):
    ...     print(step)
    Step(R2 <- R2 - (3)*R1, ExprMatrix([[1, 2, 1, 0], [0, 1, -3, 1]]))
    Step(R1 <- R1 - (2)*R2, ExprMatrix([[1, 0, 7, -2], [0, 1, -3, 1]]))
    """
    n = _square(M)
    augmented = M.col_insert(n, ExprMatrix.eye(n))
    steps = _elimination(augmented).steps(augmented, True, n)
    final = steps[-1].matrix if steps else augmented
    if any(final[i, j] != (1 if i == j else 0) for i in range(n) for j in range(n)):
        raise ValueError('the matrix is singular')
    return steps


def solve_steps(A: ExprMatrix, B: ExprMatrix) -> list[Step]:
    """The steps of the Gauss–Jordan elimination of the augmented matrix
    ``[A | B]`` of the system ``A X = B``: in the last matrix the
    solutions are read off (a row ``0 ... 0 | c`` with ``c`` nonzero
    means that there is none).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.steps import solve_steps
    >>> solve_steps(ExprMatrix.from_rows([[1, 1], [1, -1]]), ExprMatrix.column([3, 1]))[-1].matrix
    ExprMatrix([[1, 0, 2], [0, 1, 1]])
    """
    if A.rows != B.rows:
        raise ValueError('%d equations and %d right-hand sides' % (A.rows, B.rows))
    augmented = A.col_insert(A.cols, B)
    return _elimination(augmented).steps(augmented, True, A.cols)


def determinant_steps(M: ExprMatrix) -> DeterminantSteps:
    """The determinant by elimination to triangular form (see
    :class:`DeterminantSteps`).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.steps import determinant_steps
    >>> a = Symbol('a')
    >>> D = determinant_steps(ExprMatrix.from_rows([[1, a], [a, 1]]))
    >>> D.steps, D.value
    ((Step(R2 <- R2 - (a)*R1, ExprMatrix([[1, a], [0, 1 - a**2]])),), 1 - a**2)
    """
    n = _square(M)
    steps = echelon_steps(M)
    final = steps[-1].matrix if steps else M
    exchanges = sum(1 for step in steps if step.operation.kind == 'swap')
    diagonal = [final[i, i] for i in range(n)]
    sign: Expr = Integer(-1) if exchanges % 2 else Integer(1)
    return DeterminantSteps(steps, exchanges, diagonal, normal(mul(sign, *diagonal)))


def format_steps(M: ExprMatrix, steps: Sequence[Step]) -> str:
    """The matrix and the steps as text: each operation followed by the
    matrix it gives, the columns aligned.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.steps import echelon_steps, format_steps
    >>> M = ExprMatrix.from_rows([[1, 2], [3, 4]])
    >>> print(format_steps(M, echelon_steps(M)))
    [1, 2]
    [3, 4]
    R2 <- R2 - (3)*R1
    [1,  2]
    [0, -2]
    """
    def render(matrix: ExprMatrix) -> list[str]:
        cells = [[str(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)]
        widths = [max((len(cells[i][j]) for i in range(matrix.rows)), default=0) for j in range(matrix.cols)]
        return ['[' + ', '.join(cell.rjust(width) for cell, width in zip(line, widths)) + ']' for line in cells]

    lines = render(M)
    for step in steps:
        lines.append(str(step.operation))
        lines.extend(render(step.matrix))
    return '\n'.join(lines)
