"""A dense matrix of expressions, immutable: the entries in row-major
order with the shape. Only what hyperexpand's formula tables need
(SymPy's ``Matrix`` construction, ``eye``, ``zeros``, ``col_insert``,
``row_insert``, the arithmetic, ``subs``, ``diff``, ``applyfunc``,
``has``); every operation returns a new matrix. The linear algebra
(``det``, ``inv``, ``rank``, ``rref``, ``nullspace``, ``adjugate``,
``charpoly``, ``solve``) is that of :mod:`sympy_extras.matrices.linalg`,
the eigenvalues (``eigenvals``, ``eigenvects``, ``is_diagonalizable``,
``diagonalize``, ``jordan_form``) that of :mod:`sympy_extras.matrices.eigen`.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices.dense import ExprMatrix
>>> z = Symbol('z')
>>> B = ExprMatrix.column([z, z**2])
>>> M = ExprMatrix.from_rows([[0, 1], [z, 2]])
>>> (M * B).entries, ExprMatrix.from_rows([[1, 0]]) * B
((z**2, 3*z**2), ExprMatrix([[z]]))
>>> M.det(), M.inv(), M.rank()
(-z, ExprMatrix([[-2/z, 1/z], [1, 0]]), 2)
>>> ExprMatrix.eye(2).col_insert(0, ExprMatrix.zeros(2, 1)).row_insert(2, ExprMatrix.from_rows([[1, 2, 3]]))
ExprMatrix([[0, 1, 0], [0, 0, 1], [1, 2, 3]])
"""
from __future__ import annotations

from typing import Callable, Iterator, Sequence, Union

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Symbol

__all__ = ['ExprMatrix']

SubsArg = Union[dict[Basic, Basic], dict[Symbol, Expr], Sequence[tuple[Basic, Basic]]]


class ExprMatrix:
    """``rows`` by ``cols`` expressions, ``entries`` in row-major order."""

    __slots__ = ('rows', 'cols', 'entries')

    def __init__(self, rows: int, cols: int, entries: Sequence[ExprLike]) -> None:
        if len(entries) != rows * cols:
            raise ValueError('%d entries for a %d x %d matrix' % (len(entries), rows, cols))
        self.rows: int = rows
        self.cols: int = cols
        self.entries: tuple[Expr, ...] = tuple(to_expr(e) for e in entries)

    @classmethod
    def column(cls, entries: Sequence[ExprLike]) -> ExprMatrix:
        """SymPy's ``Matrix([a, b, ...])``: a column."""
        return cls(len(entries), 1, entries)

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[ExprLike]]) -> ExprMatrix:
        """SymPy's ``Matrix([[...], [...]])``."""
        cols = len(rows[0]) if rows else 0
        if any(len(r) != cols for r in rows):
            raise ValueError('rows of different lengths')
        return cls(len(rows), cols, [e for r in rows for e in r])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> ExprMatrix:
        return cls(rows, cols, [Integer(0)] * (rows * cols))

    @classmethod
    def eye(cls, n: int) -> ExprMatrix:
        return cls(n, n, [Integer(1) if i == j else Integer(0) for i in range(n) for j in range(n)])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, ij: tuple[int, int]) -> Expr:
        i, j = ij
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError('entry %s of a %d x %d matrix' % (ij, self.rows, self.cols))
        return self.entries[i * self.cols + j]

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def row(self, i: int) -> tuple[Expr, ...]:
        return self.entries[i * self.cols:(i + 1) * self.cols]

    def with_entry(self, i: int, j: int, value: ExprLike) -> ExprMatrix:
        """The matrix with the entry ``(i, j)`` replaced (SymPy's ``M[i, j] = v``)."""
        entries = list(self.entries)
        entries[i * self.cols + j] = to_expr(value)
        return ExprMatrix(self.rows, self.cols, entries)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExprMatrix) and self.shape == other.shape and self.entries == other.entries

    def __hash__(self) -> int:
        return hash((self.rows, self.cols, self.entries))

    def __repr__(self) -> str:
        rows = ('[%s]' % ', '.join(str(e) for e in self.row(i)) for i in range(self.rows))
        return 'ExprMatrix([%s])' % ', '.join(rows)

    __str__ = __repr__

    def transpose(self) -> ExprMatrix:
        return ExprMatrix(self.cols, self.rows, [self.entries[i*self.cols + j] for j in range(self.cols) for i in range(self.rows)])

    # arithmetic

    def __add__(self, other: ExprMatrix) -> ExprMatrix:
        if self.shape != other.shape:
            raise ValueError('shapes %s and %s' % (self.shape, other.shape))
        return ExprMatrix(self.rows, self.cols, [a + b for a, b in zip(self.entries, other.entries)])

    def __sub__(self, other: ExprMatrix) -> ExprMatrix:
        return self + (-other)

    def __neg__(self) -> ExprMatrix:
        return ExprMatrix(self.rows, self.cols, [-a for a in self.entries])

    def __mul__(self, other: Union[ExprMatrix, ExprLike]) -> ExprMatrix:
        if not isinstance(other, ExprMatrix):
            c = to_expr(other)
            return ExprMatrix(self.rows, self.cols, [a * c for a in self.entries])
        if self.cols != other.rows:
            raise ValueError('shapes %s and %s' % (self.shape, other.shape))
        from sympy_extras.core.add import add
        entries: list[Expr] = []
        for i in range(self.rows):
            r = self.row(i)
            for j in range(other.cols):
                entries.append(add(*[r[k] * other[k, j] for k in range(self.cols)]))
        return ExprMatrix(self.rows, other.cols, entries)

    def __rmul__(self, other: ExprLike) -> ExprMatrix:
        c = to_expr(other)
        return ExprMatrix(self.rows, self.cols, [c * a for a in self.entries])

    def __truediv__(self, other: ExprLike) -> ExprMatrix:
        c = to_expr(other)
        return ExprMatrix(self.rows, self.cols, [a / c for a in self.entries])

    # structure

    def col_insert(self, pos: int, other: ExprMatrix) -> ExprMatrix:
        if other.rows != self.rows:
            raise ValueError('%d rows inserted in a matrix of %d' % (other.rows, self.rows))
        rows = [list(self.row(i)[:pos]) + list(other.row(i)) + list(self.row(i)[pos:]) for i in range(self.rows)]
        return ExprMatrix(self.rows, self.cols + other.cols, [e for r in rows for e in r])

    def row_insert(self, pos: int, other: ExprMatrix) -> ExprMatrix:
        if self.rows and other.cols != self.cols:
            raise ValueError('%d columns inserted in a matrix of %d' % (other.cols, self.cols))
        rows = [self.row(i) for i in range(self.rows)]
        new = [other.row(i) for i in range(other.rows)]
        all_rows = rows[:pos] + new + rows[pos:]
        return ExprMatrix(len(all_rows), other.cols, [e for r in all_rows for e in r])

    # entrywise

    def applyfunc(self, f: Callable[[Expr], Expr]) -> ExprMatrix:
        return ExprMatrix(self.rows, self.cols, [f(a) for a in self.entries])

    def subs(self, *args: Union[SubsArg, ExprLike, Basic]) -> ExprMatrix:
        def s(a: Expr) -> Expr:
            r = a.subs(*args)
            assert isinstance(r, Expr)
            return r
        return self.applyfunc(s)

    def xreplace(self, rule: dict[Basic, Basic]) -> ExprMatrix:
        def s(a: Expr) -> Expr:
            r = a.xreplace(rule)
            assert isinstance(r, Expr)
            return r
        return self.applyfunc(s)

    def diff(self, x: Symbol) -> ExprMatrix:
        return self.applyfunc(lambda a: a.diff(x))

    def has(self, *patterns: Union[Basic, type]) -> bool:
        return any(a.has(*patterns) for a in self.entries)

    # linear algebra (sympy_extras.matrices.linalg, which imports this module)

    def det(self) -> Expr:
        """The determinant (fraction-free elimination)."""
        from .linalg import det
        return det(self)

    def inv(self) -> ExprMatrix:
        """The inverse; ``ValueError`` for a singular matrix."""
        from .linalg import inverse
        return inverse(self)

    def adjugate(self) -> ExprMatrix:
        from .linalg import adjugate
        return adjugate(self)

    def rank(self) -> int:
        from .linalg import rank
        return rank(self)

    def rref(self) -> tuple[ExprMatrix, tuple[int, ...]]:
        """The reduced row echelon form and the pivot columns."""
        from .linalg import rref
        return rref(self)

    def nullspace(self) -> list[ExprMatrix]:
        from .linalg import nullspace
        return nullspace(self)

    def charpoly(self, x: Expr) -> Expr:
        """The characteristic polynomial ``det(x*I - M)``."""
        from .linalg import charpoly
        return charpoly(self, x)

    def solve(self, rhs: ExprMatrix) -> ExprMatrix:
        """The solution ``X`` of ``self * X = rhs`` (the one with the free
        unknowns zero when there are several); ``ValueError`` when there
        is none."""
        from .linalg import linear_solve
        solution = linear_solve(self, rhs)
        if solution is None:
            raise ValueError('the system has no solution')
        return solution.particular

    def pinv(self) -> ExprMatrix:
        """The Moore-Penrose pseudoinverse (real entries)."""
        from .linalg import pseudoinverse
        return pseudoinverse(self)

    # eigenvalues (sympy_extras.matrices.eigen)

    def eigenvals(self) -> dict[Expr, int]:
        """The eigenvalues with their algebraic multiplicities."""
        from .eigen import eigenvalues
        return eigenvalues(self)

    def eigenvects(self) -> list[tuple[Expr, int, list[ExprMatrix]]]:
        """``(eigenvalue, multiplicity, basis of the eigenspace)``."""
        from .eigen import eigenvectors
        return eigenvectors(self)

    def is_diagonalizable(self) -> bool:
        from .eigen import is_diagonalizable
        return is_diagonalizable(self)

    def diagonalize(self) -> tuple[ExprMatrix, ExprMatrix]:
        """``(P, D)`` with ``self * P = P * D``; ``ValueError`` when the
        matrix is not diagonalisable."""
        from .eigen import diagonalize
        result = diagonalize(self)
        if result is None:
            raise ValueError('the matrix is not diagonalizable')
        return result

    def jordan_form(self) -> tuple[ExprMatrix, ExprMatrix]:
        """``(P, J)`` with ``J`` the Jordan normal form and
        ``self * P = P * J``."""
        from .eigen import jordan_form
        return jordan_form(self)

    # singular values (sympy_extras.matrices.svd)

    def singular_values(self) -> list[Expr]:
        from .svd import singular_values
        return singular_values(self)

    def svd(self) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix]:
        """The compact singular value decomposition ``(U, S, V)`` with
        ``self = U * S * V.transpose()``."""
        from .svd import svd
        return svd(self)
