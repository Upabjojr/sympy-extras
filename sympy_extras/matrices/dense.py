"""A dense matrix of expressions, immutable: the entries in row-major
order with the shape. Only what hyperexpand's formula tables need
(SymPy's ``Matrix`` construction, ``eye``, ``zeros``, ``col_insert``,
``row_insert``, the arithmetic, ``subs``, ``diff``, ``applyfunc``,
``has``); every operation returns a new matrix.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices.dense import ExprMatrix
>>> z = Symbol('z')
>>> B = ExprMatrix.column([z, z**2])
>>> M = ExprMatrix.from_rows([[0, 1], [z, 2]])
>>> (M * B).entries, ExprMatrix.from_rows([[1, 0]]) * B
((z**2, 3*z**2), ExprMatrix([[z]]))
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
