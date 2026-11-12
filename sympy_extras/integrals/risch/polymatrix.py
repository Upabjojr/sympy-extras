"""Matrices over the field of the Risch algorithm: :class:`FieldMatrix`, an
immutable dense matrix of :class:`~sympy_extras.integrals.risch.tower.
RationalFunction` elements with the operations the parametric problems use
(joins, products, the reduced row echelon form and the nullspace by
Gauss–Jordan elimination over the field).

It takes the place of SymPy's ``PolyMatrix`` (``sympy.polys.polymatrix``),
whose entries were polynomials of the tower: an element of ``k[t]`` is an
element of the field here.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.integrals.risch.polymatrix import FieldMatrix
>>> from sympy_extras.integrals.risch.tower import RationalFunction
>>> x = symbols('x')
>>> X = RationalFunction.from_expr(x)
>>> M = FieldMatrix.from_rows([[X, X**2], [1, X]])
>>> M.rref()
(FieldMatrix([[1, x], [0, 0]]), (0,))
>>> M.nullspace()
[FieldMatrix([[-x], [1]])]
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/polymatrix.py (SymPy 1.14 and Aaron Meurer's pull requests)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: rewritten as an immutable typed matrix over the field of the
# tower (RationalFunction entries); the interface of PolyMatrix kept.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Final, Iterator, Sequence, Union, overload

from .tower import RationalFunction

__all__ = ['FieldMatrix', 'Entry']

#: what the constructors take as an entry
Entry = Union[RationalFunction, int]


def _entry(e: Entry) -> RationalFunction:
    if isinstance(e, int):
        return RationalFunction.from_int(e)
    return e


_ZERO = RationalFunction.from_int(0)
_ONE = RationalFunction.from_int(1)


class FieldMatrix:
    """An immutable ``rows x cols`` matrix over the field of the tower,
    stored row by row."""

    __slots__ = ('rows', 'cols', '_entries')

    def __init__(self, rows: int, cols: int, entries: Sequence[Entry]) -> None:
        if len(entries) != rows*cols:
            raise ValueError('%d entries for a %dx%d matrix' % (len(entries), rows, cols))
        self.rows: Final[int] = rows
        self.cols: Final[int] = cols
        self._entries: Final[tuple[RationalFunction, ...]] = tuple(_entry(e) for e in entries)

    # -- construction

    @staticmethod
    def from_function(rows: int, cols: int, f: Callable[[int, int], Entry]) -> FieldMatrix:
        return FieldMatrix(rows, cols, [f(i, j) for i in range(rows) for j in range(cols)])

    @staticmethod
    def from_rows(rows: Sequence[Sequence[Entry]], cols: int = 0) -> FieldMatrix:
        """The matrix of a list of rows (``cols`` for an empty list)."""
        if not rows:
            return FieldMatrix(0, cols, [])
        n = len(rows[0])
        if any(len(r) != n for r in rows):
            raise ValueError('rows of different lengths')
        return FieldMatrix(len(rows), n, [e for r in rows for e in r])

    @staticmethod
    def row_vector(entries: Sequence[Entry]) -> FieldMatrix:
        return FieldMatrix(1, len(entries), entries)

    @staticmethod
    def column(entries: Sequence[Entry]) -> FieldMatrix:
        return FieldMatrix(len(entries), 1, entries)

    @staticmethod
    def zeros(m: int, n: int) -> FieldMatrix:
        return FieldMatrix(m, n, [_ZERO]*(m*n))

    @staticmethod
    def eye(n: int) -> FieldMatrix:
        return FieldMatrix(n, n, [_ONE if i == j else _ZERO for i in range(n) for j in range(n)])

    # -- access

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def __len__(self) -> int:
        return self.rows*self.cols

    def __iter__(self) -> Iterator[RationalFunction]:
        return iter(self._entries)

    def flat(self) -> list[RationalFunction]:
        return list(self._entries)

    def entry(self, i: int, j: int) -> RationalFunction:
        if i < 0:
            i += self.rows
        if j < 0:
            j += self.cols
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError('index (%d, %d) out of range' % (i, j))
        return self._entries[i*self.cols + j]

    def element(self, k: int) -> RationalFunction:
        """The ``k``-th entry in the order of the rows."""
        return self._entries[k]

    def elements(self, key: slice) -> list[RationalFunction]:
        """The entries of a slice, in the order of the rows."""
        return list(self._entries[key])

    @overload
    def __getitem__(self, key: tuple[int, int]) -> RationalFunction: ...
    @overload
    def __getitem__(self, key: tuple[Union[int, slice], slice]) -> FieldMatrix: ...
    @overload
    def __getitem__(self, key: tuple[slice, Union[int, slice]]) -> FieldMatrix: ...

    def __getitem__(self, key: tuple[Union[int, slice], Union[int, slice]]
                    ) -> Union[RationalFunction, FieldMatrix]:
        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            return self.entry(i, j)
        rows = range(self.rows)[i] if isinstance(i, slice) else [range(self.rows)[i]]
        cols = range(self.cols)[j] if isinstance(j, slice) else [range(self.cols)[j]]
        return FieldMatrix(len(rows), len(cols), [self._entries[r*self.cols + c] for r in rows for c in cols])

    def row(self, i: int) -> FieldMatrix:
        return self[i, :]

    def col(self, j: int) -> FieldMatrix:
        return self[:, j]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldMatrix):
            return NotImplemented
        return self.shape == other.shape and self._entries == other._entries

    def __hash__(self) -> int:
        return hash((self.shape, self._entries))

    def __repr__(self) -> str:
        rows = [[str(self.entry(i, j)) for j in range(self.cols)] for i in range(self.rows)]
        if not rows:
            return 'FieldMatrix(%d, %d, [])' % (self.rows, self.cols)
        return 'FieldMatrix([%s])' % ', '.join('[%s]' % ', '.join(r) for r in rows)

    __str__ = __repr__

    # -- arithmetic

    def __neg__(self) -> FieldMatrix:
        return FieldMatrix(self.rows, self.cols, [-e for e in self._entries])

    def __add__(self, other: FieldMatrix) -> FieldMatrix:
        if self.shape != other.shape:
            raise ValueError('matrices of shapes %s and %s' % (self.shape, other.shape))
        return FieldMatrix(self.rows, self.cols, [a + b for a, b in zip(self._entries, other._entries)])

    def __sub__(self, other: FieldMatrix) -> FieldMatrix:
        if self.shape != other.shape:
            raise ValueError('matrices of shapes %s and %s' % (self.shape, other.shape))
        return FieldMatrix(self.rows, self.cols, [a - b for a, b in zip(self._entries, other._entries)])

    def __mul__(self, other: Union[FieldMatrix, Entry]) -> FieldMatrix:
        if isinstance(other, FieldMatrix):
            if self.cols != other.rows:
                raise ValueError('matrices of shapes %s and %s' % (self.shape, other.shape))
            entries: list[RationalFunction] = []
            for i in range(self.rows):
                for j in range(other.cols):
                    total = _ZERO
                    for k in range(self.cols):
                        a = self._entries[i*self.cols + k]
                        if a:
                            b = other._entries[k*other.cols + j]
                            if b:
                                total = total + a*b
                    entries.append(total)
            return FieldMatrix(self.rows, other.cols, entries)
        c = _entry(other)
        return FieldMatrix(self.rows, self.cols, [e*c for e in self._entries])

    def __rmul__(self, other: Entry) -> FieldMatrix:
        return self*other

    def __truediv__(self, other: Entry) -> FieldMatrix:
        inverse = _entry(other).invert()
        return self*inverse

    def row_join(self, other: FieldMatrix) -> FieldMatrix:
        if self.rows != other.rows:
            if not len(self) and not self.cols:
                return other
            raise ValueError('row_join of %s and %s' % (self.shape, other.shape))
        entries: list[RationalFunction] = []
        for i in range(self.rows):
            entries.extend(self._entries[i*self.cols:(i + 1)*self.cols])
            entries.extend(other._entries[i*other.cols:(i + 1)*other.cols])
        return FieldMatrix(self.rows, self.cols + other.cols, entries)

    def col_join(self, other: FieldMatrix) -> FieldMatrix:
        if self.cols != other.cols:
            if not len(self) and not self.rows:
                return other
            raise ValueError('col_join of %s and %s' % (self.shape, other.shape))
        return FieldMatrix(self.rows + other.rows, self.cols, list(self._entries) + list(other._entries))

    def applyfunc(self, f: Callable[[RationalFunction], RationalFunction]) -> FieldMatrix:
        return FieldMatrix(self.rows, self.cols, [f(e) for e in self._entries])

    def transpose(self) -> FieldMatrix:
        return FieldMatrix(self.cols, self.rows, [self.entry(i, j) for j in range(self.cols) for i in range(self.rows)])

    # -- linear algebra over the field

    def rref(self) -> tuple[FieldMatrix, tuple[int, ...]]:
        """The reduced row echelon form and the pivot columns (Gauss–Jordan
        elimination over the field)."""
        rows = [list(self._entries[i*self.cols:(i + 1)*self.cols]) for i in range(self.rows)]
        pivots: list[int] = []
        r = 0
        for c in range(self.cols):
            if r >= self.rows:
                break
            p = next((i for i in range(r, self.rows) if rows[i][c]), None)
            if p is None:
                continue
            rows[r], rows[p] = rows[p], rows[r]
            inverse = rows[r][c].invert()
            rows[r] = [e*inverse if e else e for e in rows[r]]
            for i in range(self.rows):
                if i != r and rows[i][c]:
                    factor = rows[i][c]
                    rows[i] = [a - factor*b if b else a for a, b in zip(rows[i], rows[r])]
            pivots.append(c)
            r += 1
        return FieldMatrix(self.rows, self.cols, [e for row in rows for e in row]), tuple(pivots)

    def nullspace(self) -> list[FieldMatrix]:
        """A basis of the column vectors ``v`` with ``M v = 0``: one for
        each column without a pivot, ``1`` there (SymPy's basis)."""
        R, pivots = self.rref()
        basis: list[FieldMatrix] = []
        for free in range(self.cols):
            if free in pivots:
                continue
            v = [_ZERO]*self.cols
            v[free] = _ONE
            for k, c in enumerate(pivots):
                value = R.entry(k, free)
                if value:
                    v[c] = -value
            basis.append(FieldMatrix.column(v))
        return basis

    def rank(self) -> int:
        return len(self.rref()[1])
