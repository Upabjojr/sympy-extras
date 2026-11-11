"""The SymPy face of the lifting phase
(:mod:`sympy_extras.polys.cad.lifting`).

:func:`cylindrical_algebraic_decomposition`, :class:`Lifting` and
:func:`lifting_for` take SymPy polynomials and symbols and give cells
whose sample points are SymPy numbers (``Rational`` and ``CRootOf``, see
:mod:`sympy_extras.polys.cad.samplepoints_sympy`), around the native
liftings: a :class:`CADCell` of the face wraps one native cell (the same
face for the same native cell, so that cells can be keys), and the
stacks are those of the native lifting. :class:`CAD` is the container of
the decompositions which :mod:`sympy_extras.polys.cad.qe` and
:mod:`sympy_extras.polys.cad.cylindrical` build from the cells of a
lifting. Nothing of the lifting stays on SymPy.

Examples
========

>>> from sympy.abc import x, y
>>> from sympy_extras.polys.cad.lifting_sympy import cylindrical_algebraic_decomposition
>>> cad = cylindrical_algebraic_decomposition([x**2 + y**2 - 1], [x, y])
>>> cad
CAD(13 cells, x, y)
>>> for cell in cad:
...     print(cell.index, cell.point, cell.signs)
(1, 1) (-2, 0) (1,)
(2, 1) (-1, -1) (1,)
(2, 2) (-1, 0) (0,)
(2, 3) (-1, 1) (1,)
(3, 1) (0, -2) (1,)
(3, 2) (0, -1) (0,)
(3, 3) (0, 0) (-1,)
(3, 4) (0, 1) (0,)
(3, 5) (0, 2) (1,)
(4, 1) (1, -1) (1,)
(4, 2) (1, 0) (0,)
(4, 3) (1, 1) (1,)
(5, 1) (2, 0) (1,)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Iterator, Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._typing import ExprLike, Sign
from sympy_extras.polys.polyerrors import BasePolynomialError

from . import lifting as native
from .projection_sympy import (_core_input, core_gens, core_poly, core_projection, sympy_errors, sympy_poly)
from .samplepoints_sympy import SamplePoint, sympy_real

__all__ = ['NotWellOriented', 'CADCell', 'CAD', 'Lifting', 'lifting_for', 'cylindrical_algebraic_decomposition']


class NotWellOriented(PolynomialError):
    """:class:`~sympy_extras.polys.cad.lifting.NotWellOriented` as a SymPy
    ``PolynomialError``."""


class _Faces:
    """The cells of the face of the native cells of one lifting or
    decomposition: one face per native cell."""

    __slots__ = ('gens', '_cells')

    def __init__(self, gens: Sequence[Symbol]) -> None:
        self.gens: tuple[Symbol, ...] = tuple(gens)
        self._cells: dict[native.CADCell, CADCell] = {}

    def cell(self, cell: native.CADCell) -> CADCell:
        found = self._cells.get(cell)
        if found is None:
            parent = None if cell.parent is None else self.cell(cell.parent)
            found = CADCell(cell, parent, self.gens)
            self._cells[cell] = found
        return found


class CADCell:
    """A cell of a cylindrical algebraic decomposition: the native cell
    ``native`` with its sample point written with SymPy numbers.

    Attributes
    ==========

    index : tuple of int
        The position of the cell in the decomposition (odd entries for
        sectors, even ones for sections).
    point : tuple of Expr
        The coordinates of the sample point: rational numbers for the
        sectors and ``CRootOf`` roots (possibly times a rational number)
        for the sections.
    sample : SamplePoint
        The same sample point with its coordinates in a common algebraic
        field.
    parent : CADCell or None
        The cell of the previous level over which this cell lies.
    """

    __slots__ = ('native', 'parent', '_gens', '_point', '_sample')

    def __init__(self, cell: native.CADCell, parent: Optional[CADCell], gens: Sequence[Symbol]) -> None:
        self.native: native.CADCell = cell
        self.parent: Optional[CADCell] = parent
        self._gens: tuple[Symbol, ...] = tuple(gens)
        self._point: Optional[tuple[Expr, ...]] = None
        self._sample: Optional[SamplePoint] = None

    @property
    def index(self) -> tuple[int, ...]:
        return self.native.index

    @property
    def point(self) -> tuple[Expr, ...]:
        if self._point is None:
            if self.parent is None:
                self._point = ()
            else:
                level = len(self.native.index)
                self._point = self.parent.point + (sympy_real(self.native.point[-1], self._gens[level - 1]),)
        return self._point

    @property
    def sample(self) -> SamplePoint:
        if self._sample is None:
            with sympy_errors():
                point = self.native.sample
            self._sample = SamplePoint(point, self.point)
        return self._sample

    @property
    def _signs(self) -> tuple[Sign, ...]:
        """The signs of the projection factors of this level on the cell."""
        return self.native._signs

    @property
    def signs(self) -> tuple[Sign, ...]:
        """The signs of the input polynomials (cells of the last level of
        a :func:`cylindrical_algebraic_decomposition`)."""
        return self.native.signs

    @signs.setter
    def signs(self, value: tuple[Sign, ...]) -> None:
        self.native.signs = value

    @property
    def level(self) -> int:
        return self.native.level

    @property
    def dimension(self) -> int:
        """The dimension of the cell: the number of sectors in its index."""
        return self.native.dimension

    @property
    def is_section(self) -> bool:
        """Whether the cell is a section (a root) over its parent."""
        return self.native.is_section

    def __repr__(self) -> str:
        return "CADCell(%s, %s)" % (self.index, self.point)

    def _factor_sign(self, level: int, i: int) -> Sign:
        """Sign of the ``i``-th projection factor of level ``level`` at the
        sample point of this cell."""
        return self.native.factor_sign(level, i)


class CAD:
    """A cylindrical algebraic decomposition of `\\mathbb{R}^n`, sign-invariant
    for a set of SymPy polynomials, with the cells of the face.

    Attributes
    ==========

    gens : tuple of Symbol
        The variables, in the order of the levels.
    polys : list of Poly
        The input polynomials.
    projection : list of list of Poly
        The projection factor sets ``P_1, ..., P_n``.
    method : str
        The projection operator used, ``'mccallum'`` or ``'hong'``.
    cells : list of CADCell
        The cells of `\\mathbb{R}^n`, in lexicographic order of their indices.
    """

    def __init__(self, gens: Sequence[Symbol], polys: list[Poly], projection: list[list[Poly]],
                 method: str, levels: list[list[CADCell]]) -> None:
        self.gens: tuple[Symbol, ...] = tuple(gens)
        self.polys: list[Poly] = polys
        self.projection: list[list[Poly]] = projection
        self.method: str = method
        self._levels: list[list[CADCell]] = levels
        self.cells: list[CADCell] = levels[-1]
        self._children: dict[CADCell, list[CADCell]] = {}
        for level in levels:
            for cell in level:
                if cell.parent is not None:
                    self._children.setdefault(cell.parent, []).append(cell)

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[CADCell]:
        return iter(self.cells)

    def __repr__(self) -> str:
        return "CAD(%d cells, %s)" % (len(self.cells), ", ".join(map(str, self.gens)))

    def cells_at(self, level: int) -> list[CADCell]:
        """The cells of `\\mathbb{R}^k` for ``level = k``, ``1 <= k <= n``."""
        if not 1 <= level <= len(self.gens):
            raise ValueError("level must be between 1 and %d" % len(self.gens))
        return self._levels[level - 1]

    def children(self, cell: CADCell) -> list[CADCell]:
        """The cells of the next level lying over ``cell``."""
        return list(self._children.get(cell, []))


class Lifting:
    """:class:`~sympy_extras.polys.cad.lifting.Lifting` on SymPy
    polynomials: the stacks are those of the native lifting, with the
    cells of the face.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.projection_sympy import projection_sets
    >>> from sympy_extras.polys.cad.lifting_sympy import Lifting
    >>> lifting = Lifting(projection_sets([x**2 + y**2 - 1], [x, y]), [x, y], 'mccallum')
    >>> line = lifting.stack(lifting.root)
    >>> [cell.point for cell in line]
    [(-2,), (-1,), (0,), (1,), (2,)]
    >>> [(cell.point, cell.index) for cell in lifting.stack(line[2])]
    [((0, -2), (3, 1)), ((0, -1), (3, 2)), ((0, 0), (3, 3)), ((0, 1), (3, 4)), ((0, 2), (3, 5))]
    >>> lifting.lifted
    10
    """

    def __init__(self, projection: Sequence[Sequence[Poly]], gens: Sequence[Symbol], method: str,
                 lifting: Optional[native.Lifting] = None) -> None:
        self.projection: list[list[Poly]] = [list(level) for level in projection]
        self.gens: tuple[Symbol, ...] = tuple(gens)
        self.method: str = method
        if lifting is None:
            lifting = native.Lifting(core_projection(self.projection), core_gens(self.gens), method)
        self.native: native.Lifting = lifting
        self._faces: _Faces = _Faces(self.gens)
        self._stacks: dict[CADCell, list[CADCell]] = {}
        self.root: CADCell = self._faces.cell(lifting.root)

    @property
    def lifted(self) -> int:
        """The number of cells built so far."""
        return self.native.lifted

    def is_lifted(self, parent: CADCell) -> bool:
        """Whether the stack over ``parent`` was built."""
        return self.native.is_lifted(parent.native)

    def stack(self, parent: CADCell) -> list[CADCell]:
        """The cells of the next level over ``parent``, from the lowest
        one up (:class:`NotWellOriented` as in the native lifting)."""
        found = self._stacks.get(parent)
        if found is not None:
            return found
        try:
            cells = self.native.stack(parent.native)
        except native.NotWellOriented as error:
            raise NotWellOriented(str(error)) from error
        except BasePolynomialError as error:
            raise PolynomialError(str(error)) from error
        found = [self._faces.cell(cell) for cell in cells]
        self._stacks[parent] = found
        return found

    def levels(self) -> list[list[CADCell]]:
        """The cells built so far, by level."""
        result: list[list[CADCell]] = []
        current = [self.root]
        for _ in self.gens:
            cells = [cell for parent in current if self.is_lifted(parent) for cell in self.stack(parent)]
            result.append(cells)
            current = cells
        return result

    def lift_all(self) -> list[list[CADCell]]:
        """Build every stack: the full decomposition."""
        current = [self.root]
        for _ in self.gens:
            current = [cell for parent in current for cell in self.stack(parent)]
        return self.levels()


#: the faces of the native liftings kept by
#: :func:`~sympy_extras.polys.cad.lifting.lifting_for`
_liftings: OrderedDict[native.Lifting, Lifting] = OrderedDict()

#: how many faces are kept (as many as native liftings)
_KEPT = native.KEPT


def lifting_for(projection: Sequence[Sequence[Poly]], gens: Sequence[Symbol], method: str) -> Lifting:
    """:func:`~sympy_extras.polys.cad.lifting.lifting_for` on SymPy
    polynomials: the face of the native lifting kept for the same sets.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.projection_sympy import projection_sets
    >>> from sympy_extras.polys.cad.lifting_sympy import lifting_for
    >>> projection = projection_sets([x**2 + y**2 - 1], [x, y])
    >>> lifting_for(projection, [x, y], 'mccallum') is lifting_for(projection, [x, y], 'mccallum')
    True
    """
    lifting = native.lifting_for(core_projection(projection), core_gens(gens), method)
    found = _liftings.get(lifting)
    if found is None:
        found = Lifting(projection, gens, method, lifting)
        _liftings[lifting] = found
        while len(_liftings) > _KEPT:
            _liftings.popitem(last=False)
    else:
        _liftings.move_to_end(lifting)
    return found


def _factor_signs(polys: Sequence[Poly], projection: Sequence[Sequence[Poly]],
                  gens: Sequence[Symbol]) -> native.FactorSigns:
    """:func:`~sympy_extras.polys.cad.lifting.factor_signs` on SymPy
    polynomials."""
    generators = core_gens(gens)
    with sympy_errors():
        return native.factor_signs([core_poly(p, generators) for p in polys], core_projection(projection),
                                   generators)


def cylindrical_algebraic_decomposition(polys: Iterable[Union[ExprLike, Poly]], gens: Sequence[Symbol],
                                        method: Optional[str] = None) -> CAD:
    """:func:`~sympy_extras.polys.cad.lifting.cylindrical_algebraic_decomposition`
    on SymPy polynomials and symbols: the cells give their sample points
    as SymPy numbers."""
    generators = tuple(gens)
    if not generators:
        raise ValueError("at least one generator is needed")
    inputs = [_core_input(p) for p in polys]
    try:
        cad = native.cylindrical_algebraic_decomposition(inputs, core_gens(generators), method)
    except native.NotWellOriented as error:
        raise NotWellOriented(str(error)) from error
    except BasePolynomialError as error:
        raise PolynomialError(str(error)) from error
    faces = _Faces(generators)
    levels = [[faces.cell(cell) for cell in level] for level in cad.levels()]
    projection = [[sympy_poly(p, generators[:k]) for p in level] for k, level in enumerate(cad.projection, start=1)]
    return CAD(generators, [sympy_poly(p, generators) for p in cad.polys], projection, cad.method, levels)
