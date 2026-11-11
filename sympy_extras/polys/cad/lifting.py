"""Lifting phase of cylindrical algebraic decomposition.

Given the projection factor sets `P_1, \\ldots, P_n` of a set of
polynomials (see :func:`~.projection_sets`), the lifting phase builds the
decomposition level by level. The real line is split at the real roots of
`P_1` into *sections* (the roots) and *sectors* (the open intervals between
them). Over every cell `c` of `\\mathbb{R}^{k-1}` the polynomials of `P_k`
are evaluated at the sample point of `c`; their real roots split the
cylinder `c \\times \\mathbb{R}` into sections and sectors again, each with
its own sample point. The polynomials are sign-invariant on every cell, so
the sign of every input polynomial on a cell is read off its sample point.

The module is native: polynomials are the typed
:class:`~sympy_extras.polys.polytools.Poly` in symbols of the core, the
coordinates of the sample points are real
:class:`~sympy_extras.polys.numberfields.qqbar.Algebraic` numbers, and
SymPy is not imported; the SymPy face is
:mod:`sympy_extras.polys.cad.lifting_sympy`.
"""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from typing import Iterable, Iterator, Optional, Sequence

from sympy_extras.core.expr import Expr
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.polytools import Poly

from .projection import PolyLike, projection_sets, to_polys
from .samplepoints import (SamplePoint, Sign, Specialization, _order, rational_between, rational_below,
    rational_above)

__all__ = ['NotWellOriented', 'CADCell', 'CAD', 'Lifting', 'lifting_for', 'factor_signs',
           'cylindrical_algebraic_decomposition', 'FactorSigns']

#: for every input polynomial, the sign of its constant factor and the
#: positions ``(level, index, exponent)`` of its irreducible factors in
#: the projection factor sets
FactorSigns = list[tuple[Sign, list[tuple[int, int, int]]]]


class NotWellOriented(PolynomialError):
    """Raised when a projection factor vanishes identically on a cell of
    positive dimension, so that McCallum's projection is not guaranteed to
    give a sign-invariant decomposition."""


class CADCell:
    """A cell of a cylindrical algebraic decomposition.

    Attributes
    ==========

    index : tuple of int
        The position of the cell in the decomposition, one integer per
        level: the `k`-th entry is odd for a sector and even for a section
        of the cylinder over the parent cell, counted from 1 upwards.
    point : tuple of Algebraic
        The coordinates of an exact sample point of the cell: rational
        numbers for the sectors and real algebraic numbers for the
        sections.
    sample : SamplePoint
        The same sample point with its coordinates in a common algebraic
        field, for exact sign evaluations. It is computed when it is first
        asked for: the signs of the projection factors on a cell are known
        from the roots of the stack, and the field of a section, which
        takes a primitive element, is only needed to lift over it.
    parent : CADCell or None
        The cell of the previous level over which this cell lies.
    """

    __slots__ = ('index', 'point', '_sample', 'parent', '_signs', 'signs')

    def __init__(self, index: tuple[int, ...], point: tuple[Algebraic, ...], sample: Optional[SamplePoint],
                 parent: Optional[CADCell], signs: tuple[Sign, ...]) -> None:
        self.index: tuple[int, ...] = index
        self.point: tuple[Algebraic, ...] = point
        self._sample: Optional[SamplePoint] = sample
        self.parent: Optional[CADCell] = parent
        # signs of the projection factors of this level at the sample point
        self._signs: tuple[Sign, ...] = signs
        # signs of the input polynomials, filled by CAD for the top level
        # (empty for the cells of the lower levels)
        self.signs: tuple[Sign, ...] = ()

    @property
    def sample(self) -> SamplePoint:
        if self._sample is None:
            if self.parent is None:
                self._sample = SamplePoint()
            else:
                self._sample = self.parent.sample.extend(self.point[-1])
        return self._sample

    @property
    def level(self) -> int:
        return len(self.index)

    @property
    def dimension(self) -> int:
        """The dimension of the cell: the number of sectors in its index."""
        return sum(i % 2 for i in self.index)

    @property
    def is_section(self) -> bool:
        """Whether the cell is a section (a root) over its parent."""
        return self.index[-1] % 2 == 0

    def __repr__(self) -> str:
        return "CADCell(%s, %s)" % (self.index, self.point)

    def factor_sign(self, level: int, i: int) -> Sign:
        """Sign of the ``i``-th projection factor of level ``level`` at the
        sample point of this cell."""
        cell: CADCell = self
        while cell.level > level:
            if cell.parent is None:
                raise ValueError("no cell of level %d below %s" % (level, self))
            cell = cell.parent
        return cell._signs[i]


class CAD:
    """A cylindrical algebraic decomposition of `\\mathbb{R}^n`, sign-invariant
    for a set of polynomials.

    Instances are built by :func:`cylindrical_algebraic_decomposition`.

    Attributes
    ==========

    gens : tuple of Expr
        The variables, in the order of the levels.
    polys : list of Poly
        The input polynomials.
    projection : list of list of Poly
        The projection factor sets ``P_1, ..., P_n``.
    method : str
        The projection operator used, ``'mccallum'`` or ``'hong'``.
    cells : list of CADCell
        The cells of `\\mathbb{R}^n`, in lexicographic order of their indices.
        The ``signs`` attribute of each cell is the tuple of signs of the
        input polynomials on the cell.
    """

    def __init__(self, gens: Sequence[Expr], polys: list[Poly], projection: list[list[Poly]],
                 method: str, levels: list[list[CADCell]]) -> None:
        self.gens: tuple[Expr, ...] = tuple(gens)
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

    def levels(self) -> list[list[CADCell]]:
        """The cells of every level."""
        return [list(level) for level in self._levels]

    def children(self, cell: CADCell) -> list[CADCell]:
        """The cells of the next level lying over ``cell``."""
        return list(self._children.get(cell, []))


#: a section of a stack: the root and the indices of the projection factors
#: of the level which vanish there
_Section = tuple[Algebraic, frozenset[int]]


def _sections(roots: Sequence[tuple[Algebraic, int]]) -> list[_Section]:
    """The sections of a stack, in increasing order, from the roots of its
    polynomials paired with the indices of the polynomials: a root of
    several polynomials is one section."""
    values = [root for root, _ in roots]
    return [(values[group[0]], frozenset(roots[i][1] for i in group)) for group in _order(values)]


class Lifting:
    """The lifting phase, one stack at a time and on demand: the partial
    decompositions of :mod:`sympy_extras.polys.cad.qe` only ask for the
    stacks which the truth value of the formula depends on [1]_.

    The signs of the projection factors on the cells of a stack are read
    off the real roots of the factors over the sample point of the parent
    cell: a factor vanishes on the sections which are its roots, has the
    sign of its leading coefficient above its last root and changes sign at
    its roots of odd multiplicity. No polynomial is evaluated at the sample
    point of a cell of the stack, which is only computed (a primitive
    element for a section over a cell with algebraic coordinates) when a
    stack is built over the cell.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.projection import projection_sets
    >>> from sympy_extras.polys.cad.lifting import Lifting
    >>> x, y = symbols('x y')
    >>> lifting = Lifting(projection_sets([x**2 + y**2 - 1], [x, y]), [x, y], 'mccallum')
    >>> line = lifting.stack(lifting.root)
    >>> [[float(c) for c in cell.point] for cell in line]
    [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    >>> [cell.index for cell in lifting.stack(line[2])]
    [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
    >>> lifting.lifted
    10

    References
    ==========

    .. [1] G. E. Collins, H. Hong, Partial cylindrical algebraic
           decomposition for quantifier elimination, J. Symbolic Comput. 12
           (1991), pp. 299-328.
    """

    def __init__(self, projection: Sequence[Sequence[Poly]], gens: Sequence[Expr], method: str) -> None:
        self.projection: list[list[Poly]] = [list(level) for level in projection]
        self.gens: tuple[Expr, ...] = tuple(gens)
        self.method: str = method
        self.root: CADCell = CADCell((), (), SamplePoint(), None, ())
        self._stacks: dict[CADCell, list[CADCell]] = {}
        #: the number of cells built so far
        self.lifted: int = 0

    def is_lifted(self, parent: CADCell) -> bool:
        """Whether the stack over ``parent`` was built."""
        return parent in self._stacks

    def stack(self, parent: CADCell) -> list[CADCell]:
        """The cells of the next level over ``parent``, from the lowest
        one up. :class:`NotWellOriented` is raised when a factor of
        McCallum's projection vanishes identically over a cell of positive
        dimension."""
        found = self._stacks.get(parent)
        if found is not None:
            return found
        k = parent.level + 1
        if k > len(self.gens):
            raise ValueError("the cell is of the last level")
        polys = self.projection[k - 1]
        level_gens = self.gens[:k]
        sample = parent.sample
        specializations: list[Specialization] = []
        roots_found: list[tuple[Algebraic, int]] = []
        for i, f in enumerate(polys):
            specialized = sample.specialization(f, level_gens)
            if specialized.degree < 0 and self.method == 'mccallum' and parent.dimension > 0:
                raise NotWellOriented("%s vanishes identically on a cell of dimension %d"
                                      % (f.as_expr(), parent.dimension))
            specializations.append(specialized)
            roots_found.extend((r, i) for r, _ in specialized.roots)
        sections = _sections(roots_found)
        roots = [root for root, _ in sections]
        values: list[Algebraic] = []
        if not roots:
            values.append(Algebraic.from_rational(Fraction(0)))
        else:
            values.append(Algebraic.from_rational(rational_below(roots[0])))
            for j, root_ in enumerate(roots):
                values.append(root_)
                if j + 1 < len(roots):
                    values.append(Algebraic.from_rational(rational_between(root_, roots[j + 1])))
            values.append(Algebraic.from_rational(rational_above(roots[-1])))
        # the number of the roots of each factor below the current cell
        passed = [0] * len(polys)
        cells: list[CADCell] = []
        for j, value in enumerate(values, start=1):
            vanishing: frozenset[int] = sections[j // 2 - 1][1] if j % 2 == 0 else frozenset()
            signs: list[Sign] = []
            for i, specialized in enumerate(specializations):
                if i in vanishing:
                    signs.append(0)
                    passed[i] += 1
                else:
                    signs.append(specialized.sign_before(passed[i]))
            cells.append(CADCell(parent.index + (j,), parent.point + (value,), None, parent, tuple(signs)))
        self._stacks[parent] = cells
        self.lifted += len(cells)
        return cells

    def levels(self) -> list[list[CADCell]]:
        """The cells built so far, by level, in lexicographic order of
        their indices."""
        result: list[list[CADCell]] = []
        current = [self.root]
        for _ in self.gens:
            cells = [cell for parent in current if parent in self._stacks for cell in self._stacks[parent]]
            result.append(cells)
            current = cells
        return result

    def lift_all(self) -> list[list[CADCell]]:
        """Build every stack: the full decomposition."""
        current = [self.root]
        for _ in self.gens:
            current = [cell for parent in current for cell in self.stack(parent)]
        return self.levels()


#: the key of a lifting: the projection factor sets, the variables and the
#: projection operator
_LiftingKey = tuple[tuple[tuple[Poly, ...], ...], tuple[Expr, ...], str]

#: the liftings of the last questions, for the ones which come back with the
#: same projection factor sets (the theory checks of ``satisfiable`` and the
#: handlers of ``refine`` ask about the same polynomials many times); a
#: stack is the same whenever it is built, so the answers do not depend on
#: what was asked before
_liftings: OrderedDict[_LiftingKey, Lifting] = OrderedDict()

#: how many liftings are kept
KEPT = 64


def lifting_for(projection: Sequence[Sequence[Poly]], gens: Sequence[Expr], method: str) -> Lifting:
    """The :class:`Lifting` of the projection factor sets ``projection``:
    the one of an earlier question with the same sets, whose stacks are
    kept, or a new one. The last :data:`KEPT` liftings are kept.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.projection import projection_sets
    >>> from sympy_extras.polys.cad.lifting import lifting_for
    >>> x, y = symbols('x y')
    >>> projection = projection_sets([x**2 + y**2 - 1], [x, y])
    >>> lifting_for(projection, [x, y], 'mccallum') is lifting_for(projection, [x, y], 'mccallum')
    True
    """
    key: _LiftingKey = (tuple(tuple(level) for level in projection), tuple(gens), method)
    found = _liftings.get(key)
    if found is None:
        found = Lifting(projection, gens, method)
        _liftings[key] = found
        while len(_liftings) > KEPT:
            _liftings.popitem(last=False)
    else:
        _liftings.move_to_end(key)
    return found


def factor_signs(polys: Sequence[Poly], projection: Sequence[Sequence[Poly]], gens: Sequence[Expr]) -> FactorSigns:
    """For every input polynomial, the sign of its constant factor and the
    positions ``(level, index, exponent)`` of its irreducible factors in
    the projection sets.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.projection import projection_sets, to_polys
    >>> from sympy_extras.polys.cad.lifting import factor_signs
    >>> x, y = symbols('x y')
    >>> polys = to_polys([1 - x**2, x**2 + y**2 - 1], [x, y])
    >>> factor_signs(polys, projection_sets(polys, [x, y]), [x, y])
    [(-1, [(1, 0, 1), (1, 1, 1)]), (1, [(2, 0, 1)])]
    """
    positions: dict[Poly, tuple[int, int]] = {}
    for k, level in enumerate(projection, start=1):
        for i, g in enumerate(level):
            positions[Poly(g.as_expr(), *gens)] = (k, i)
    result: FactorSigns = []
    for f in polys:
        coeff, factors = f.factor_list()
        c: Sign = 0 if f.is_zero else (1 if coeff.is_positive else -1)
        items: list[tuple[int, int, int]] = []
        for g, e in factors:
            if g.rep.terms[max(g.rep.terms)] < 0:
                g = -g
                if e % 2:
                    c = -c
            items.append(positions[g] + (e,))
        result.append((c, items))
    return result


def cylindrical_algebraic_decomposition(polys: Iterable[PolyLike], gens: Sequence[Expr],
                                        method: Optional[str] = None) -> CAD:
    """Cylindrical algebraic decomposition sign-invariant for ``polys``.

    Parameters
    ==========

    polys : list of Expr or Poly
        Polynomials with rational coefficients in ``gens``.
    gens : list of Symbol
        The variables. The decomposition is cylindrical with respect to this
        order: the cells of `\\mathbb{R}^n` project onto cells of
        `\\mathbb{R}^{n-1}` in the first ``n - 1`` variables, and so on.
    method : ``'mccallum'``, ``'hong'`` or None
        The projection operator, see :mod:`sympy_extras.polys.cad.projection`. By
        default McCallum's projection is tried first and Hong's is used if
        the polynomials are not well-oriented for it. With
        ``method='mccallum'`` :class:`NotWellOriented` is raised instead.

    Returns
    =======

    A :class:`CAD` whose ``cells`` are sign-invariant for every polynomial:
    each cell is a connected set on which every input polynomial is
    constantly positive, negative or zero, and every point of
    `\\mathbb{R}^n` belongs to exactly one cell.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.lifting import cylindrical_algebraic_decomposition
    >>> x, y = symbols('x y')
    >>> cad = cylindrical_algebraic_decomposition([x**2 + y**2 - 1], [x, y])
    >>> cad
    CAD(13 cells, x, y)
    >>> for cell in cad:
    ...     print(cell.index, [str(c.rational_value()) for c in cell.point], cell.signs)
    (1, 1) ['-2', '0'] (1,)
    (2, 1) ['-1', '-1'] (1,)
    (2, 2) ['-1', '0'] (0,)
    (2, 3) ['-1', '1'] (1,)
    (3, 1) ['0', '-2'] (1,)
    (3, 2) ['0', '-1'] (0,)
    (3, 3) ['0', '0'] (-1,)
    (3, 4) ['0', '1'] (0,)
    (3, 5) ['0', '2'] (1,)
    (4, 1) ['1', '-1'] (1,)
    (4, 2) ['1', '0'] (0,)
    (4, 3) ['1', '1'] (1,)
    (5, 1) ['2', '0'] (1,)

    The cells of index ``(2, 2)``, ``(3, 2)``, ``(3, 4)`` and ``(4, 2)``
    are the two points and the two arcs of the circle, ``(3, 3)`` is the
    open disc and the others cover its complement.
    """
    generators = list(gens)
    if not generators:
        raise ValueError("at least one generator is needed")
    inputs = to_polys(polys, generators)
    if method is None:
        methods = ['mccallum', 'hong']
    elif method in ('mccallum', 'hong'):
        methods = [method]
    else:
        raise ValueError("unknown projection method %r" % (method,))

    levels: list[list[CADCell]] = []
    projection: list[list[Poly]] = []
    m = methods[0]
    for m in methods:
        projection = projection_sets(inputs, generators, method=m)
        try:
            levels = Lifting(projection, generators, m).lift_all()
        except NotWellOriented:
            if m == methods[-1]:
                raise
            continue
        break

    signs_of_factors = factor_signs(inputs, projection, generators)
    for cell in levels[-1]:
        signs: list[Sign] = []
        for c, items in signs_of_factors:
            s = c
            for level, i, e in items:
                s *= cell.factor_sign(level, i)**e
            signs.append(s)
        cell.signs = tuple(signs)
    return CAD(generators, inputs, projection, m, levels)
