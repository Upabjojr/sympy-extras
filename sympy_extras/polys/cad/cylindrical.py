"""Cylindrical descriptions of semialgebraic sets.

The set of the real points at which a (possibly quantified) formula in
polynomial relations holds is a union of cells of a cylindrical algebraic
decomposition of the space of its free variables. A cell is cylindrical:
its first coordinate ranges over an interval between two real algebraic
numbers, or is one; over a point of the cell of the first ``j - 1``
coordinates the ``j``-th coordinate ranges between two consecutive real
roots of the projection polynomials of level ``j``, or is one of them.
By delineability the ``k``-th real root of a projection polynomial is a
continuous function on the cell below, so that the cell is described by

.. math::

    a_1 < x_1 < b_1 \\;\\wedge\\; a_2(x_1) < x_2 < b_2(x_1) \\;\\wedge\\; \\cdots

with *root functions* for bounds (the description of Mathematica's
``CylindricalDecomposition`` and ``Reduce``, which writes them with
parametric ``Root`` objects; the extended Tarski formulas of [Brown]_).
This is a solution formula for any number of free variables: the signs
of the projection factors, which :func:`~sympy_extras.polys.cad.quantifier_elimination`
uses, do not always tell the true cells from the false ones, the root
functions do.

A root function is written explicitly when the polynomial has degree one
or two in its variable on the cell (the coefficients of a projection
polynomial have constant signs on the cell, which tell its degree there
and the branch of the quadratic formula), and as an
:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot` otherwise. Consecutive cells of a stack with the
same description of their higher coordinates are joined (a sector, the
section which closes it and the next sector give one interval).

The module is native: formulas, expressions and sets are those of the
core, on the native decompositions of :mod:`sympy_extras.polys.cad.qe`;
SymPy is not imported. The SymPy face is
:mod:`sympy_extras.polys.cad.cylindrical_sympy`.

References
==========

.. [Brown] C. W. Brown, *Solution formula construction for truth
   invariant CAD's*, PhD thesis, University of Delaware, 1999 (the
   extended language with indexed roots).
.. [Collins] G. E. Collins, *Quantifier elimination for real closed
   fields by cylindrical algebraic decomposition*, LNCS 33, 1975
   (delineability, the stack construction).
.. [Strzebonski] A. Strzeboński, *Cylindrical algebraic decomposition
   using validated numerics*, J. Symbolic Comput. 41 (2006) (the form of
   the output of ``CylindricalDecomposition``).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from sympy_extras._timeout import attempt
from sympy_extras.core.containers import Tuple
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, BooleanLike, and_, false, or_, true
from sympy_extras.core.numbers import ComplexInfinity, Infinity, NaN, NegativeInfinity, Rational, oo
from sympy_extras.core.power import sqrt
from sympy_extras.core.relational import eq, ge, gt, le, lt
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.polytools import Poly
from sympy_extras.sets.conditionset import condition_set
from sympy_extras.sets.sets import EmptySet, Reals, Set, finite_set, interval, point_set, product_set, union

from .indexedroot import IndexedRoot, indexed_root, real_number
from .lifting import CAD, CADCell
from .qe import CellTruth, QuantifierSpec, _truth_values
from .samplepoints import _order

__all__ = ['IndexedRoot', 'cylindrical_formula', 'cylindrical_set', 'cylindrical_cases', 'CylindricalCase']

#: a case of :func:`cylindrical_cases`: a condition on the parameters and
#: the set of the solutions under it
CylindricalCase = tuple[Boolean, Set]

#: the seconds given to the core to tell whether two bounds meet
_PROOF_SECONDS = 2.0


def _symbol(v: Expr) -> Symbol:
    if not isinstance(v, Symbol):
        raise TypeError("expected a symbol, got %s" % (v,))
    return v


class _Root:
    """A real root over a cell: the ``index``-th of ``poly``."""

    __slots__ = ('poly', 'index', 'value', 'degree')

    def __init__(self, poly: Poly, index: int, value: Algebraic, degree: int) -> None:
        self.poly = poly
        self.index = index
        self.value = value
        self.degree = degree


def _coefficients(poly: Poly, gens: Sequence[Expr]) -> list[Expr]:
    """The coefficients of the polynomial in its last generator, the
    leading one first, as expressions in the others."""
    return Poly(poly.as_expr(), gens[-1]).all_coeffs()


def _degree_over(poly: Poly, parent: CADCell, gens: Sequence[Expr]) -> tuple[int, list[Expr], list[int]]:
    """The degree of the polynomial in the last generator over the parent
    cell, with the coefficients from that degree down and their signs
    there: the coefficients of a projection polynomial have constant signs
    on the cell, the sample point tells them."""
    coefficients = _coefficients(poly, gens)
    lower = list(gens[:-1])
    signs: list[int] = []
    for c in coefficients:
        if not lower:
            if not isinstance(c, Rational):
                raise ValueError("a rational coefficient expected, got %s" % (c,))  # pragma: no cover
            signs.append((c.p > 0) - (c.p < 0))
        else:
            signs.append(parent.sample.sign(c, lower))
    while signs and signs[0] == 0:
        signs.pop(0)
        coefficients.pop(0)
    return len(coefficients) - 1, coefficients, signs


def _root_sequence(parent: CADCell, polys: Sequence[Poly], gens: Sequence[Expr]) -> list[_Root]:
    """The distinct real roots of the polynomials of a level over the
    parent cell, in increasing order (they are the sections of the stack),
    each as a root of the polynomial of least degree which has it."""
    found: list[_Root] = []
    for poly in polys:
        roots = parent.sample.real_roots(poly, gens)
        if not roots:
            continue
        degree = _degree_over(poly, parent, gens)[0]
        for index, value in enumerate(roots):
            found.append(_Root(poly, index, value, degree))
    return [min((found[i] for i in group), key=lambda root: root.degree)
            for group in _order([root.value for root in found])]


def _root_function(root: _Root, parent: CADCell, gens: Sequence[Expr]) -> Expr:
    """The root as an expression in the variables of the parent cell."""
    x = _symbol(gens[-1])
    if len(gens) == 1:
        return real_number(root.value, x.name)
    degree, coefficients, signs = _degree_over(root.poly, parent, gens)
    if degree == 1:
        return -coefficients[1]/coefficients[0]
    if degree == 2:
        a, b, c = coefficients
        roots = parent.sample.real_roots(root.poly, gens)
        if roots is not None and len(roots) == 1:
            return -b/(2*a)
        # the smaller root has the sign of -a before its square root
        branch = (-1 if signs[0] > 0 else 1)*(1 if root.index == 0 else -1)
        return (-b + branch*sqrt(factor_terms(b**2 - 4*a*c)))/(2*a)
    return indexed_root(root.poly.as_expr(), x, root.index)


class _Run:
    """Consecutive cells of a stack with one description of the higher
    coordinates: the variable between two root functions (``None`` for an
    infinite end), or equal to one (``point``)."""

    __slots__ = ('lower', 'lower_closed', 'upper', 'upper_closed', 'inner')

    def __init__(self, lower: Optional[Expr], lower_closed: bool, upper: Optional[Expr], upper_closed: bool,
                 inner: _Description) -> None:
        self.lower = lower
        self.lower_closed = lower_closed
        self.upper = upper
        self.upper_closed = upper_closed
        self.inner = inner

    @property
    def point(self) -> bool:
        return self.lower is not None and self.lower == self.upper

    def key(self) -> tuple[Optional[Expr], bool, Optional[Expr], bool, _Key]:
        return (self.lower, self.lower_closed, self.upper, self.upper_closed, _key(self.inner))


#: the true cells over a cell: all of the stack (``True``), none
#: (``False``), or runs; ``_Key`` is its hashable form
_Description = Union[bool, list[_Run]]
_Key = Union[bool, tuple[tuple[Optional[Expr], bool, Optional[Expr], bool, '_Key'], ...]]


def _key(description: _Description) -> _Key:
    if isinstance(description, bool):
        return description
    return tuple(run.key() for run in description)


def _is_zero(gap: Expr) -> Callable[[], Optional[bool]]:
    def decide() -> Optional[bool]:
        return gap.is_zero
    return decide


def _specialized(description: _Description, x: Symbol, value: Expr, own: bool) -> Optional[_Description]:
    """The description at ``x = value``, a point of the closure of the
    cell it was made on: the bounds are continuous functions there when
    they are written explicitly and stay finite, a run whose bounds meet
    is a point when both are closed and empty otherwise. ``None`` when
    this is not decided. The index of an :class:`IndexedRoot` is that of
    the root on the cell, not at its boundary, where roots may meet: such
    a bound is specialized on its ``own`` cell only."""
    if isinstance(description, bool):
        return description
    runs: list[_Run] = []
    for run in description:
        ends: list[Optional[Expr]] = []
        for end in (run.lower, run.upper):
            if end is None:
                ends.append(None)
                continue
            if not own and end.has(IndexedRoot) and end.has(x):
                return None
            moved = end.subs(x, value)
            if not isinstance(moved, Expr):
                return None  # pragma: no cover
            if moved.has(NaN, ComplexInfinity, Infinity, NegativeInfinity) or moved.is_extended_real is False:
                return None
            ends.append(moved)
        lower, upper = ends
        inner = _specialized(run.inner, x, value, own)
        if inner is None:
            return None
        if lower is not None and upper is not None and lower != upper:
            gap = upper - lower
            if not gap.free_symbols:
                met = attempt(_is_zero(gap), _PROOF_SECONDS, native=True)
                if met is None:
                    return None
                if met:
                    upper = lower
        if lower is not None and lower == upper and not (run.lower_closed and run.upper_closed):
            continue
        if inner is False:
            continue
        moved_run = _Run(lower, run.lower_closed, upper, run.upper_closed, inner)
        # two branches which meet at the boundary give one point
        if moved_run.point and any(moved_run.key() == other.key() for other in runs):
            continue
        runs.append(moved_run)
    return runs if runs else False


def _stack(cad: CAD, children: dict[CADCell, list[CADCell]], truth: dict[CADCell, bool],
           parent: CADCell, level: int, last: int) -> _Description:
    """The description of the true cells over ``parent``."""
    gens = cad.gens[:level]
    x = _symbol(gens[-1])
    cells = children.get(parent, [])
    descriptions: list[_Description] = []
    for cell in cells:
        if cell in truth:
            # a cell of the last level, or one which a partial decomposition
            # did not lift: the formula has one truth value over it
            descriptions.append(truth[cell])
        else:
            descriptions.append(_stack(cad, children, truth, cell, level + 1, last))
    if all(d is False for d in descriptions):
        return False
    roots = _root_sequence(parent, cad.projection[level - 1], gens)
    functions: dict[int, Expr] = {}

    def bound(position: int) -> Expr:
        # the section at ``position`` of the stack (the odd positions, from 0)
        if position not in functions:
            functions[position] = _root_function(roots[(position - 1)//2], parent, gens)
        return functions[position]

    # a section whose description is that of a neighbour at the boundary
    # joins it: the closed disc is one run, not an open one and two points
    keys = [_key(d) for d in descriptions]
    for i, cell in enumerate(cells):
        if not cell.is_section or descriptions[i] is False:
            continue
        here = _specialized(descriptions[i], x, bound(i), True)
        if here is None:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(cells) and descriptions[j] is not False and keys[j] != keys[i]:
                there = _specialized(descriptions[j], x, bound(i), False)
                if there is not None and _key(there) == _key(here):
                    descriptions[i], keys[i] = descriptions[j], keys[j]
                    break
    runs: list[_Run] = []
    i = 0
    while i < len(cells):
        if descriptions[i] is False:
            i += 1
            continue
        j = i
        while j + 1 < len(cells) and keys[j + 1] == keys[i]:
            j += 1
        inner = descriptions[i]
        if i == j and cells[i].is_section:
            value = bound(i)
            # at a number, the higher bounds are written there
            if value.is_number and not value.has(IndexedRoot):
                at_value = _specialized(inner, x, value, True)
                inner = inner if at_value is None else at_value
            runs.append(_Run(value, True, value, True, inner))
        else:
            lower = bound(i) if cells[i].is_section else bound(i - 1) if i > 0 else None
            upper = bound(j) if cells[j].is_section else bound(j + 1) if j + 1 < len(cells) else None
            runs.append(_Run(lower, cells[i].is_section, upper, cells[j].is_section, inner))
        i = j + 1
    if len(runs) == 1 and runs[0].lower is None and runs[0].upper is None and runs[0].inner is True:
        return True
    return runs


def _formula(description: _Description, gens: Sequence[Symbol], level: int) -> Boolean:
    if isinstance(description, bool):
        return true if description else false
    x = gens[level - 1]
    parts: list[Boolean] = []
    for run in description:
        conditions: list[Boolean] = []
        if run.point:
            assert run.lower is not None
            conditions.append(eq(x, run.lower))
        else:
            if run.lower is not None:
                conditions.append(ge(x, run.lower) if run.lower_closed else gt(x, run.lower))
            if run.upper is not None:
                conditions.append(le(x, run.upper) if run.upper_closed else lt(x, run.upper))
        parts.append(and_(*conditions, _formula(run.inner, gens, level + 1)))
    return or_(*parts)


def cylindrical_formula(formula: BooleanLike, gens: Sequence[Symbol], quantifiers: QuantifierSpec = (),
                        method: Optional[str] = None, partial: bool = True) -> Boolean:
    """A cylindrical description of the set of the real points
    ``gens`` at which the formula (with its ``quantifiers``, over other
    variables) holds: a disjunction of conjunctions which bound the first
    variable by numbers, the second by root functions of the first, and so
    on; ``true`` for the whole space and ``false`` for the empty set. The
    counterpart of Mathematica's ``CylindricalDecomposition``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.polys.cad.cylindrical import cylindrical_formula
    >>> x, y, z = symbols('x y z')
    >>> cylindrical_formula(x**2 + y**2 < 1, [x, y])
    (x > -1) & (x < 1) & (y < sqrt(1 - x**2)) & (y > -sqrt(1 - x**2))
    >>> cylindrical_formula((x**2 + y**2 <= 1) & (y >= x), [x, y])
    ((y >= x) & (x <= sqrt(2)/2) & (x > -sqrt(2)/2) & (y <= sqrt(1 - x**2))) | ((x >= -1) & (x <= -sqrt(2)/2) & (y <= sqrt(1 - x**2)) & (y >= -sqrt(1 - x**2)))

    The example of sympy-extras#9, whose solution set the signs of the
    projection factors do not describe:

    >>> cylindrical_formula(Eq(z**2, x) & (z > y), [x, y], [('exists', z)])
    (x >= 0) & (y < sqrt(x))

    A bound which is not a root of a polynomial of degree two (at ``x = 2``
    the cubic has the roots -2 and 1, the second one double):

    >>> cylindrical_formula((y**3 - 3*y + x > 0) & (x > 1), [x, y])
    (Eq(x, 2) & ((y > 1) | ((y > -2) & (y < 1)))) | ((x > 2) & (y > IndexedRoot(x + y**3 - 3*y, y, 0))) | ((x > 1) & (x < 2) & ((y > IndexedRoot(x + y**3 - 3*y, y, 2)) | ((y > IndexedRoot(x + y**3 - 3*y, y, 0)) & (y < IndexedRoot(x + y**3 - 3*y, y, 1)))))
    """
    free, description = _description(formula, gens, quantifiers, method, partial)
    return _formula(description, free, 1)


def _description(formula: BooleanLike, gens: Sequence[Symbol], quantifiers: QuantifierSpec,
                 method: Optional[str], partial: bool = True) -> tuple[list[Symbol], _Description]:
    variables = [_symbol(g) for g in gens]
    # a cell over which the formula has one truth value is not lifted: the
    # description of the stack over it is that truth value
    cad, free, _, cells = _truth_values(formula, variables, quantifiers, method, partial, 'both')
    if not free:
        raise ValueError("the formula has no free variable among %s" % (variables,))
    return free, _described(cad, len(free), cells)


def _described(cad: CAD, last: int, cells: Sequence[CellTruth]) -> _Description:
    """The description of the true cells of the space of the first
    ``last`` variables of a decomposition."""
    truth = {cell: value for cell, value in cells}
    children: dict[CADCell, list[CADCell]] = {}
    for level in range(1, last + 1):
        for cell in cad.cells_at(level):
            assert cell.parent is not None
            children.setdefault(cell.parent, []).append(cell)
    root = cad.cells_at(1)[0].parent
    assert root is not None
    return _stack(cad, children, truth, root, 1, last)


def described_by_root_functions(cad: CAD, free: Sequence[Symbol], cells: Sequence[CellTruth]) -> Boolean:
    """The cylindrical formula of the true ``cells`` of the space of the
    ``free`` variables of a decomposition (for
    :func:`~sympy_extras.polys.cad.qe.quantifier_elimination`, when the
    signs of the projection factors do not describe them)."""
    return _formula(_described(cad, len(free), cells), free, 1)


def _isolated(description: _Description, gens: Sequence[Symbol]) -> tuple[list[tuple[Expr, ...]], _Description]:
    """The points of the description whose coordinates are all numbers,
    and the description without them."""
    if isinstance(description, bool):
        return ([()] if description else []), False
    points: list[tuple[Expr, ...]] = []
    rest: list[_Run] = []
    for run in description:
        if run.point and run.lower is not None and run.lower.is_number and run.inner is not True:
            inner = _specialized(run.inner, gens[0], run.lower, True)
            if inner is not None and _all_points(inner):
                points.extend((run.lower,) + tail for tail in _points(inner, gens[1:]))
                continue
        rest.append(run)
    return points, (rest if rest else False)


def _all_points(description: _Description) -> bool:
    if isinstance(description, bool):
        return description
    return all(run.point and run.lower is not None and run.lower.is_number and _all_points(run.inner)
               for run in description)


def _points(description: _Description, gens: Sequence[Symbol]) -> list[tuple[Expr, ...]]:
    """The points of a description whose runs are all points: a coordinate
    is a function of the ones before, whose values are put in it."""
    if isinstance(description, bool):
        return [()]
    found: list[tuple[Expr, ...]] = []
    for run in description:
        value = run.lower
        assert value is not None
        for tail in _points(run.inner, gens[1:]):
            moved: list[Expr] = []
            for coordinate in tail:
                substituted = coordinate.subs(gens[0], value)
                if not isinstance(substituted, Expr):
                    raise TypeError("expected an expression, got %s" % (substituted,))  # pragma: no cover
                moved.append(substituted)
            found.append((value,) + tuple(moved))
    return found


def _space(n: int) -> Set:
    return product_set(*[Reals]*n)


def cylindrical_set(formula: BooleanLike, gens: Sequence[Symbol], method: Optional[str] = None,
                    partial: bool = True) -> Set:
    """The set of the real points ``gens`` at which the formula holds:
    its isolated points with algebraic coordinates as a finite set, and
    the rest as a condition set with the cylindrical description of
    :func:`cylindrical_formula`. The first symbols of ``gens`` may be
    parameters of a problem in the others: the description bounds them
    first.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.cylindrical import cylindrical_set
    >>> x, y = symbols('x y')
    >>> cylindrical_set((x**2 + y**2 <= 1) & (x + y >= 1), [x, y])
    ConditionSet((x, y), (x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2)), ProductSet(Reals, Reals))
    >>> cylindrical_set((x**2 + y**2)*((x - 3)**2 + y**2 - 1) <= 0, [x, y])
    Union(ConditionSet((x, y), (x >= 2) & (x <= 4) & (y <= sqrt(-x**2 + 6*x - 8)) & (y >= -sqrt(-x**2 + 6*x - 8)), ProductSet(Reals, Reals)), {(0, 0)})
    >>> cylindrical_set(x**2 + y**2 < 0, [x, y])
    EmptySet
    """
    free, description = _description(formula, gens, (), method, partial)
    space = _space(len(free))
    if description is True:
        return space
    points, rest = _isolated(description, free)
    finite: Set = point_set(*[Tuple(*point) for point in points]) if points else EmptySet
    if rest is False:
        return finite
    region = condition_set(Tuple(*free), _formula(rest, free, 1), space)
    return union(finite, region) if points else region


def _as_set(description: _Description, gens: Sequence[Symbol]) -> Set:
    """The set of the points ``gens`` of a description: a finite set when
    every coordinate is given, a union of intervals for one variable, a
    condition set otherwise."""
    space: Set = Reals if len(gens) == 1 else _space(len(gens))
    if isinstance(description, bool):
        return space if description else EmptySet
    if _every_run_is_a_point(description):
        points = _points(description, gens)
        if len(gens) == 1:
            return finite_set(*[point[0] for point in points])
        return point_set(*[Tuple(*point) for point in points])
    if len(gens) == 1:
        pieces: list[Set] = []
        for run in description:
            if run.point:
                assert run.lower is not None
                pieces.append(finite_set(run.lower))
            else:
                pieces.append(interval(-oo if run.lower is None else run.lower,
                                       oo if run.upper is None else run.upper,
                                       not run.lower_closed, not run.upper_closed))
        return union(*pieces)
    return condition_set(Tuple(*gens), _formula(description, gens, 1), space)


def _every_run_is_a_point(description: _Description) -> bool:
    if isinstance(description, bool):
        return description
    return all(run.point and _every_run_is_a_point(run.inner) for run in description)


def cylindrical_cases(formula: BooleanLike, parameters: Sequence[Symbol], unknowns: Sequence[Symbol],
                      method: Optional[str] = None, partial: bool = True,
                      quantifiers: QuantifierSpec = ()) -> list[CylindricalCase]:
    """The real solutions in the ``unknowns`` of a formula in polynomial
    relations, for every real value of its ``parameters``: pairs of a
    condition on the parameters (cylindrical: the first one between
    numbers, the second one between functions of the first) and of the set
    of the solutions under it, in terms of the parameters. The values of
    the parameters for which there is no solution are in no pair.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.polys.cad.cylindrical import cylindrical_cases
    >>> a, x = symbols('a x')
    >>> cylindrical_cases(Eq(x**2, a), [a], [x])
    [(a >= 0, {-sqrt(a), sqrt(a)})]
    >>> cylindrical_cases(x**2 <= a, [a], [x])
    [(a >= 0, Interval(-sqrt(a), sqrt(a)))]
    >>> cylindrical_cases(Eq(a*x, 1), [a], [x])
    [((a > 0) | (a < 0), {1/a})]

    With ``quantifiers`` over other variables of the formula (the values
    ``t`` taken by ``y**2 + a*y`` for each ``a``):

    >>> t, y = symbols('t y')
    >>> cylindrical_cases(Eq(y**2 + a*y, t), [a], [t], quantifiers=[('exists', y)])
    [(True, Interval(-a**2/4, oo))]
    """
    names = [_symbol(p) for p in parameters]
    variables = [_symbol(u) for u in unknowns]
    free, description = _description(formula, names + variables, quantifiers, method, partial)
    if free != names + variables:
        raise ValueError("the parameters and the unknowns are the symbols of the formula")
    cases: list[CylindricalCase] = []

    def walk(current: _Description, level: int, conditions: list[Boolean]) -> None:
        if level == len(names):
            found = _as_set(current, variables)
            if found is not EmptySet:
                cases.append((and_(*conditions), found))
            return
        if isinstance(current, bool):
            if current:
                walk(True, len(names), conditions)
            return
        for run in current:
            walk(run.inner, level + 1, conditions + [_formula([_Run(run.lower, run.lower_closed, run.upper,
                                                                  run.upper_closed, True)], names, level + 1)])

    walk(description, 0, [])
    # the cases with one set of solutions are one case
    merged: list[CylindricalCase] = []
    for condition, found in cases:
        for position, (other, same) in enumerate(merged):
            if same == found:
                merged[position] = (or_(other, condition), found)
                break
        else:
            merged.append((condition, found))
    return merged
