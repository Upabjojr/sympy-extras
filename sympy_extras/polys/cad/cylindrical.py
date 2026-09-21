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
:class:`IndexedRoot` otherwise. Consecutive cells of a stack with the
same description of their higher coordinates are joined (a sector, the
section which closes it and the next sector give one interval).

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

from typing import Optional, Sequence, Union

from mpmath.libmp.libhyper import NoConvergence
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.containers import Tuple
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import And, Boolean, Or, false, true
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import ComplexRootOf
from sympy.printing.printer import Printer
from sympy.sets.conditionset import ConditionSet
from sympy.sets.sets import FiniteSet, ProductSet, Set, Union as SetUnion

from sympy_extras._timeout import attempt
from sympy_extras._typing import QuantifierSpec, as_expr, as_symbol
from sympy_extras.polys.roots import radical_form

from .lifting import CAD, CADCell
from .qe import CellTruth, _truth_values
from .samplepoints import RealAlgebraic, _SortKey, compare_real

__all__ = ['IndexedRoot', 'cylindrical_formula', 'cylindrical_set']

#: the seconds given to SymPy to tell whether two bounds meet
_PROOF_SECONDS = 2.0

#: the bound variables of the root functions, by the name of the variable:
#: one per name, so that equal root functions are equal expressions
_BOUND: dict[str, Dummy] = {}


def _bound_variable(name: str) -> Dummy:
    if name not in _BOUND:
        _BOUND[name] = Dummy(name, real=True)
    return _BOUND[name]


class IndexedRoot(Function):
    """``IndexedRoot(f, t, k)``: the ``k``-th distinct real root, counted
    from 0 in increasing order, of the polynomial ``f`` in ``t`` whose
    coefficients depend on other symbols, the counterpart of Mathematica's
    parametric ``Root[f, k]``. It is a function of those symbols where the
    polynomial has more than ``k`` real roots, and becomes a number when
    they are given rational values.

    The variable ``t`` is bound: it is replaced by a dummy of the same
    name, which a substitution of ``t`` in the formula around does not
    reach.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad import IndexedRoot
    >>> r = IndexedRoot(y**3 + x*y - 1, y, 0); r
    IndexedRoot(x*y + y**3 - 1, y, 0)
    >>> r.subs(x, 0), r.subs(x, 1).evalf(10), r.free_symbols       # a CRootOf at x = 1
    (1, 0.6823278038, {x})
    >>> IndexedRoot(y**3 - 3*y + x, y, 2).subs(x, 5)     # one real root only
    IndexedRoot(y**3 - 3*y + 5, y, 2)
    """

    @classmethod
    def eval(cls, f: Expr, t: Expr, k: Expr) -> Optional[Expr]:
        if not (isinstance(t, Symbol) and isinstance(k, Integer) and k >= 0):
            raise ValueError("IndexedRoot(f, t, k) takes a symbol t and an index k, from 0")
        bound = _bound_variable(t.name)
        if t != bound:
            return cls(as_expr(f.xreplace({t: bound})), bound, k)
        if not f.has(bound):
            # a polynomial which the values of the parameters made constant
            # has no root: the object stays as it is, like one whose index
            # is beyond the number of the roots
            return None
        if f.free_symbols == {bound}:
            return _numerical_root(f, bound, int(k))
        return None

    def _eval_is_extended_real(self) -> bool:
        return True

    @property
    def expr(self) -> Expr:
        return as_expr(self.args[0])

    @property
    def variable(self) -> Dummy:
        variable = self.args[1]
        assert isinstance(variable, Dummy)
        return variable

    @property
    def index(self) -> int:
        return int(as_expr(self.args[2]))

    @property
    def free_symbols(self) -> set[Basic]:
        return set(self.expr.free_symbols) - {self.variable}

    @property
    def bound_symbols(self) -> list[Symbol]:
        return [self.variable]

    def _eval_evalf(self, prec: int) -> Optional[Expr]:
        # coefficients which are not rational numbers: the real ones among
        # the numerical roots. A multiple root (which is what the polynomial
        # has over a section of its discriminant) is found with half of the
        # working digits, slowly: twice the digits asked for, and as many
        # steps as it takes
        if self.free_symbols:
            return None
        digits = max(15, prec_to_dps(prec))
        working = 2 * digits + 10
        try:
            numerical = Poly(self.expr.evalf(working + 10), self.variable).nroots(n=working, maxsteps=5000)
        except (BasePolynomialError, NoConvergence, ValueError, TypeError, ArithmeticError):
            return None
        tolerance = Float(10)**(3 - digits)
        values: list[Expr] = []
        for root in numerical:
            real_part, imaginary_part = as_expr(root).as_real_imag()
            if abs(imaginary_part) < tolerance * (1 + abs(real_part)):
                values.append(as_expr(real_part))
        values.sort(key=lambda value: float(value))
        distinct: list[Expr] = []
        for value in values:
            if not distinct or abs(value - distinct[-1]) > tolerance * (1 + abs(value)):
                distinct.append(value)
        if self.index >= len(distinct):
            return None
        return as_expr(distinct[self.index].evalf(digits))

    def _sympystr(self, printer: Printer) -> str:
        shown = Symbol(self.variable.name)
        return "IndexedRoot(%s, %s, %s)" % (printer._print(self.expr.xreplace({self.variable: shown})),
                                            printer._print(shown), self.index)


def _numerical_root(expression: Expr, t: Symbol, index: int) -> Optional[Expr]:
    """The ``index``-th distinct real root of a polynomial with rational
    coefficients, when it has that many."""
    try:
        # in a plain symbol: the root object shows its generator
        plain = Symbol(t.name)
        poly = Poly(expression.xreplace({t: plain}), plain, domain='QQ')
    except BasePolynomialError:
        return None
    if poly.degree() < 1:
        return None
    roots = [as_expr(r) for r, _ in poly.real_roots(multiple=False, radicals=False)]
    roots.sort(key=_SortKey)
    if index >= len(roots):
        return None
    root = roots[index]
    form = radical_form(root) if isinstance(root, ComplexRootOf) else None
    return root if form is None else form


class _Root:
    """A real root over a cell: the ``index``-th of ``poly``."""

    __slots__ = ('poly', 'index', 'value', 'degree')

    def __init__(self, poly: Poly, index: int, value: RealAlgebraic, degree: int) -> None:
        self.poly = poly
        self.index = index
        self.value = value
        self.degree = degree


def _coefficients(poly: Poly, gens: Sequence[Symbol]) -> list[Expr]:
    """The coefficients of the polynomial in its last generator, the
    leading one first, as expressions in the others."""
    return [as_expr(c) for c in Poly(poly.as_expr(), gens[-1]).all_coeffs()]


def _degree_over(poly: Poly, parent: CADCell, gens: Sequence[Symbol]) -> tuple[int, list[Expr], list[int]]:
    """The degree of the polynomial in the last generator over the parent
    cell, with the coefficients from that degree down and their signs
    there: the coefficients of a projection polynomial have constant signs
    on the cell, the sample point tells them."""
    coefficients = _coefficients(poly, gens)
    lower = list(gens[:-1])
    signs: list[int] = []
    for c in coefficients:
        if not lower:
            signs.append(int(bool(c > 0)) - int(bool(c < 0)))
        else:
            signs.append(int(parent.sample.sign(Poly(c, *lower), lower)))
    while signs and signs[0] == 0:
        signs.pop(0)
        coefficients.pop(0)
    return len(coefficients) - 1, coefficients, signs


def _root_sequence(parent: CADCell, polys: Sequence[Poly], gens: Sequence[Symbol]) -> list[_Root]:
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
            for position, other in enumerate(found):
                if compare_real(value, other.value) == 0:
                    if degree < other.degree:
                        found[position] = _Root(poly, index, value, degree)
                    break
            else:
                found.append(_Root(poly, index, value, degree))
    found.sort(key=lambda root: _SortKey(root.value))
    return found


def _root_function(root: _Root, parent: CADCell, gens: Sequence[Symbol]) -> Expr:
    """The root as an expression in the variables of the parent cell."""
    if len(gens) == 1:
        value = as_expr(sympify(root.value))
        form = radical_form(value) if isinstance(value, ComplexRootOf) else None
        return value if form is None else form
    degree, coefficients, signs = _degree_over(root.poly, parent, gens)
    if degree == 1:
        return as_expr(-coefficients[1] / coefficients[0])
    if degree == 2:
        a, b, c = coefficients
        roots = parent.sample.real_roots(root.poly, gens)
        if roots is not None and len(roots) == 1:
            return as_expr(-b / (2 * a))
        # the smaller root has the sign of -a before its square root
        branch = (-1 if signs[0] > 0 else 1) * (1 if root.index == 0 else -1)
        return as_expr((-b + branch * sqrt(factor_terms(b**2 - 4 * a * c))) / (2 * a))
    return IndexedRoot(root.poly.as_expr(), gens[-1], root.index)


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
            moved = as_expr(end.subs(x, value))
            if moved.has(S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity) or moved.is_extended_real is False:
                return None
            ends.append(moved)
        lower, upper = ends
        inner = _specialized(run.inner, x, value, own)
        if inner is None:
            return None
        if lower is not None and upper is not None and lower != upper:
            gap = as_expr(upper - lower)
            if not gap.free_symbols:
                met = attempt(lambda: gap.is_zero, _PROOF_SECONDS)
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
    x = gens[-1]
    cells = children.get(parent, [])
    descriptions: list[_Description] = []
    for cell in cells:
        if level == last:
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
            functions[position] = _root_function(roots[(position - 1) // 2], parent, gens)
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
            if value.is_number and not value.has(ComplexRootOf):
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
            conditions.append(Eq(x, run.lower))
        else:
            if run.lower is not None:
                conditions.append(x >= run.lower if run.lower_closed else x > run.lower)
            if run.upper is not None:
                conditions.append(x <= run.upper if run.upper_closed else x < run.upper)
        parts.append(And(*conditions, _formula(run.inner, gens, level + 1)))
    return Or(*parts)


def cylindrical_formula(formula: Union[Boolean, bool], gens: Sequence[Symbol],
                        quantifiers: QuantifierSpec = (), method: Optional[str] = None) -> Boolean:
    """A cylindrical description of the set of the real points
    ``gens`` at which the formula (with its ``quantifiers``, over other
    variables) holds: a disjunction of conjunctions which bound the first
    variable by numbers, the second by root functions of the first, and so
    on; ``True`` for the whole space and ``False`` for the empty set. The
    counterpart of Mathematica's ``CylindricalDecomposition``.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.cad import cylindrical_formula
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
    free, description = _description(formula, gens, quantifiers, method)
    return _formula(description, free, 1)


def _description(formula: Union[Boolean, bool], gens: Sequence[Symbol], quantifiers: QuantifierSpec,
                 method: Optional[str]) -> tuple[list[Symbol], _Description]:
    variables = [as_symbol(g) for g in gens]
    cad, free, _, cells = _truth_values(formula, variables, quantifiers, method)
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
    :func:`~sympy_extras.polys.cad.quantifier_elimination`, when the signs
    of the projection factors do not describe them)."""
    return _formula(_described(cad, len(free), cells), free, 1)


def _isolated(description: _Description, x: Symbol) -> tuple[list[tuple[Expr, ...]], _Description]:
    """The points of the description whose coordinates are all numbers,
    and the description without them."""
    if isinstance(description, bool):
        return ([()] if description else []), False
    points: list[tuple[Expr, ...]] = []
    rest: list[_Run] = []
    for run in description:
        if run.point and run.lower is not None and run.lower.is_number and run.inner is not True:
            inner = _specialized(run.inner, x, run.lower, True)
            if inner is not None and _all_points(inner):
                points.extend((run.lower,) + tail for tail in _points(inner))
                continue
        rest.append(run)
    return points, (rest if rest else False)


def _all_points(description: _Description) -> bool:
    if isinstance(description, bool):
        return description
    return all(run.point and run.lower is not None and run.lower.is_number and _all_points(run.inner)
               for run in description)


def _points(description: _Description) -> list[tuple[Expr, ...]]:
    if isinstance(description, bool):
        return [()]
    return [(as_expr(run.lower),) + tail for run in description for tail in _points(run.inner)]


def cylindrical_set(formula: Union[Boolean, bool], gens: Sequence[Symbol], method: Optional[str] = None) -> Set:
    """The set of the real points ``gens`` at which the formula holds:
    its isolated points with algebraic coordinates as a finite set, and
    the rest as a condition set with the cylindrical description of
    :func:`cylindrical_formula`. The first symbols of ``gens`` may be
    parameters of a problem in the others: the description bounds them
    first.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad import cylindrical_set
    >>> cylindrical_set((x**2 + y**2 <= 1) & (x + y >= 1), [x, y])
    ConditionSet((x, y), (x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2)), ProductSet(Reals, Reals))
    >>> cylindrical_set((x**2 + y**2)*((x - 3)**2 + y**2 - 1) <= 0, [x, y])
    Union(ConditionSet((x, y), (x >= 2) & (x <= 4) & (y <= sqrt(-x**2 + 6*x - 8)) & (y >= -sqrt(-x**2 + 6*x - 8)), ProductSet(Reals, Reals)), {(0, 0)})
    >>> cylindrical_set(x**2 + y**2 < 0, [x, y])
    EmptySet
    """
    free, description = _description(formula, gens, (), method)
    space = ProductSet(*[S.Reals] * len(free))
    if description is True:
        return space
    points, rest = _isolated(description, free[0])
    finite: Set = FiniteSet(*[Tuple(*point) for point in points]) if points else S.EmptySet
    if rest is False:
        return finite
    region = ConditionSet(Tuple(*free), _formula(rest, free, 1), space)
    return SetUnion(finite, region) if points else region
