"""Quantifier elimination over the reals by cylindrical algebraic
decomposition.

A formula is a Boolean combination of polynomial relations. Given a prefix
of quantifiers over some of its variables, the truth of the formula on a
cell of the decomposition of the space of the free variables is obtained by
propagating the truth values of the cells of the full decomposition
downwards, since every polynomial, and so every atom of the formula, has a
constant sign on every cell.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.logic.boolalg import (And, Or, Not, Implies, Equivalent, Xor,
    Boolean, BooleanFunction, BooleanTrue, BooleanFalse)
from sympy.polys.polytools import Poly
from sympy.sets.sets import Interval, FiniteSet, Set, Union as SetUnion

from sympy_extras._typing import (QuantifierPrefix, QuantifierSpec, Sign,
    as_boolean, as_symbol, free_symbols, sorted_symbols)

from .lifting import CAD, CADCell, cylindrical_algebraic_decomposition
from .projection import _to_polys

#: the truth value of a formula on a cell
CellTruth = tuple[CADCell, bool]
#: a sign test on an atom
SignTest = Callable[[int], bool]

_RELATIONS: dict[type, SignTest] = {
    Lt: lambda s: s < 0,
    Gt: lambda s: s > 0,
    Le: lambda s: s <= 0,
    Ge: lambda s: s >= 0,
    Eq: lambda s: s == 0,
    Ne: lambda s: s != 0,
}


class _Compiled:
    """A formula compiled into a tree evaluated on sign vectors."""

    __slots__ = ()

    def __call__(self, signs: Sequence[Sign]) -> bool:
        raise NotImplementedError


class _Const(_Compiled):
    __slots__ = ('value',)

    def __init__(self, value: bool) -> None:
        self.value = value

    def __call__(self, signs: Sequence[Sign]) -> bool:
        return self.value


class _Atom(_Compiled):
    """The sign test ``test`` on the polynomial of index ``index``."""

    __slots__ = ('index', 'test')

    def __init__(self, index: int, test: SignTest) -> None:
        self.index = index
        self.test = test

    def __call__(self, signs: Sequence[Sign]) -> bool:
        return self.test(signs[self.index])


class _Not(_Compiled):
    __slots__ = ('arg',)

    def __init__(self, arg: _Compiled) -> None:
        self.arg = arg

    def __call__(self, signs: Sequence[Sign]) -> bool:
        return not self.arg(signs)


class _Connective(_Compiled):
    """An n-ary connective: ``kind`` is one of ``'and'``, ``'or'``,
    ``'xor'``, ``'implies'`` and ``'equivalent'``."""

    __slots__ = ('kind', 'args')

    def __init__(self, kind: str, args: list[_Compiled]) -> None:
        if kind not in _CONNECTIVES.values():
            raise ValueError("unknown connective %r" % kind)
        self.kind = kind
        self.args = args

    def __call__(self, signs: Sequence[Sign]) -> bool:
        kind = self.kind
        values = [a(signs) for a in self.args]
        if kind == 'and':
            return all(values)
        if kind == 'or':
            return any(values)
        if kind == 'xor':
            return sum(values) % 2 == 1
        if kind == 'implies':
            return (not values[0]) or values[1]
        return all(v == values[0] for v in values)


_CONNECTIVES: dict[type[BooleanFunction], str] = {And: 'and', Or: 'or', Xor: 'xor',
    Implies: 'implies', Equivalent: 'equivalent'}


def _compile(formula: Union[Boolean, bool], gens: Sequence[Symbol], polys: list[Poly],
             index: dict[Poly, int]) -> _Compiled:
    """Compile ``formula`` into a :class:`_Compiled` tree, collecting the
    polynomials of the atoms into ``polys`` (``index`` maps them to their
    position)."""
    if isinstance(formula, (BooleanTrue, BooleanFalse)) or formula in (True, False):
        return _Const(bool(formula))
    if isinstance(formula, Relational):
        try:
            test = _RELATIONS[type(formula)]
        except KeyError:
            raise ValueError("unsupported relation %s" % formula)
        p = Poly(formula.lhs - formula.rhs, *gens)
        [p] = _to_polys([p], gens)
        if p not in index:
            index[p] = len(polys)
            polys.append(p)
        return _Atom(index[p], test)
    if isinstance(formula, Not):
        return _Not(_compile(formula.args[0], gens, polys, index))
    for cls, kind in _CONNECTIVES.items():
        if isinstance(formula, cls):
            return _Connective(kind, [_compile(as_boolean(a), gens, polys, index) for a in formula.args])
    raise ValueError("unsupported formula %s" % formula)


def _quantifiers(quantifiers: QuantifierSpec) -> QuantifierPrefix:
    """Normalize the quantifier prefix to a list of ``(kind, var)`` pairs."""
    result: QuantifierPrefix = []
    for kind, variables in quantifiers:
        if kind not in ('exists', 'forall'):
            raise ValueError("unknown quantifier %r" % (kind,))
        items = list(variables) if isinstance(variables, (list, tuple, set)) else [variables]
        for v in items:
            result.append((kind, as_symbol(v)))
    return result


def _truth_values(formula: Union[Boolean, bool], free: Optional[Sequence[Symbol]],
                  quantifiers: QuantifierSpec, method: Optional[str]
                  ) -> tuple[CAD, list[Symbol], Optional[bool], list[CellTruth]]:
    """The decomposition of the space of the free variables, the free
    variables, the truth value of the quantified formula if there are no
    free variables (else ``None``) and its truth value on each cell of the
    space of the free variables (an empty list if there are none)."""
    formula_ = as_boolean(formula)
    prefix = _quantifiers(quantifiers)
    bound = [v for _, v in prefix]
    if free is None:
        free_list = sorted_symbols(free_symbols(formula_) - set(bound))
    else:
        free_list = [as_symbol(v) for v in free]
    gens = free_list + bound
    if len(set(gens)) != len(gens):
        raise ValueError("a variable is both free and quantified")
    extra = formula_.free_symbols - set(gens)
    if extra:
        raise ValueError("variables not declared: %s" % ", ".join(sorted(map(str, extra))))

    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    compiled = _compile(formula_, gens, polys, index)
    cad = cylindrical_algebraic_decomposition(polys, gens, method=method)

    n, k = len(gens), len(free_list)
    truth: dict[Optional[CADCell], bool] = {}
    for cell in cad.cells:
        truth[cell] = compiled(cell.signs)
    for level in range(n, k, -1):
        kind = prefix[level - k - 1][0]
        combine: Callable[[list[bool]], bool] = any if kind == 'exists' else all
        lower: dict[Optional[CADCell], list[bool]] = {}
        for cell in cad.cells_at(level):
            lower.setdefault(cell.parent, []).append(truth[cell])
        truth = {parent: combine(values) for parent, values in lower.items()}
    if k == 0:
        [value] = truth.values()
        return cad, free_list, value, []
    return cad, free_list, None, [(cell, truth[cell]) for cell in cad.cells_at(k)]


def _interval_union(cells: Sequence[CellTruth], x: Symbol) -> Set:
    """The union of the true cells of the real line as a set."""
    intervals: list[Set] = []
    i = 0
    while i < len(cells):
        if not cells[i][1]:
            i += 1
            continue
        j = i
        while j + 1 < len(cells) and cells[j + 1][1]:
            j += 1
        first, last = cells[i][0], cells[j][0]
        if first.is_section:
            left, left_open = first.point[0], False
        elif i > 0:
            left, left_open = cells[i - 1][0].point[0], True
        else:
            left, left_open = S.NegativeInfinity, True
        if last.is_section:
            right, right_open = last.point[0], False
        elif j + 1 < len(cells):
            right, right_open = cells[j + 1][0].point[0], True
        else:
            right, right_open = S.Infinity, True
        if first is last and first.is_section:
            intervals.append(FiniteSet(left))
        else:
            intervals.append(Interval(left, right, left_open, right_open))
        i = j + 1
    return SetUnion(*intervals)


def _as_relational(sets: Set, x: Symbol) -> Boolean:
    """Relational form of a union of intervals and points, without
    conditions involving infinity."""
    terms: list[Boolean] = []
    for part in (sets.args if isinstance(sets, SetUnion) else [sets]):
        if isinstance(part, FiniteSet):
            terms.extend(Eq(x, v) for v in part.args)
            continue
        if not isinstance(part, Interval):
            raise TypeError("unexpected set %s" % (part,))
        conditions: list[Boolean] = []
        if part.start != S.NegativeInfinity:
            conditions.append(x > part.start if part.left_open else x >= part.start)
        if part.end != S.Infinity:
            conditions.append(x < part.end if part.right_open else x <= part.end)
        terms.append(And(*conditions))
    return Or(*terms)


def _sign_vector(cell: CADCell) -> tuple[Sign, ...]:
    """Signs of the projection factors of all levels at the cell."""
    signs: list[tuple[Sign, ...]] = []
    current: Optional[CADCell] = cell
    while current is not None and current.level > 0:
        signs.append(current._signs)
        current = current.parent
    return tuple(s for level in reversed(signs) for s in level)


_SIGN_RELATIONS: dict[frozenset[int], type] = {
    frozenset([-1]): Lt, frozenset([0]): Eq, frozenset([1]): Gt,
    frozenset([-1, 0]): Le, frozenset([0, 1]): Ge, frozenset([-1, 1]): Ne,
}


def _sign_formula(cad: CAD, cells: Sequence[CellTruth], k: int) -> Optional[Boolean]:
    """A formula in the projection factors of the first ``k`` levels that
    holds exactly on the true cells, or ``None`` if the truth of the cells
    is not determined by the signs of those factors."""
    factors = [f for level in cad.projection[:k] for f in level]
    true_vectors: set[tuple[Sign, ...]] = set()
    false_vectors: set[tuple[Sign, ...]] = set()
    for cell, value in cells:
        (true_vectors if value else false_vectors).add(_sign_vector(cell))
    if true_vectors & false_vectors:
        return None
    if not true_vectors:
        return S.false
    if not false_vectors:
        return S.true

    Conditions = dict[int, frozenset[int]]

    def covers(conditions: Conditions, vector: tuple[Sign, ...]) -> bool:
        return all(vector[i] in allowed for i, allowed in conditions.items())

    def consistent(conditions: Conditions) -> bool:
        return not any(covers(conditions, v) for v in false_vectors)

    # each true sign vector is a conjunction of sign conditions; merge
    # conjunctions differing in one factor, then drop redundant conditions,
    # and repeat until nothing changes
    implicants: list[Conditions] = [{i: frozenset([s]) for i, s in enumerate(v)} for v in sorted(true_vectors)]
    while True:
        merged = True
        while merged:
            merged = False
            for a in range(len(implicants)):
                for b in range(a + 1, len(implicants)):
                    p, q = implicants[a], implicants[b]
                    if p.keys() != q.keys():
                        continue
                    diff = [i for i in p if p[i] != q[i]]
                    if len(diff) == 1:
                        [i] = diff
                        new = dict(p)
                        new[i] = p[i] | q[i]
                        if len(new[i]) == 3:
                            del new[i]
                        if consistent(new):
                            implicants[a] = new
                            del implicants[b]
                            merged = True
                            break
                if merged:
                    break
        reduced: list[Conditions] = []
        for conditions in implicants:
            for i in sorted(conditions, reverse=True):
                trial = dict(conditions)
                del trial[i]
                if consistent(trial):
                    conditions = trial
            if conditions not in reduced:
                reduced.append(conditions)
        if reduced == implicants:
            break
        implicants = reduced
    # drop implicants whose true cells are all covered by the others
    def covered(conditions: Conditions) -> set[tuple[Sign, ...]]:
        return {v for v in true_vectors if covers(conditions, v)}

    final = list(reduced)
    dropped = True
    while dropped:
        dropped = False
        for conditions in final:
            rest = [c for c in final if c is not conditions]
            cover = set().union(*[covered(c) for c in rest]) if rest else set()
            if covered(conditions) <= cover:
                final.remove(conditions)
                dropped = True
                break
    terms: list[Boolean] = []
    for conditions in final:
        atoms = [_SIGN_RELATIONS[allowed](factors[i].as_expr(), 0)
                 for i, allowed in sorted(conditions.items())]
        terms.append(And(*atoms))
    return Or(*terms)


def truth_tables(formulas: Sequence[Union[Boolean, bool]], gens: Sequence[Symbol],
                 method: Optional[str] = None) -> tuple[list[CADCell], list[list[bool]]]:
    """Truth values of several quantifier-free formulas on the cells of a
    single decomposition sign-invariant for the polynomials of all of them.

    Returns ``(cells, tables)`` where ``cells`` are the cells of
    `\\mathbb{R}^n` and ``tables[i][j]`` is the truth value of
    ``formulas[i]`` on ``cells[j]``. This is useful to compare formulas or
    to check implications between them with one decomposition.

    >>> from sympy.abc import x
    >>> from sympy_extras.polys.cad import truth_tables
    >>> cells, (a, b) = truth_tables([x > 0, x**3 > 0], [x])
    >>> [c.point for c in cells]
    [(-1,), (0,), (1,)]
    >>> a == b
    True
    """
    gens = [as_symbol(g) for g in gens]
    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    compiled = [_compile(as_boolean(f), gens, polys, index) for f in formulas]
    cad = cylindrical_algebraic_decomposition(polys, gens, method=method)
    tables: list[list[bool]] = []
    for c in compiled:
        table: list[bool] = []
        for cell in cad.cells:
            table.append(c(cell.signs))
        tables.append(table)
    return cad.cells, tables


def quantifier_elimination(formula: Union[Boolean, bool], quantifiers: QuantifierSpec = (),
                           free: Optional[Sequence[Symbol]] = None, method: Optional[str] = None) -> Boolean:
    """Eliminate the quantifiers of a formula over the real numbers.

    Parameters
    ==========

    formula : Boolean
        A Boolean combination (``And``, ``Or``, ``Not``, ``Implies``,
        ``Equivalent``, ``Xor``) of polynomial equations and inequalities
        with rational coefficients.
    quantifiers : list of pairs ``(kind, variables)``
        The quantifier prefix, outermost first. ``kind`` is ``'exists'``
        or ``'forall'`` and ``variables`` is a symbol or a list of symbols.
    free : list of Symbol, optional
        The free variables, in the order used for the decomposition. By
        default the free symbols of the formula which are not quantified,
        sorted by name.
    method : ``'mccallum'``, ``'hong'`` or None
        The projection operator, see
        :func:`~.cylindrical_algebraic_decomposition`.

    Returns
    =======

    ``S.true`` or ``S.false`` if all variables are quantified, otherwise a
    quantifier-free formula in the free variables which is equivalent to
    the input over the reals. With one free variable it describes a union
    of intervals with exact endpoints; with more the formula is built from
    sign conditions on the projection factors of the decomposition, and
    ``NotImplementedError`` is raised if those factors are not enough to
    express it.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, b, c, x, y
    >>> from sympy_extras.polys.cad import quantifier_elimination as qe
    >>> qe(x**2 + b*x + c > 0, [('forall', x)])
    b**2 - 4*c < 0
    >>> qe(Eq(a*x**2 + b*x + c, 0), [('exists', x)])
    Eq(c, 0) | (4*a*c - b**2 < 0) | ((a > 0) & Eq(4*a*c - b**2, 0)) | ((a < 0) & (4*a*c - b**2 <= 0))
    >>> qe(Eq(x**2 + y**2, 1), [('exists', y)])
    (x >= -1) & (x <= 1)
    >>> qe(x**2 + y**2 < 1, [('forall', x), ('exists', y)])
    False
    >>> qe(Eq(y, x**2), [('forall', x), ('exists', y)])
    True
    """
    cad, free_vars, value, cells = _truth_values(formula, free, quantifiers, method)
    if not free_vars:
        return S.true if value else S.false
    if len(free_vars) == 1:
        result = _interval_union(cells, free_vars[0])
        if result == S.Reals:
            return S.true
        if result == S.EmptySet:
            return S.false
        return _as_relational(result, free_vars[0])
    formula_ = _sign_formula(cad, cells, len(free_vars))
    if formula_ is None:
        raise NotImplementedError(
            "the solution set is not described by the signs of the projection factors")
    return formula_


def decide(formula: Union[Boolean, bool], quantifiers: QuantifierSpec, method: Optional[str] = None) -> bool:
    """Truth value of a formula with all its variables quantified.

    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad import decide
    >>> decide(Eq(x, 2*y), [('forall', x), ('exists', y)])
    True
    >>> decide(x**2 + y**2 < 0, [('exists', [x, y])])
    False
    """
    _, _, value, _ = _truth_values(formula, [], quantifiers, method)
    return bool(value)


def sample_points(formula: Union[Boolean, bool], gens: Sequence[Symbol],
                  method: Optional[str] = None) -> list[dict[Symbol, Expr]]:
    """Sample points of the cells on which a quantifier-free formula holds.

    Returns a list of dicts mapping the variables ``gens`` to exact
    coordinates, one for each cell of the decomposition satisfying
    ``formula``; the list is empty if the formula is not satisfiable.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad import sample_points
    >>> sample_points((x**2 + y**2 < 1) & (x > y), [x, y])
    [{x: 0, y: -1/2}, {x: CRootOf(2*x**2 - 1, 1), y: 0}, {x: 3/4, y: 0}]
    >>> sample_points(x**2 + y**2 < 0, [x, y])
    []
    """
    gens = [as_symbol(g) for g in gens]
    _, _, _, cells = _truth_values(formula, gens, [], method)
    return [dict(zip(gens, cell.point)) for cell, value in cells if value]


def solution_set(formula: Union[Boolean, bool], x: Symbol, quantifiers: QuantifierSpec = (),
                 method: Optional[str] = None) -> Set:
    """The set of values of the free variable ``x`` for which the quantified
    formula holds, as a union of intervals and points with exact endpoints.

    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad import solution_set
    >>> solution_set(Eq(3*x**2 + 2*x*y + y**2 - x + y - 7, 0), x, [('exists', y)])
    Interval(CRootOf(8*x**2 - 8*x - 29, 0), CRootOf(8*x**2 - 8*x - 29, 1))
    >>> solution_set(x**2 > 2, x)
    Union(Interval.open(-oo, CRootOf(x**2 - 2, 0)), Interval.open(CRootOf(x**2 - 2, 1), oo))
    """
    _, free_vars, _, cells = _truth_values(formula, [x], quantifiers, method)
    return _interval_union(cells, free_vars[0])
