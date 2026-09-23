"""Quantifier elimination over the reals by cylindrical algebraic
decomposition.

A formula is a Boolean combination of polynomial relations. Given a prefix
of quantifiers over some of its variables, the truth of the formula on a
cell of the decomposition of the space of the free variables is obtained by
propagating the truth values of the cells of the full decomposition
downwards, since every polynomial, and so every atom of the formula, has a
constant sign on every cell.

With several free variables the answer is a formula in the signs of the
projection factors of the free levels (a solution formula), which is
found when no true cell of the space of the free variables has the sign
vector of a false one; when one does, the projection is augmented after
Hong [1]_ with the derivatives of the factors which vanish between the
two cells, and the space of the free variables is decomposed again
(:func:`_separating_refinement`).

References
==========

.. [1] H. Hong, Simple solution formula construction in cylindrical
       algebraic decomposition based quantifier elimination, ISSAC 1992,
       pp. 177-188.
"""
from __future__ import annotations

from itertools import combinations
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

from .lifting import (CAD, CADCell, Lifting, NotWellOriented, _factor_signs,
    cylindrical_algebraic_decomposition, lifting_for)
from .projection import _PROJECTIONS, _to_polys, augmented_projection_sets, projection_sets

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

    def trial(self, signs: Sequence[Optional[Sign]]) -> Optional[bool]:
        """The truth value when the known signs determine it (the trial
        evaluation of Collins and Hong): ``None`` stands for a sign, and
        for a truth value, which is not known."""
        raise NotImplementedError


class _Const(_Compiled):
    __slots__ = ('value',)

    def __init__(self, value: bool) -> None:
        self.value = value

    def __call__(self, signs: Sequence[Sign]) -> bool:
        return self.value

    def trial(self, signs: Sequence[Optional[Sign]]) -> Optional[bool]:
        return self.value


class _Atom(_Compiled):
    """The sign test ``test`` on the polynomial of index ``index``."""

    __slots__ = ('index', 'test')

    def __init__(self, index: int, test: SignTest) -> None:
        self.index = index
        self.test = test

    def __call__(self, signs: Sequence[Sign]) -> bool:
        return self.test(signs[self.index])

    def trial(self, signs: Sequence[Optional[Sign]]) -> Optional[bool]:
        sign = signs[self.index]
        return None if sign is None else self.test(sign)


class _Not(_Compiled):
    __slots__ = ('arg',)

    def __init__(self, arg: _Compiled) -> None:
        self.arg = arg

    def __call__(self, signs: Sequence[Sign]) -> bool:
        return not self.arg(signs)

    def trial(self, signs: Sequence[Optional[Sign]]) -> Optional[bool]:
        value = self.arg.trial(signs)
        return None if value is None else not value


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

    def trial(self, signs: Sequence[Optional[Sign]]) -> Optional[bool]:
        kind = self.kind
        values = [a.trial(signs) for a in self.args]
        if kind == 'and':
            if any(v is False for v in values):
                return False
            return None if any(v is None for v in values) else True
        if kind == 'or':
            if any(v is True for v in values):
                return True
            return None if any(v is None for v in values) else False
        if kind == 'implies':
            if values[0] is False or values[1] is True:
                return True
            return None if values[0] is None or values[1] is None else False
        if kind == 'equivalent' and any(v is True for v in values) and any(v is False for v in values):
            return False
        if any(v is None for v in values):
            return None
        if kind == 'xor':
            return sum(1 for v in values if v) % 2 == 1
        return True


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
                  quantifiers: QuantifierSpec, method: Optional[str], partial: bool = True,
                  prune: str = 'none') -> tuple[CAD, list[Symbol], Optional[bool], list[CellTruth]]:
    """The decomposition of the space of the free variables, the free
    variables, the truth value of the quantified formula if there are no
    free variables (else ``None``) and its truth value on each cell of the
    space of the free variables (an empty list if there are none).

    With ``partial`` the decomposition is a partial one (see
    :func:`_partial_truth_values`), and ``prune`` tells what is not lifted
    in the space of the free variables either: nothing (``'none'``), the
    cells on which the formula is already false (``'false'``: the list has
    such a cell of a lower level in the place of the cells over it), or the
    cells on which its truth value is already known (``'both'``)."""
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
    if prune not in ('none', 'false', 'both'):
        raise ValueError("unknown pruning %r" % (prune,))

    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    compiled = _compile(formula_, gens, polys, index)
    if partial:
        equational = _equational_constraint(formula_, gens) if bound else None
        return _partial_truth_values(compiled, polys, gens, free_list, prefix, method, prune, equational)
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


def _known_signs(cell: CADCell, factor_signs: Sequence[tuple[Sign, Sequence[tuple[int, int, int]]]]
                 ) -> list[Optional[Sign]]:
    """The signs of the input polynomials which the projection factors of
    the levels of the cell determine (``factor_signs`` from
    :func:`~.lifting._factor_signs`): a polynomial whose factors are all
    of those levels, or one of whose factors vanishes there."""
    signs: list[Optional[Sign]] = []
    for constant, items in factor_signs:
        sign: Optional[Sign] = constant
        for level, i, exponent in items:
            if level > cell.level:
                if sign != 0:
                    sign = None
                continue
            factor = cell._factor_sign(level, i)
            if factor == 0:
                sign = 0
            elif sign is not None:
                sign *= factor**exponent
        signs.append(sign)
    return signs


class _PartialEvaluation:
    """The truth value of a quantified formula on the cells of a lifting,
    building the stacks which it depends on and no other [1]_:

    * the formula is evaluated on a cell of any level as soon as the signs
      known there determine it (trial evaluation, in three-valued logic:
      the sign of an input polynomial is known when its factors of the
      levels reached are, or when one of them vanishes);
    * over a cell of the space of the bound variables the stack is searched
      for one cell which settles the quantifier, a true one for ``exists``
      and a false one for ``forall``, the sectors first: their sample
      points are rational, while lifting over a section takes an algebraic
      extension.

    References
    ==========

    .. [1] G. E. Collins, H. Hong, Partial cylindrical algebraic
           decomposition for quantifier elimination, J. Symbolic Comput. 12
           (1991), pp. 299-328.
    """

    def __init__(self, compiled: _Compiled, lifting: Lifting,
                 factor_signs: list[tuple[Sign, list[tuple[int, int, int]]]], free: int,
                 prefix: QuantifierPrefix, prune: str) -> None:
        self.compiled = compiled
        self.lifting = lifting
        self.factor_signs = factor_signs
        self.free = free
        self.prefix = prefix
        self.prune = prune

    def known_signs(self, cell: CADCell) -> list[Optional[Sign]]:
        """The signs of the input polynomials which the projection factors
        of the levels of the cell determine."""
        return _known_signs(cell, self.factor_signs)

    def truth(self, cell: CADCell) -> bool:
        """The truth value on a cell of the space of the free variables or
        above: of the formula with the quantifiers of the variables beyond
        the level of the cell."""
        value = self.compiled.trial(self.known_signs(cell))
        if value is not None:
            return value
        kind = self.prefix[cell.level - self.free][0]
        stack = self.lifting.stack(cell)
        settling = kind == 'exists'
        for child in stack[0::2] + stack[1::2]:
            if self.truth(child) == settling:
                return settling
        return not settling

    def free_cells(self, cell: CADCell, found: list[CellTruth]) -> None:
        """The cells of the space of the free variables over ``cell`` (or
        ``cell`` itself when it is not lifted), in lexicographic order,
        with their truth values."""
        if cell.level == self.free:
            found.append((cell, self.truth(cell)))
            return
        if self.prune != 'none' and cell.level > 0:
            value = self.compiled.trial(self.known_signs(cell))
            if value is False or (value is True and self.prune == 'both'):
                found.append((cell, value))
                return
        for child in self.lifting.stack(cell):
            self.free_cells(child, found)


def _equational_constraint(formula: Boolean, gens: Sequence[Symbol]) -> Optional[Poly]:
    """A polynomial which the formula implies to vanish, of positive
    degree in the last variable: the polynomial of an equation which is
    the formula or one of the terms of its conjunction (McCallum's
    equational constraint, which reduces the projection of the last
    level). ``None`` when there is none."""
    atoms = [formula] if isinstance(formula, Eq) else list(formula.args) if isinstance(formula, And) else []
    for atom in atoms:
        if isinstance(atom, Eq):
            [p] = _to_polys([Poly(atom.lhs - atom.rhs, *gens)], gens)
            if p.degree(gens[-1]) > 0:
                return p
    return None


def _partial_truth_values(compiled: _Compiled, polys: list[Poly], gens: list[Symbol], free: list[Symbol],
                          prefix: QuantifierPrefix, method: Optional[str], prune: str,
                          equational: Optional[Poly] = None
                          ) -> tuple[CAD, list[Symbol], Optional[bool], list[CellTruth]]:
    """:func:`_truth_values` by a partial decomposition. The projection is
    that of the full one, reduced at the last level by the ``equational``
    constraint if there is one (the last variable must be quantified: the
    cells of the last level are not sign-invariant off its sections, where
    the formula is false); McCallum's is given up for Hong's when a factor
    vanishes identically over a cell of positive dimension which is lifted
    (over the others it does no harm: the stack over a cell only depends on
    the cells below it). The lifting is the one kept from an earlier
    question with the same projection factor sets, when there was one
    (:func:`~.lifting_for`)."""
    if method is None:
        methods = ['mccallum', 'hong']
    elif method in ('mccallum', 'hong'):
        methods = [method]
    else:
        raise ValueError("unknown projection method %r" % (method,))
    if not gens:
        raise ValueError("at least one generator is needed")
    polys = _to_polys(polys, gens)
    for m in methods:
        projection = projection_sets(polys, gens, method=m, equational=equational)
        lifting = lifting_for(projection, gens, m)
        evaluation = _PartialEvaluation(compiled, lifting, _factor_signs(polys, projection, gens), len(free),
                                        prefix, prune)
        cells: list[CellTruth] = []
        value: Optional[bool] = None
        try:
            if free:
                evaluation.free_cells(lifting.root, cells)
            else:
                value = evaluation.truth(lifting.root)
        except NotWellOriented:
            if m == methods[-1]:
                raise
            continue
        return CAD(gens, polys, projection, m, lifting.levels()), free, value, cells
    raise RuntimeError("unreachable")  # pragma: no cover


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


#: the true and the false cells of the space of the free variables with
#: one sign vector of the projection factors
_SignClasses = dict[tuple[Sign, ...], tuple[list[CADCell], list[CADCell]]]


def _sign_classes(cells: Sequence[CellTruth]) -> _SignClasses:
    """The cells grouped by the signs of the projection factors of their
    levels, the true ones and the false ones apart."""
    classes: _SignClasses = {}
    for cell, value in cells:
        trues, falses = classes.setdefault(_sign_vector(cell), ([], []))
        (trues if value else falses).append(cell)
    return classes


def _conflicts(classes: _SignClasses) -> list[tuple[CADCell, CADCell]]:
    """The pairs of a true and a false cell with the same sign vector:
    the signs of the projection factors do not tell them apart."""
    return [(t, f) for trues, falses in classes.values() for t in trues for f in falses]


def _ancestors(cell: CADCell) -> list[CADCell]:
    """The cell and the cells below it, from level 1 up."""
    chain: list[CADCell] = []
    current: Optional[CADCell] = cell
    while current is not None and current.level > 0:
        chain.append(current)
        current = current.parent
    chain.reverse()
    return chain


def _separating_factors(cad: CAD, a: CADCell, b: CADCell) -> tuple[int, list[Poly]]:
    """The first level at which two cells with the same sign vector lie
    in different cells (of one stack, since the cells below are the
    same), and the projection factors of that level which vanish on the
    sections of the stack between the two, these included: by Thom's
    lemma the signs of such a factor and of its derivatives of all orders
    tell the two apart, so that a round of derivatives of these factors
    adds a polynomial to the projection as long as the conflict lasts."""
    chain_a, chain_b = _ancestors(a), _ancestors(b)
    for level, (cell_a, cell_b) in enumerate(zip(chain_a, chain_b), start=1):
        if cell_a.index != cell_b.index:
            break
    else:
        raise ValueError("the cells are one cell")  # pragma: no cover
    stack = cad.cells_at(1) if cell_a.parent is None else cad.children(cell_a.parent)
    low, high = sorted([cell_a.index[-1], cell_b.index[-1]])
    factors: list[Poly] = []
    for cell in stack:
        if cell.is_section and low <= cell.index[-1] <= high:
            for i, sign in enumerate(cell._signs):
                if sign == 0 and cad.projection[level - 1][i] not in factors:
                    factors.append(cad.projection[level - 1][i])
    return level, factors


def _inherited_truth(lifting: Lifting, counts: Sequence[int], truth: dict[tuple[int, ...], bool],
                     k: int) -> list[CellTruth]:
    """The truth values of the cells of level ``k`` of a lifting which
    refines the decomposition whose cells have the values ``truth`` (by
    their indices): the first ``counts[j]`` factors of level ``j + 1`` of
    the refining projection are those of the refined one, so that a cell
    of the refinement is a section of the refined decomposition when one
    of them vanishes there (and not identically over the cell below), and
    the index of the cell of the refined decomposition which contains it
    counts those sections."""
    found: list[CellTruth] = []

    def walk(parent: CADCell, index: tuple[int, ...]) -> None:
        stack = lifting.stack(parent)
        count = counts[parent.level]
        # a factor with the sign 0 on a sector vanishes identically over
        # the parent, and has no section
        kept = [i for i in range(count) if any(cell._signs[i] != 0 for cell in stack)]
        below = 0
        for child in stack:
            if any(child._signs[i] == 0 for i in kept):
                below += 1
                child_index = index + (2 * below,)
            else:
                child_index = index + (2 * below + 1,)
            if child.level == k:
                found.append((child, truth[child_index]))
            else:
                walk(child, child_index)

    walk(lifting.root, ())
    return found


#: the rounds of augmentation of the projection which are tried
_ROUNDS = 16


def _separating_refinement(cad: CAD, cells: Sequence[CellTruth], k: int) -> Optional[tuple[CAD, list[CellTruth]]]:
    """A refinement of the decomposition of the space of the first ``k``
    variables (the free ones) on which the signs of the projection factors
    tell the true cells from the false ones, with its cells and their
    truth values, by the augmented projection of Hong [1]_: as long as a
    true and a false cell have the same sign vector, polynomials are added
    to the projection factor sets of the free levels and the space is
    decomposed again, each cell of the refinement taking the truth value
    of the cell of the previous decomposition which contains it. The
    polynomials added are, at the first round, the full projection of the
    level above the free space (the reduced projection for an equational
    constraint leaves out resultants and discriminants which the formula
    may need), and then the derivatives of the projection factors which
    vanish between two cells with one sign vector, at the first level
    where the two lie in different cells (Brown's choice of the
    polynomials which resolve a conflict [2]_): the signs of a factor
    and of its derivatives of all orders tell two cells of one stack
    apart (Thom's lemma), so that the rounds end. ``None`` if they do
    not within :data:`_ROUNDS`.

    References
    ==========

    .. [1] H. Hong, Simple solution formula construction in cylindrical
           algebraic decomposition based quantifier elimination, ISSAC
           1992, pp. 177-188.
    .. [2] C. W. Brown, Solution formula construction for truth invariant
           CAD's, PhD thesis, University of Delaware, 1999.
    """
    gens = list(cad.gens)
    free = gens[:k]
    projection = [list(level) for level in cad.projection[:k]]
    truth = {cell.index: value for cell, value in cells}
    counts = [len(level) for level in projection]
    method = cad.method
    current, current_cells = cad, list(cells)
    differentiated: set[Poly] = set()
    extra: list[Poly] = []
    above_added = False
    for _ in range(_ROUNDS):
        conflicts = _conflicts(_sign_classes(current_cells))
        if not conflicts:
            return current, current_cells
        added: list[Poly] = []
        if not above_added and k < len(gens):
            above_added = True
            operator = _PROJECTIONS[method]
            for g in operator(cad.projection[k], gens[k]):
                g = Poly(g.as_expr(), *free)
                if not g.is_ground and g not in projection[_free_level(g, free) - 1] and g not in added:
                    added.append(g)
        if not added:
            for t, f in conflicts:
                level, factors = _separating_factors(current, t, f)
                for g in factors:
                    if g not in differentiated:
                        differentiated.add(g)
                        derivative = g.diff(free[level - 1])
                        if not derivative.is_ground:
                            added.append(derivative)
        if not added:
            return None  # pragma: no cover
        extra.extend(added)
        while True:
            refined = augmented_projection_sets(projection, free, extra, method)
            lifting = lifting_for(refined, free, method)
            try:
                levels = lifting.lift_all()
            except NotWellOriented:
                if method == 'hong':
                    raise
                method = 'hong'
                continue
            break
        current_cells = _inherited_truth(lifting, counts, truth, k)
        current = CAD(gens, cad.polys, refined + [list(level) for level in cad.projection[k:]], method,
                      levels + [[] for _ in gens[k:]])
    return None


def _free_level(g: Poly, free: Sequence[Symbol]) -> int:
    """The last of the ``free`` variables which ``g`` depends on, from 1."""
    for j in range(len(free), 0, -1):
        if g.degree(free[j - 1]) > 0:
            return j
    return 0


def _sign_formula(cad: CAD, cells: Sequence[CellTruth], k: int) -> Optional[Boolean]:
    """A formula in the projection factors of the first ``k`` levels that
    holds exactly on the true cells, or ``None`` if the truth of the cells
    is not determined by the signs of those factors: a disjunction of
    prime implicants (conjunctions of sign conditions which hold on no
    false cell) covering the true cells with the fewest conditions
    (:func:`_prime_implicants`, :func:`_minimum_cover`), or a greedy
    cover when the prime implicants are too many."""
    factors = [f for level in cad.projection[:k] for f in level]
    classes = _sign_classes(cells)
    if _conflicts(classes):
        return None
    true_vectors = {vector for vector, (trues, _) in classes.items() if trues}
    false_vectors = {vector for vector, (_, falses) in classes.items() if falses}
    if not true_vectors:
        return S.false
    if not false_vectors:
        return S.true

    trues = sorted(true_vectors)
    falses = sorted(false_vectors)
    primes = _prime_implicants(trues, falses)
    if primes is None:
        cover = _greedy_implicants(trues, falses)
    else:
        cover = _minimum_cover(primes, trues)
    terms: list[Boolean] = []
    for implicant in cover:
        atoms = [_SIGN_RELATIONS[_SIGNS_OF[mask]](factors[i].as_expr(), 0)
                 for i, mask in enumerate(implicant) if mask != _ALL]
        terms.append(And(*atoms))
    return Or(*terms)


#: the allowed signs of a factor as a bit mask: bit 0 for -1, bit 1 for 0
#: and bit 2 for 1
_BIT: dict[Sign, int] = {-1: 1, 0: 2, 1: 4}
_ALL = 7
_SIGNS_OF: dict[int, frozenset[int]] = {mask: frozenset(s for s, bit in _BIT.items() if mask & bit)
                                        for mask in range(1, 8)}
#: an implicant: the allowed signs of every factor, a conjunction of sign
#: conditions
_Implicant = tuple[int, ...]
#: the transversals and the prime implicants beyond which the greedy
#: minimisation is used instead
_PRIME_LIMIT = 4000


def _covers(implicant: _Implicant, vector: Sequence[Sign]) -> bool:
    return all(mask & _BIT[s] for mask, s in zip(implicant, vector))


def _atoms(implicant: _Implicant) -> int:
    return sum(1 for mask in implicant if mask != _ALL)


def _minimal_transversals(edges: Sequence[frozenset[int]], limit: int) -> Optional[list[frozenset[int]]]:
    """The minimal hitting sets of the hyperedges ``edges`` (Berge's
    algorithm: the minimal transversals of the first edges are extended
    edge by edge), or ``None`` when more than ``limit`` of them appear on
    the way."""
    transversals: list[frozenset[int]] = [frozenset()]
    for edge in edges:
        hitting = [t for t in transversals if t & edge]
        extended: list[frozenset[int]] = []
        for t in transversals:
            if t & edge:
                continue
            for e in edge:
                candidate = t | {e}
                if any(h < candidate for h in hitting) or candidate in extended:
                    continue
                extended.append(candidate)
        extended = [c for c in extended if not any(d < c for d in extended)]
        transversals = hitting + extended
        if len(transversals) > limit:
            return None
    return transversals


def _prime_implicants(trues: Sequence[tuple[Sign, ...]], falses: Sequence[tuple[Sign, ...]]
                      ) -> Optional[list[_Implicant]]:
    """The prime implicants: the conjunctions of sign conditions which
    hold on no false vector and on a true one, and which no condition
    can be dropped or widened from. An implicant which holds on a true
    vector `v` excludes, for every false vector, one of its signs which
    differ from those of `v`: the sets of the pairs (factor, sign) which
    a false vector offers are hyperedges, and the prime implicants
    holding on `v` are their minimal transversals. ``None`` when there
    are more than :data:`_PRIME_LIMIT` of them, or of the transversals
    on the way."""
    if not trues:
        return []
    m = len(trues[0])
    found: dict[_Implicant, None] = {}
    for v in trues:
        edges_found: set[frozenset[int]] = set()
        for f in falses:
            edges_found.add(frozenset(3 * i + f[i] + 1 for i in range(m) if f[i] != v[i]))
        edges = sorted(edges_found, key=len)
        # an edge containing another is hit whenever the other is
        edges = [e for e in edges if not any(d < e for d in edges)]
        transversals = _minimal_transversals(edges, _PRIME_LIMIT)
        if transversals is None:
            return None
        for t in transversals:
            masks = [_ALL] * m
            for code in t:
                masks[code // 3] &= ~(1 << (code % 3))
            found[tuple(masks)] = None
        if len(found) > _PRIME_LIMIT:
            return None
    return list(found)


#: the number of prime implicants up to which the cover is found exactly
_EXACT_COVER = 18


def _minimum_cover(primes: Sequence[_Implicant], trues: Sequence[tuple[Sign, ...]]) -> list[_Implicant]:
    """A cover of the true vectors by prime implicants with the fewest
    sign conditions (then the fewest implicants): a greedy cover without
    its redundant members, bettered by every smaller cover when the
    primes are at most :data:`_EXACT_COVER`."""
    full = (1 << len(trues)) - 1
    coverage = [sum(1 << j for j, v in enumerate(trues) if _covers(p, v)) for p in primes]

    def union(indices: Sequence[int]) -> int:
        total = 0
        for i in indices:
            total |= coverage[i]
        return total

    def cost(indices: Sequence[int]) -> tuple[int, int]:
        return sum(_atoms(primes[i]) for i in indices), len(indices)

    order = sorted(range(len(primes)), key=lambda i: (-bin(coverage[i]).count('1'), _atoms(primes[i]), primes[i]))
    chosen: list[int] = []
    covered = 0
    while covered != full:
        best = max(order, key=lambda i: (bin(coverage[i] & ~covered).count('1'), -_atoms(primes[i])))
        chosen.append(best)
        covered |= coverage[best]
    for i in list(chosen):
        rest = [j for j in chosen if j != i]
        if union(rest) == full:
            chosen = rest
    best_cost = cost(chosen)
    if len(primes) <= _EXACT_COVER:
        for size in range(1, len(chosen) + 1):
            for combination in combinations(order, size):
                if union(combination) == full and cost(combination) < best_cost:
                    best_cost, chosen = cost(combination), list(combination)
    return [primes[i] for i in sorted(chosen, key=lambda i: primes[i])]


def _greedy_implicants(trues: Sequence[tuple[Sign, ...]], falses: Sequence[tuple[Sign, ...]]) -> list[_Implicant]:
    """A cover of the true vectors by implicants, when the prime
    implicants are too many: each true sign vector is a conjunction of
    sign conditions; conjunctions differing in one factor are merged,
    then the redundant conditions dropped, until nothing changes, and
    the implicants whose true vectors the others cover are removed."""
    Conditions = dict[int, frozenset[int]]

    def covers(conditions: Conditions, vector: tuple[Sign, ...]) -> bool:
        return all(vector[i] in allowed for i, allowed in conditions.items())

    def consistent(conditions: Conditions) -> bool:
        return not any(covers(conditions, v) for v in falses)

    implicants: list[Conditions] = [{i: frozenset([s]) for i, s in enumerate(v)} for v in trues]
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

    def covered(conditions: Conditions) -> set[tuple[Sign, ...]]:
        return {v for v in trues if covers(conditions, v)}

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
    m = len(trues[0]) if trues else 0
    result: list[_Implicant] = []
    for conditions in final:
        result.append(tuple(sum(_BIT[s] for s in conditions[i]) if i in conditions else _ALL for i in range(m)))
    return result


class _Tables(_Compiled):
    """Several formulas evaluated together: true when the signs known
    determine every one of them (for :func:`truth_tables`, which lifts
    nothing over a cell where they are all determined)."""

    __slots__ = ('args',)

    def __init__(self, args: Sequence[_Compiled]) -> None:
        self.args = list(args)

    def __call__(self, signs: Sequence[Sign]) -> bool:
        return True

    def trial(self, signs: Sequence[Optional[Sign]]) -> Optional[bool]:
        return None if any(a.trial(signs) is None for a in self.args) else True


def truth_tables(formulas: Sequence[Union[Boolean, bool]], gens: Sequence[Symbol],
                 method: Optional[str] = None, partial: bool = True) -> tuple[list[CADCell], list[list[bool]]]:
    """Truth values of several quantifier-free formulas on the cells of a
    single decomposition sign-invariant for the polynomials of all of them.

    Returns ``(cells, tables)`` where ``cells`` are cells of the
    decomposition which cover `\\mathbb{R}^n` and ``tables[i][j]`` is the
    truth value of ``formulas[i]`` on ``cells[j]``. This is useful to
    compare formulas or to check implications between them with one
    decomposition. The decomposition is partial: a cell of a lower level
    over which every formula has one truth value (the signs of the
    projection factors of its levels determine them) is not lifted, and
    stands in the list for the cells over it; with ``partial=False`` the
    cells are those of `\\mathbb{R}^n` of the full decomposition.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad import truth_tables
    >>> cells, (a, b) = truth_tables([x > 0, x**3 > 0], [x])
    >>> [c.point for c in cells]
    [(-1,), (0,), (1,)]
    >>> a == b
    True
    >>> cells, (a, b) = truth_tables([x > 0, x*y > 0], [x, y])
    >>> [(c.point, s, t) for c, s, t in zip(cells, a, b)]
    [((-1, -1), False, True), ((-1, 0), False, False), ((-1, 1), False, False), ((0,), False, False), ((1, -1), True, False), ((1, 0), True, False), ((1, 1), True, True)]
    >>> [c.point for c in truth_tables([x > 0, x*y > 0], [x, y], partial=False)[0]]
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    """
    gens = [as_symbol(g) for g in gens]
    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    compiled = [_compile(as_boolean(f), gens, polys, index) for f in formulas]
    tables: list[list[bool]] = []
    if not partial:
        cad = cylindrical_algebraic_decomposition(polys, gens, method=method)
        for c in compiled:
            tables.append([c(cell.signs) for cell in cad.cells])
        return cad.cells, tables
    cad, _, _, cells = _partial_truth_values(_Tables(compiled), polys, gens, gens, [], method, 'both')
    factor_signs = _factor_signs(cad.polys, cad.projection, cad.gens)
    for c in compiled:
        table: list[bool] = []
        for cell, _ in cells:
            value = c.trial(_known_signs(cell, factor_signs))
            if value is None:
                raise RuntimeError("the cell %s does not determine %s" % (cell, c))  # pragma: no cover
            table.append(value)
        tables.append(table)
    return [cell for cell, _ in cells], tables


def quantifier_elimination(formula: Union[Boolean, bool], quantifiers: QuantifierSpec = (),
                           free: Optional[Sequence[Symbol]] = None, method: Optional[str] = None,
                           partial: bool = True) -> Boolean:
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
    sign conditions on the projection factors of the decomposition (a
    minimum cover of the true cells by prime implicants), and when those
    factors do not tell the true cells from the false ones the projection
    is augmented with the derivatives of the factors until they do
    (Hong's solution formula construction, :func:`_separating_refinement`;
    the root functions of the cylindrical description of
    :func:`~sympy_extras.polys.cad.cylindrical_formula` are the last
    resort, should the augmentation not end).

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, b, c, x, y
    >>> from sympy_extras.polys.cad import quantifier_elimination as qe
    >>> qe(x**2 + b*x + c > 0, [('forall', x)])
    b**2 - 4*c < 0
    >>> qe(Eq(a*x**2 + b*x + c, 0), [('exists', x)])
    Eq(c, 0) | (4*a*c - b**2 < 0) | (Ne(b, 0) & (4*a*c - b**2 <= 0))
    >>> qe(Eq(x**2 + y**2, 1), [('exists', y)])
    (x >= -1) & (x <= 1)
    >>> qe(x**2 + y**2 < 1, [('forall', x), ('exists', y)])
    False
    >>> qe(Eq(y, x**2), [('forall', x), ('exists', y)])
    True
    >>> from sympy.abc import z
    >>> qe(Eq(z**2, x) & (z > y), [('exists', z)], free=[x, y])
    ((x >= 0) & (y < 0)) | (x - y**2 > 0)
    """
    cad, free_vars, value, cells = _truth_values(formula, free, quantifiers, method, partial)
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
        # the signs of the projection factors do not tell the true cells
        # from the false ones (sympy-extras#9): the augmented projection
        # adds the polynomials whose signs do
        refined = _separating_refinement(cad, cells, len(free_vars))
        if refined is not None:
            cad, cells = refined
            formula_ = _sign_formula(cad, cells, len(free_vars))
    if formula_ is None:
        # the root functions of the factors describe the set
        from .cylindrical import described_by_root_functions
        return described_by_root_functions(cad, free_vars, cells)
    return formula_


def decide(formula: Union[Boolean, bool], quantifiers: QuantifierSpec, method: Optional[str] = None,
           partial: bool = True) -> bool:
    """Truth value of a formula with all its variables quantified.

    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad import decide
    >>> decide(Eq(x, 2*y), [('forall', x), ('exists', y)])
    True
    >>> decide(x**2 + y**2 < 0, [('exists', [x, y])])
    False
    """
    _, _, value, _ = _truth_values(formula, [], quantifiers, method, partial)
    return bool(value)


def sample_points(formula: Union[Boolean, bool], gens: Sequence[Symbol],
                  method: Optional[str] = None, partial: bool = True) -> list[dict[Symbol, Expr]]:
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
    _, _, _, cells = _truth_values(formula, gens, [], method, partial, 'false')
    return [dict(zip(gens, cell.point)) for cell, value in cells if value]


def solution_set(formula: Union[Boolean, bool], x: Symbol, quantifiers: QuantifierSpec = (),
                 method: Optional[str] = None, partial: bool = True) -> Set:
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
    _, free_vars, _, cells = _truth_values(formula, [x], quantifiers, method, partial)
    return _interval_union(cells, free_vars[0])
