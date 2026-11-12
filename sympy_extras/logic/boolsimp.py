"""SymPy's ``simplify`` of Boolean formulas, on the objects of the core:
a typed port of the ``_eval_simplify`` methods of ``sympy.logic.boolalg``
(``BooleanFunction``, ``And``, ``Or``), ``simplify_univariate`` and the
pattern-based simplification of the relations of ``And`` and ``Or``. It
simplifies the conditions of ``piecewise_simplify_arguments`` and the
compound formulas of :func:`sympy_extras.assumptions.refine.simplify`.

:func:`simplify_boolean` simplifies a formula as SymPy's ``simplify``
does:

1. a formula in one symbol without equations is written as the union of
   the intervals where it holds (``simplify_univariate``): SymPy finds the
   intervals with ``Piecewise._intervals`` and ``_solve_inequality``,
   here each relation is solved for the symbol when it is linear in it,
   or when it is a polynomial with rational coefficients whose real roots
   the native :func:`~sympy_extras.polys.polyroots.roots` finds, and the
   truth of the formula is decided on the breakpoints and between them;
   the intervals are maximal (SymPy writes one interval per term of the
   disjunctive form, which its patterns merge afterwards), and a formula
   true at an isolated point is left as it is, as SymPy leaves it. The
   inclusion of the endpoints is decided on the formula itself: SymPy
   tests them on the relation built for the previous interval, so that
   its ``simplify(~((x > 0) & (x < 1)))`` is ``(x >= 1) | (x < 0)``,
   false at ``x = 0``. With ``sympy_intervals`` (the conditions of
   ``piecewise_simplify_arguments``, which must be SymPy's) the intervals
   are SymPy's, from the port of ``Piecewise._intervals``, with the
   endpoints still decided on the formula and a formula true at an
   isolated point still left as it is (SymPy's ``simplify`` of
   ``(x**2 <= 0) & (x < 2)`` is ``False``);
2. the arguments are simplified (the relations by
   :func:`~sympy_extras.assumptions.refine.simplify_relational`, the
   port of ``Relational._eval_simplify``) and the formula is minimised by
   :func:`~sympy_extras.logic.boolalg.simplify_logic`, with its relations
   simplified once more (``deep=True``: SymPy's ``simplify`` of a
   relation is not idempotent, ``x - y > -x`` is ``y < 2*x``, which is
   ``2*x > y``);
3. in a conjunction, the equations in one symbol are solved for it and
   substituted into the others (``Eq(x, 1) & (y > x)`` is
   ``Eq(x, 1) & (y > 1)``); a disjunction writes ``a <= b`` as
   ``Eq(a, b) | (a < b)`` first;
4. pairs and triples of relations are replaced by shorter equivalent
   ones by SymPy's patterns (``(x < 2) & Ne(x, 2)`` is ``x < 2``,
   ``(x > 2) | (x < -2)`` is ``Abs(x) > 2``); a replacement is kept only
   when it is shorter (``count_ops``) and has no ``ITE``, ``Min`` or
   ``Max`` left, which happens when its conditions are decided. The
   matching of the patterns, SymPy's ``Wild`` matching on the pairs of
   relations, is structural here: ``-b`` matches only once ``b`` is
   bound, as in SymPy.

``Xor`` and the other connectives take step 1 and 2 only (SymPy rewrites
``Xor`` in algebraic normal form first, which is not ported); a Boolean
which is not a connective or a relation (``Contains``) is left as it is,
as SymPy does.

Examples
========

>>> from sympy_extras.core import symbols, Symbol, Eq, Ne, and_, or_, not_
>>> from sympy_extras.logic.boolsimp import simplify_boolean
>>> x, y = symbols('x y')
>>> simplify_boolean(and_(x > 1, x < 0)), simplify_boolean(and_(x > 0, Ne(x, 0)))
(False, x > 0)
>>> simplify_boolean(or_(x > 2, x < -2)), simplify_boolean(and_(Eq(x, 1), y > x))
(Abs(x) > 2, Eq(x, 1) & (y > 1))
>>> r = Symbol('r', real=True)
>>> simplify_boolean(not_(and_(r > 0, r < 1))), simplify_boolean(and_(r <= 1, r <= 2))
((r >= 1) | (r <= 0), r <= 1)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/logic/boolalg.py (BooleanFunction._eval_simplify,
# And._eval_simplify, Or._eval_simplify, simplify_univariate,
# _apply_patternbased_simplification, _apply_patternbased_twoterm_simplification,
# _apply_patternbased_threeterm_simplification, _simplify_patterns_and,
# _simplify_patterns_and3, _simplify_patterns_or), sympy/core/relational.py
# (_canonical) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing on the core's Booleans; structural matching of
# the patterns instead of Wild; the univariate intervals computed natively
# (maximal, polynomial relations, the endpoints tested on the formula);
# Xor without the algebraic normal form.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations, product
from typing import Callable, Final, Optional, Sequence

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.logic import (And, Boolean, BooleanAtom, BooleanFunction, ITE, Or, Xor, and_, false, ite,
                                     not_, or_, to_boolean, true)
from sympy_extras.core.mul import neg
from sympy_extras.core.numbers import Float, Integer, NegativeInfinity, Rational, oo
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq, ge, gt, le, lt, ne
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.elementary.miscellaneous import Max, Min, max_, min_

__all__ = ['simplify_boolean', 'simplify_univariate']

#: the simplification of a relation (``Relational._eval_simplify``)
RelationSimplifier = Callable[[Relational], Boolean]
#: the measure of the size of a formula (``count_ops``)
Measure = Callable[[Boolean], int]
#: the expressions bound to the names ``a``, ``b``, ``c`` of a pattern
Bindings = dict[str, Expr]


class _Options:
    """The relation simplifier, the measure and the ratio of ``simplify``,
    and whether the univariate intervals are SymPy's."""

    def __init__(self, relation: RelationSimplifier, measure: Measure, ratio: float,
                 sympy_intervals: bool) -> None:
        self.relation: Final[RelationSimplifier] = relation
        self.measure: Final[Measure] = measure
        self.ratio: Final[float] = ratio
        self.sympy_intervals: Final[bool] = sympy_intervals


def simplify_boolean(formula: Boolean, relation: Optional[RelationSimplifier] = None,
                     measure: Optional[Measure] = None, ratio: float = 1.7,
                     sympy_intervals: bool = False) -> Boolean:
    """SymPy's ``simplify`` of a Boolean formula (see the module
    documentation). ``relation`` simplifies the relations (by default
    :func:`~sympy_extras.assumptions.refine.simplify_relational` with the
    default options of ``simplify``) and ``measure`` is the size of a
    formula (``count_ops`` by default); ``sympy_intervals`` is passed to
    :func:`simplify_univariate`.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq, or_, and_
    >>> from sympy_extras.logic.boolsimp import simplify_boolean
    >>> x = symbols('x')
    >>> simplify_boolean(or_(x < 1, and_(x >= 1, x < 2))), simplify_boolean(or_(x >= 1, Eq(x, 0)))
    (x < 2, Eq(x, 0) | (x >= 1))
    """
    if relation is None or measure is None:
        from sympy_extras.assumptions.refine import _measure, simplify_relational

        def default_relation(r: Relational) -> Boolean:
            return simplify_relational(r, ratio, False, False, True)
        relation = default_relation if relation is None else relation
        measure = _measure if measure is None else measure
    return _simplify(formula, _Options(relation, measure, ratio, sympy_intervals))


def _simplify(b: Boolean, options: _Options) -> Boolean:
    """``simplify(b)``: the ``_eval_simplify`` of its class."""
    if isinstance(b, Relational):
        return options.relation(b)
    if isinstance(b, And):
        return _simplify_and(b, options)
    if isinstance(b, Or):
        return _simplify_or(b, options)
    if isinstance(b, BooleanFunction):
        return _simplify_function(b, options)
    return b


# PORTED FROM SYMPY: sympy/logic/boolalg.py:BooleanFunction._eval_simplify
def _simplify_function(b: BooleanFunction, options: _Options) -> Boolean:
    """The univariate intervals, the arguments simplified and the formula
    minimised by ``simplify_logic`` (with its relations simplified once
    more, ``deep=True``)."""
    from sympy_extras.logic.boolalg import simplify_logic
    rv = simplify_univariate(b, options.sympy_intervals)
    if not isinstance(rv, BooleanFunction):
        return _simplify(rv, options)
    rv = to_boolean(rv._evaluated(tuple(_simplify(a, options) for a in rv.args)))
    relations = [r for r in ordered(rv.atoms(Relational)) if isinstance(r, Relational)]
    if relations:
        rule: dict[Basic, Basic] = {r: options.relation(r) for r in relations}
        rv = to_boolean(rv._xreplace_evaluated(rule))
    return simplify_logic(rv)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:And._eval_simplify
def _simplify_and(b: And, options: _Options) -> Boolean:
    """The equations in one symbol solved and substituted into the other
    relations, then the patterns of ``And``."""
    rv = _simplify_function(b, options)
    if not isinstance(rv, And):
        return rv
    relations = [a for a in rv.args if isinstance(a, Relational)]
    others = [a for a in rv.args if not isinstance(a, Relational)]
    if not relations:
        return rv
    eqs = [r for r in relations if isinstance(r, Eq)]
    other: list[Boolean] = [r for r in relations if not isinstance(r, Eq)]
    if eqs:
        rv = _substituted_equations(eqs, other, others, options)
    if not isinstance(rv, And):
        return rv
    return _apply_patterns(rv, _AND_PATTERNS, options.measure, false, _AND3_PATTERNS)


def _linear_solution(e: Eq, x: Symbol) -> Optional[Boolean]:
    """``Eq(x, -b/m)`` for an equation ``m*x + b = 0`` linear in ``x``
    (SymPy's ``linear_coeffs``), ``None`` when it is not linear."""
    from sympy_extras.core.add import add
    from sympy_extras.core.power import pow
    from sympy_extras.assumptions.refine import _lin_eq2dict, _NonlinearError
    try:
        constant, coefficients = _lin_eq2dict(add(e.lhs, neg(e.rhs)), {x})
    except _NonlinearError:
        return None
    m = coefficients.get(x, Integer(0))
    if m == Integer(0) or m.has(x) or constant.has(x):
        return None
    return eq(x, neg(constant)*pow(m, Integer(-1)))


def _substituted_equations(eqs: list[Eq], other: list[Boolean], others: list[Boolean],
                           options: _Options) -> Boolean:
    """The part of ``And._eval_simplify`` which solves the equations in
    one symbol and substitutes them into the other relations."""
    reps: dict[Basic, Basic] = {}
    # the equations grouped by the number of their symbols, each group in
    # SymPy's order of (free symbols, equation)
    sifted: dict[int, list[Boolean]] = {}
    for equation in sorted(eqs, key=lambda r: (sorted(s.name for s in r.free_symbols), ordered(eqs).index(r))):
        sifted.setdefault(len(equation.free_symbols), []).append(equation)
    found: list[Boolean] = []
    nonlinear: list[Boolean] = []
    while 1 in sifted:
        for e_ in sifted.pop(1):
            e: Boolean = e_
            x = next(iter(e.free_symbols))
            if isinstance(e, Eq) and (e.lhs != x or x in e.rhs.free_symbols) and x not in reps:
                solved = _linear_solution(e, x)
                if solved is not None:
                    if options.measure(solved) <= options.ratio*options.measure(e):
                        e = solved
                    else:
                        found.append(e)
                        continue
            if x in reps:
                found.append(to_boolean(e._xreplace_evaluated({x: reps[x]})))
            elif isinstance(e, Eq) and e.lhs == x and x not in e.rhs.free_symbols:
                reps[x] = e.rhs
                found.append(e)
            else:
                # x is not yet identified, but may be later
                nonlinear.append(e)
        resifted: dict[int, list[Boolean]] = {}
        for k in sifted:
            for old in sifted[k]:
                e2 = to_boolean(old._xreplace_evaluated(reps))
                resifted.setdefault(len(e2.free_symbols), []).append(e2)
        sifted = resifted
    for k in sifted:
        found.extend(sifted[k])
    nonlinear = [to_boolean(n._xreplace_evaluated(reps)) for n in nonlinear]
    other = [to_boolean(o._xreplace_evaluated(reps)) for o in other]
    return and_(*([_canonical(f) for f in found + nonlinear + other] + others))


def _canonical(b: Boolean) -> Boolean:
    """The formula with its relations in canonical form (SymPy's
    ``_canonical``; a relation itself by its ``canonical``)."""
    if isinstance(b, Relational):
        return b.canonical
    relations = b.atoms(Relational)
    if not relations:
        return b
    rule: dict[Basic, Basic] = {r: r.canonical for r in relations if isinstance(r, Relational)}
    return to_boolean(b._xreplace_evaluated(rule))


# PORTED FROM SYMPY: sympy/logic/boolalg.py:Or._eval_simplify
def _simplify_or(b: Or, options: _Options) -> Boolean:
    """``a <= b`` written as ``Eq(a, b) | (a < b)`` first, then the
    patterns of ``Or``."""
    lege = [r for r in b.atoms(Le, Ge) if isinstance(r, (Le, Ge))]
    if lege:
        rule: dict[Basic, Basic] = {r: or_(eq(r.lhs, r.rhs), _strict(r)) for r in lege}
        return _simplify(to_boolean(b._xreplace_evaluated(rule)), options)
    rv = _simplify_function(b, options)
    if not isinstance(rv, Or):
        return rv
    return _apply_patterns(rv, _OR_PATTERNS, options.measure, true, ())


def _strict(r: Relational) -> Boolean:
    if isinstance(r, Le):
        return lt(r.lhs, r.rhs)
    if isinstance(r, Ge):
        return gt(r.lhs, r.rhs)
    return r


# ---------------------------------------------------------------------------
# patterns

#: a relation of a pattern: its class and the names of its sides
#: (``'a'``, ``'b'``, ``'c'`` or ``'-b'``)
RelationPattern = tuple[type[Relational], str, str]


class _Pattern:
    """Relations with named sides and their replacement, built from the
    bound expressions with the evaluating functions."""

    def __init__(self, relations: Sequence[RelationPattern], replacement: Callable[[Bindings], Boolean]) -> None:
        self.relations: Final[tuple[RelationPattern, ...]] = tuple(relations)
        self.replacement: Final[Callable[[Bindings], Boolean]] = replacement


def _match(pattern: _Pattern, relations: Sequence[Relational]) -> Optional[Bindings]:
    """The bindings under which the relations are the pattern, matched
    from left to right (a negated name only once it is bound, as SymPy's
    ``Wild`` matching of ``-b``)."""
    bound: Bindings = {}
    for (cls, left, right), r in zip(pattern.relations, relations):
        if type(r) is not cls:
            return None
        for spec, value in ((left, r.lhs), (right, r.rhs)):
            if spec.startswith('-'):
                name = spec[1:]
                if name not in bound or neg(bound[name]) != value:
                    return None
            elif spec in bound:
                if bound[spec] != value:
                    return None
            else:
                bound[spec] = value
    return bound


def _apply_patterns(rv: Boolean, patterns: Sequence[_Pattern], measure: Measure, dominating: Boolean,
                    threeterm: Sequence[_Pattern]) -> Boolean:
    """SymPy's ``_apply_patternbased_simplification``: the relations of
    ``rv`` (an ``And`` or an ``Or``) in canonical form, triples and pairs
    replaced by the patterns while it shortens the formula."""
    connective = And if isinstance(rv, And) else Or
    relations = [a for a in rv.args if isinstance(a, Relational)]
    others = [to_boolean(a) for a in rv.args if not isinstance(a, Relational)]
    if len(relations) <= 1:
        return rv
    real = [r for r in relations if not any(s.is_real is False for s in r.free_symbols)]
    nonreal: list[Boolean] = [r for r in relations if any(s.is_real is False for s in r.free_symbols)]
    rels = [r.canonical for r in real]
    combine = and_ if connective is And else or_
    if threeterm and len(rels) >= 3:
        found = _threeterm(rels, threeterm, combine, connective, dominating, measure)
        if found is None:
            return dominating
        rels = found
    found = _twoterm(rels, patterns, combine, connective, dominating, measure)
    if found is None:
        return dominating
    ordered_rels = [r for r in ordered(found) if isinstance(r, Relational)]
    return combine(*([_canonical(r) for r in ordered_rels] + others + nonreal))


Combine = Callable[..., Boolean]


def _replace(rels: list[Relational], results: list[tuple[int, list[int], Boolean]], connective: type[Basic],
             dominating: Boolean) -> list[Relational]:
    """The relations with the best replacement made (the first of the
    largest saving)."""
    best = sorted(results, key=lambda r: -r[0])[0]
    _, indices, new = best
    for index in sorted(indices, reverse=True):
        del rels[index]
    if new != not_(dominating):
        parts = [to_boolean(a) for a in new.args] if isinstance(new, connective) else [new]
        rels.extend(p for p in parts if isinstance(p, Relational))
    return rels


def _usable(new: Boolean) -> bool:
    """A replacement is used when its conditions are decided: no ``ITE``,
    ``Min`` or ``Max`` left, and only relations (SymPy's loop would fail
    on anything else)."""
    if isinstance(new, ITE) or new.has(Min, Max):
        return False
    parts = new.args if isinstance(new, (And, Or)) else (new,)
    return all(isinstance(p, (Relational, BooleanAtom)) for p in parts)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_apply_patternbased_twoterm_simplification
def _twoterm(rels_: list[Relational], patterns: Sequence[_Pattern], combine: Combine, connective: type[Basic],
             dominating: Boolean, measure: Measure) -> Optional[list[Relational]]:
    """``None`` when a pair gives the dominating value."""
    rels = list(rels_)
    changed = True
    while changed and len(rels) >= 2:
        changed = False
        # use only < or <=
        rels = [r.reversed if isinstance(r, (Ge, Gt)) else r for r in rels]
        rels = [r for r in ordered(rels) if isinstance(r, Relational)]
        # Eq and Ne are tested reversed as well
        candidates = [(r,) if not isinstance(r, (Eq, Ne)) else (r, r.reversed) for r in rels]
        results: list[tuple[int, list[int], Boolean]] = []
        for (i, pi), (j, pj) in combinations(list(enumerate(candidates)), 2):
            for pattern in patterns:
                for p1, p2 in product(pi, pj):
                    bound = _match(pattern, (p1, p2))
                    if bound is None:
                        continue
                    new = pattern.replacement(bound)
                    if new == dominating:
                        return None
                    if _usable(new):
                        saving = measure(combine(p1, p2)) - measure(new)
                        if saving > 0:
                            results.append((saving, [i, j], new))
        if results:
            rels = _replace(rels, results, connective, dominating)
            changed = True
    return rels


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_apply_patternbased_threeterm_simplification
def _threeterm(rels_: list[Relational], patterns: Sequence[_Pattern], combine: Combine, connective: type[Basic],
               dominating: Boolean, measure: Measure) -> Optional[list[Relational]]:
    """``None`` when a triple gives the dominating value."""
    rels = list(rels_)
    changed = True
    while changed and len(rels) >= 3:
        changed = False
        # use only > or >=
        rels = [r.reversed if isinstance(r, (Le, Lt)) else r for r in rels]
        rels = [r for r in ordered(rels) if isinstance(r, Relational)]
        results: list[tuple[int, list[int], Boolean]] = []
        candidates = [(r,) if not isinstance(r, (Eq, Ne)) else (r, r.reversed) for r in rels]
        for (i, pi), (j, pj), (k, pk) in permutations(list(enumerate(candidates)), 3):
            for pattern in patterns:
                for p1, p2, p3 in product(pi, pj, pk):
                    bound = _match(pattern, (p1, p2, p3))
                    if bound is None:
                        continue
                    new = pattern.replacement(bound)
                    if new == dominating:
                        return None
                    if _usable(new):
                        saving = measure(combine(p1, p2, p3)) - measure(new)
                        if saving > 0:
                            results.append((saving, [i, j, k], new))
        if results:
            rels = _replace(rels, results, connective, dominating)
            changed = True
    return rels


def _p(relations: Sequence[RelationPattern], replacement: Callable[[Bindings], Boolean]) -> _Pattern:
    return _Pattern(relations, replacement)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_simplify_patterns_and
_AND_PATTERNS: Final[tuple[_Pattern, ...]] = (
    _p(((Eq, 'a', 'b'), (Lt, 'a', 'b')), lambda w: false),
    _p(((Lt, 'b', 'a'), (Lt, 'a', 'b')), lambda w: false),
    _p(((Eq, 'a', 'b'), (Le, 'b', 'a')), lambda w: eq(w['a'], w['b'])),
    _p(((Le, 'b', 'a'), (Le, 'a', 'b')), lambda w: eq(w['a'], w['b'])),
    _p(((Le, 'a', 'b'), (Lt, 'a', 'b')), lambda w: lt(w['a'], w['b'])),
    _p(((Le, 'a', 'b'), (Ne, 'a', 'b')), lambda w: lt(w['a'], w['b'])),
    _p(((Lt, 'a', 'b'), (Ne, 'a', 'b')), lambda w: lt(w['a'], w['b'])),
    # sign
    _p(((Eq, 'a', 'b'), (Eq, 'a', '-b')), lambda w: and_(eq(w['a'], Integer(0)), eq(w['b'], Integer(0)))),
    # Min/Max/ITE
    _p(((Le, 'b', 'a'), (Le, 'c', 'a')), lambda w: ge(w['a'], max_(w['b'], w['c']))),
    _p(((Le, 'b', 'a'), (Lt, 'c', 'a')), lambda w: ite(gt(w['b'], w['c']), ge(w['a'], w['b']), gt(w['a'], w['c']))),
    _p(((Lt, 'b', 'a'), (Lt, 'c', 'a')), lambda w: gt(w['a'], max_(w['b'], w['c']))),
    _p(((Le, 'a', 'b'), (Le, 'a', 'c')), lambda w: le(w['a'], min_(w['b'], w['c']))),
    _p(((Le, 'a', 'b'), (Lt, 'a', 'c')), lambda w: ite(lt(w['b'], w['c']), le(w['a'], w['b']), lt(w['a'], w['c']))),
    _p(((Lt, 'a', 'b'), (Lt, 'a', 'c')), lambda w: lt(w['a'], min_(w['b'], w['c']))),
    _p(((Le, 'a', 'b'), (Le, 'c', 'a')),
       lambda w: ite(eq(w['b'], w['c']), eq(w['a'], w['b']),
                     ite(lt(w['b'], w['c']), false, and_(le(w['a'], w['b']), ge(w['a'], w['c']))))),
    _p(((Le, 'c', 'a'), (Le, 'a', 'b')),
       lambda w: ite(eq(w['b'], w['c']), eq(w['a'], w['b']),
                     ite(lt(w['b'], w['c']), false, and_(le(w['a'], w['b']), ge(w['a'], w['c']))))),
    _p(((Lt, 'a', 'b'), (Lt, 'c', 'a')),
       lambda w: ite(lt(w['b'], w['c']), false, and_(lt(w['a'], w['b']), gt(w['a'], w['c'])))),
    _p(((Lt, 'c', 'a'), (Lt, 'a', 'b')),
       lambda w: ite(lt(w['b'], w['c']), false, and_(lt(w['a'], w['b']), gt(w['a'], w['c'])))),
    _p(((Le, 'a', 'b'), (Lt, 'c', 'a')),
       lambda w: ite(le(w['b'], w['c']), false, and_(le(w['a'], w['b']), gt(w['a'], w['c'])))),
    _p(((Le, 'c', 'a'), (Lt, 'a', 'b')),
       lambda w: ite(le(w['b'], w['c']), false, and_(lt(w['a'], w['b']), ge(w['a'], w['c'])))),
    _p(((Eq, 'a', 'b'), (Eq, 'a', 'c')), lambda w: ite(eq(w['b'], w['c']), eq(w['a'], w['b']), false)),
    _p(((Lt, 'a', 'b'), (Lt, '-b', 'a')),
       lambda w: ite(gt(w['b'], Integer(0)), lt(abs_(w['a']), w['b']), false)),
    _p(((Le, 'a', 'b'), (Le, '-b', 'a')),
       lambda w: ite(ge(w['b'], Integer(0)), le(abs_(w['a']), w['b']), false)),
)

# PORTED FROM SYMPY: sympy/logic/boolalg.py:_simplify_patterns_and3
_AND3_PATTERNS: Final[tuple[_Pattern, ...]] = (
    _p(((Ge, 'a', 'b'), (Ge, 'b', 'c'), (Gt, 'c', 'a')), lambda w: false),
    _p(((Ge, 'a', 'b'), (Gt, 'b', 'c'), (Gt, 'c', 'a')), lambda w: false),
    _p(((Gt, 'a', 'b'), (Gt, 'b', 'c'), (Gt, 'c', 'a')), lambda w: false),
    # lower bounds
    _p(((Ge, 'a', 'b'), (Ge, 'a', 'c'), (Ge, 'b', 'c')), lambda w: and_(ge(w['a'], w['b']), ge(w['b'], w['c']))),
    _p(((Ge, 'a', 'b'), (Ge, 'a', 'c'), (Gt, 'b', 'c')), lambda w: and_(ge(w['a'], w['b']), gt(w['b'], w['c']))),
    _p(((Ge, 'a', 'b'), (Gt, 'a', 'c'), (Gt, 'b', 'c')), lambda w: and_(ge(w['a'], w['b']), gt(w['b'], w['c']))),
    _p(((Ge, 'a', 'c'), (Gt, 'a', 'b'), (Gt, 'b', 'c')), lambda w: and_(gt(w['a'], w['b']), gt(w['b'], w['c']))),
    _p(((Ge, 'b', 'c'), (Gt, 'a', 'b'), (Gt, 'a', 'c')), lambda w: and_(gt(w['a'], w['b']), ge(w['b'], w['c']))),
    _p(((Gt, 'a', 'b'), (Gt, 'a', 'c'), (Gt, 'b', 'c')), lambda w: and_(gt(w['a'], w['b']), gt(w['b'], w['c']))),
    # upper bounds
    _p(((Ge, 'b', 'a'), (Ge, 'c', 'a'), (Ge, 'b', 'c')), lambda w: and_(ge(w['c'], w['a']), ge(w['b'], w['c']))),
    _p(((Ge, 'b', 'a'), (Ge, 'c', 'a'), (Gt, 'b', 'c')), lambda w: and_(ge(w['c'], w['a']), gt(w['b'], w['c']))),
    _p(((Ge, 'b', 'a'), (Gt, 'c', 'a'), (Gt, 'b', 'c')), lambda w: and_(gt(w['c'], w['a']), gt(w['b'], w['c']))),
    _p(((Ge, 'c', 'a'), (Gt, 'b', 'a'), (Gt, 'b', 'c')), lambda w: and_(ge(w['c'], w['a']), gt(w['b'], w['c']))),
    _p(((Ge, 'b', 'c'), (Gt, 'b', 'a'), (Gt, 'c', 'a')), lambda w: and_(gt(w['c'], w['a']), ge(w['b'], w['c']))),
    _p(((Gt, 'b', 'a'), (Gt, 'c', 'a'), (Gt, 'b', 'c')), lambda w: and_(gt(w['c'], w['a']), gt(w['b'], w['c']))),
    # circular
    _p(((Ge, 'a', 'b'), (Ge, 'b', 'c'), (Ge, 'c', 'a')), lambda w: and_(eq(w['a'], w['b']), eq(w['b'], w['c']))),
)

# PORTED FROM SYMPY: sympy/logic/boolalg.py:_simplify_patterns_or
_OR_PATTERNS: Final[tuple[_Pattern, ...]] = (
    _p(((Le, 'b', 'a'), (Le, 'a', 'b')), lambda w: true),
    _p(((Le, 'b', 'a'), (Ne, 'a', 'b')), lambda w: true),
    _p(((Eq, 'a', 'b'), (Le, 'a', 'b')), lambda w: le(w['a'], w['b'])),
    _p(((Eq, 'a', 'b'), (Lt, 'a', 'b')), lambda w: le(w['a'], w['b'])),
    _p(((Lt, 'b', 'a'), (Lt, 'a', 'b')), lambda w: ne(w['a'], w['b'])),
    _p(((Lt, 'b', 'a'), (Ne, 'a', 'b')), lambda w: ne(w['a'], w['b'])),
    _p(((Le, 'a', 'b'), (Lt, 'a', 'b')), lambda w: le(w['a'], w['b'])),
    _p(((Eq, 'a', 'b'), (Ne, 'a', 'c')), lambda w: ite(eq(w['b'], w['c']), true, ne(w['a'], w['c']))),
    _p(((Ne, 'a', 'b'), (Ne, 'a', 'c')), lambda w: ite(eq(w['b'], w['c']), ne(w['a'], w['b']), true)),
    # Min/Max/ITE
    _p(((Le, 'b', 'a'), (Le, 'c', 'a')), lambda w: ge(w['a'], min_(w['b'], w['c']))),
    _p(((Le, 'b', 'a'), (Lt, 'c', 'a')), lambda w: ite(gt(w['b'], w['c']), lt(w['c'], w['a']), le(w['b'], w['a']))),
    _p(((Lt, 'b', 'a'), (Lt, 'c', 'a')), lambda w: gt(w['a'], min_(w['b'], w['c']))),
    _p(((Le, 'a', 'b'), (Le, 'a', 'c')), lambda w: le(w['a'], max_(w['b'], w['c']))),
    _p(((Le, 'a', 'b'), (Lt, 'a', 'c')), lambda w: ite(ge(w['b'], w['c']), le(w['a'], w['b']), lt(w['a'], w['c']))),
    _p(((Lt, 'a', 'b'), (Lt, 'a', 'c')), lambda w: lt(w['a'], max_(w['b'], w['c']))),
    _p(((Le, 'a', 'b'), (Le, 'c', 'a')),
       lambda w: ite(ge(w['b'], w['c']), true, or_(le(w['a'], w['b']), ge(w['a'], w['c'])))),
    _p(((Le, 'c', 'a'), (Le, 'a', 'b')),
       lambda w: ite(ge(w['b'], w['c']), true, or_(le(w['a'], w['b']), ge(w['a'], w['c'])))),
    _p(((Lt, 'a', 'b'), (Lt, 'c', 'a')),
       lambda w: ite(gt(w['b'], w['c']), true, or_(lt(w['a'], w['b']), gt(w['a'], w['c'])))),
    _p(((Lt, 'c', 'a'), (Lt, 'a', 'b')),
       lambda w: ite(gt(w['b'], w['c']), true, or_(lt(w['a'], w['b']), gt(w['a'], w['c'])))),
    _p(((Le, 'a', 'b'), (Lt, 'c', 'a')),
       lambda w: ite(ge(w['b'], w['c']), true, or_(le(w['a'], w['b']), gt(w['a'], w['c'])))),
    _p(((Le, 'c', 'a'), (Lt, 'a', 'b')),
       lambda w: ite(ge(w['b'], w['c']), true, or_(lt(w['a'], w['b']), ge(w['a'], w['c'])))),
    _p(((Lt, 'b', 'a'), (Lt, 'a', '-b')),
       lambda w: ite(ge(w['b'], Integer(0)), gt(abs_(w['a']), w['b']), true)),
    _p(((Le, 'b', 'a'), (Le, 'a', '-b')),
       lambda w: ite(gt(w['b'], Integer(0)), ge(abs_(w['a']), w['b']), true)),
)


# ---------------------------------------------------------------------------
# univariate formulas

# PORTED FROM SYMPY: sympy/logic/boolalg.py:simplify_univariate
def simplify_univariate(expr: Boolean, sympy_intervals: bool = False) -> Boolean:
    """A formula in one symbol without equations as the union of the
    intervals where it holds, in SymPy's form (``x < b``,
    ``(a <= x) & (x < b)``, ...); the formula itself when a relation
    cannot be solved for the symbol or it holds at an isolated point.

    The intervals are maximal and computed natively (the linear relations
    and the polynomial ones whose real roots are found). With
    ``sympy_intervals`` they are SymPy's, those of ``Piecewise._intervals``
    (:func:`~sympy_extras.functions.elementary.piecewise.piecewise_intervals`,
    one or more per term of the disjunctive form), for a caller whose
    output must be SymPy's (``piecewise_simplify_arguments``); in both the
    endpoints are decided on the formula (SymPy decides them on the
    relation of the previous interval).

    Examples
    ========

    >>> from sympy_extras.core import symbols, and_, or_
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.logic.boolsimp import simplify_univariate
    >>> x = symbols('x')
    >>> simplify_univariate(and_(x > 1, x < 3)), simplify_univariate(or_(x**2 > 4, x < 0))
    ((1 < x) & (x < 3), (x > 2) | (x < 0))
    >>> simplify_univariate(and_(x > 1, x < 0)), simplify_univariate(and_(sin(x) > 0, x < 0))
    (False, (x < 0) & (sin(x) > 0))
    >>> simplify_univariate(or_(x <= -3, x < 1)), simplify_univariate(or_(x <= -3, x < 1), True)
    (x < 1, (x <= -3) | (x < 1))
    """
    if not isinstance(expr, BooleanFunction):
        return expr
    if expr.atoms(Eq, Ne):
        return expr
    free = expr.free_symbols
    if len(free) != 1:
        return expr
    x = next(iter(free))
    truth = _truth_intervals(expr, x)
    if truth is not None and truth.isolated:
        # true at an isolated point: SymPy's intervals stop at an equation
        # (its Piecewise._intervals drops the point: simplify of
        # (x**2 <= 0) & (x < 2) is False there)
        return expr
    if sympy_intervals and not expr.has(Xor):
        # (SymPy's Piecewise._intervals does not take an Xor: such a
        # formula takes the native intervals)
        return _sympy_univariate(expr, x)
    if truth is None:
        return expr
    found = truth.intervals
    if not found:
        return false
    args: list[Boolean] = []
    for a, b, incl_a, incl_b in found:
        c: Boolean
        if isinstance(a, NegativeInfinity):
            if b == oo:
                c = true
            else:
                c = _relation(Le, x, b) if incl_b else _relation(Lt, x, b)
        elif b == oo:
            c = _relation(Ge, x, a) if incl_a else _relation(Gt, x, a)
        else:
            lower = _relation(Le, a, x) if incl_a else _relation(Lt, a, x)
            upper = _relation(Le, x, b) if incl_b else _relation(Lt, x, b)
            c = and_(lower, upper)
        args.append(c)
    return or_(*args)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:simplify_univariate
def _sympy_univariate(expr: BooleanFunction, x: Symbol) -> Boolean:
    """SymPy's ``simplify_univariate`` on the intervals of
    ``Piecewise._intervals``, the endpoints included as the formula says."""
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_intervals
    from sympy_extras.core.numbers import Infinity
    ok, pieces = piecewise_intervals(Piecewise((Integer(0), expr)), x, err_on_Eq=True)
    if not ok or isinstance(pieces, str):
        return expr
    if not pieces:
        return false

    def holds(at: Expr) -> bool:
        return expr.subs(x, at) is true
    args: list[Boolean] = []
    for a, b, _, _ in pieces:
        c: Boolean
        if isinstance(a, NegativeInfinity):
            if isinstance(b, Infinity):
                c = true
            else:
                c = (x <= b) if holds(b) else (x < b)
        else:
            incl_a = holds(a)
            incl_b = holds(b)
            if incl_a and incl_b:
                c = (x >= a) if b.is_infinite is True else and_(a <= x, x <= b)
            elif incl_a:
                c = and_(a <= x, x < b)
            elif incl_b:
                c = (x > a) if b.is_infinite is True else and_(a < x, x <= b)
            else:
                c = and_(a < x, x < b)
        args.append(c)
    return or_(*args)


def _relation(cls: type[Relational], lhs: Expr, rhs: Expr) -> Boolean:
    """``lhs < rhs`` (or the class given) as SymPy's operators build it:
    decided when the core decides it, else with the sides as given
    (``1 < x``, which the evaluating ``lt`` writes ``x > 1``)."""
    decided = to_boolean(cls(lhs, rhs).doit())
    return decided if isinstance(decided, BooleanAtom) else cls(lhs, rhs)


#: an interval of truth: its ends and whether they are included
Interval = tuple[Expr, Expr, bool, bool]


class _Solved:
    """A relation in one symbol through the signs of its difference
    ``p = lhs - rhs``: its real roots and a way to decide the sign of
    ``p`` at a point which is not a root."""

    def __init__(self, relation: Relational, x: Symbol, p: Expr, roots: list[Expr]) -> None:
        self.relation: Final[Relational] = relation
        self.x: Final[Symbol] = x
        self.p: Final[Expr] = p
        self.roots: Final[list[Expr]] = roots

    def holds(self, sign: int) -> bool:
        """The truth of ``p op 0`` for the sign of ``p``."""
        r = self.relation
        if isinstance(r, Lt):
            return sign < 0
        if isinstance(r, Le):
            return sign <= 0
        if isinstance(r, Gt):
            return sign > 0
        return sign >= 0


def _solved(r: Relational, x: Symbol) -> Optional[_Solved]:
    """The roots of the difference of a relation which is linear in
    ``x`` with a real numerical coefficient, or polynomial in ``x`` with
    rational coefficients and all its real roots found."""
    from sympy_extras.core.add import add
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.polytools import Poly
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.assumptions.refine import _lin_eq2dict, _NonlinearError
    if not isinstance(r, (Lt, Le, Gt, Ge)):
        return None
    p = add(r.lhs, neg(r.rhs))
    try:
        constant, coefficients = _lin_eq2dict(p, {x})
        m = coefficients.get(x, Integer(0))
        if m.is_number and m.is_extended_real and m != Integer(0) and constant.is_number \
                and constant.is_extended_real is True:
            return _Solved(r, x, p, [neg(constant)/m])
    except _NonlinearError:
        pass
    try:
        poly = Poly(p, x)
    except (BasePolynomialError, ValueError, TypeError, NotImplementedError):
        return None
    if poly.degree() < 1 or not all(isinstance(c, Rational) for c in poly.all_coeffs()):
        return None
    found = roots(poly)
    if not isinstance(found, dict) or sum(found.values()) != poly.degree():
        return None
    real: list[Expr] = []
    for root in found:
        if root.is_extended_real is True:
            real.append(root)
        elif root.is_extended_real is not False:
            return None
    return _Solved(r, x, p, real)


def _approximation(e: Expr) -> Optional[Fraction]:
    """A rational approximation of a real number to 30 digits, ``None``
    when it cannot be evaluated reliably."""
    from sympy_extras._numeric import reliable_value
    value = reliable_value(e, 30)
    if isinstance(value, Rational):
        return Fraction(value.p, value.q)
    if isinstance(value, Float):
        return value.as_fraction()
    return None


class _Truth:
    """The maximal intervals where a formula holds, or that it holds at an
    isolated point (``isolated``, the intervals then not computed)."""

    def __init__(self, intervals: list[Interval], isolated: bool) -> None:
        self.intervals: Final[list[Interval]] = intervals
        self.isolated: Final[bool] = isolated


def _truth_intervals(expr: Boolean, x: Symbol) -> Optional[_Truth]:
    """The maximal intervals where ``expr`` holds; ``None`` when a
    relation is not solved or the breakpoints are not ordered."""
    relations = [r for r in ordered(expr.atoms(Relational)) if isinstance(r, Relational)]
    solved: list[_Solved] = []
    for r in relations:
        if x not in r.free_symbols:
            return None
        s = _solved(r, x)
        if s is None:
            return None
        solved.append(s)
    # the breakpoints, in increasing order, with their approximations
    points: list[tuple[Fraction, Expr]] = []
    for s in solved:
        for root in s.roots:
            if any(root == p for _, p in points):
                continue
            approximation = _approximation(root)
            if approximation is None:
                return None
            points.append((approximation, root))
    points.sort(key=lambda p: p[0])
    for (low, _), (high, _) in zip(points, points[1:]):
        if high - low < Fraction(1, 10**20):
            # two forms of one number, or too close to be ordered here
            return None

    def sign_at(s: _Solved, k: int) -> Optional[int]:
        """The sign of the difference at position ``k``: ``2*i + 1`` the
        breakpoint ``i``, even positions between them."""
        from sympy_extras._numeric import reliable_value
        if k % 2 == 1:
            point = points[k//2][1]
            if any(point == root for root in s.roots):
                return 0
        else:
            i = k//2
            if not points:
                sample = Fraction(0)
            elif i == 0:
                sample = points[0][0] - 1
            elif i == len(points):
                sample = points[-1][0] + 1
            else:
                sample = (points[i - 1][0] + points[i][0])/2
            point = Rational(sample.numerator, sample.denominator)
        value = reliable_value(as_expr(s.p._xreplace_evaluated({x: point})), 15)
        if value is None or value.is_zero:
            return None
        return 1 if value.is_positive else -1

    truth: list[bool] = []
    for k in range(2*len(points) + 1):
        values: dict[Basic, Basic] = {}
        for s in solved:
            sign = sign_at(s, k)
            if sign is None:
                return None
            values[s.relation] = true if s.holds(sign) else false
        decided = expr._xreplace_evaluated(values)
        if not isinstance(decided, BooleanAtom):
            return None
        truth.append(bool(decided))
    intervals: list[Interval] = []
    k = 0
    while k < len(truth):
        if not truth[k]:
            k += 1
            continue
        start = k
        while k + 1 < len(truth) and truth[k + 1]:
            k += 1
        end = k
        if start == end and start % 2 == 1:
            return _Truth([], True)
        a: Expr = -oo if start == 0 else points[(start - 1)//2][1]
        b: Expr = oo if end == len(truth) - 1 else points[end//2][1]
        intervals.append((a, b, start % 2 == 1, end % 2 == 1))
        k += 1
    return _Truth(intervals, False)
