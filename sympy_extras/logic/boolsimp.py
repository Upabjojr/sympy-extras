"""SymPy's ``simplify`` of a Boolean formula of the core:
:func:`simplify_boolean`.

A typed port of the simplification of SymPy's ``sympy.logic.boolalg``
(``BooleanFunction._eval_simplify``, ``And._eval_simplify``,
``Or._eval_simplify``, ``simplify_univariate`` and the pattern-based
simplification of the relations of ``And`` and ``Or``): a formula in one
real variable without equations is first rewritten as the union of the
intervals where it holds; the arguments are simplified (the relations by
the port of ``Relational._eval_simplify``,
:func:`~sympy_extras.assumptions.refine.simplify_relational`), the
formula minimised by :func:`~sympy_extras.logic.boolalg.simplify_logic`;
in a conjunction the equations of one symbol are solved and substituted
into the other relations, and pairs and triples of relations of a
conjunction or a disjunction are replaced by shorter ones (``x < y`` and
``Ne(x, y)`` by ``x < y``, ``x <= b`` and ``x <= c`` by
``x <= Min(b, c)`` when it evaluates to a relation, ...), the largest
saving first, as long as one applies.

Differences from SymPy: the inclusion of the endpoints of the intervals
is decided on the formula itself (SymPy tests them on the relation built
for the previous interval, so that ``simplify(~((x > 0) & (x < 1)))`` is
``(x >= 1) | (x < 0)`` there, false at ``x = 0``); a formula with an
``Xor`` raises ``NotImplementedError`` (SymPy's ``Xor._eval_simplify``
goes through the algebraic normal form, not ported); a Boolean which is
not a connective or a relation (``Contains``) is left as it is, as SymPy
does.

Examples
========

>>> from sympy_extras.core import Symbol, Ne, Eq
>>> from sympy_extras.logic.boolsimp import simplify_boolean
>>> x, y = Symbol('x', real=True), Symbol('y', real=True)
>>> simplify_boolean((x < y) & Ne(x, y)), simplify_boolean((x <= 1) & (x <= 2))
(x < y, x <= 1)
>>> simplify_boolean(~((x > 0) & (x < 1))), simplify_boolean(Eq(x, 1) & (x + y > 2))
((x >= 1) | (x <= 0), Eq(x, 1) & (y + 1 > 2))
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
# Changes: typed; the measure is count_ops; the endpoints of
# simplify_univariate tested on the formula; no Xor.
# ---------------------------------------------------------------------
from __future__ import annotations

from itertools import combinations, permutations, product
from typing import Optional, Sequence

from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanAtom, BooleanFunction, Equivalent, Implies, Not, Or,
                                     Xor, and_, false, not_, or_, to_boolean, true)
from sympy_extras.core.numbers import Infinity, Integer, NegativeInfinity
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Symbol, Wild

__all__ = ['simplify_boolean', 'simplify_univariate']

#: a pattern of two or three relations and its replacement
Pattern = tuple[Tuple, Boolean]

_RATIO = 1.7


def _measure(b: Basic) -> int:
    """SymPy's default measure, ``count_ops``."""
    from sympy_extras.core.function import count_ops
    value = count_ops(b)
    return value.p if isinstance(value, Integer) else 0


def simplify_boolean(expr: Boolean, ratio: float = _RATIO) -> Boolean:
    """SymPy's ``simplify(expr)`` of a Boolean formula (see the module).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.logic.boolsimp import simplify_boolean
    >>> x = Symbol('x', real=True)
    >>> simplify_boolean((x > 1) | (x > 2) | (x < 0))
    (x > 1) | (x < 0)
    """
    if isinstance(expr, BooleanAtom):
        return expr
    if isinstance(expr, Relational):
        from sympy_extras.assumptions.refine import simplify_relational
        return simplify_relational(expr, ratio, False, False, True)
    if isinstance(expr, Xor):
        raise NotImplementedError("SymPy's Xor._eval_simplify (the algebraic normal form) is not ported")
    if isinstance(expr, And):
        return _simplify_and(expr, ratio)
    if isinstance(expr, Or):
        return _simplify_or(expr, ratio)
    if isinstance(expr, (Not, Implies, Equivalent, ITE)):
        return _simplify_function(expr, ratio)
    return expr


# PORTED FROM SYMPY: sympy/logic/boolalg.py:BooleanFunction._eval_simplify
def _simplify_function(expr: BooleanFunction, ratio: float) -> Boolean:
    from .boolalg import simplify_logic
    rv = simplify_univariate(expr)
    if not isinstance(rv, BooleanFunction):
        return simplify_boolean(rv, ratio)
    args = tuple(simplify_boolean(to_boolean(a), ratio) for a in rv.args)
    return simplify_logic(_relations_simplified(to_boolean(rv._evaluated(args)), ratio))


# PORTED FROM SYMPY: sympy/logic/boolalg.py:simplify_logic (deep=True: the relations simplified first)
def _relations_simplified(expr: Boolean, ratio: float) -> Boolean:
    """Each relation of the formula simplified (SymPy's ``simplify_logic``
    with ``deep=True``, the default, before the minimisation; SymPy's
    ``simplify`` of a relation is not idempotent: ``x - y > -x`` is
    ``y < 2*x``, which is ``2*x > y``)."""
    from sympy_extras.assumptions.refine import simplify_relational
    relations = [r for r in expr.atoms(Relational) if isinstance(r, Relational)]
    if not relations:
        return expr
    reps: dict[Basic, Basic] = {r: simplify_relational(r, ratio, False, False, True) for r in relations}
    return to_boolean(expr._xreplace_evaluated(reps))


def _linear_in(e: Relational, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """SymPy's ``linear_coeffs(lhs - rhs, x)``, ``None`` (its
    ``ValueError``) when it is not linear."""
    from sympy_extras.assumptions.refine import _linear_coeffs, _NonlinearError
    try:
        m, b = _linear_coeffs(e.lhs - e.rhs, [x])
    except _NonlinearError:
        return None
    return m, b


# PORTED FROM SYMPY: sympy/logic/boolalg.py:And._eval_simplify
def _simplify_and(expr: And, ratio: float) -> Boolean:
    rv = _simplify_function(expr, ratio)
    if not isinstance(rv, And):
        return rv
    args = [to_boolean(a) for a in rv.args]
    rel = [a for a in args if isinstance(a, Relational)]
    non_rel = [a for a in args if not isinstance(a, Relational)]
    if not rel:
        return rv
    eqs_in = [r for r in rel if isinstance(r, Eq)]
    other: list[Boolean] = [r for r in rel if not isinstance(r, Eq)]
    if eqs_in:
        reps: dict[Basic, Basic] = {}
        # grouped by the number of their free symbols
        sifted: dict[int, list[tuple[set[Symbol], Relational]]] = {}
        for e in ordered(eqs_in):
            assert isinstance(e, Relational)
            sifted.setdefault(len(e.free_symbols), []).append((set(e.free_symbols), e))
        eqs: list[Boolean] = []
        nonlineqs: list[Boolean] = []
        while 1 in sifted:
            for free, e in sifted.pop(1):
                x = free.pop()
                if (e.lhs != x or x in e.rhs.free_symbols) and x not in reps:
                    coefficients = _linear_in(e, x)
                    if coefficients is not None:
                        m, b = coefficients
                        enew = to_boolean(e._evaluated((x, -b / m)))
                        if _measure(enew) <= ratio * _measure(e):
                            if not isinstance(enew, Relational):
                                eqs.append(enew)
                                continue
                            e = enew
                        else:
                            eqs.append(e)
                            continue
                if x in reps:
                    eqs.append(to_boolean(e.subs(x, reps[x])))
                elif e.lhs == x and x not in e.rhs.free_symbols:
                    reps[x] = e.rhs
                    eqs.append(e)
                else:
                    # x is not yet identified, but may be later
                    nonlineqs.append(e)
            resifted: dict[int, list[tuple[set[Symbol], Relational]]] = {}
            for group in sifted.values():
                for _, e in group:
                    replaced = to_boolean(e._xreplace_evaluated(reps))
                    if isinstance(replaced, Relational):
                        resifted.setdefault(len(replaced.free_symbols), []).append(
                            (set(replaced.free_symbols), replaced))
                    else:
                        eqs.append(replaced)
            sifted = resifted
        for group in sifted.values():
            eqs.extend(e for _, e in group)
        nonlineqs = [to_boolean(e.subs(reps)) for e in nonlineqs]
        other = [to_boolean(e.subs(reps)) for e in other]
        rebuilt = and_(*([_canonical_relation(e) for e in eqs + nonlineqs + other] + non_rel))
        if not isinstance(rebuilt, And):
            return rebuilt
        rv = rebuilt
    return _apply_patternbased_simplification(rv, _patterns_and(), false, _patterns_and3())


# PORTED FROM SYMPY: sympy/logic/boolalg.py:Or._eval_simplify
def _simplify_or(expr: Or, ratio: float) -> Boolean:
    lege = [a for a in expr.atoms(Le, Ge) if isinstance(a, (Le, Ge))]
    if lege:
        from sympy_extras.core.relational import eq, gt, lt
        reps: dict[Basic, Basic] = {r: or_(eq(r.lhs, r.rhs), lt(r.lhs, r.rhs) if isinstance(r, Le) else gt(r.lhs, r.rhs))
                                    for r in lege}
        replaced = to_boolean(expr._xreplace_evaluated(reps))
        return simplify_boolean(replaced, ratio)
    rv = _simplify_function(expr, ratio)
    if not isinstance(rv, Or):
        return rv
    return _apply_patternbased_simplification(rv, _patterns_or(), true, None)


def _canonical_relation(b: Boolean) -> Boolean:
    return b.canonical if isinstance(b, Relational) else b


# PORTED FROM SYMPY: sympy/core/relational.py:_canonical
def _canonical(cond: Boolean) -> Boolean:
    """The relations of the formula in canonical form."""
    reps: dict[Basic, Basic] = {r: r.canonical for r in cond.atoms(Relational) if isinstance(r, Relational)}
    return to_boolean(cond._xreplace_evaluated(reps)) if reps else cond


# PORTED FROM SYMPY: sympy/logic/boolalg.py:simplify_univariate
def simplify_univariate(expr: Boolean) -> Boolean:
    """A formula in one variable without equations as the union of the
    intervals where it holds, the formula itself otherwise (endpoints
    included as the formula says: SymPy decides them on the relation of
    the previous interval).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.logic.boolsimp import simplify_univariate
    >>> x = Symbol('x', real=True)
    >>> simplify_univariate((x >= 1) | (x <= 0))
    (x >= 1) | (x <= 0)
    """
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_intervals
    if not isinstance(expr, BooleanFunction):
        return expr
    if any(isinstance(a, (Eq, Ne)) for a in expr.atoms(Eq, Ne)):
        return expr
    free = list(expr.free_symbols)
    if len(free) != 1 or not isinstance(free[0], Symbol):
        return expr
    x = free[0]
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


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_apply_patternbased_simplification
def _apply_patternbased_simplification(rv: BooleanFunction, patterns: Sequence[Pattern], dominating: Boolean,
                                       threeterm: Optional[Sequence[Pattern]]) -> Boolean:
    """The relations of a conjunction (``dominating`` false) or a
    disjunction (``dominating`` true) replaced by the patterns."""
    args = [to_boolean(a) for a in rv.args]
    rel_all = [a for a in args if isinstance(a, Relational)]
    non_rel = [a for a in args if not isinstance(a, Relational)]
    if len(rel_all) <= 1:
        return rv
    rel: list[Boolean] = [r.canonical for r in rel_all if not any(s.is_real is False for s in r.free_symbols)]
    non_real_rel: list[Boolean] = [r for r in rel_all if any(s.is_real is False for s in r.free_symbols)]
    is_and = isinstance(rv, And)
    if threeterm is not None and len(rel) >= 3:
        rel = _patterns_applied(rel, threeterm, is_and, dominating, 3)
    rel = _patterns_applied(rel, patterns, is_and, dominating, 2)
    combined = [_canonical(r) for r in ordered(rel) if isinstance(r, Boolean)] + non_rel + non_real_rel
    return and_(*combined) if is_and else or_(*combined)


def _variants(r: Boolean) -> tuple[Boolean, ...]:
    """The relation, and reversed for an equation or an inequation (they
    are matched both ways)."""
    if isinstance(r, (Eq, Ne)):
        return (r, r.reversed)
    return (r,)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_apply_patternbased_twoterm_simplification
# PORTED FROM SYMPY: sympy/logic/boolalg.py:_apply_patternbased_threeterm_simplification
def _patterns_applied(rel: list[Boolean], patterns: Sequence[Pattern], is_and: bool, dominating: Boolean,
                      terms: int) -> list[Boolean]:
    """The two-term (``terms`` 2, relations made ``<`` or ``<=``) or
    three-term (``terms`` 3, made ``>`` or ``>=``) replacements, the
    largest saving of ``count_ops`` first, while one applies."""
    from sympy_extras.functions.elementary.miscellaneous import Max, Min

    def combine(parts: Sequence[Boolean]) -> Boolean:
        return and_(*parts) if is_and else or_(*parts)
    changed = True
    while changed and len(rel) >= terms:
        changed = False
        if terms == 2:
            rel = [r.reversed if isinstance(r, (Ge, Gt)) else r for r in rel]
        else:
            rel = [r.reversed if isinstance(r, (Le, Lt)) else r for r in rel]
        rel = [to_boolean(r) for r in ordered(rel)]
        variants = [_variants(r) for r in rel]
        results: list[tuple[int, list[int], Boolean]] = []
        chosen = combinations(enumerate(variants), 2) if terms == 2 else permutations(enumerate(variants), 3)
        for group in chosen:
            indices = [i for i, _ in group]
            for pattern, simp in patterns:
                matches: list[tuple[dict[Basic, Basic], Tuple]] = []
                for parts in product(*[v for _, v in group]):
                    oldexpr = Tuple(*parts)
                    found = oldexpr.match(pattern)
                    if found:
                        matches.append((dict(found.items()), oldexpr))
                for found, oldexpr in matches:
                    np = to_boolean(simp._xreplace_evaluated(found))
                    if np == dominating:
                        return [dominating]
                    if not isinstance(np, ITE) and not np.has(Max, Min):
                        saving = _measure(combine([to_boolean(a) for a in oldexpr.args])) - _measure(np)
                        if saving > 0:
                            results.append((saving, indices, np))
        if results:
            # the most saving first (a stable sort, as SymPy's)
            results.sort(key=lambda t: t[0], reverse=True)
            _, idx, newrel = results[0]
            for index in sorted(idx, reverse=True):
                del rel[index]
            if newrel != not_(dominating):
                if (is_and and isinstance(newrel, And)) or (not is_and and isinstance(newrel, Or)):
                    rel.extend(to_boolean(a) for a in newrel.args)
                else:
                    rel.append(newrel)
            changed = True
    return rel


_WILDS: list[Wild] = []


def _wilds() -> tuple[Wild, Wild, Wild]:
    if not _WILDS:
        _WILDS.extend([Wild('a'), Wild('b'), Wild('c')])
    return _WILDS[0], _WILDS[1], _WILDS[2]


_AND_PATTERNS: list[Pattern] = []
_AND3_PATTERNS: list[Pattern] = []
_OR_PATTERNS: list[Pattern] = []


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_simplify_patterns_and
def _patterns_and() -> list[Pattern]:
    """The two-term patterns of a conjunction (no ``Ge``, ``Gt``)."""
    from sympy_extras.functions.elementary.complexes import Abs
    from sympy_extras.functions.elementary.miscellaneous import Max, Min
    if _AND_PATTERNS:
        return _AND_PATTERNS
    a, b, c = _wilds()
    zero = Integer(0)
    _AND_PATTERNS.extend([
        (Tuple(Eq(a, b), Lt(a, b)), false),
        (Tuple(Lt(b, a), Lt(a, b)), false),
        (Tuple(Eq(a, b), Le(b, a)), Eq(a, b)),
        (Tuple(Le(b, a), Le(a, b)), Eq(a, b)),
        (Tuple(Le(a, b), Lt(a, b)), Lt(a, b)),
        (Tuple(Le(a, b), Ne(a, b)), Lt(a, b)),
        (Tuple(Lt(a, b), Ne(a, b)), Lt(a, b)),
        # sign
        (Tuple(Eq(a, b), Eq(a, -b)), And(Eq(a, zero), Eq(b, zero))),
        # Min/Max/ITE
        (Tuple(Le(b, a), Le(c, a)), Ge(a, Max(b, c))),
        (Tuple(Le(b, a), Lt(c, a)), ITE(Gt(b, c), Ge(a, b), Gt(a, c))),
        (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Max(b, c))),
        (Tuple(Le(a, b), Le(a, c)), Le(a, Min(b, c))),
        (Tuple(Le(a, b), Lt(a, c)), ITE(Lt(b, c), Le(a, b), Lt(a, c))),
        (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Min(b, c))),
        (Tuple(Le(a, b), Le(c, a)), ITE(Eq(b, c), Eq(a, b), ITE(Lt(b, c), false, And(Le(a, b), Ge(a, c))))),
        (Tuple(Le(c, a), Le(a, b)), ITE(Eq(b, c), Eq(a, b), ITE(Lt(b, c), false, And(Le(a, b), Ge(a, c))))),
        (Tuple(Lt(a, b), Lt(c, a)), ITE(Lt(b, c), false, And(Lt(a, b), Gt(a, c)))),
        (Tuple(Lt(c, a), Lt(a, b)), ITE(Lt(b, c), false, And(Lt(a, b), Gt(a, c)))),
        (Tuple(Le(a, b), Lt(c, a)), ITE(Le(b, c), false, And(Le(a, b), Gt(a, c)))),
        (Tuple(Le(c, a), Lt(a, b)), ITE(Le(b, c), false, And(Lt(a, b), Ge(a, c)))),
        (Tuple(Eq(a, b), Eq(a, c)), ITE(Eq(b, c), Eq(a, b), false)),
        (Tuple(Lt(a, b), Lt(-b, a)), ITE(Gt(b, zero), Lt(Abs(a), b), false)),
        (Tuple(Le(a, b), Le(-b, a)), ITE(Ge(b, zero), Le(Abs(a), b), false)),
    ])
    return _AND_PATTERNS


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_simplify_patterns_and3
def _patterns_and3() -> list[Pattern]:
    """The three-term patterns of a conjunction (no ``Le``, ``Lt``)."""
    if _AND3_PATTERNS:
        return _AND3_PATTERNS
    a, b, c = _wilds()
    _AND3_PATTERNS.extend([
        (Tuple(Ge(a, b), Ge(b, c), Gt(c, a)), false),
        (Tuple(Ge(a, b), Gt(b, c), Gt(c, a)), false),
        (Tuple(Gt(a, b), Gt(b, c), Gt(c, a)), false),
        # lower bounds
        (Tuple(Ge(a, b), Ge(a, c), Ge(b, c)), And(Ge(a, b), Ge(b, c))),
        (Tuple(Ge(a, b), Ge(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))),
        (Tuple(Ge(a, b), Gt(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))),
        (Tuple(Ge(a, c), Gt(a, b), Gt(b, c)), And(Gt(a, b), Gt(b, c))),
        (Tuple(Ge(b, c), Gt(a, b), Gt(a, c)), And(Gt(a, b), Ge(b, c))),
        (Tuple(Gt(a, b), Gt(a, c), Gt(b, c)), And(Gt(a, b), Gt(b, c))),
        # upper bounds
        (Tuple(Ge(b, a), Ge(c, a), Ge(b, c)), And(Ge(c, a), Ge(b, c))),
        (Tuple(Ge(b, a), Ge(c, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))),
        (Tuple(Ge(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))),
        (Tuple(Ge(c, a), Gt(b, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))),
        (Tuple(Ge(b, c), Gt(b, a), Gt(c, a)), And(Gt(c, a), Ge(b, c))),
        (Tuple(Gt(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))),
        # circular
        (Tuple(Ge(a, b), Ge(b, c), Ge(c, a)), And(Eq(a, b), Eq(b, c))),
    ])
    return _AND3_PATTERNS


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_simplify_patterns_or
def _patterns_or() -> list[Pattern]:
    """The two-term patterns of a disjunction (no ``Ge``, ``Gt``)."""
    from sympy_extras.functions.elementary.complexes import Abs
    from sympy_extras.functions.elementary.miscellaneous import Max, Min
    if _OR_PATTERNS:
        return _OR_PATTERNS
    a, b, c = _wilds()
    zero = Integer(0)
    _OR_PATTERNS.extend([
        (Tuple(Le(b, a), Le(a, b)), true),
        (Tuple(Le(b, a), Ne(a, b)), true),
        (Tuple(Eq(a, b), Le(a, b)), Le(a, b)),
        (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)),
        (Tuple(Lt(b, a), Lt(a, b)), Ne(a, b)),
        (Tuple(Lt(b, a), Ne(a, b)), Ne(a, b)),
        (Tuple(Le(a, b), Lt(a, b)), Le(a, b)),
        (Tuple(Eq(a, b), Ne(a, c)), ITE(Eq(b, c), true, Ne(a, c))),
        (Tuple(Ne(a, b), Ne(a, c)), ITE(Eq(b, c), Ne(a, b), true)),
        # Min/Max/ITE
        (Tuple(Le(b, a), Le(c, a)), Ge(a, Min(b, c))),
        (Tuple(Le(b, a), Lt(c, a)), ITE(Gt(b, c), Lt(c, a), Le(b, a))),
        (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Min(b, c))),
        (Tuple(Le(a, b), Le(a, c)), Le(a, Max(b, c))),
        (Tuple(Le(a, b), Lt(a, c)), ITE(Ge(b, c), Le(a, b), Lt(a, c))),
        (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Max(b, c))),
        (Tuple(Le(a, b), Le(c, a)), ITE(Ge(b, c), true, Or(Le(a, b), Ge(a, c)))),
        (Tuple(Le(c, a), Le(a, b)), ITE(Ge(b, c), true, Or(Le(a, b), Ge(a, c)))),
        (Tuple(Lt(a, b), Lt(c, a)), ITE(Gt(b, c), true, Or(Lt(a, b), Gt(a, c)))),
        (Tuple(Lt(c, a), Lt(a, b)), ITE(Gt(b, c), true, Or(Lt(a, b), Gt(a, c)))),
        (Tuple(Le(a, b), Lt(c, a)), ITE(Ge(b, c), true, Or(Le(a, b), Gt(a, c)))),
        (Tuple(Le(c, a), Lt(a, b)), ITE(Ge(b, c), true, Or(Lt(a, b), Ge(a, c)))),
        (Tuple(Lt(b, a), Lt(a, -b)), ITE(Ge(b, zero), Gt(Abs(a), b), true)),
        (Tuple(Le(b, a), Le(a, -b)), ITE(Gt(b, zero), Ge(Abs(a), b), true)),
    ])
    return _OR_PATTERNS
