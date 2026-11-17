"""Minimal sums of products and products of sums of the Booleans of the
core: ``simplify_logic``, ``SOPform`` and ``POSform``; the negation,
conjunctive and disjunctive normal forms ``to_nnf``, ``to_cnf`` and
``to_dnf`` (SymPy's, by the ``to_nnf`` rules of each connective and the
distribution of ``And`` over ``Or`` or of ``Or`` over ``And``).

A typed port of the two-level minimisation of SymPy's
``sympy.logic.boolalg``: the relations of the formula are replaced by
propositional variables (a relation and its negation by one variable and
its negation), the truth table of the formula over its variables is
computed, and the Quine–McCluskey method [Quine]_ [McCluskey]_ merges the
minterms into prime implicants (``_simplified_pairs``), of which a cover is
chosen by row and column dominance in the prime implicant table, the
implicant covering most minterms taken when no dominance is left
(``_rem_redundancy``). The result is the sum of products (``'dnf'``) or the
product of sums (``'cnf'``) of the cover; without a form, the sum of
products when at least half of the rows are true, else the product of
sums, as in SymPy. A formula with more than eight variables is returned
as it is unless ``force`` is given.

The relations are numbered in SymPy's ``ordered`` order (SymPy pops them
from a set, an order which depends on the hashes), so that the result does
not depend on the hashes; the minimisation itself is SymPy's.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.logic.boolalg import simplify_logic, SOPform, POSform
>>> x, y, z = symbols('x y z')
>>> simplify_logic((~x & ~y & ~z) | (~x & ~y & z))
~x & ~y
>>> simplify_logic(x | y, dontcare=y)
x
>>> simplify_logic((x > 0) | ((x <= 0) & (y > 1)))
(x > 0) | (y > 1)
>>> w = symbols('w')
>>> SOPform([w, x, y, z], [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
...         [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]])
(y & z) | (~w & ~x)
>>> POSform([w, x, y, z], [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
...         [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]])
z & (y | ~w)

References
==========

.. [Quine] W. V. Quine, *The problem of simplifying truth functions*,
   American Mathematical Monthly 59 (1952), 521-531.
.. [McCluskey] E. J. McCluskey, *Minimization of Boolean functions*, Bell
   System Technical Journal 35 (1956), 1417-1444.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/logic/boolalg.py (simplify_logic, SOPform, POSform, to_nnf, is_nnf,
# to_cnf, to_dnf, eliminate_implications, distribute_and_over_or, distribute_or_over_and and the
# to_nnf methods of the connectives) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, the terms as lists of 0, 1 and 3 only (no dict or
# integer forms), the relations numbered in ordered order, deep=False only
# (no simplification of the relations, in to_cnf and to_dnf with
# simplify=True too); the to_nnf methods are one function over the classes.
# ---------------------------------------------------------------------
from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from typing import Optional, Sequence

from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanAtom, BooleanFalse, BooleanFunction, BooleanLike,
    BooleanTrue, Equivalent, Implies, Not, Or, Xor, and_, false, not_, or_, to_boolean, true)
from sympy_extras.core.relational import Relational
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy

__all__ = ['simplify_logic', 'SOPform', 'POSform', 'is_literal', 'is_cnf', 'is_dnf', 'is_nnf', 'to_nnf', 'to_cnf',
           'to_dnf', 'eliminate_implications', 'distribute_and_over_or', 'distribute_or_over_and']

#: a term of the truth table: 0, 1, or 3 for a variable which does not matter
Term = list[int]

#: SymPy's limit on the number of variables without ``force``
_MAX_VARIABLES = 8


# PORTED FROM SYMPY: sympy/logic/boolalg.py:is_literal
def is_literal(expr: Boolean) -> bool:
    """Whether ``expr`` is a literal: an atom (anything which is not a
    connective) or the negation of one.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.boolalg import is_literal
    >>> a, b = symbols('a b')
    >>> is_literal(~a), is_literal(a & b)
    (True, False)
    """
    if isinstance(expr, Not):
        return is_literal(expr.arg)
    if not expr.args:
        return True
    return not isinstance(expr, BooleanFunction) and all(not a.args for a in expr.args)


def _is_form(expr: Boolean, outer: type[BooleanFunction], inner: type[BooleanFunction]) -> bool:
    """SymPy's ``_is_form``: ``expr`` is a literal, a disjunction (an
    ``inner``) of literals, or an ``outer`` of those."""
    if isinstance(expr, outer):
        args: tuple[Boolean, ...] = expr.args
    else:
        args = (expr,)
    for arg in args:
        if isinstance(arg, inner):
            if not all(is_literal(a) for a in arg.args):
                return False
        elif not is_literal(arg):
            return False
    return True


# PORTED FROM SYMPY: sympy/logic/boolalg.py:is_cnf
def is_cnf(expr: Boolean) -> bool:
    """Whether ``expr`` is in conjunctive normal form."""
    return _is_form(expr, And, Or)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:is_dnf
def is_dnf(expr: Boolean) -> bool:
    """Whether ``expr`` is in disjunctive normal form."""
    return _is_form(expr, Or, And)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_find_predicates
def _find_predicates(expr: Boolean) -> list[Boolean]:
    """The atoms of the formula (everything which is not a connective),
    without duplicates."""
    found: dict[Boolean, None] = {}

    def walk(e: Boolean) -> None:
        if isinstance(e, BooleanFunction):
            for a in e.args:
                walk(a)
        else:
            found.setdefault(e, None)
    walk(expr)
    return list(found)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_check_pair
def _check_pair(minterm1: Term, minterm2: Term) -> int:
    """The index of the only bit where the terms differ, ``-1`` if they
    differ in no bit or in several."""
    index = -1
    for x, i in enumerate(minterm1):
        if i != minterm2[x]:
            if index == -1:
                index = x
            else:
                return -1
    return index


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_convert_to_varsSOP
def _convert_to_vars_sop(minterm: Term, variables: Sequence[Boolean]) -> Boolean:
    return and_(*[variables[n] if val == 1 else not_(variables[n]) for n, val in enumerate(minterm) if val != 3])


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_convert_to_varsPOS
def _convert_to_vars_pos(maxterm: Term, variables: Sequence[Boolean]) -> Boolean:
    return or_(*[variables[n] if val == 0 else not_(variables[n]) for n, val in enumerate(maxterm) if val != 3])


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_simplified_pairs
def _simplified_pairs(terms: list[Term]) -> list[Term]:
    """The prime implicants of the terms (Quine–McCluskey): pairs of terms
    which differ in one bit merged, until no pair merges."""
    if not terms:
        return []
    simplified_terms: list[Term] = []
    todo: list[Optional[int]] = list(range(len(terms)))
    # the number of ones: two terms only merge if they differ by one
    termdict: defaultdict[int, list[int]] = defaultdict(list)
    for n, term in enumerate(terms):
        termdict[sum(1 for t in term if t == 1)].append(n)
    variables = len(terms[0])
    for k in range(variables):
        for i in termdict[k]:
            for j in termdict[k + 1]:
                index = _check_pair(terms[i], terms[j])
                if index != -1:
                    todo[i] = todo[j] = None
                    newterm = terms[i][:]
                    newterm[index] = 3
                    if newterm not in simplified_terms:
                        simplified_terms.append(newterm)
    if simplified_terms:
        simplified_terms = _simplified_pairs(simplified_terms)
    simplified_terms.extend([terms[i] for i in todo if i is not None])
    return simplified_terms


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_rem_redundancy
def _rem_redundancy(l1: list[Term], terms: list[Term]) -> list[Term]:
    """A cover of the terms by the prime implicants ``l1``: dominated rows
    and dominated columns of the prime implicant table removed in turn,
    then the implicant covering most terms chosen."""
    if not terms:
        return []
    nterms = len(terms)
    nl1 = len(l1)
    dommatrix = [[0]*nl1 for _ in range(nterms)]
    colcount = [0]*nl1
    rowcount = [0]*nterms
    for primei, prime in enumerate(l1):
        for termi, term in enumerate(terms):
            if all(t == 3 or t == mt for t, mt in zip(prime, term)):
                dommatrix[termi][primei] = 1
                colcount[primei] += 1
                rowcount[termi] += 1
    anythingchanged = True
    while anythingchanged:
        anythingchanged = False
        for rowi in range(nterms):
            if rowcount[rowi]:
                row = dommatrix[rowi]
                for row2i in range(nterms):
                    if rowi != row2i and rowcount[rowi] and (rowcount[rowi] <= rowcount[row2i]):
                        row2 = dommatrix[row2i]
                        if all(row2[n] >= row[n] for n in range(nl1)):
                            # row2 dominates row: row2 removed
                            rowcount[row2i] = 0
                            anythingchanged = True
                            for primei, prime_entry in enumerate(row2):
                                if prime_entry:
                                    dommatrix[row2i][primei] = 0
                                    colcount[primei] -= 1
        colcache: dict[int, list[int]] = {}
        for coli in range(nl1):
            if colcount[coli]:
                if coli in colcache:
                    col = colcache[coli]
                else:
                    col = [dommatrix[i][coli] for i in range(nterms)]
                    colcache[coli] = col
                for col2i in range(nl1):
                    if coli != col2i and colcount[col2i] and (colcount[coli] >= colcount[col2i]):
                        if col2i in colcache:
                            col2 = colcache[col2i]
                        else:
                            col2 = [dommatrix[i][col2i] for i in range(nterms)]
                            colcache[col2i] = col2
                        if all(col[n] >= col2[n] for n in range(nterms)):
                            # col dominates col2: col2 removed
                            colcount[col2i] = 0
                            anythingchanged = True
                            for termi, term_entry in enumerate(col2):
                                if term_entry and dommatrix[termi][col2i]:
                                    dommatrix[termi][col2i] = 0
                                    rowcount[termi] -= 1
        if not anythingchanged:
            # the prime implicant covering most terms, if it covers two
            maxterms = 0
            bestcolidx = -1
            for coli in range(nl1):
                s = colcount[coli]
                if s > maxterms:
                    bestcolidx = coli
                    maxterms = s
            if bestcolidx != -1 and maxterms > 1:
                for primei in range(nl1):
                    if primei != bestcolidx:
                        for termi, term_entry in enumerate(colcache[bestcolidx]):
                            if term_entry and dommatrix[termi][primei]:
                                dommatrix[termi][primei] = 0
                                anythingchanged = True
                                rowcount[termi] -= 1
                                colcount[primei] -= 1
    return [l1[i] for i in range(nl1) if colcount[i]]


def _checked_terms(terms: Sequence[Sequence[int]], bits: int) -> list[Term]:
    found: list[Term] = []
    for t in terms:
        if len(t) != bits:
            raise ValueError("Each term must contain {bits} bits as there are\n{bits} variables "
                             "(or be an integer).".format(bits=bits))
        found.append(list(t))
    return found


def _sop_form(variables: Sequence[Boolean], minterms: list[Term], dontcares: list[Term]) -> Boolean:
    new = _simplified_pairs(minterms + dontcares)
    essential = _rem_redundancy(new, minterms)
    return or_(*[_convert_to_vars_sop(x, variables) for x in essential])


def _pos_form(variables: Sequence[Boolean], minterms: list[Term], dontcares: list[Term]) -> Boolean:
    maxterms: list[Term] = []
    for t in product((0, 1), repeat=len(variables)):
        term = list(t)
        if term not in minterms and term not in dontcares:
            maxterms.append(term)
    new = _simplified_pairs(maxterms + dontcares)
    essential = _rem_redundancy(new, maxterms)
    return and_(*[_convert_to_vars_pos(x, variables) for x in essential])


# PORTED FROM SYMPY: sympy/logic/boolalg.py:SOPform
def SOPform(variables: Sequence[Boolean], minterms: Sequence[Sequence[int]],
            dontcares: Sequence[Sequence[int]] = ()) -> Boolean:
    """The smallest sum of products found by the Quine–McCluskey method
    which is true on the ``minterms`` (lists of 0 and 1 for the
    ``variables``) and false elsewhere, the ``dontcares`` free."""
    if not minterms:
        return false
    m = _checked_terms(minterms, len(variables))
    d = _checked_terms(dontcares, len(variables))
    for t in d:
        if t in m:
            raise ValueError('%s in minterms is also in dontcares' % t)
    return _sop_form(variables, m, d)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:POSform
def POSform(variables: Sequence[Boolean], minterms: Sequence[Sequence[int]],
            dontcares: Sequence[Sequence[int]] = ()) -> Boolean:
    """The smallest product of sums found by the Quine–McCluskey method
    which is true on the ``minterms`` and false elsewhere, the
    ``dontcares`` free."""
    if not minterms:
        return false
    m = _checked_terms(minterms, len(variables))
    d = _checked_terms(dontcares, len(variables))
    for t in d:
        if t in m:
            raise ValueError('%s in minterms is also in dontcares' % t)
    return _pos_form(variables, m, d)


def _replaced(expr: Boolean, rule: dict[Basic, Basic]) -> Boolean:
    return to_boolean(expr._xreplace_evaluated(rule))


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_get_truthtable
def _get_truthtable(variables: list[Boolean], expr: Boolean, const: list[int]) -> list[Term]:
    """The rows of the variables (with the constants first) where the
    formula is true."""
    rows: list[tuple[Boolean, Term]] = [(expr, [])]
    for v in reversed(variables):
        table = [(_replaced(e, {v: false}), [0] + bits) for e, bits in rows if not isinstance(e, BooleanFalse)]
        table.extend([(_replaced(e, {v: true}), [1] + bits) for e, bits in rows
                      if not isinstance(e, BooleanFalse)])
        rows = table
    res: list[Term] = []
    for e, bits in rows:
        if not isinstance(e, BooleanAtom):
            raise TypeError('the formula %s is not decided by its atoms' % (e,))
        if isinstance(e, BooleanTrue):
            res.append(const + bits)
    return [] if res == [[]] else res


# PORTED FROM SYMPY: sympy/logic/boolalg.py:simplify_logic
def simplify_logic(expr: Boolean, form: Optional[str] = None, force: bool = False,
                   dontcare: Optional[Boolean] = None) -> Boolean:
    """The formula as a minimal sum of products (``form='dnf'``) or product
    of sums (``form='cnf'``), by default the one SymPy's
    ``simplify_logic(expr, deep=False)`` chooses; a formula with more than
    eight variables (atoms, a relation and its negation counting once) is
    returned as it is unless ``force``. The rows where ``dontcare`` is true
    are free.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.logic.boolalg import simplify_logic
    >>> a, b, c = symbols('a b c')
    >>> simplify_logic((a & b) | (a & ~b) | c)
    a | c
    >>> simplify_logic((a | b) & (a | ~b), form='dnf')
    a
    >>> simplify_logic(Eq(a, 1) | (~Eq(a, 1) & Eq(b, 2)))
    Eq(a, 1) | Eq(b, 2)
    """
    if form not in (None, 'cnf', 'dnf'):
        raise ValueError("form can be cnf or dnf only")
    if form:
        form_ok = is_cnf(expr) if form == 'cnf' else is_dnf(expr)
        if form_ok and (not isinstance(expr, BooleanFunction) or all(is_literal(a) for a in expr.args)):
            return expr
    if not isinstance(expr, BooleanFunction):
        return expr
    # the relations replaced by variables, a relation and its negation by
    # one variable and its negation
    repl: dict[Basic, Basic] = {}
    undo: dict[Basic, Basic] = {}
    relations = expr.atoms(Relational)
    if dontcare is not None:
        relations |= dontcare.atoms(Relational)
    for var in ordered(relations):
        if var in repl or not isinstance(var, Relational):
            continue
        d = Dummy()
        undo[d] = var
        repl[var] = d
        nvar = var.negated
        if nvar in relations:
            repl[nvar] = Not(d)
    replaced = _replaced(expr, repl)
    dc = None if dontcare is None else _replaced(dontcare, repl)
    variables = _find_predicates(replaced)
    if not force and len(variables) > _MAX_VARIABLES:
        return _replaced(replaced, undo)
    if dc is not None:
        variables = variables + [v for v in _find_predicates(dc) if v not in variables]
        if not force and len(variables) > _MAX_VARIABLES:
            variables = _find_predicates(replaced)
            dc = None
    ordered_variables = [to_boolean(v) for v in ordered(variables)]
    constants = [v for v in ordered_variables if isinstance(v, BooleanAtom)]
    free = [v for v in ordered_variables if not isinstance(v, BooleanAtom)]
    all_variables = constants + free
    const = [1 if isinstance(c, BooleanTrue) else 0 for c in constants]
    truthtable = _get_truthtable(free, replaced, const)
    if dc is not None:
        dctruthtable = _get_truthtable(free, dc, const)
        truthtable = [t for t in truthtable if t not in dctruthtable]
    else:
        dctruthtable = []
    big = len(truthtable) >= 2**(len(all_variables) - 1)
    if form == 'dnf' or form is None and big:
        return _replaced(_sop_form(all_variables, truthtable, dctruthtable), undo)
    if not truthtable:
        return false
    return _replaced(_pos_form(all_variables, truthtable, dctruthtable), undo)


# ---------------------------------------------------------------------
# the normal forms

# PORTED FROM SYMPY: sympy/logic/boolalg.py:BooleanFunction._to_nnf
def _lattice_nnf(conjunction: bool, args: Sequence[Boolean], simplify: bool) -> Boolean:
    """SymPy's ``And._to_nnf`` and ``Or._to_nnf``: the arguments in
    negation normal form; with ``simplify``, flattened, and the zero of the
    connective when an argument and its negation occur."""
    cls: type[BooleanFunction] = And if conjunction else Or
    kept: list[Boolean] = []
    seen: set[Boolean] = set()
    for arg in args:
        if not is_literal(arg):
            arg = _nnf(arg, simplify)
        if simplify:
            parts: tuple[Boolean, ...] = arg.args if isinstance(arg, cls) else (arg,)
            for a in parts:
                if not_(a) in seen:
                    return false if conjunction else true
                if a not in seen:
                    seen.add(a)
                    kept.append(a)
        elif arg not in seen:
            seen.add(arg)
            kept.append(arg)
    return and_(*kept) if conjunction else or_(*kept)


def _xor_clauses(args: Sequence[Boolean], first: int) -> list[Boolean]:
    """The clauses of ``Xor`` (``first = 0``) or of its negation
    (``first = 1``): one for each choice of an even (odd) number of negated
    arguments."""
    clauses: list[Boolean] = []
    for i in range(first, len(args) + 1, 2):
        for negated in combinations(args, i):
            clauses.append(or_(*[not_(a) if a in negated else a for a in args]))
    return clauses


# PORTED FROM SYMPY: sympy/logic/boolalg.py:And.to_nnf, Or.to_nnf, Not.to_nnf, Xor.to_nnf,
# Implies.to_nnf, Equivalent.to_nnf, ITE.to_nnf
def _nnf(expr: Boolean, simplify: bool) -> Boolean:
    """The ``to_nnf`` method of the class of ``expr``."""
    if isinstance(expr, And):
        return _lattice_nnf(True, expr.args, simplify)
    if isinstance(expr, Or):
        return _lattice_nnf(False, expr.args, simplify)
    if isinstance(expr, Not):
        if is_literal(expr):
            return expr
        inner = expr.arg
        args: tuple[Boolean, ...] = inner.args if isinstance(inner, BooleanFunction) else ()
        if isinstance(inner, And):
            return _lattice_nnf(False, [not_(a) for a in args], simplify)
        if isinstance(inner, Or):
            return _lattice_nnf(True, [not_(a) for a in args], simplify)
        if isinstance(inner, Implies):
            return _lattice_nnf(True, [args[0], not_(args[1])], simplify)
        if isinstance(inner, Equivalent):
            return _lattice_nnf(True, [or_(*args), or_(*[not_(a) for a in args])], simplify)
        if isinstance(inner, Xor):
            return _lattice_nnf(True, _xor_clauses(args, 1), simplify)
        if isinstance(inner, ITE):
            a, b, c = args
            return _lattice_nnf(True, [or_(a, not_(c)), or_(not_(a), not_(b))], simplify)
        raise ValueError('Illegal operator %s in expression' % (type(inner).__name__,))
    if isinstance(expr, Xor):
        return _lattice_nnf(True, _xor_clauses(expr.args, 0), simplify)
    if isinstance(expr, Implies):
        p, q = expr.args
        return _lattice_nnf(False, [not_(p), q], simplify)
    if isinstance(expr, Equivalent):
        eargs = expr.args
        clauses = [or_(not_(p), q) for p, q in zip(eargs, eargs[1:])]
        clauses.append(or_(not_(eargs[-1]), eargs[0]))
        return _lattice_nnf(True, clauses, simplify)
    if isinstance(expr, ITE):
        a, b, c = expr.args
        return _lattice_nnf(True, [or_(not_(a), b), or_(a, c)], simplify)
    return expr


# PORTED FROM SYMPY: sympy/logic/boolalg.py:is_nnf
def is_nnf(expr: Boolean, simplified: bool = True) -> bool:
    """Whether ``expr`` is in negation normal form: ``And``, ``Or`` and
    ``Not`` of literals only; with ``simplified``, no ``And`` or ``Or``
    with an argument and its negation.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.boolalg import is_nnf
    >>> a, b, c = symbols('a b c')
    >>> is_nnf(a & b | ~c), is_nnf((a | ~a) & (b | c)), is_nnf((a | ~a) & (b | c), False), is_nnf(~(a & b) | c)
    (True, False, True, False)
    """
    if is_literal(expr):
        return True
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, (And, Or)):
            args = e.args
            if simplified and any(not_(a) in args for a in args):
                return False
            stack.extend(args)
        elif not is_literal(e):
            return False
    return True


# PORTED FROM SYMPY: sympy/logic/boolalg.py:to_nnf
def to_nnf(expr: BooleanLike, simplify: bool = True) -> Boolean:
    """The negation normal form: ``Xor``, ``Implies``, ``Equivalent`` and
    ``ITE`` written with ``And`` and ``Or``, and the negations pushed to
    the literals; with ``simplify``, no redundant clauses.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Equivalent, ITE, Not
    >>> from sympy_extras.logic.boolalg import to_nnf
    >>> a, b, c, d = symbols('a b c d')
    >>> to_nnf(~((~a & ~b) | (c & d))), to_nnf(ITE(a, b, c)), to_nnf(Not(ITE(a, b, c)))
    ((a | b) & (~c | ~d), (a | c) & (b | ~a), (a | ~c) & (~a | ~b))
    >>> to_nnf(Equivalent(a >> b, b >> a))
    (a | ~b | (a & ~b)) & (b | ~a | (b & ~a))
    """
    e = to_boolean(expr)
    if is_nnf(e, simplify):
        return e
    return _nnf(e, simplify)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:eliminate_implications
def eliminate_implications(expr: BooleanLike) -> Boolean:
    """``Implies``, ``Equivalent``, ``Xor`` and ``ITE`` written with
    ``And``, ``Or`` and ``Not`` (``to_nnf`` without simplification).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Implies
    >>> from sympy_extras.logic.boolalg import eliminate_implications
    >>> a, b = symbols('a b')
    >>> eliminate_implications(Implies(a, b))
    b | ~a
    """
    return to_nnf(expr, simplify=False)


def _build(conjunction: bool, args: Sequence[Boolean]) -> Boolean:
    return and_(*args) if conjunction else or_(*args)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_distribute
def _distribute(e: Boolean, conjunction: bool) -> Boolean:
    """``distribute_and_over_or`` (``conjunction``: into an ``And`` of
    ``Or``) and ``distribute_or_over_and``."""
    outer: type[BooleanFunction] = And if conjunction else Or
    inner: type[BooleanFunction] = Or if conjunction else And
    if isinstance(e, inner):
        for arg in e.args:
            if isinstance(arg, outer):
                rest = _build(not conjunction, [a for a in e.args if a is not arg])
                return _build(conjunction, [_distribute(_build(not conjunction, [c, rest]), conjunction)
                                            for c in arg.args])
        return e
    if isinstance(e, outer):
        return _build(conjunction, [_distribute(a, conjunction) for a in e.args])
    return e


# PORTED FROM SYMPY: sympy/logic/boolalg.py:distribute_and_over_or
def distribute_and_over_or(expr: BooleanLike) -> Boolean:
    """An ``Or`` of ``And`` distributed into an ``And`` of ``Or``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.boolalg import distribute_and_over_or
    >>> a, b, c = symbols('a b c')
    >>> distribute_and_over_or((a & b) | c)
    (a | c) & (b | c)
    """
    return _distribute(to_boolean(expr), True)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:distribute_or_over_and
def distribute_or_over_and(expr: BooleanLike) -> Boolean:
    """An ``And`` of ``Or`` distributed into an ``Or`` of ``And``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.boolalg import distribute_or_over_and
    >>> a, b, c = symbols('a b c')
    >>> distribute_or_over_and((a | b) & c)
    (a & c) | (b & c)
    """
    return _distribute(to_boolean(expr), False)


def _check_size(expr: Boolean, force: bool) -> None:
    if not force and len(_find_predicates(expr)) > _MAX_VARIABLES:
        raise ValueError('To simplify a logical expression with more than 8 variables may take a long time '
                         'and requires the use of `force=True`.')


# PORTED FROM SYMPY: sympy/logic/boolalg.py:to_cnf
def to_cnf(expr: BooleanLike, simplify: bool = False, force: bool = False) -> Boolean:
    """The conjunctive normal form; with ``simplify``, the minimal one by
    :func:`simplify_logic` (more than eight variables need ``force``; the
    relations are not simplified: SymPy's ``deep=True`` is not ported).

    Examples
    ========

    >>> from sympy_extras.core import symbols, ITE
    >>> from sympy_extras.logic.boolalg import to_cnf
    >>> a, b, d = symbols('a b d')
    >>> to_cnf(~(a | b) | d), to_cnf((a | b) & (a | ~a), True), to_cnf(ITE(a, b, d))
    ((d | ~a) & (d | ~b), a | b, (a | d) & (b | ~a))
    """
    e = to_boolean(expr)
    if not isinstance(e, BooleanFunction):
        return e
    if simplify:
        _check_size(e, force)
        return simplify_logic(e, 'cnf', force=force)
    if is_cnf(e):
        return e
    return distribute_and_over_or(eliminate_implications(e))


# PORTED FROM SYMPY: sympy/logic/boolalg.py:to_dnf
def to_dnf(expr: BooleanLike, simplify: bool = False, force: bool = False) -> Boolean:
    """The disjunctive normal form; with ``simplify``, the minimal one by
    :func:`simplify_logic` (more than eight variables need ``force``).

    Examples
    ========

    >>> from sympy_extras.core import symbols, ITE
    >>> from sympy_extras.logic.boolalg import to_dnf
    >>> a, b, c = symbols('a b c')
    >>> to_dnf(b & (a | c)), to_dnf((a & b) | (a & ~b) | (b & c) | (~b & c), True)
    ((a & b) | (b & c), a | c)
    >>> to_dnf(ITE(a, b, c))
    (a & b) | (b & c) | (a & ~a) | (c & ~a)
    """
    e = to_boolean(expr)
    if not isinstance(e, BooleanFunction):
        return e
    if simplify:
        _check_size(e, force)
        return simplify_logic(e, 'dnf', force=force)
    if is_dnf(e):
        return e
    return distribute_or_over_and(eliminate_implications(e))
