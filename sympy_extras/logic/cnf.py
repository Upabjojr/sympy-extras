"""Conjunctive normal forms of the Booleans of the core, encoded with
integers for the SAT solver of :mod:`sympy_extras.logic.dpll2`.

A formula is put in negation normal form and the conjunctions are
distributed over the disjunctions, without auxiliary variables, as SymPy's
``CNF.to_CNF`` (:mod:`sympy.assumptions.cnf`) does: the variables of the
clauses are the atoms of the formula (everything which is not a
connective), so that the models of the clauses are the models of the
formula. Clauses and literals are kept without duplicates in the order in
which they are first met, so that the encoding does not depend on the
hashes of the atoms. A tautological clause (``a | ~a``) is kept, so that
every atom of the formula is a variable of the clauses.

An optional expansion replaces atoms by equivalent formulas before the
distribution (SymPy's composite predicates: ``Q.real(x)`` is
``Q.negative(x) | Q.zero(x) | Q.positive(x)``).

The module is a typed port of the conversion of
:mod:`sympy.assumptions.cnf` (``to_NNF``, ``distribute_AND_over_OR``,
``EncodedCNF``) on the core's Booleans.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.logic.cnf import to_clauses
>>> a, b, c = symbols('a b c')
>>> cnf = to_clauses((a & b) | ~c)
>>> cnf.atoms, cnf.clauses
((c, a, b), ((-1, 2), (-1, 3)))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/assumptions/cnf.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, an immutable Clauses value with integer literals
# instead of Literal/AND/OR objects, polarity-driven distribution, clauses
# and literals in a deterministic order.
# ---------------------------------------------------------------------
from __future__ import annotations

from itertools import combinations
from typing import Callable, Final, Optional, Sequence

from sympy_extras.core.logic import (ITE, And, Boolean, BooleanFalse, BooleanTrue, Equivalent, Implies, Not, Or,
    Xor)

__all__ = ['Clauses', 'to_clauses', 'conjunction', 'Expansion']

#: a clause: literals, ``+i`` for the atom number ``i`` (from 1), ``-i`` for its negation
Clause = frozenset[int]
#: a replacement of an atom by an equivalent formula, ``None`` to keep it
Expansion = Callable[[Boolean], Optional[Boolean]]


class Clauses:
    """A conjunction of clauses over numbered atoms: an immutable value.

    ``atoms[i - 1]`` is the atom of the literals ``i`` and ``-i``; each
    clause is a tuple of literals sorted by atom, the negative literal
    first. No clause is empty unless the conjunction is unsatisfiable,
    in which case ``clauses`` is the single empty clause.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.cnf import Clauses
    >>> a, b = symbols('a b')
    >>> Clauses((a, b), ((1, -2), (2,))).clauses
    ((1, -2), (2,))
    """

    __slots__ = ('atoms', 'clauses')

    def __init__(self, atoms: Sequence[Boolean], clauses: Sequence[Sequence[int]]) -> None:
        self.atoms: Final[tuple[Boolean, ...]] = tuple(atoms)
        self.clauses: Final[tuple[tuple[int, ...], ...]] = tuple(tuple(c) for c in clauses)

    @property
    def is_false(self) -> bool:
        """Whether a clause is empty (the conjunction is unsatisfiable)."""
        return any(not c for c in self.clauses)

    def __repr__(self) -> str:
        return 'Clauses(%r, %r)' % (self.atoms, self.clauses)


def _dedup(clauses: Sequence[Clause]) -> list[Clause]:
    seen: dict[Clause, None] = {}
    for c in clauses:
        seen.setdefault(c, None)
    return list(seen)


def _conj(parts: Sequence[list[Clause]]) -> list[Clause]:
    """The clauses of a conjunction."""
    result: list[Clause] = []
    for p in parts:
        result.extend(p)
    return _dedup(result)


def _disj(parts: Sequence[list[Clause]]) -> list[Clause]:
    """The clauses of a disjunction, by distribution (SymPy's ``CNF._or``)."""
    result: list[Clause] = [frozenset()]
    for p in parts:
        result = _dedup([a | b for a in result for b in p])
    return result


class _Encoder:
    """The numbering of the atoms met while a formula is converted."""

    __slots__ = ('atoms', 'index', 'expand')

    def __init__(self, expand: Optional[Expansion]) -> None:
        self.atoms: Final[list[Boolean]] = []
        self.index: Final[dict[Boolean, int]] = {}
        self.expand: Final[Optional[Expansion]] = expand

    def variable(self, atom: Boolean) -> int:
        found = self.index.get(atom)
        if found is not None:
            return found
        self.atoms.append(atom)
        n = len(self.atoms)
        self.index[atom] = n
        return n

    # PORTED FROM SYMPY: sympy/assumptions/cnf.py:to_NNF, distribute_AND_over_OR
    def clauses(self, f: Boolean, positive: bool) -> list[Clause]:
        """The clauses of ``f`` (``positive``) or of ``~f``."""
        if isinstance(f, BooleanTrue):
            return [] if positive else [frozenset()]
        if isinstance(f, BooleanFalse):
            return [frozenset()] if positive else []
        if isinstance(f, Not):
            return self.clauses(f.arg, not positive)
        if isinstance(f, And):
            parts = [self.clauses(a, positive) for a in f.args]
            return _conj(parts) if positive else _disj(parts)
        if isinstance(f, Or):
            parts = [self.clauses(a, positive) for a in f.args]
            return _disj(parts) if positive else _conj(parts)
        if isinstance(f, Implies):
            a, b = f.args
            if positive:
                return _disj([self.clauses(a, False), self.clauses(b, True)])
            return _conj([self.clauses(a, True), self.clauses(b, False)])
        if isinstance(f, Equivalent):
            args = f.args
            # a chain of implications closed by the first argument
            pairs = [(args[i], args[(i + 1) % len(args)]) for i in range(len(args))]
            if positive:
                return _conj([_disj([self.clauses(a, False), self.clauses(b, True)]) for a, b in pairs])
            return _disj([_conj([self.clauses(a, True), self.clauses(b, False)]) for a, b in pairs])
        if isinstance(f, Xor):
            args = f.args
            # one clause excluding each assignment with an even number of
            # true arguments: those in ``neg``
            evens = [neg for i in range(0, len(args) + 1, 2) for neg in combinations(range(len(args)), i)]
            if positive:
                return _conj([_disj([self.clauses(s, j not in neg) for j, s in enumerate(args)]) for neg in evens])
            return _disj([_conj([self.clauses(s, j in neg) for j, s in enumerate(args)]) for neg in evens])
        if isinstance(f, ITE):
            c, t, e = f.args
            if positive:
                return _conj([_disj([self.clauses(c, False), self.clauses(t, True)]),
                              _disj([self.clauses(c, True), self.clauses(e, True)])])
            return _disj([_conj([self.clauses(c, True), self.clauses(t, False)]),
                          _conj([self.clauses(c, False), self.clauses(e, False)])])
        if self.expand is not None:
            expanded = self.expand(f)
            if expanded is not None:
                return self.clauses(expanded, positive)
        v = self.variable(f)
        return [frozenset((v if positive else -v,))]


def _sorted_clause(c: Clause) -> tuple[int, ...]:
    return tuple(sorted(c, key=lambda lit: (abs(lit), lit)))


def _compact(atoms: Sequence[Boolean], clauses: Sequence[Clause]) -> Clauses:
    """Renumber the atoms in the order in which the clauses use them,
    dropping those which no clause uses."""
    if any(not c for c in clauses):
        return Clauses((), ((),))
    order: dict[int, int] = {}
    for c in clauses:
        for lit in _sorted_clause(c):
            if abs(lit) not in order:
                order[abs(lit)] = len(order) + 1
    renumbered = [tuple(sorted((order[abs(lit)] if lit > 0 else -order[abs(lit)] for lit in c),
                               key=lambda lit: (abs(lit), lit))) for c in clauses]
    new_atoms = [atoms[v - 1] for v in order]
    return Clauses(new_atoms, renumbered)


def to_clauses(formula: Boolean, expand: Optional[Expansion] = None) -> Clauses:
    """The conjunctive normal form of a formula: its atoms and its clauses.

    Parameters
    ==========

    formula : Boolean
        A formula of the core; everything but the connectives and the
        truth values is an atom.
    expand : callable, optional
        A replacement of atoms by equivalent formulas, applied (again on
        its results) before the distribution; it returns ``None`` to keep
        an atom.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Equivalent
    >>> from sympy_extras.logic.cnf import to_clauses
    >>> a, b = symbols('a b')
    >>> to_clauses(Equivalent(a, b)).clauses, to_clauses(a & ~a).clauses
    (((-1, 2), (1, -2)), ((1,), (-1,)))
    >>> to_clauses(a | ~a).clauses, to_clauses(a & False).is_false
    (((-1, 1),), True)
    """
    encoder = _Encoder(expand)
    clauses = encoder.clauses(formula, True)
    return _compact(encoder.atoms, clauses)


def conjunction(parts: Sequence[Clauses]) -> Clauses:
    """The conjunction of conjunctive normal forms, on one numbering of
    their atoms.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.cnf import to_clauses, conjunction
    >>> a, b = symbols('a b')
    >>> c = conjunction([to_clauses(a | b), to_clauses(~b)])
    >>> c.atoms, c.clauses
    ((a, b), ((1, 2), (-2,)))
    """
    atoms: list[Boolean] = []
    index: dict[Boolean, int] = {}
    clauses: list[Clause] = []
    for part in parts:
        numbers: list[int] = []
        for atom in part.atoms:
            found = index.get(atom)
            if found is None:
                atoms.append(atom)
                found = len(atoms)
                index[atom] = found
            numbers.append(found)
        for c in part.clauses:
            clauses.append(frozenset(numbers[abs(lit) - 1] if lit > 0 else -numbers[abs(lit) - 1] for lit in c))
    return _compact(atoms, _dedup(clauses))
