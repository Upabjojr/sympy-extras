"""A SAT solver: the DPLL algorithm with watched literals and the VSIDS
heuristic, enumerating every model of a conjunction of clauses.

The search assigns the free literal of lowest score first (the score of a
literal is minus the number of clauses it occurs in, ties broken by the
smaller literal, so the negative one), propagates the unit clauses through
two watched literals ("sentinels") per clause, and on a conflict or after
a model flips the most recent decision which has not been flipped yet [1]_.
Every model assigns all the variables, so ``all_models`` enumerates the
total assignments satisfying the clauses, each once.

The module is a typed port of SymPy's ``dpll2`` with its default settings
(VSIDS, no clause learning, no theory); the input is a
:class:`~sympy_extras.logic.cnf.Clauses`, whose numbering of the atoms is
deterministic, so the models come in the same order in every run.

References
==========

.. [1] M. Davis, G. Logemann, D. Loveland, *A machine program for
       theorem-proving*, Communications of the ACM 5 (1962), 394-397;
       https://en.wikipedia.org/wiki/DPLL_algorithm

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.logic.cnf import to_clauses
>>> from sympy_extras.logic.dpll2 import propositional_models, is_satisfiable
>>> a, b = symbols('a b')
>>> list(propositional_models(to_clauses(a | b)))
[{a: True, b: True}, {a: True, b: False}, {a: False, b: True}]
>>> is_satisfiable(to_clauses(a & ~a))
False
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py, sympy/logic/inference.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, the default configuration only (VSIDS, no clause
# learning, no LRA theory), integer scores, deterministic iteration orders.
# ---------------------------------------------------------------------
from __future__ import annotations

from heapq import heappop, heappush
from typing import Final, Iterator, Sequence

from sympy_extras.core.logic import Boolean

from .cnf import Clauses

__all__ = ['SATSolver', 'propositional_models', 'is_satisfiable', 'PropositionalModel']

#: a model: the atoms to their truth values
PropositionalModel = dict[Boolean, bool]


class _Level:
    """A decision level: its decision literal (``0`` for the base level),
    the literals it set and whether the decision was flipped."""

    __slots__ = ('decision', 'var_settings', 'flipped')

    def __init__(self, decision: int, flipped: bool = False) -> None:
        self.decision: Final[int] = decision
        self.var_settings: Final[set[int]] = set()
        self.flipped: Final[bool] = flipped


class SATSolver:
    """The state of a DPLL search over clauses on the variables
    ``1, ..., variables``.

    Examples
    ========

    >>> from sympy_extras.logic.dpll2 import SATSolver
    >>> list(SATSolver([[2, -3], [1], [3, -3], [2, -2], [3, -2]], 3).find_models())
    [{1: True, 2: False, 3: False}, {1: True, 2: True, 3: True}]
    """

    def __init__(self, clauses: Sequence[Sequence[int]], variables: int) -> None:
        self.var_settings: Final[set[int]] = set()
        self.is_unsatisfied: bool = False
        self._unit_prop_queue: list[int] = []
        self.sentinels: Final[dict[int, set[int]]] = {}
        self.occurrence_count: Final[dict[int, int]] = {}
        self.variable_set: Final[list[bool]] = [False]*(variables + 1)
        self.clauses: Final[list[list[int]]] = [list(c) for c in clauses]
        # PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._initialize_clauses
        for i, clause in enumerate(self.clauses):
            if len(clause) == 1:
                self._unit_prop_queue.append(clause[0])
                continue
            self._sentinels(clause[0]).add(i)
            self._sentinels(clause[-1]).add(i)
            for lit in clause:
                self.occurrence_count[lit] = self.occurrence_count.get(lit, 0) + 1
        # PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._vsids_init
        self.lit_heap: Final[list[tuple[int, int]]] = []
        self.lit_scores: Final[dict[int, int]] = {}
        for var in range(1, variables + 1):
            for lit in (var, -var):
                self.lit_scores[lit] = -self.occurrence_count.get(lit, 0)
                heappush(self.lit_heap, (self.lit_scores[lit], lit))
        self.levels: Final[list[_Level]] = [_Level(0)]
        self.num_decisions: int = 0

    def _sentinels(self, lit: int) -> set[int]:
        found = self.sentinels.get(lit)
        if found is None:
            found = set()
            self.sentinels[lit] = found
        return found

    @property
    def _current_level(self) -> _Level:
        return self.levels[-1]

    # PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._find_model
    def find_models(self) -> Iterator[dict[int, bool]]:
        """The models, one at a time: the variables to their values."""
        flip_var = False
        self._simplify()
        if self.is_unsatisfied:
            return
        while True:
            if flip_var:
                flip_var = False
                lit = self._current_level.decision
            else:
                lit = self._vsids_calculate()
                self.num_decisions += 1
                if lit == 0:
                    yield {abs(v): v > 0 for v in sorted(self.var_settings, key=abs)}
                    while self._current_level.flipped:
                        self._undo()
                    if len(self.levels) == 1:
                        return
                    flip_lit = -self._current_level.decision
                    self._undo()
                    self.levels.append(_Level(flip_lit, flipped=True))
                    flip_var = True
                    continue
                self.levels.append(_Level(lit))
            self._assign_literal(lit)
            self._simplify()
            if self.is_unsatisfied:
                self.is_unsatisfied = False
                while self._current_level.flipped:
                    self._undo()
                    if len(self.levels) == 1:
                        return
                flip_lit = -self._current_level.decision
                self._undo()
                self.levels.append(_Level(flip_lit, flipped=True))
                flip_var = True

    def _clause_sat(self, cls: int) -> bool:
        return any(lit in self.var_settings for lit in self.clauses[cls])

    def _is_sentinel(self, lit: int, cls: int) -> bool:
        found = self.sentinels.get(lit)
        return found is not None and cls in found

    # PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._assign_literal
    def _assign_literal(self, lit: int) -> None:
        self.var_settings.add(lit)
        self._current_level.var_settings.add(lit)
        self.variable_set[abs(lit)] = True
        for cls in sorted(self._sentinels(-lit)):
            if self._clause_sat(cls):
                continue
            other_sentinel = 0
            for newlit in self.clauses[cls]:
                if newlit != -lit:
                    if self._is_sentinel(newlit, cls):
                        other_sentinel = newlit
                    elif not self.variable_set[abs(newlit)]:
                        self.sentinels[-lit].remove(cls)
                        self._sentinels(newlit).add(cls)
                        other_sentinel = 0
                        break
            if other_sentinel:
                self._unit_prop_queue.append(other_sentinel)

    # PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._undo
    def _undo(self) -> None:
        for lit in sorted(self._current_level.var_settings, key=abs):
            self.var_settings.remove(lit)
            self._vsids_lit_unset(lit)
            self.variable_set[abs(lit)] = False
        self.levels.pop()

    def _simplify(self) -> None:
        # pure literal elimination is disabled in SymPy's solver: unit
        # propagation is the only simplification
        while self._unit_prop():
            pass

    # PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._unit_prop
    def _unit_prop(self) -> bool:
        result = len(self._unit_prop_queue) > 0
        while self._unit_prop_queue:
            next_lit = self._unit_prop_queue.pop()
            if -next_lit in self.var_settings:
                self.is_unsatisfied = True
                self._unit_prop_queue = []
                return False
            self._assign_literal(next_lit)
        return result

    # PORTED FROM SYMPY: sympy/logic/algorithms/dpll2.py:SATSolver._vsids_calculate
    def _vsids_calculate(self) -> int:
        if not self.lit_heap:
            return 0
        while self.variable_set[abs(self.lit_heap[0][1])]:
            heappop(self.lit_heap)
            if not self.lit_heap:
                return 0
        return heappop(self.lit_heap)[1]

    def _vsids_lit_unset(self, lit: int) -> None:
        var = abs(lit)
        heappush(self.lit_heap, (self.lit_scores[var], var))
        heappush(self.lit_heap, (self.lit_scores[-var], -var))


def propositional_models(cnf: Clauses) -> Iterator[PropositionalModel]:
    """The models of a conjunctive normal form, one at a time, each
    assigning every atom (none when it is unsatisfiable), as SymPy's
    ``satisfiable(..., all_models=True)`` with the ``dpll2`` solver.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.cnf import to_clauses
    >>> from sympy_extras.logic.dpll2 import propositional_models
    >>> a, b, c = symbols('a b c')
    >>> list(propositional_models(to_clauses((a | b) & ~c & (~a | ~b))))
    [{c: False, a: True, b: False}, {c: False, a: False, b: True}]
    >>> list(propositional_models(to_clauses(a & ~a)))
    []
    """
    if cnf.is_false:
        return
    solver = SATSolver(cnf.clauses, len(cnf.atoms))
    for model in solver.find_models():
        yield {cnf.atoms[v - 1]: value for v, value in model.items()}


def is_satisfiable(cnf: Clauses) -> bool:
    """Whether a conjunctive normal form has a model.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.cnf import to_clauses
    >>> from sympy_extras.logic.dpll2 import is_satisfiable
    >>> a, b = symbols('a b')
    >>> is_satisfiable(to_clauses(a >> b)), is_satisfiable(to_clauses(a & (a >> b) & ~b))
    (True, False)
    """
    for _ in propositional_models(cnf):
        return True
    return False
