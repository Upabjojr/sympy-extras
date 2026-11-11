r"""Finite Markov chains, exactly: classes of states, periods, stationary
distributions, absorption probabilities, hitting times and limits, for
transition matrices of rational numbers or of expressions in symbols.

A chain on the states $0, \dots, n - 1$ is given by its transition
matrix $P$, with $P_{ij}$ the probability of the step $i \to j$ and rows
of sum one.

* **Structure.** The states split into *communicating classes*, the
  strongly connected components of the graph of the nonzero entries
  (Tarjan's algorithm [1]_). A class is *recurrent* when no step leaves
  it and *transient* otherwise; the *period* of a class is the gcd of
  the lengths of its cycles, found from the levels of a breadth-first
  search [2]_.
* **Stationary distributions.** Each recurrent class carries exactly one
  distribution $\pi$ with $\pi P = \pi$; the stationary distributions
  of the chain are their convex combinations [3]_. They are computed by
  the fraction-free elimination of :mod:`sympy_extras.matrices.linalg`,
  so that parameters are allowed.
* **Absorption.** With $Q$ the transitions between transient states and
  $R$ those from transient to recurrent states, the *fundamental matrix*
  $N = (I - Q)^{-1}$ counts the expected visits; $N \mathbf{1}$ is the
  expected number of steps before absorption and $N R$, summed over
  each recurrent class, the probabilities of ending in it [3]_.
* **Hitting.** For a set $T$ of target states the probabilities $h_i$
  of ever reaching it are the minimal nonnegative solution of $h_i =
  \sum_j P_{ij} h_j$ ($i \notin T$), $h_i = 1$ ($i \in T$), and the mean
  hitting times $k_i$ solve $k_i = 1 + \sum_j P_{ij} k_j$ where the
  target is reached with probability one [2]_; both are linear systems
  once the states which cannot reach $T$ are set apart.
* **Limits.** $\lim_n \frac1n \sum_{k < n} P^k$ always exists: row $i$
  is the mixture of the stationary distributions of the recurrent
  classes with the absorption probabilities of $i$. It is $\lim P^n$
  when every recurrent class has the period one.

With symbols in the matrix an entry which is not identically zero is
taken as a possible step: the structure is the generic one.

Examples
========

>>> from sympy_extras.core import Rational, Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.markov import MarkovChain
>>> a, b = Symbol('a'), Symbol('b')
>>> two = MarkovChain(ExprMatrix.from_rows([[1 - a, a], [b, 1 - b]]))
>>> two.stationary_distribution()
ExprMatrix([[b/(a + b), a/(a + b)]])
>>> h = Rational(1, 2)
>>> ruin = MarkovChain(ExprMatrix.from_rows([[1, 0, 0, 0], [h, 0, h, 0], [0, h, 0, h], [0, 0, 0, 1]]))   # gambler's ruin
>>> ruin.classes()
[CommunicatingClass([0], recurrent, period 1), CommunicatingClass([3], recurrent, period 1), CommunicatingClass([1, 2], transient, period 2)]
>>> ruin.absorption_probabilities(), ruin.expected_steps()
(ExprMatrix([[2/3, 1/3], [1/3, 2/3]]), ExprMatrix([[2], [2]]))
>>> ruin.hitting_probabilities([3])
ExprMatrix([[0], [1/3], [2/3], [1]])

References
==========

.. [1] R. Tarjan, *Depth-first search and linear graph algorithms*, SIAM
       Journal on Computing 1 (1972), 146-160.
.. [2] J. R. Norris, *Markov Chains*, Cambridge University Press, 1997,
       Sections 1.2 (class structure), 1.3 (hitting times and absorption
       probabilities) and 1.8 (periodicity, convergence to equilibrium).
.. [3] J. G. Kemeny, J. L. Snell, *Finite Markov Chains*, Van Nostrand,
       1960, Chapters III (absorbing chains, the fundamental matrix) and
       V (ergodic chains).
"""
from __future__ import annotations

from math import gcd
from typing import Final, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike
from sympy_extras.core.numbers import Integer

from .dense import ExprMatrix
from .linalg import _square, inverse, linear_solve, normal, nullspace

__all__ = ['CommunicatingClass', 'MarkovChain']


class CommunicatingClass:
    """A communicating class: its ``states`` (sorted), whether it is
    ``recurrent`` (closed) and its ``period``."""

    __slots__ = ('states', 'recurrent', 'period')

    def __init__(self, states: Sequence[int], recurrent: bool, period: int) -> None:
        self.states: Final[tuple[int, ...]] = tuple(states)
        self.recurrent: Final[bool] = recurrent
        self.period: Final[int] = period

    def __repr__(self) -> str:
        return 'CommunicatingClass(%s, %s, period %d)' % (list(self.states), 'recurrent' if self.recurrent else 'transient',
                                                         self.period)


def _components(successors: Sequence[Sequence[int]]) -> list[list[int]]:
    """The strongly connected components (Tarjan, without recursion)."""
    n = len(successors)
    index: list[int] = [-1]*n
    low: list[int] = [0]*n
    on_stack: list[bool] = [False]*n
    stack: list[int] = []
    components: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work: list[tuple[int, int]] = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            node, position = work[-1]
            if position < len(successors[node]):
                work[-1] = (node, position + 1)
                target = successors[node][position]
                if index[target] == -1:
                    index[target] = low[target] = counter
                    counter += 1
                    stack.append(target)
                    on_stack[target] = True
                    work.append((target, 0))
                elif on_stack[target]:
                    low[node] = min(low[node], index[target])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component: list[int] = []
                    while True:
                        member = stack.pop()
                        on_stack[member] = False
                        component.append(member)
                        if member == node:
                            break
                    components.append(sorted(component))
    return components


def _period(states: Sequence[int], successors: Sequence[Sequence[int]]) -> int:
    """The gcd of the lengths of the cycles of a strongly connected set:
    the gcd of ``level(u) + 1 - level(v)`` over its edges ``u -> v``."""
    inside = set(states)
    level = {states[0]: 0}
    queue = [states[0]]
    result = 0
    while queue:
        node = queue.pop(0)
        for target in successors[node]:
            if target not in inside:
                continue
            if target not in level:
                level[target] = level[node] + 1
                queue.append(target)
            else:
                result = gcd(result, level[node] + 1 - level[target])
    return result


class MarkovChain:
    """The chain with the transition matrix ``P`` (square, each row of
    sum one; ``ValueError`` otherwise)."""

    def __init__(self, P: ExprMatrix) -> None:
        n = _square(P)
        matrix = P.applyfunc(normal)
        for i in range(n):
            if normal(add(*matrix.row(i)) - 1) != 0:
                raise ValueError('the row %d of the transition matrix has the sum %s, not 1' % (i, normal(add(*matrix.row(i)))))
        self.P: Final[ExprMatrix] = matrix
        self.size: Final[int] = n
        self._successors: Final[list[list[int]]] = [[j for j in range(n) if matrix[i, j] != 0] for i in range(n)]
        found: list[CommunicatingClass] = []
        for component in _components(self._successors):
            inside = set(component)
            closed = all(target in inside for state in component for target in self._successors[state])
            found.append(CommunicatingClass(component, closed, _period(component, self._successors)))
        found.sort(key=lambda c: (not c.recurrent, c.states))
        self._classes: Final[list[CommunicatingClass]] = found

    def __repr__(self) -> str:
        return 'MarkovChain(%r)' % (self.P,)

    # -- structure

    def classes(self) -> list[CommunicatingClass]:
        """The communicating classes: the recurrent ones first, each
        group in the order of its smallest state."""
        return list(self._classes)

    def recurrent_classes(self) -> list[CommunicatingClass]:
        return [c for c in self._classes if c.recurrent]

    def transient_states(self) -> list[int]:
        return sorted(state for c in self._classes if not c.recurrent for state in c.states)

    def recurrent_states(self) -> list[int]:
        return sorted(state for c in self._classes if c.recurrent for state in c.states)

    @property
    def is_irreducible(self) -> bool:
        """One communicating class."""
        return len(self._classes) == 1

    @property
    def is_absorbing(self) -> bool:
        """Every recurrent class is a single absorbing state."""
        return all(len(c.states) == 1 for c in self._classes if c.recurrent)

    @property
    def is_aperiodic(self) -> bool:
        """Every recurrent class has the period one."""
        return all(c.period == 1 for c in self._classes if c.recurrent)

    @property
    def is_ergodic(self) -> bool:
        """Irreducible and aperiodic: the powers of ``P`` converge to
        the matrix whose rows are the stationary distribution."""
        return self.is_irreducible and self.is_aperiodic

    def period(self) -> int:
        """The period of an irreducible chain; ``ValueError`` otherwise."""
        if not self.is_irreducible:
            raise ValueError('the chain is not irreducible: see classes()')
        return self._classes[0].period

    # -- stationary distributions

    def _class_distribution(self, recurrent: CommunicatingClass) -> list[Expr]:
        states = recurrent.states
        k = len(states)
        block = ExprMatrix(k, k, [self.P[i, j] for i in states for j in states])
        kernel = nullspace((block.transpose() - ExprMatrix.eye(k)).applyfunc(normal))
        if len(kernel) != 1:
            raise ArithmeticError('the recurrent class %s has %d invariant vectors' % (list(states), len(kernel)))
        vector = [kernel[0][i, 0] for i in range(k)]
        total = add(*vector)
        zero: Expr = Integer(0)
        distribution = [zero]*self.size
        for state, value in zip(states, vector):
            distribution[state] = normal(value/total)
        return distribution

    def stationary_distributions(self) -> list[ExprMatrix]:
        """The stationary distribution (a row) carried by each recurrent
        class, in the order of :meth:`recurrent_classes`; the stationary
        distributions of the chain are their convex combinations."""
        return [ExprMatrix(1, self.size, self._class_distribution(c)) for c in self.recurrent_classes()]

    def stationary_distribution(self) -> ExprMatrix:
        """The stationary distribution, a row ``pi`` with ``pi P = pi``,
        when there is one recurrent class; ``ValueError`` otherwise.

        >>> from sympy_extras.core import Rational
        >>> from sympy_extras.matrices import ExprMatrix
        >>> from sympy_extras.matrices.markov import MarkovChain
        >>> h = Rational(1, 2)
        >>> MarkovChain(ExprMatrix.from_rows([[0, 1, 0], [h, 0, h], [0, 1, 0]])).stationary_distribution()
        ExprMatrix([[1/4, 1/2, 1/4]])
        """
        distributions = self.stationary_distributions()
        if len(distributions) != 1:
            raise ValueError('the chain has %d recurrent classes: see stationary_distributions()' % len(distributions))
        return distributions[0]

    # -- absorption

    def fundamental_matrix(self) -> ExprMatrix:
        r"""$N = (I - Q)^{-1}$ on the transient states (in increasing
        order): $N_{ij}$ is the expected number of visits to $j$ from
        $i$."""
        transient = self.transient_states()
        k = len(transient)
        Q = ExprMatrix(k, k, [self.P[i, j] for i in transient for j in transient])
        return inverse((ExprMatrix.eye(k) - Q).applyfunc(normal)) if k else ExprMatrix.zeros(0, 0)

    def expected_steps(self) -> ExprMatrix:
        """The expected number of steps before leaving the transient
        states, from each of them (a column, in increasing order of the
        states)."""
        N = self.fundamental_matrix()
        return ExprMatrix.column([normal(add(*N.row(i))) for i in range(N.rows)])

    def absorption_probabilities(self) -> ExprMatrix:
        """The probability, from each transient state (rows, in
        increasing order), of ending in each recurrent class (columns, in
        the order of :meth:`recurrent_classes`)."""
        transient = self.transient_states()
        recurrent = self.recurrent_classes()
        N = self.fundamental_matrix()
        entries: list[Expr] = []
        for row in range(len(transient)):
            for target in recurrent:
                total: list[Expr] = []
                for position, state in enumerate(transient):
                    step = add(*[self.P[state, j] for j in target.states])
                    if step != 0 and N[row, position] != 0:
                        total.append(N[row, position]*step)
                entries.append(normal(add(*total)) if total else Integer(0))
        return ExprMatrix(len(transient), len(recurrent), entries)

    # -- hitting

    def _can_reach(self, targets: Sequence[int]) -> set[int]:
        reached = set(targets)
        changed = True
        while changed:
            changed = False
            for state in range(self.size):
                if state not in reached and any(t in reached for t in self._successors[state]):
                    reached.add(state)
                    changed = True
        return reached

    def _targets(self, targets: Sequence[int]) -> list[int]:
        chosen = sorted(set(targets))
        if not chosen or chosen[0] < 0 or chosen[-1] >= self.size:
            raise ValueError('the target states are among 0, ..., %d' % (self.size - 1))
        return chosen

    def hitting_probabilities(self, targets: Sequence[int]) -> ExprMatrix:
        """The probability, from each state (a column), of ever being in
        one of the ``targets``."""
        chosen = self._targets(targets)
        able = self._can_reach(chosen)
        unknown = [state for state in range(self.size) if state in able and state not in chosen]
        zero: Expr = Integer(0)
        result = [Integer(1) if state in chosen else zero for state in range(self.size)]
        if unknown:
            # h_i - sum_{j unknown} P_ij h_j = sum_{j in targets} P_ij
            k = len(unknown)
            A = ExprMatrix(k, k, [(Integer(1) if i == j else zero) - self.P[i, j] for i in unknown for j in unknown])
            rhs = ExprMatrix.column([add(*[self.P[i, j] for j in chosen]) for i in unknown])
            solution = linear_solve(A.applyfunc(normal), rhs.applyfunc(normal))
            if solution is None or not solution.is_unique:
                raise ArithmeticError('the hitting probabilities are not determined')
            for position, state in enumerate(unknown):
                result[state] = solution.particular[position, 0]
        return ExprMatrix.column(result)

    def mean_hitting_times(self, targets: Sequence[int]) -> ExprMatrix:
        """The expected number of steps, from each state (a column), to
        reach one of the ``targets``: ``oo`` where they are missed with
        positive probability."""
        from sympy_extras.core.numbers import oo
        chosen = self._targets(targets)
        probabilities = self.hitting_probabilities(chosen)
        certain = [state for state in range(self.size) if state not in chosen and normal(probabilities[state, 0] - 1) == 0]
        zero: Expr = Integer(0)
        result: list[Expr] = [zero if state in chosen else oo for state in range(self.size)]
        if certain:
            k = len(certain)
            A = ExprMatrix(k, k, [(Integer(1) if i == j else zero) - self.P[i, j] for i in certain for j in certain])
            solution = linear_solve(A.applyfunc(normal), ExprMatrix.column([Integer(1)]*k))
            if solution is None or not solution.is_unique:
                raise ArithmeticError('the mean hitting times are not determined')
            for position, state in enumerate(certain):
                result[state] = solution.particular[position, 0]
        return ExprMatrix.column(result)

    def mean_return_time(self, state: int) -> Expr:
        """The expected number of steps to come back to a recurrent
        state: the inverse of its stationary probability (Kac's formula);
        ``oo`` for a transient state."""
        from sympy_extras.core.numbers import oo
        for recurrent in self.recurrent_classes():
            if state in recurrent.states:
                return normal(1/self._class_distribution(recurrent)[state])
        if not 0 <= state < self.size:
            raise ValueError('no state %d' % state)
        return oo

    # -- limits

    def limiting_matrix(self) -> ExprMatrix:
        r"""The Cesàro limit $\lim_n \frac1n \sum_{k<n} P^k$, which is
        $\lim_n P^n$ when the chain :attr:`is_aperiodic`: row $i$ is the
        long run distribution from the state $i$.

        >>> from sympy_extras.core import Rational
        >>> from sympy_extras.matrices import ExprMatrix
        >>> from sympy_extras.matrices.markov import MarkovChain
        >>> h = Rational(1, 2)
        >>> MarkovChain(ExprMatrix.from_rows([[1, 0, 0], [h, 0, h], [0, 0, 1]])).limiting_matrix()
        ExprMatrix([[1, 0, 0], [1/2, 0, 1/2], [0, 0, 1]])
        """
        recurrent = self.recurrent_classes()
        distributions = [self._class_distribution(c) for c in recurrent]
        transient = self.transient_states()
        absorption = self.absorption_probabilities()
        zero: Expr = Integer(0)
        rows: list[list[Expr]] = [[zero]*self.size for _ in range(self.size)]
        for distribution, group in zip(distributions, recurrent):
            for state in group.states:
                rows[state] = list(distribution)
        for position, state in enumerate(transient):
            rows[state] = [normal(add(*[absorption[position, c]*distributions[c][j] for c in range(len(recurrent))]))
                           for j in range(self.size)]
        return ExprMatrix(self.size, self.size, [e for row in rows for e in row])

    def distribution_after(self, initial: Sequence[ExprLike], steps: int) -> ExprMatrix:
        """The distribution (a row) after ``steps`` steps from the
        distribution ``initial``."""
        if len(initial) != self.size or steps < 0:
            raise ValueError('an initial distribution on %d states and a number of steps are needed' % self.size)
        current = ExprMatrix(1, self.size, list(initial))
        for _ in range(steps):
            current = (current*self.P).applyfunc(normal)
        return current
