"""Markov chains: gambler's ruin and the textbook chains, the limit of
the powers computed by SymPy, the identities pi P = pi and the first
step equations of the hitting quantities."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.core.expr import Expr
from sympy_extras.interop import to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.linalg import normal
from sympy_extras.matrices.markov import MarkovChain

a, b, p = Symbol('a'), Symbol('b'), Symbol('p')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def _random_chain(rnd: random.Random, n: int, density: float) -> ExprMatrix:
    rows: list[list[Expr]] = []
    for i in range(n):
        weights = [rnd.randint(1, 4) if rnd.random() < density else 0 for _ in range(n)]
        if not any(weights):
            weights[rnd.randrange(n)] = 1
        total = sum(weights)
        rows.append([Rational(w, total) for w in weights])
    return ExprMatrix.from_rows(rows)


def test_gamblers_ruin() -> None:
    # Norris, Example 1.3.3: with fair bets the ruin probability from i is 1 - i/N, the duration i (N - i)
    N = 6
    h = Rational(1, 2)
    rows: list[list[Expr]] = [[Integer(0)]*(N + 1) for _ in range(N + 1)]
    rows[0][0] = rows[N][N] = Integer(1)
    for i in range(1, N):
        rows[i][i - 1] = rows[i][i + 1] = h
    chain = MarkovChain(ExprMatrix.from_rows(rows))
    assert chain.is_absorbing and not chain.is_irreducible
    assert [c.states for c in chain.recurrent_classes()] == [(0,), (N,)] and chain.transient_states() == list(range(1, N))
    absorption = chain.absorption_probabilities()
    steps = chain.expected_steps()
    for i in range(1, N):
        assert absorption[i - 1, 1] == Rational(i, N) and absorption[i - 1, 0] == 1 - Rational(i, N)
        assert steps[i - 1, 0] == i*(N - i)
    assert chain.hitting_probabilities([N]) == ExprMatrix.column([Rational(i, N) for i in range(N + 1)])
    times = chain.mean_hitting_times([0, N])
    assert [times[i, 0] for i in range(N + 1)] == [i*(N - i) for i in range(N + 1)]
    assert chain.mean_hitting_times([N])[1, 0] == oo
    # biased bets: the ruin probability from 1 of 3 is q**2/(p**2 + p q + q**2) ... checked through the first step equations
    biased = MarkovChain(ExprMatrix.from_rows([[1, 0, 0, 0], [1 - p, 0, p, 0], [0, 1 - p, 0, p], [0, 0, 0, 1]]))
    win = biased.hitting_probabilities([3])
    assert normal(win[1, 0] - p*win[2, 0]) == 0 and normal(win[2, 0] - (1 - p)*win[1, 0] - p) == 0
    assert normal(win[1, 0] - p**2/(1 - p + p**2)) == 0


def test_two_state_chain() -> None:
    chain = MarkovChain(ExprMatrix.from_rows([[1 - a, a], [b, 1 - b]]))
    assert chain.is_irreducible and chain.is_ergodic and chain.period() == 1
    pi = chain.stationary_distribution()
    assert pi == ExprMatrix.from_rows([[b/(a + b), a/(a + b)]])
    assert (pi*chain.P).applyfunc(normal) == pi
    assert normal(chain.mean_return_time(0) - (a + b)/b) == 0
    assert chain.limiting_matrix() == ExprMatrix.from_rows([[b/(a + b), a/(a + b)]]*2)
    assert normal(chain.mean_hitting_times([1])[0, 0] - 1/a) == 0


def test_periodic_chains() -> None:
    cycle = MarkovChain(ExprMatrix.from_rows([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert cycle.period() == 3 and not cycle.is_aperiodic and not cycle.is_ergodic
    third = Rational(1, 3)
    assert cycle.stationary_distribution() == ExprMatrix.from_rows([[third, third, third]])
    assert cycle.limiting_matrix() == ExprMatrix.from_rows([[third]*3]*3)             # the Cesaro limit
    # the random walk on a path of four vertices has the period two; the walk on a triangle is aperiodic
    h = Rational(1, 2)
    path = MarkovChain(ExprMatrix.from_rows([[0, 1, 0, 0], [h, 0, h, 0], [0, h, 0, h], [0, 0, 1, 0]]))
    assert path.period() == 2 and path.stationary_distribution() == ExprMatrix.from_rows([[Rational(1, 6), third, third, Rational(1, 6)]])
    triangle = MarkovChain(ExprMatrix.from_rows([[0, h, h], [h, 0, h], [h, h, 0]]))
    assert triangle.period() == 1 and triangle.is_ergodic
    with pytest.raises(ValueError):
        MarkovChain(ExprMatrix.from_rows([[1, 0, 0], [h, 0, h], [0, 0, 1]])).period()


def test_random_chains_against_powers() -> None:
    rnd = random.Random(131)
    seen_transient = seen_several = 0
    for _ in range(25):
        n = rnd.randint(2, 5)
        chain = MarkovChain(_random_chain(rnd, n, rnd.choice([0.3, 0.5, 0.9])))
        S = _sympy_matrix(chain.P)
        recurrent = chain.recurrent_classes()
        seen_transient += bool(chain.transient_states())
        seen_several += len(recurrent) > 1
        # the classes are those of mutual reachability, computed from the powers of the adjacency matrix
        reach = (sympy.eye(n) + S.applyfunc(lambda e: sympy.Integer(1) if e != 0 else sympy.Integer(0)))**n
        for group in chain.classes():
            for i in range(n):
                together = reach[group.states[0], i] != 0 and reach[i, group.states[0]] != 0
                assert together == (i in group.states)
            closed = all(reach[state, j] == 0 for state in group.states for j in range(n) if j not in group.states)
            assert closed == group.recurrent
        for pi in chain.stationary_distributions():
            assert (pi*chain.P).applyfunc(normal) == pi and normal(sum(pi.entries, Integer(0)) - 1) == 0
            assert all(e.is_nonnegative for e in pi.entries)
        limit = _sympy_matrix(chain.limiting_matrix())
        assert limit*S == limit and all(sum(limit.row(i)) == 1 for i in range(n))
        if chain.is_aperiodic:
            power = (S.evalf(40))**4096
            assert all(abs(power[i, j] - limit[i, j]) < sympy.Float('1e-15') for i in range(n) for j in range(n))
        # first step equations of the hitting probabilities and times of the state 0
        hit, time = chain.hitting_probabilities([0]), chain.mean_hitting_times([0])
        for i in range(1, n):
            assert normal(hit[i, 0] - sum((chain.P[i, j]*hit[j, 0] for j in range(n)), Integer(0))) == 0
            if time[i, 0] != oo:
                assert hit[i, 0] == 1
                assert normal(time[i, 0] - 1 - sum((chain.P[i, j]*time[j, 0] for j in range(1, n) if chain.P[i, j] != 0),
                                                   Integer(0))) == 0
            else:
                assert hit[i, 0] != 1
        # Kac: the mean return time is 1 + the mean hitting time after one step
        for group in recurrent:
            state = group.states[0]
            back = chain.mean_hitting_times([state])
            expected = 1 + sum((chain.P[state, j]*back[j, 0] for j in range(n) if chain.P[state, j] != 0), Integer(0))
            assert normal(chain.mean_return_time(state) - expected) == 0
    assert seen_transient > 3 and seen_several > 2


def test_absorption_and_rejections() -> None:
    h = Rational(1, 2)
    chain = MarkovChain(ExprMatrix.from_rows([[h, h, 0, 0], [h, h, 0, 0], [h, 0, 0, h], [0, 0, 0, 1]]))
    # from the state 2: to the class {0, 1} or to 3 with probability 1/2 each, after one step
    assert chain.absorption_probabilities() == ExprMatrix.from_rows([[h, h]])
    assert chain.expected_steps() == ExprMatrix.column([1]) and chain.fundamental_matrix() == ExprMatrix.eye(1)
    assert chain.limiting_matrix().row(2) == (Rational(1, 4), Rational(1, 4), 0, h)
    assert chain.mean_return_time(2) == oo and chain.mean_return_time(0) == 2
    assert chain.distribution_after([0, 0, 1, 0], 2) == ExprMatrix.from_rows([[Rational(1, 4), Rational(1, 4), 0, h]])
    with pytest.raises(ValueError):
        chain.stationary_distribution()
    with pytest.raises(ValueError):
        MarkovChain(ExprMatrix.from_rows([[h, h], [h, 1]]))
    with pytest.raises(ValueError):
        MarkovChain(ExprMatrix.from_rows([[1, 0, 0], [0, 1, 0]]))
    with pytest.raises(ValueError):
        chain.hitting_probabilities([7])
    with pytest.raises(ValueError):
        chain.hitting_probabilities([])
