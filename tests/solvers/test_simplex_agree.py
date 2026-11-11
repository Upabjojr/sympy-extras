"""The Rust port of ``solvers.simplex`` (exact linear and integer
programming with certificates, Fourier-Motzkin elimination) against the
Python module: the docstring examples on both implementations, random
linear programs of every status (values, points, duals, rays), strict
feasibility, small integer programs, and projections of random systems
(the rows and their order)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def sx(impl: str) -> Any:
    return load(impl + ".solvers.simplex")


def test_known(impl: str) -> None:
    s = sx(impl)
    result = s.linear_program([3, 2], [[1, 1], [1, 0], [0, 1], [-1, 0], [0, -1]], [4, 3, 3, 0, 0])
    assert (result.status, result.value, result.point, result.dual) == ('optimal', 11, [3, 1], [2, 1, 0, 0, 0])
    assert repr(result) == "LinearProgramResult('optimal', value=11)"
    assert s.linear_program([1], [[1], [-1]], [0, -1]).status == 'infeasible'
    assert s.linear_program([1, 0], [[0, 1]], [1]).status == 'unbounded'
    r = s.linear_program([1, 1], [[2, 1], [1, 3], [-1, 0], [0, -1]], [4, 6, 0, 0])
    assert (r.value, r.point, r.dual) == (F(14, 5), [F(6, 5), F(8, 5)], [F(2, 5), F(1, 5), 0, 0])
    assert repr(r) == "LinearProgramResult('optimal', value=14/5)"
    assert s.linear_feasible([[1], [-1]], [1, -1], [False, False]) == [1]
    assert s.linear_feasible([[1], [-1]], [1, -1], [True, False]) is None
    assert s.linear_feasible([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], [True, True, True]) == [F(1, 3), F(1, 3)]
    assert repr(s.integer_program([5, 4], [[6, 4], [1, 2], [-1, 0], [0, -1]], [24, 6, 0, 0])) == \
        "IntegerProgramResult('optimal', value=20, point=[4, 0])"
    assert s.integer_program([1, 1], [[2, 2], [-2, -2]], [1, -1]).status == 'infeasible'
    assert s.integer_program([1, 0], [[0, 1], [0, -1]], [F(5, 2), 0]).status == 'unbounded'
    triangle = [[-1, 0], [0, -1], [1, 1]]
    shadow = s.eliminate_variables(triangle, [0, 0, 1], [1])
    assert sorted(zip(shadow.A, shadow.b)) == [([-1], 0), ([1], 1)]
    open_one = s.eliminate_variables(triangle, [0, 0, 1], [1], [False, True, False])
    assert (open_one.contains([1]), open_one.contains([F(1, 2)])) == (False, True)
    assert not s.eliminate_variables([[1], [-1]], [0, -1], [0]).is_feasible()
    with pytest.raises(ValueError):
        s.linear_program([1], [[1, 2]], [1])
    with pytest.raises(ValueError):
        s.eliminate_variables([[1]], [1], [3])


def entry(rng: random.Random) -> Fraction:
    return F(rng.randint(-4, 4), rng.choice([1, 1, 1, 2, 3]))


def lp_data(result: Any) -> Any:
    return (result.status, result.value, result.point, result.dual, result.ray, repr(result))


@pytest.mark.parametrize("seed", range(30))
def test_linear_programs_agree(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("linear programs", seed))
    for _ in range(6):
        n, m = rng.randint(1, 4), rng.randint(0, 6)
        c = [entry(rng) for _ in range(n)]
        A = [[entry(rng) for _ in range(n)] for _ in range(m)]
        b = [entry(rng) for _ in range(m)]
        if rng.random() < 0.5:
            # a box: more programs with an optimum (and a dual certificate)
            for j in range(n):
                A.append([F(1) if k == j else F(0) for k in range(n)])
                b.append(F(rng.randint(0, 5)))
                A.append([F(-1) if k == j else F(0) for k in range(n)])
                b.append(F(rng.randint(0, 5)))
            m = len(A)
        strict = [rng.random() < 0.4 for _ in range(m)]
        results = [(lp_data(sx(impl).linear_program(c, A, b)), sx(impl).linear_feasible(A, b, strict))
                   for impl in IMPLEMENTATIONS]
        assert results[0] == results[1], (c, A, b)


@pytest.mark.parametrize("seed", range(15))
def test_integer_programs_agree(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("integer programs", seed))
    for _ in range(3):
        n = rng.randint(1, 3)
        c = [F(rng.randint(-3, 3)) for _ in range(n)]
        A = [[F(rng.randint(-3, 3)) for _ in range(n)] for _ in range(rng.randint(1, 3))]
        b = [F(rng.randint(-5, 9), rng.choice([1, 2])) for _ in A]
        if rng.random() < 0.7:
            # a box, so that most programs are bounded
            for j in range(n):
                A.append([F(1) if k == j else F(0) for k in range(n)])
                b.append(F(rng.randint(1, 4)))
                A.append([F(-1) if k == j else F(0) for k in range(n)])
                b.append(F(rng.randint(0, 3)))
        results = []
        for impl in IMPLEMENTATIONS:
            result = sx(impl).integer_program(c, A, b)
            results.append((result.status, result.value, result.point, repr(result)))
        assert results[0] == results[1], (c, A, b)


@pytest.mark.parametrize("seed", range(20))
def test_elimination_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("fourier motzkin", seed))
    for _ in range(3):
        n, m = rng.randint(2, 4), rng.randint(1, 6)
        A = [[F(rng.randint(-3, 3)) for _ in range(n)] for _ in range(m)]
        b = [F(rng.randint(-4, 6)) for _ in range(m)]
        strict = [rng.random() < 0.3 for _ in range(m)]
        variables = sorted(rng.sample(range(n), rng.randint(1, n - 1)))
        points = [[F(rng.randint(-3, 3), rng.choice([1, 2])) for _ in range(n - len(variables))] for _ in range(5)]
        results = []
        for impl in IMPLEMENTATIONS:
            system = sx(impl).eliminate_variables(A, b, variables, strict)
            results.append((system.A, system.b, system.strict, system.is_feasible(),
                            [system.contains(p) for p in points], repr(system)))
        assert results[0] == results[1], (A, b, strict, variables)
