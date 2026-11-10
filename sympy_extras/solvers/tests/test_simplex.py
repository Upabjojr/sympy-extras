"""Exact linear programming: every answer is checked through its
certificate (duality, Farkas' lemma, a ray), and small problems against
the vertices enumerated by brute force."""
from __future__ import annotations

import random
from fractions import Fraction
from itertools import combinations

from sympy.matrices.dense import Matrix
from sympy.testing.pytest import raises

from sympy_extras.solvers.simplex import LinearProgramResult, linear_feasible, linear_program


def _dot(u: list[Fraction], v: list[Fraction]) -> Fraction:
    return sum((a*c for a, c in zip(u, v)), Fraction(0))


def _check(c: list[Fraction], A: list[list[Fraction]], b: list[Fraction], result: LinearProgramResult) -> None:
    """The certificate of the result proves its status."""
    n = len(c)
    columns = [[row[j] for row in A] for j in range(n)]
    if result.status == 'optimal':
        assert result.point is not None and result.dual is not None and result.value is not None
        assert all(_dot(row, result.point) <= bound for row, bound in zip(A, b))
        assert all(y >= 0 for y in result.dual)
        assert [_dot(result.dual, column) for column in columns] == c
        assert _dot(c, result.point) == result.value == _dot(result.dual, b)
    elif result.status == 'infeasible':
        assert result.dual is not None
        assert all(y >= 0 for y in result.dual)
        assert all(_dot(result.dual, column) == 0 for column in columns)
        assert _dot(result.dual, b) < 0
    else:
        assert result.status == 'unbounded' and result.point is not None and result.ray is not None
        assert all(_dot(row, result.point) <= bound for row, bound in zip(A, b))
        assert all(_dot(row, result.ray) <= 0 for row in A)
        assert _dot(c, result.ray) > 0


def test_textbook_problems() -> None:
    # maximise 3 x + 2 y on x + y <= 4, x <= 3, y <= 3, x, y >= 0: 11 at (3, 1)
    result = linear_program([3, 2], [[1, 1], [1, 0], [0, 1], [-1, 0], [0, -1]], [4, 3, 3, 0, 0])
    assert (result.status, result.value, result.point) == ('optimal', 11, [3, 1])
    assert result.dual == [2, 1, 0, 0, 0]
    # Chvatal's first example: maximise 5 x1 + 4 x2 + 3 x3: 13 at (2, 0, 1)
    result = linear_program([5, 4, 3], [[2, 3, 1], [4, 1, 2], [3, 4, 2], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                            [5, 11, 8, 0, 0, 0])
    assert (result.status, result.value, result.point) == ('optimal', 13, [2, 0, 1])
    # a degenerate problem on which the largest-coefficient rule cycles (Beale)
    F = Fraction
    A = [[F(1, 4), F(-8), F(-1), F(9)], [F(1, 2), F(-12), F(-1, 2), F(3)], [F(0), F(0), F(1), F(0)],
         [F(-1), F(0), F(0), F(0)], [F(0), F(-1), F(0), F(0)], [F(0), F(0), F(-1), F(0)], [F(0), F(0), F(0), F(-1)]]
    result = linear_program([F(3, 4), F(-20), F(1, 2), F(-6)], A, [0, 0, 1, 0, 0, 0, 0])
    assert result.status == 'optimal' and result.value == Fraction(5, 4)
    assert linear_program([1], [[1], [-1]], [0, -1]).status == 'infeasible'
    assert linear_program([1, 0], [[0, 1]], [1]).status == 'unbounded'
    # no constraint, no unknown
    assert linear_program([0], [], []).status == 'optimal'
    assert linear_program([1], [], []).status == 'unbounded'
    assert linear_program([], [[], []], [1, 0]).value == 0
    assert linear_program([], [[]], [-1]).status == 'infeasible'
    assert repr(linear_program([1], [[1]], [2])) == "LinearProgramResult('optimal', value=2)"
    raises(ValueError, lambda: linear_program([1], [[1, 2]], [1]))
    raises(ValueError, lambda: linear_program([1], [[1]], [1, 2]))


def test_certificates_of_random_programs() -> None:
    rng = random.Random(81)
    seen = {'optimal': 0, 'infeasible': 0, 'unbounded': 0}
    for _ in range(400):
        n, m = rng.randint(1, 4), rng.randint(1, 7)
        A = [[Fraction(rng.randint(-4, 4), rng.choice([1, 1, 2])) for _ in range(n)] for _ in range(m)]
        b = [Fraction(rng.randint(-5, 8)) for _ in range(m)]
        c = [Fraction(rng.randint(-3, 3)) for _ in range(n)]
        result = linear_program(c, A, b)
        _check(c, A, b, result)
        seen[result.status] += 1
    assert min(seen.values()) > 20


def test_against_vertex_enumeration() -> None:
    rng = random.Random(82)
    compared = 0
    for _ in range(150):
        n = rng.randint(1, 3)
        m = rng.randint(n + 1, 6)
        # a box keeps the problem bounded
        A = [[Fraction(rng.randint(-4, 4)) for _ in range(n)] for _ in range(m)]
        b = [Fraction(rng.randint(-3, 8)) for _ in range(m)]
        for j in range(n):
            A.append([Fraction(1 if k == j else 0) for k in range(n)])
            b.append(Fraction(10))
            A.append([Fraction(-1 if k == j else 0) for k in range(n)])
            b.append(Fraction(10))
        c = [Fraction(rng.randint(-3, 3)) for _ in range(n)]
        best = None
        for rows in combinations(range(len(A)), n):
            square = Matrix([[A[i][j] for j in range(n)] for i in rows])
            if square.det() == 0:
                continue
            solution = square.LUsolve(Matrix([b[i] for i in rows]))
            vertex = [Fraction(int(v.p), int(v.q)) for v in solution]
            if all(_dot(row, vertex) <= bound for row, bound in zip(A, b)):
                value = _dot(c, vertex)
                best = value if best is None or value > best else best
        result = linear_program(c, A, b)
        if best is None:
            assert result.status == 'infeasible'
        else:
            assert result.status == 'optimal' and result.value == best
            compared += 1
    assert compared > 40


def test_strict_inequalities() -> None:
    assert linear_feasible([[1], [-1]], [1, -1]) == [1]
    assert linear_feasible([[1], [-1]], [1, -1], [True, False]) is None
    assert linear_feasible([[1], [-1]], [0, -1], [True, True]) is None             # x < 0, x > 1
    point = linear_feasible([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], [True, True, True])
    assert point is not None and point[0] > 0 and point[1] > 0 and point[0] + point[1] < 1
    assert linear_feasible([[1, 1], [-1, -1]], [1, -1], [True, False]) is None
    assert linear_feasible([], []) == []
    raises(ValueError, lambda: linear_feasible([[1]], [1], [True, False]))
    rng = random.Random(83)
    for _ in range(200):
        n, m = rng.randint(1, 3), rng.randint(1, 6)
        A = [[Fraction(rng.randint(-3, 3)) for _ in range(n)] for _ in range(m)]
        b = [Fraction(rng.randint(-3, 5)) for _ in range(m)]
        strict = [rng.random() < 0.5 for _ in range(m)]
        point = linear_feasible(A, b, strict)
        if point is not None:
            for row, bound, flag in zip(A, b, strict):
                value = _dot(row, point)
                assert value < bound if flag else value <= bound
        else:
            # no point of a fine grid either
            grid = [Fraction(k, 2) for k in range(-12, 13)]
            points = [[x] for x in grid] if n == 1 else None
            if points is not None:
                for candidate in points:
                    assert not all((_dot(row, candidate) < bound) if flag else (_dot(row, candidate) <= bound)
                                   for row, bound, flag in zip(A, b, strict))


def test_integer_programs_known() -> None:
    from sympy_extras.solvers.simplex import integer_program
    # a knapsack: maximise 10 a + 13 b + 7 c with 3 a + 4 b + 2 c <= 9, 0 <= a, b, c <= 1... relaxed to counts <= 3
    weights = [[3, 4, 2], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = integer_program([10, 13, 7], weights, [9, 0, 0, 0, 3, 3, 3])
    best = max((10*a + 13*b + 7*c, [a, b, c]) for a in range(4) for b in range(4) for c in range(4)
               if 3*a + 4*b + 2*c <= 9)
    assert result.status == 'optimal' and result.value == best[0]
    assert result.point is not None and 10*result.point[0] + 13*result.point[1] + 7*result.point[2] == best[0]
    # the relaxation has the optimum 21 at (3, 3/2); the integer optimum is 20
    assert integer_program([5, 4], [[6, 4], [1, 2], [-1, 0], [0, -1]], [24, 6, 0, 0]).value == 20
    # no integer point: a hyperplane x + y = 1/2, an unbounded strip 1/3 <= x - y <= 2/3
    assert integer_program([1, 1], [[2, 2], [-2, -2]], [1, -1]).status == 'infeasible'
    assert integer_program([0, 0], [[3, -3], [-3, 3]], [2, -1]).status == 'infeasible'
    assert integer_program([1, 1], [[3, -3], [-3, 3]], [2, -1]).status == 'infeasible'       # unbounded relaxation
    unbounded = integer_program([1, 0], [[0, 1], [0, -1]], [Fraction(5, 2), 0])
    assert unbounded.status == 'unbounded' and unbounded.point is not None and 0 <= unbounded.point[1] <= 2
    assert integer_program([1], [[1], [-1]], [0, -1]).status == 'infeasible'
    # a thin triangle with a single integer point far from the vertex of the relaxation
    thin = integer_program([0, 1], [[-10, 21], [10, -19], [0, -1]], [0, 0, 0])
    assert thin.status == 'optimal' and thin.point == [0, 0]
    assert repr(thin) == "IntegerProgramResult('optimal', value=0, point=[0, 0])"


def test_integer_programs_against_search() -> None:
    from sympy_extras.solvers.simplex import integer_program
    rng = random.Random(84)
    solved = 0
    for _ in range(120):
        n = rng.randint(1, 3)
        m = rng.randint(1, 4)
        A = [[Fraction(rng.randint(-4, 4)) for _ in range(n)] for _ in range(m)]
        b = [Fraction(rng.randint(-6, 12), rng.choice([1, 2, 3])) for _ in range(m)]
        # a box, so that the search below is complete
        for j in range(n):
            A.append([Fraction(1 if k == j else 0) for k in range(n)])
            b.append(Fraction(5))
            A.append([Fraction(-1 if k == j else 0) for k in range(n)])
            b.append(Fraction(5))
        c = [Fraction(rng.randint(-3, 3)) for _ in range(n)]
        best = None
        point = [-5]*n
        while True:
            vector = [Fraction(v) for v in point]
            if all(_dot(row, vector) <= bound for row, bound in zip(A, b)):
                value = _dot(c, vector)
                best = value if best is None or value > best else best
            k = 0
            while k < n and point[k] == 5:
                point[k] = -5
                k += 1
            if k == n:
                break
            point[k] += 1
        result = integer_program(c, A, b)
        if best is None:
            assert result.status == 'infeasible'
        else:
            assert result.status == 'optimal' and result.value == best
            assert result.point is not None
            found = [Fraction(v) for v in result.point]
            assert all(_dot(row, found) <= bound for row, bound in zip(A, b)) and _dot(c, found) == best
            solved += 1
    assert solved > 60
