"""The Rust port of ``lattice`` (LLL, Hermite and Smith normal forms,
integer linear systems) and of van Hoeij's recombination in
``factortools`` against the Python modules: the docstring examples on
both implementations, random lattices (dependent generators, other
parameters, large entries) and systems, and the factorisations of
Swinnerton-Dyer polynomials, their products and random polynomials with
many modular factors."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def outcome(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except (ValueError, ArithmeticError) as e:
        return type(e).__name__


def test_known(impl: str) -> None:
    la = mod(impl, "lattice")
    assert la.lll([[1, 0, 3], [0, 1, 5], [0, 0, 7]]) == [[-1, -1, -1], [-1, 2, 0], [-2, 0, 1]]
    assert la.lll([[2, 0], [0, 2], [1, 1]]) == [[1, 1], [-1, 1]]
    assert la.hermite_normal_form([[2, 0], [0, 2], [1, 1]]) == [[1, 1], [0, 2]]
    assert la.hermite_normal_form([[1, 2, 3], [2, 4, 6]]) == [[1, 2, 3]]
    reduction = la.lll_reduction([[1, 0, 1000], [0, 1, 1001]])
    assert reduction.basis == [[-1, 1, 1], [334, -333, 667]]
    assert reduction.norms_squared() == [3, F(reduction.gram[2], 3)]
    reduction = la.lll_reduction([[1, 1, 1], [-1, 0, 2], [3, 5, 6]])
    assert reduction.gram == [1, 1, 2, 9] and reduction.short_part(F(2)) == [[0, 1, 0], [1, 0, 1]]
    assert la.lll([[1, 0, 0, 1000000], [0, 1, 0, 1414214], [0, 0, 1, 2828427]])[0] == [0, -2, 1, -1]
    assert la.is_lll_reduced([[1, 0], [0, 1]]) and not la.is_lll_reduced([[1, 0], [5, 1]])
    assert la.integer_kernel([[2, 4, 6]], 3) == [[-1, -1, 1], [-2, 1, 0]]
    assert la.integer_kernel([], 2) == [[1, 0], [0, 1]]
    assert repr(la.solve_integer_system([[3, 5]], [1], 2)) == "IntegerSolutions([2, -1], [[-5, 3]])"
    assert la.solve_integer_system([[2, 4]], [1], 2) is None
    assert la.smith_normal_form([[2, 4, 4], [-6, 6, 12], [10, -4, -16]]) == [2, 6, 12]
    assert la.smith_normal_form([[0, 0]]) == []
    with pytest.raises(ValueError):
        la.lll([[1, 0]], F(1, 4))
    with pytest.raises(ValueError):
        la.hermite_normal_form([[1, 0], [1]])


def random_rows(rng: random.Random, n: int, m: int, bound: int) -> list[list[int]]:
    rows = [[rng.randint(-bound, bound) for _ in range(m)] for _ in range(n)]
    if n >= 2 and rng.random() < 0.3:
        # a dependent generator
        a, b = rng.randint(-3, 3), rng.randint(-3, 3)
        rows.append([a*x + b*y for x, y in zip(rows[0], rows[1])])
    return rows


def lattice_results(impl: str, rows: list[list[int]], delta: Fraction) -> Any:
    la = mod(impl, "lattice")
    reduction = outcome(la.lll_reduction, rows, delta)
    data = reduction if isinstance(reduction, str) else (
        reduction.basis, reduction.gram, reduction.norms_squared(), reduction.short_part(F(50)))
    return (data, outcome(la.hermite_normal_form, rows), outcome(la.smith_normal_form, rows),
            la.is_lll_reduced(rows, delta))


@pytest.mark.parametrize("seed", range(20))
def test_lattice_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("lattice", seed))
    for _ in range(6):
        n, m = rng.randint(0, 5), rng.randint(1, 6)
        bound = rng.choice([3, 20, 10**12])
        rows = random_rows(rng, n, m, bound)
        delta = rng.choice([F(3, 4), F(3, 4), F(99, 100), F(1), F(1, 3)])
        assert lattice_results("sympy_extras", rows, delta) == lattice_results("sympy_extras_rs", rows, delta), rows


def system_results(impl: str, matrix: list[list[int]], rhs: list[int], unknowns: int) -> Any:
    la = mod(impl, "lattice")
    solutions = outcome(la.solve_integer_system, matrix, rhs, unknowns)
    if solutions is not None and not isinstance(solutions, str):
        solutions = (solutions.particular, solutions.kernel)
    return outcome(la.integer_kernel, matrix, unknowns), solutions


@pytest.mark.parametrize("seed", range(20))
def test_integer_systems_agree(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("integer systems", seed))
    for _ in range(6):
        equations, unknowns = rng.randint(0, 3), rng.randint(1, 5)
        matrix = [[rng.randint(-6, 6) for _ in range(unknowns)] for _ in range(equations)]
        if rng.random() < 0.6:
            x = [rng.randint(-5, 5) for _ in range(unknowns)]
            rhs = [sum(a*b for a, b in zip(row, x)) for row in matrix]
        else:
            rhs = [rng.randint(-10, 10) for _ in range(equations)]
        results = [system_results(impl, matrix, rhs, unknowns) for impl in IMPLEMENTATIONS]
        assert results[0] == results[1], (matrix, rhs)


def swinnerton_dyer(n: int) -> list[int]:
    """The minimal polynomial of ``sqrt(2) + ... + sqrt(p_n)`` (as in
    ``sympy_extras/polys/tests/test_factor.py``)."""
    from sympy_extras.polys.densearith import dup_mul, dup_mul_ground, dup_sub
    from sympy_extras.polys.densebasic import dup_degree, dup_strip
    from sympy_extras.polys.domains import ZZ
    f = [1, 0]
    for p in (2, 3, 5, 7, 11, 13, 17)[:n]:
        d = dup_degree(f)
        even = [0]*(d + 1)
        odd = [0]*(d + 1)
        for i, c in enumerate(reversed(f)):
            binomial = 1
            for j in range(i, -1, -1):
                term = c*binomial*p**((i - j)//2)
                if (i - j) % 2:
                    odd[j] += term
                else:
                    even[j] += term
                binomial = binomial*j//(i - j + 1)
        A = dup_strip(even[::-1], ZZ)
        B = dup_strip(odd[::-1], ZZ)
        f = dup_sub(dup_mul(A, A, ZZ), dup_mul_ground(dup_mul(B, B, ZZ), p, ZZ), ZZ)
    return f


def factor_results(f: list[int]) -> list[Any]:
    results = []
    for impl in IMPLEMENTATIONS:
        ft, domains = mod(impl, "factortools"), mod(impl, "domains")
        results.append((ft.dup_zz_van_hoeij(f), ft.dup_zz_zassenhaus(f), ft.dup_zz_factor(f, domains.ZZ)))
    return results


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_swinnerton_dyer_agrees(rust: Any, n: int) -> None:
    f = swinnerton_dyer(n)
    results = factor_results(f)
    assert results[0] == results[1]
    assert results[1][0] == [f]


def test_products_agree(rust: Any) -> None:
    from sympy_extras.polys.densearith import dup_mul
    from sympy_extras.polys.densebasic import dup_degree
    from sympy_extras.polys.densetools import dup_primitive, dup_shift
    from sympy_extras.polys.domains import ZZ
    s3, s4, s5 = swinnerton_dyer(3), swinnerton_dyer(4), swinnerton_dyer(5)
    f = dup_mul(dup_mul(s4, dup_shift(s4, 1, ZZ), ZZ), dup_shift(s3, -2, ZZ), ZZ)
    results = factor_results(f)
    assert results[0] == results[1] and len(results[1][0]) == 3
    shifted = dup_shift(s5, 1, ZZ)
    scaled = dup_primitive([c*2**(dup_degree(shifted) - i) for i, c in enumerate(shifted)], ZZ)[1]
    results = factor_results(dup_mul(s5, scaled, ZZ))
    assert results[0] == results[1] and len(results[1][0]) == 2


@pytest.mark.parametrize("seed", range(10))
def test_random_van_hoeij_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.polys.densearith import dup_mul
    from sympy_extras.polys.densebasic import dup_degree
    from sympy_extras.polys.densetools import dup_primitive, dup_shift
    from sympy_extras.polys.domains import ZZ
    from sympy_extras.polys.sqfreetools import dup_sqf_p
    rng = random.Random(stable_seed("van hoeij", seed))
    for trial in range(9):
        f: list[int] = [1]
        kind = trial % 3
        for _ in range(rng.randint(8, 16) if kind == 0 else rng.randint(1, 4)):
            if kind == 0:
                g = [rng.randint(1, 3), rng.randint(-40, 40)]
            elif kind == 1:
                g = dup_shift(swinnerton_dyer(rng.randint(2, 3)), rng.randint(-3, 3), ZZ)
                scale = rng.choice([1, 2, 3])
                g = [c*scale**i for i, c in enumerate(g)]
            else:
                g = [rng.randint(1, 5)] + [rng.randint(-9, 9) for _ in range(rng.randint(1, 7))]
            f = dup_mul(f, g, ZZ)
        f = dup_primitive(f, ZZ)[1]
        if dup_degree(f) < 2 or not dup_sqf_p(f, ZZ):
            continue
        results = factor_results(f)
        assert results[0] == results[1], f
