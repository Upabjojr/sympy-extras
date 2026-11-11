"""The Rust port of ``solvers.exponential_diophantine`` (Pillai's
equation, the integer zeros of exponential polynomials) against the
Python module: the docstring examples on both implementations, random
equations of every kind (independent and dependent bases, with
coefficients, built from a known solution or not), and random exponential
polynomials of either sign of the variable."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def ed(impl: str) -> Any:
    return load(impl + ".solvers.exponential_diophantine")


def test_known(impl: str) -> None:
    m = ed(impl)
    assert m.pillai(3, 2, 1) == [(1, 1), (2, 3)]
    assert m.pillai(2, 3, 13) == [(4, 1), (8, 5)]
    assert m.pillai(2, 5, 3) == [(2, 0), (3, 1), (7, 3)]
    assert m.pillai(5, 7, 1) == []
    assert m.pillai(2, 3, -1) == [(1, 1), (3, 2)]
    assert m.pillai(2, 3, 5) == [(3, 1), (5, 3)]
    assert m.pillai(3, 2, -5) == [(1, 3), (3, 5)]
    assert m.pillai(4, 2, 12) == [(2, 2)]
    assert m.pillai(10, 3, 73) == [(2, 3)]
    assert m.pillai(5, 2, -3, s=2) == [(0, 1), (1, 2), (3, 6)]
    assert m.pillai(5, 11, 4, r=5) == [(0, 0), (2, 2)]
    one = [F(1)]
    assert m.exponential_polynomial_roots([([F(1)], F(2)), ([F(-1), F(0), F(0)], F(1))]) == [2, 4]
    assert m.exponential_polynomial_roots([(one, F(3)), (one, F(4)), ([F(-1)], F(5))]) == [2]
    assert m.exponential_polynomial_roots([([F(1)], F(2)), ([F(-1), F(-1)], F(1))]) == [0, 1]
    assert m.exponential_polynomial_roots([([F(1)], F(2)), ([F(1)], F(1, 2)), ([F(-5, 2)], F(1))]) == [-1, 1]
    assert m.exponential_polynomial_roots([([F(1)], F(2)), ([F(1)], F(3))]) == []
    for bad in [(1, 3, 1), (2, 3, 0)]:
        with pytest.raises(ValueError):
            m.pillai(*bad)
    with pytest.raises(ValueError):
        m.exponential_polynomial_roots([([F(1)], F(2)), ([F(-1)], F(2))])


def outcome(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except (ValueError, ArithmeticError) as e:
        return (type(e).__name__, str(e))


@pytest.mark.parametrize("seed", range(12))
def test_pillai_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("pillai", seed))
    for _ in range(4):
        a, b = rng.choice([2, 3, 5, 6, 7, 10, 4, 8, 9, 27]), rng.choice([2, 3, 5, 7, 11, 4, 9, 16])
        r, s = (1, 1) if rng.random() < 0.6 else (rng.randint(1, 6), rng.randint(1, 6))
        if rng.random() < 0.6:
            # from a solution, so that there is one
            x, y = rng.randint(0, 6), rng.randint(0, 6)
            c = r*a**x - s*b**y
        else:
            c = rng.choice([-1, 1]) * rng.randint(1, 200)
        if c == 0:
            c = 1
        results = [outcome(ed(impl).pillai, a, b, c, r=r, s=s) for impl in IMPLEMENTATIONS]
        assert results[0] == results[1], (a, b, c, r, s)


@pytest.mark.parametrize("seed", range(15))
def test_exponential_polynomials_agree(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("exponential polynomials", seed))
    for _ in range(4):
        terms = []
        for _ in range(rng.randint(1, 3)):
            p = [F(rng.randint(-4, 4), rng.choice([1, 2])) for _ in range(rng.randint(1, 3))]
            base = rng.choice([F(1), F(2), F(3), F(1, 2), F(5), F(2, 3), F(4)])
            terms.append((p, base))
        results = [outcome(ed(impl).exponential_polynomial_roots, terms) for impl in IMPLEMENTATIONS]
        assert results[0] == results[1], terms
