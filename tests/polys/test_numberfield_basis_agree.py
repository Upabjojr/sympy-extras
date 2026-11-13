"""The Rust port of ``numberfields.basis`` (integral bases and field
discriminants by the Round 2 algorithm) against the Python module: the
docstring examples on both implementations, and random irreducible and
square-free polynomials of degree up to five (non-monic ones included,
with square factors in their discriminants)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def nb(impl: str) -> Any:
    return load(impl + ".polys.numberfields.basis")


def test_known(impl: str) -> None:
    m = nb(impl)
    assert m.integral_basis([1, 0, -5]) == [[F(1)], [F(1, 2), F(1, 2)]]
    assert (m.field_discriminant([1, 0, -5]), m.field_discriminant([1, 0, 1]), m.field_discriminant([1, 0, 0, -2])) == (
        5, -4, -108)
    assert m.integral_basis([1, -1, -2, -8]) == [[F(1)], [F(1), F(0)], [F(1, 2), F(1, 2), F(0)]]
    assert m.field_discriminant([1, -1, -2, -8]) == -503
    assert m.integral_basis([1, 0, 3]) == [[F(1)], [F(1, 2), F(1, 2)]]
    assert m.integral_basis([4, 0, -5]) == [[F(1)], [F(1), F(1, 2)]]
    assert (m.field_discriminant([1, 1, 1, 1, 1]), m.field_discriminant([1, 0, -2]), m.field_discriminant([1, 0, -1, -1])) \
        == (125, 8, -23)
    assert (m.polynomial_discriminant([1, 0, -5]), m.polynomial_discriminant([1, 0, 0, -2]),
            m.polynomial_discriminant([2, 0, 1])) == (20, -108, -8)
    with pytest.raises(ValueError):
        m.integral_basis([1, -2, 1])
    with pytest.raises(ValueError):
        m.field_discriminant([3])


def outcome(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except ValueError as e:
        return ('ValueError', str(e))


@pytest.mark.parametrize("seed", range(20))
def test_round2_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("round 2", seed))
    for _ in range(4):
        n = rng.randint(2, 5)
        f = [rng.choice([1, 1, 1, 2, 3, -1])] + [rng.choice([0, 0, 1, -1, 2, -3, 4, 8, -12]) for _ in range(n)]
        if f[-1] == 0:
            f[-1] = rng.choice([2, -6, 12])
        results = [(outcome(nb(impl).polynomial_discriminant, f), outcome(nb(impl).integral_basis, f),
                    outcome(nb(impl).field_discriminant, f)) for impl in IMPLEMENTATIONS]
        assert results[0] == results[1], f
