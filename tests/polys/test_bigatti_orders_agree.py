"""The Rust ports of ``bigatti`` (the Hilbert numerator of monomial
ideals, split out of ``hilbert``) and of the orders now defined in
``rings`` (``as_order``, ``WeightOrder``, ``BlockOrder``,
``elimination_order``) against the Python modules: the docstring
examples, random monomial ideals, and the comparisons of random
monomials under random orders."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_known(impl: str) -> None:
    b = mod(impl, "bigatti")
    assert b.minimal_generators([(2, 1), (1, 0), (0, 3), (1, 3)]) == [(1, 0), (0, 3)]
    assert b.hilbert_numerator_coefficients([(2, 0), (0, 2)]) == [1, 0, -2, 0, 1]
    assert b.hilbert_numerator_coefficients([(1, 1, 0), (0, 1, 1)]) == [1, 0, -2, 1]
    assert b.hilbert_numerator_coefficients([]) == [1]
    assert b.hilbert_numerator_coefficients([(0, 0)]) == []
    r = mod(impl, "rings")
    assert str(r.elimination_order(1, 3)) == "BlockOrder((grevlex, 1), (grevlex, 2))"
    assert str(r.elimination_order(0, 3)) == "grevlex"
    assert str(r.WeightOrder((1, 3))) == "WeightOrder((1, 3), lex)"
    assert str(r.as_order("grlex")) == "grlex"
    with pytest.raises(ValueError):
        r.as_order("nonsense")
    with pytest.raises(ValueError):
        r.WeightOrder((1, -1))
    with pytest.raises(ValueError):
        r.BlockOrder([(r.lex, 0)])


@pytest.mark.parametrize("seed", range(20))
def test_bigatti_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("bigatti", seed))
    for _ in range(10):
        n = rng.randint(1, 5)
        gens = [tuple(rng.choice([0, 0, 1, 2, 3, 5]) for _ in range(n)) for _ in range(rng.randint(0, 8))]
        results = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            b = mod(impl, "bigatti")
            results.append((b.minimal_generators(gens), b.hilbert_numerator_coefficients(gens)))
        assert results[0] == results[1], gens


def random_order(rng: random.Random, impl: str, nvars: int, depth: int = 0) -> Any:
    r = mod(impl, "rings")
    kind = rng.choice(["named", "named", "weight", "block", "elimination"] if depth < 2 else ["named"])
    if kind == "named":
        return r.as_order(rng.choice(["lex", "grlex", "grevlex"]))
    if kind == "weight":
        return r.WeightOrder(tuple(rng.randint(0, 4) for _ in range(nvars)), random_order(rng, impl, nvars, depth + 1))
    if kind == "elimination" or nvars < 2:
        return r.elimination_order(rng.randint(0, nvars), nvars, random_order(rng, impl, nvars, depth + 1))
    first = rng.randint(1, nvars - 1)
    return r.BlockOrder([(random_order(rng, impl, first, depth + 1), first),
                         (random_order(rng, impl, nvars - first, depth + 1), nvars - first)])


def comparison(impl: str, order: Any, a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if impl == "sympy_extras_rs":
        return int(order.compare(list(a), list(b)))
    ka, kb = order(a), order(b)
    return (ka > kb) - (ka < kb)


@pytest.mark.parametrize("seed", range(20))
def test_orders_agree(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("orders", seed))
    for _ in range(5):
        nvars = rng.randint(1, 4)
        state = rng.getstate()
        orders = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            rng.setstate(state)
            orders.append(random_order(rng, impl, nvars))
        assert str(orders[0]) == str(orders[1])
        for _ in range(20):
            a = tuple(rng.randint(0, 3) for _ in range(nvars))
            b = tuple(rng.randint(0, 3) for _ in range(nvars))
            assert comparison("sympy_extras", orders[0], a, b) == comparison("sympy_extras_rs", orders[1], a, b), (
                str(orders[0]), a, b)
