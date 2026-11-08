"""The Rust port of ``multifactor`` (Wang's EEZ factorisation over ZZ)
against the Python package: the docstring example on both implementations,
and the factorisations of random products with repeated factors in 2-3
variables (the factors sorted as SymPy sorts them, so equal whatever the
evaluation points)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_known(impl: str) -> None:
    d, rings, m = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "multifactor")
    R = rings.PolyRing(d.ZZ, ['x', 'y'])
    x, y = R.gens
    f = (x**3*y).mul_ground(2) - (x*y**3).mul_ground(2) + (x*x).mul_ground(4) - (y*y).mul_ground(4)
    c, factors = m.zz_factor(f)
    assert (c, [(str(g), k) for g, k in factors]) == (2, [('x - y', 1), ('x + y', 1), ('x*y + 2', 1)])


def random_factor(rng: random.Random, R: Any) -> Any:
    p = R.zero
    while not p or p.is_ground:
        p = R.zero
        for _ in range(rng.randint(2, 3)):
            m = R.one
            for x in R.gens:
                m = m * x**rng.randint(0, 2)
            p = p + m.mul_ground(rng.choice([-3, -2, -1, 1, 1, 2, 3, 5]))
    return p


@pytest.mark.parametrize("nvars", [2, 3])
@pytest.mark.parametrize("seed", range(12))
def test_zz_factor_agrees(rust: Any, nvars: int, seed: int) -> None:
    names = ['x', 'y', 'z'][:nvars]
    rng = random.Random(seed * 10 + nvars)
    for _ in range(3):
        state = rng.getstate()
        results = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            rng.setstate(state)
            d, rings, m = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "multifactor")
            R = rings.PolyRing(d.ZZ, names)
            f = R.from_int(rng.choice([1, -1, 2, -6]))
            for _ in range(rng.randint(1, 3)):
                f = f * random_factor(rng, R)**rng.choice([1, 1, 2])
            c, factors = m.zz_factor(f)
            results.append((c, [(dict(g.terms), k) for g, k in factors], dict(m.sqf_part(f).terms),
                            [m.factor_sort_key(g) for g, _ in factors]))
        assert results[0] == results[1], str(f)
