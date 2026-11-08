"""The Rust port of ``multisqf`` against the Python package: the docstring
example on both implementations, and the square-free decompositions of
random products with repeated factors over ZZ, QQ, ZZ[a] and QQ[a]."""

from __future__ import annotations

import importlib
import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load


def mod(impl: str, name: str) -> Any:
    load(impl)
    return importlib.import_module(impl + ".polys." + name)


def test_known(impl: str) -> None:
    m = mod(impl, "multisqf")
    D = m.IntegerSqfDomain(['x', 'y'])
    x, y = D.ring.gens
    f = (x + y)**2 * (x - y)**3 * (y + D.ring.one)**2
    c, factors = m.sqf_list(f.mul_ground(2), D)
    assert str(c) == '2'
    assert [(str(g), k) for g, k in factors] == [('x*y + x + y**2 + y', 2), ('x - y', 3)]


def make(impl: str, kind: str, names: list[str]) -> Any:
    m = mod(impl, "multisqf")
    if kind == "ZZ":
        return m.IntegerSqfDomain(names)
    if kind == "QQ":
        return m.RationalSqfDomain(names)
    s = mod(impl, "domains.sparsepolynomialring")
    params = s.IntegerPolynomialRing(['a']) if kind == "ZZ[a]" else s.RationalPolynomialRing(['a'])
    return m.ParameterSqfDomain(params, names)


@pytest.mark.parametrize("kind", ["ZZ", "QQ", "ZZ[a]", "QQ[a]"])
@pytest.mark.parametrize("nvars", [1, 2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_sqf_list_agrees(rust: Any, kind: str, nvars: int, seed: int) -> None:
    names = ['x', 'y', 'z'][:nvars]
    rng = random.Random(seed * 10 + nvars)
    nall = nvars + (1 if kind in ("ZZ[a]", "QQ[a]") else 0)
    for _ in range(4):
        factors = []
        for _ in range(rng.randint(1, 3)):
            terms = {}
            for _ in range(rng.randint(1, 3)):
                c: Any = rng.randint(-3, 3)
                if kind.startswith("QQ"):
                    c = Fraction(c, rng.randint(1, 2))
                terms[tuple(rng.randint(0, 1) for _ in range(nall))] = c
            factors.append((terms, rng.randint(1, 3)))
        constant = rng.choice([1, 2, -3, 6])
        all_flag = rng.random() < 0.3
        results = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            D = make(impl, kind, names)
            R = D.ring
            f = R.from_int(constant)
            for terms, k in factors:
                f = f * R.from_dict(terms)**k
            coeff, sqf = mod(impl, "multisqf").sqf_list(f, D, all=all_flag)
            results.append((dict(coeff.terms), [(dict(g.terms), k) for g, k in sqf]))
        assert results[0] == results[1], (kind, factors, constant)
