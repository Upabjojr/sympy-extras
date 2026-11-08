"""The Rust port of ``domains/sparsepolynomialring`` against the Python
package: SymPy's gcd and lcm conventions of the four ring flavours on
random products sharing factors."""

from __future__ import annotations

import importlib
import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load


def sparse(impl: str) -> Any:
    load(impl)
    return importlib.import_module(impl + ".polys.domains.sparsepolynomialring")


def domains(impl: str) -> Any:
    return importlib.import_module(impl + ".polys.domains")


def coefficient(rng: random.Random, kind: str) -> Any:
    if kind == "ZZ":
        return rng.randint(-4, 4)
    if kind == "QQ":
        return Fraction(rng.randint(-4, 4), rng.randint(1, 3))
    return rng.randrange(5)


def terms(rng: random.Random, kind: str, nvars: int, size: int = 4) -> dict[tuple[int, ...], Any]:
    result: dict[tuple[int, ...], Any] = {}
    for _ in range(rng.randint(1, size)):
        result[tuple(rng.randint(0, 2) for _ in range(nvars))] = coefficient(rng, kind)
    return result


def make(impl: str, flavour: str, kind: str, names: list[str], gens: list[str]) -> Any:
    s, d = sparse(impl), domains(impl)
    ground = d.GF(5) if kind == "GF(5)" else getattr(d, kind)
    if flavour == "plain":
        return s.SparsePolynomialRing(ground, names)
    if flavour == "sympy":
        return s.IntegerPolynomialRing(names) if kind == "ZZ" else s.RationalPolynomialRing(names)
    params = s.IntegerPolynomialRing(names) if kind == "ZZ" else s.RationalPolynomialRing(names)
    return s.IteratedPolynomialRing(params, gens)


def outcome(call: Any) -> Any:
    try:
        return ("ok", call())
    except (ValueError, ZeroDivisionError) as e:
        return ("raises", type(e).__name__)


# the gcd needs an ordered ground domain (signs of the leading coefficients):
# GF(p) is not supported, in Python either
CASES = [("plain", "ZZ"), ("plain", "QQ"), ("sympy", "ZZ"), ("sympy", "QQ"),
         ("iterated", "ZZ"), ("iterated", "QQ")]


@pytest.mark.parametrize("flavour, kind", CASES)
@pytest.mark.parametrize("shape", [(["y"], ["x"]), (["y", "z"], ["x"]), (["z"], ["x", "y"])])
@pytest.mark.parametrize("seed", range(4))
def test_sparse_ring_agrees(rust: Any, flavour: str, kind: str, shape: tuple[list[str], list[str]],
                            seed: int) -> None:
    names, gens = shape
    nvars = len(names) + (len(gens) if flavour == "iterated" else 0)
    rings = [make(impl, flavour, kind, names, gens) for impl in ("sympy_extras", "sympy_extras_rs")]
    assert rings[0].name == rings[1].name
    rng = random.Random(seed * 100 + nvars)
    for _ in range(12):
        common, a, b = terms(rng, kind, nvars, 3), terms(rng, kind, nvars), terms(rng, kind, nvars)
        zero_a = rng.random() < 0.1
        results = []
        for R in rings:
            h = R.ring.from_dict(common)
            f = R.ring.zero if zero_a else R.ring.from_dict(a) * h
            g = R.ring.from_dict(b) * h
            row = [dict(R.gcd(f, g).terms), dict(R.lcm(f, g).terms), dict(R.gcd(g, R.ring.zero).terms),
                   dict(R.canonical_unit(g).terms), R.is_negative(g), R.is_unit(g),
                   outcome(lambda: dict(R.exquo(f, g).terms)),
                   outcome(lambda: dict(R.exquo(g * h, h).terms)) if h else None]
            if flavour == "iterated":
                row += [dict(R.ground_LC(g).terms), dict(R.ground_content(g).terms)]
            else:
                c, p = R.primitive(g)
                row += [c, dict(p.terms)]
            results.append(row)
        assert results[0] == results[1], (common, a, b)
