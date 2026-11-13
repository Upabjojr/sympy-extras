"""The Rust ports of ``regularchains.recursive`` (polynomials over ZZ in
their main variable: lazy pseudo-divisions, prem, subresultants, factors)
and ``regularchains.operations`` (squarefree regular chains:
regularize, intersect, triangularize, in both modes) against the Python
modules: random polynomials, and random systems in two and three
variables, the chains compared through their reprs and in order."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def ring(impl: str, names: list[str]) -> Any:
    return mod(impl, "rings").PolyRing(mod(impl, "domains").ZZ, names)


def random_terms(rng: random.Random, n: int, size: int) -> dict[tuple[int, ...], int]:
    terms: dict[tuple[int, ...], int] = {}
    for _ in range(rng.randint(1, size)):
        m = tuple(rng.randint(0, 2) for _ in range(n))
        terms[m] = terms.get(m, 0) + rng.choice([-3, -2, -1, 1, 1, 2, 4])
    return {m: c for m, c in terms.items() if c}


def outcome(function: Any, *args: Any) -> Any:
    try:
        value = function(*args)
    except (ValueError, ZeroDivisionError, ArithmeticError) as e:
        return (type(e).__name__, str(e))
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return str(value)


@pytest.mark.parametrize("seed", range(20))
def test_recursive_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("recursive", seed))
    n = rng.choice([2, 3])
    names = ['x', 'y', 'z'][:n]
    f_terms, g_terms = random_terms(rng, n, 4), random_terms(rng, n, 3)
    i = rng.randrange(n)
    results = []
    for impl in IMPLEMENTATIONS:
        rc = mod(impl, "regularchains.recursive")
        R = ring(impl, names)
        f, g = R.from_dict(f_terms) if f_terms else R.one, R.from_dict(g_terms) if g_terms else R.one
        h = f*g
        results.append([
            rc.main_variable(f), str(rc.initial(f)), str(rc.tail(f)), str(rc.normalize(f)),
            outcome(rc.primitive_part, h, i), outcome(rc.pseudo_divide, f, g, i), outcome(rc.pseudo_remainder, f, g, i),
            outcome(rc.prem, f, g, i), outcome(rc.subresultants, f, g, i), outcome(rc.subresultant_chain, h, g, i),
            outcome(rc.distinct_factors, h), str(rc.gcd(h, f)), outcome(rc.exact_quotient, h, g),
            outcome(rc.sqf_part, h), str(rc.sort_key(f)),
        ])
    assert results[0] == results[1], (f_terms, g_terms, i)


def random_equation(rng: random.Random, n: int) -> dict[tuple[int, ...], int]:
    """A polynomial with three to five terms and a constant term."""
    terms = random_terms(rng, n, 4)
    terms[tuple(0 for _ in range(n))] = rng.choice([-2, -1, 1, 3])
    return {m: c for m, c in terms.items() if c}


#: seed 12 is a zero-dimensional system in three variables which takes a
#: minute in Python (18 s in Rust)
@pytest.mark.parametrize("seed", [s for s in range(40) if s != 12])
def test_triangularize_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("triangularize", seed))
    n = rng.choice([2, 2, 3])
    names = ['x', 'y', 'z'][:n]
    systems = [random_equation(rng, n) for _ in range(rng.randint(1, n))]
    inequations = [random_equation(rng, n)] if rng.random() < 0.3 else []
    bound = rng.choice([None, len(systems)])
    probe = random_equation(rng, n)
    results = []
    for impl in IMPLEMENTATIONS:
        ops = mod(impl, "regularchains.operations")
        R = ring(impl, names)
        d = ops.Decomposer(R, bound)
        chains = d.triangularize([R.from_dict(t) for t in systems], [R.from_dict(t) for t in inequations])
        p = R.from_dict(probe)
        results.append(([repr(c) for c in chains], [[(repr(c), z) for c, z in d.regularize(p, chain)] for chain in chains],
                        [[repr(c) for c in d.intersect(p, chain)] for chain in chains]))
    assert results[0] == results[1], (systems, inequations, bound)
