"""The Rust port of ``algebraicfactor`` (Trager's factorisation and gcds
of multivariate polynomials over a number field) against the Python
module: the docstring example, and the factorisations and gcds of random
products over QQ(sqrt(d)) and QQ(2**(1/3)) in one and two variables, with
repeated and monomial factors."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def field(impl: str, minpoly: list[Fraction]) -> Any:
    from sympy_extras.core import Symbol
    return mod(impl, "domains.algebraicfield").AlgebraicField(minpoly, Symbol('alpha'))


def test_known(impl: str) -> None:
    K = field(impl, [F(1), F(0), F(-2)])
    af = mod(impl, "algebraicfactor")
    R = af.AlgebraicPolynomialRing(K, ['x', 'y'])
    x, y = R.ring.gens
    s = R.ring.ground(K.alpha)
    assert R.gcd((x - s*y)*(x + y), (x - s*y)*(x - y)) == x - s*y
    lc, factors = af.ext_factor_list(x*x - (y*y).mul_ground(K.from_int(2)))
    assert lc == K.one and sorted(k for _, k in factors) == [1, 1]
    assert af.ext_gcd(R.ring.zero, x*x - y).LC == K.one
    with pytest.raises(TypeError):
        af.ext_factor_list(mod(impl, "rings").PolyRing(mod(impl, "domains").QQ, ['x']).gens[0])


FIELDS = [[F(1), F(0), F(-2)], [F(1), F(0), F(-3)], [F(1), F(0), F(1)], [F(1), F(0), F(0), F(-2)]]


def random_factor(rng: random.Random, n: int, degree: int) -> dict[tuple[int, ...], list[Fraction]]:
    terms: dict[tuple[int, ...], list[Fraction]] = {}
    while not any(any(m) for m in terms):
        terms = {}
        for _ in range(rng.randint(2, 3)):
            m = tuple(rng.randint(0, 2) for _ in range(n))
            c = [F(rng.randint(-2, 2)) for _ in range(degree)]
            while c and c[0] == 0:
                c = c[1:]
            if c:
                terms[m] = c
    return terms


@pytest.mark.parametrize("seed", range(30))
def test_ext_factor_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("algebraic factor", seed))
    minpoly = rng.choice(FIELDS)
    n = rng.choice([1, 2, 2])
    degree = len(minpoly) - 1
    parts = [(random_factor(rng, n, degree), rng.choice([1, 1, 2])) for _ in range(rng.randint(1, 2))]
    monomial = tuple(rng.choice([0, 0, 1]) for _ in range(n))
    other = random_factor(rng, n, degree)
    results = []
    for impl in IMPLEMENTATIONS:
        K = field(impl, minpoly)
        R = mod(impl, "rings").PolyRing(K, ['x', 'y'][:n])
        f = R.term(monomial, K.one)
        for terms, k in parts:
            g = R.from_dict(terms)
            for _ in range(k):
                f = f*g
        h = R.from_dict(other)*R.from_dict(parts[0][0])
        af = mod(impl, "algebraicfactor")
        lc, factors = af.ext_factor_list(f)
        results.append((list(lc), [(dict(g.terms), k) for g, k in factors], dict(af.ext_gcd(f, h).terms)))
    assert results[0] == results[1], (minpoly, parts, monomial)
