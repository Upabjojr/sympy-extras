"""The Rust port of ``gaussianfactor`` (Trager's factorisation over
``ZZ_I`` and ``QQ_I``) and of ``Domain.nullspace`` against the Python
modules: the docstring examples on both implementations, the
factorisations of random products of Gaussian polynomials in two and three
variables (coefficient, factors, multiplicities, order and sort keys), and
the null spaces of random matrices over ``ZZ``, ``QQ``, ``GF(p)`` and
``QQ_I``."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction

#: a random polynomial as plain data: monomial -> (real part, imaginary part)
Terms = dict[tuple[int, ...], tuple[Fraction, Fraction]]


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def plain(c: Any) -> Any:
    """A coefficient as comparable plain data."""
    if hasattr(c, "x") and hasattr(c, "y"):
        return (F(c.x), F(c.y))
    return F(c)


def gaussian(impl: str, c: tuple[Fraction, Fraction]) -> Any:
    return mod(impl, "domains.gaussiandomains").GaussianRational(c[0], c[1])


def build(impl: str, names: list[str], factors: list[tuple[Terms, int]]) -> Any:
    """The product of the factors (with exponents) in ``QQ_I[names]``."""
    g = mod(impl, "domains.gaussiandomains")
    R = mod(impl, "rings").PolyRing(g.QQ_I, names)
    f = R.one
    for terms, k in factors:
        f = f * R.from_dict({m: gaussian(impl, c) for m, c in terms.items()})**k
    return f


def result(impl: str, outcome: tuple[Any, list[tuple[Any, int]]]) -> Any:
    gf = mod(impl, "gaussianfactor")
    c, factors = outcome
    return (plain(c),
            [({m: plain(v) for m, v in g.terms.items()}, k) for g, k in factors],
            [gf.gaussian_sort_key(item) for item in factors])


def test_known(impl: str) -> None:
    g, rings, gf = mod(impl, "domains.gaussiandomains"), mod(impl, "rings"), mod(impl, "gaussianfactor")
    R = rings.PolyRing(g.QQ_I, ['x', 'y'])
    x, y = R.gens
    i = R.ground(g.GaussianRational(F(0), F(1)))
    lc, factors = gf.gaussian_ext_factor(x*x + y*y)
    assert plain(lc) == (1, 0)
    assert [(dict(f.terms), k) for f, k in factors] == [
        ({(1, 0): g.GaussianRational(F(1), F(0)), (0, 1): g.GaussianRational(F(0), F(-1))}, 1),
        ({(1, 0): g.GaussianRational(F(1), F(0)), (0, 1): g.GaussianRational(F(0), F(1))}, 1)]
    _, factors = gf.gaussian_ext_factor((x*x + y*y)*(x - y)*(x - y))
    assert [k for _, k in factors] == [1, 1, 2] and factors[2][0] == x - y
    two = R.ground(g.GaussianRational(F(2)))
    c, factors = gf.gaussian_factor_list(two*x*(x*x + y*y), True)
    assert plain(c) == (2, 0) and factors[0] == (x, 1) and len(factors) == 3
    f = (x*x + y*y)*(x*y - i)*(x*y - i)
    lc, factors = gf.gaussian_ext_factor(f)
    product = R.one
    for h, k in factors:
        product = product * h**k
    assert product == f and all(h.is_monic for h, _ in factors)
    assert gf.gaussian_factor_list(R.zero, False) == (g.QQ_I.zero, [])


def random_factor(rng: random.Random, nvars: int, rational: bool) -> Terms:
    """A random polynomial with 2-3 terms and small Gaussian coefficients
    (as in ``sympy_extras/polys/tests/test_gaussianfactor.py``)."""
    while True:
        terms: Terms = {}
        for _ in range(rng.randint(2, 3)):
            c = (F(rng.randint(-3, 3)), F(rng.randint(-3, 3)))
            if rational and rng.random() < 0.3:
                d = rng.choice([2, 3])
                c = (c[0] / d, c[1] / d)
            m = tuple(rng.randint(0, 2) for _ in range(nvars))
            old = terms.get(m, (F(0), F(0)))
            terms[m] = (old[0] + c[0], old[1] + c[1])
        terms = {m: c for m, c in terms.items() if c != (F(0), F(0))}
        if any(any(m) for m in terms):
            return terms


@pytest.mark.parametrize("seed", range(16))
def test_factor_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("gaussianfactor", seed))
    for _ in range(4):
        nvars = rng.choice([2, 2, 3])
        names = ['x', 'y', 'z'][:nvars]
        rational = rng.random() < 0.3
        factors = [(random_factor(rng, nvars, rational), rng.choice([1, 1, 2]))]
        for _ in range(rng.randint(0, 2 if nvars == 2 else 1)):
            factors.append((random_factor(rng, nvars, rational), 1))
        results = []
        for impl in IMPLEMENTATIONS:
            gf = mod(impl, "gaussianfactor")
            f = build(impl, names, factors)
            outcome = [result(impl, gf.gaussian_ext_factor(f)), result(impl, gf.gaussian_factor_list(f, False))]
            if not rational:
                outcome.append(result(impl, gf.gaussian_factor_list(f, True)))
            results.append(outcome)
        assert results[0] == results[1], factors


def test_integral_needs_gaussian_integers(rust: Any) -> None:
    errors = []
    for impl in IMPLEMENTATIONS:
        f = build(impl, ['x', 'y'], [({(1, 0): (F(1, 2), F(0)), (0, 1): (F(1), F(0))}, 1)])
        with pytest.raises(Exception) as info:
            mod(impl, "gaussianfactor").gaussian_factor_list(f, True)
        errors.append(type(info.value).__name__)
    assert errors[0] == errors[1]


def domains(impl: str) -> dict[str, Any]:
    d = mod(impl, "domains")
    return {"ZZ": d.ZZ, "QQ": d.QQ, "GF(7)": d.GF(7), "QQ_I": mod(impl, "domains.gaussiandomains").QQ_I}


def entry(impl: str, name: str, rng: random.Random) -> Any:
    value = rng.choice([0, 0, 0, 1, -1, 2, 3, -4])
    if name == "QQ_I":
        return gaussian(impl, (F(value), F(rng.choice([0, 0, 1, -2]))))
    if name == "QQ":
        return F(value, rng.choice([1, 1, 2, 3]))
    return domains(impl)[name].from_int(value)


@pytest.mark.parametrize("name", ["ZZ", "QQ", "GF(7)", "QQ_I"])
@pytest.mark.parametrize("seed", range(6))
def test_nullspace_agrees(rust: Any, name: str, seed: int) -> None:
    rng = random.Random(stable_seed("nullspace", name, seed))
    for _ in range(8):
        nrows, ncols = rng.randint(0, 4), rng.randint(1, 5)
        state = rng.getstate()
        results: list[object] = []
        for impl in IMPLEMENTATIONS:
            rng.setstate(state)
            K = domains(impl)[name]
            rows = [[entry(impl, name, rng) for _ in range(ncols)] for _ in range(nrows)]
            try:
                results.append([[plain(v) for v in vector] for vector in K.nullspace(rows, ncols)])
            except ValueError:
                results.append("ValueError")
        assert results[0] == results[1]
