"""The Rust port of ``rings`` and ``heuristicgcd`` against the Python
package: the docstring examples on both implementations, and random
polynomials compared through their terms, strings and representations."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any, Callable

import pytest

from ..conftest import load


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_rings_known(impl: str) -> None:
    QQ = mod(impl, "domains").QQ
    r = mod(impl, "rings")
    R = r.PolyRing(QQ, ['x', 'y'], r.grevlex)
    x, y = R.gens
    f = x**2*y - R.one
    assert repr(f) == "PolyElement({(2, 1): Fraction(1, 1), (0, 0): Fraction(-1, 1)}, QQ[x,y])"
    assert (f.LM, f.LC, f.degree(), str(f)) == ((2, 1), Fraction(1), 3, 'x**2*y - 1')
    assert (x + y)**2 == x**2 + (x*y).mul_ground(QQ.from_int(2)) + y**2
    q, rem = (x**2*y + x).div([x*y - R.one])
    assert (str(q[0]), str(rem)) == ('x', '2*x')


def test_heugcd_known(impl: str) -> None:
    ZZ = mod(impl, "domains").ZZ
    r, h = mod(impl, "rings"), mod(impl, "heuristicgcd")
    R = r.PolyRing(ZZ, ['x', 'y'])
    x, y = R.gens
    f, g = (x + y)*(x.mul_ground(2) - y), (x + y)*(x + R.from_int(3))
    gcd, cff, cfg = h.heugcd(f, g)
    assert (str(gcd), str(cff), str(cfg)) == ('x + y', '2*x - y', 'x + 3')


DOMAINS: list[str] = ["ZZ", "QQ", "GF(7)"]
ORDERS: list[str] = ["lex", "grlex", "grevlex"]


def domain(impl: str, name: str) -> Any:
    d = mod(impl, "domains")
    return d.GF(int(name[3:-1])) if name.startswith("GF(") else getattr(d, name)


def coefficient(rng: random.Random, name: str) -> Any:
    if name == "ZZ":
        return rng.randint(-5, 5)
    if name == "QQ":
        return Fraction(rng.randint(-5, 5), rng.randint(1, 3))
    return rng.randrange(int(name[3:-1]))


def random_terms(rng: random.Random, name: str, nvars: int) -> dict[tuple[int, ...], Any]:
    terms: dict[tuple[int, ...], Any] = {}
    for _ in range(rng.randint(0, 6)):
        m = tuple(rng.randint(0, 3) for _ in range(nvars))
        terms[m] = coefficient(rng, name)
    return terms


def view(f: Any) -> tuple[Any, ...]:
    return (dict(f.terms), str(f), repr(f), f.LM, f.LC, f.degree(), list(f.monomials()), f.is_ground, f.is_monic)


def outcome(call: Callable[[], Any]) -> Any:
    try:
        return ("ok", call())
    except (ValueError, ZeroDivisionError) as e:
        return ("raises", type(e).__name__)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("name", DOMAINS)
@pytest.mark.parametrize("nvars", [1, 2, 3])
def test_ring_arithmetic_agrees(rust: Any, name: str, order: str, nvars: int) -> None:
    rng = random.Random(hash((name, order, nvars)) & 0xffff)
    names = ['x', 'y', 'z'][:nvars]
    rings = []
    for impl in ("sympy_extras", "sympy_extras_rs"):
        r = mod(impl, "rings")
        rings.append(r.PolyRing(domain(impl, name), names, getattr(r, order)))
    assert repr(rings[0]) == repr(rings[1])
    for _ in range(40):
        a, b, c = (random_terms(rng, name, nvars) for _ in range(3))
        n, i = rng.randint(0, 3), rng.randrange(nvars)
        point = coefficient(rng, name)
        ground = coefficient(rng, name)
        results = []
        for R in rings:
            f, g, h = R.from_dict(a), R.from_dict(b), R.from_dict(c)
            results.append([
                view(f), view(f + g), view(f - g), view(f*g), view(-f), view(f**n),
                view(f.diff(i)), view(f.evaluate(i, point)), view(f.mul_ground(ground)),
                view(f.mul_monom(tuple(range(nvars)))), outcome(lambda: view(f.exquo_ground(ground))),
                outcome(lambda: view(f.monic())), (f.primitive()[0], view(f.primitive()[1])),
                outcome(lambda: [[view(q) for q in d[0]] + [view(d[1])] for d in [f.div([g, h])]]),
                outcome(lambda: view((f*g).exquo(g))), f == g, bool(f), f.degree_in(i),
                f.coeff(tuple(0 for _ in names)),
            ])
        assert results[0] == results[1], (a, b, c)


@pytest.mark.parametrize("nvars", [1, 2, 3])
@pytest.mark.parametrize("seed", range(10))
def test_heugcd_agrees(rust: Any, nvars: int, seed: int) -> None:
    rng = random.Random(seed * 10 + nvars)
    names = ['x', 'y', 'z'][:nvars]
    for _ in range(8):
        common, a, b = (random_terms(rng, "ZZ", nvars) or {(0,) * nvars: 1} for _ in range(3))
        results = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            r, hg = mod(impl, "rings"), mod(impl, "heuristicgcd")
            R = r.PolyRing(domain(impl, "ZZ"), names, r.lex)
            h = R.from_dict(common)
            f, g = R.from_dict(a)*h, R.from_dict(b)*h
            if not f or not g:
                results.append(None)
                continue
            try:
                gcd, cff, cfg = hg.heugcd(f, g)
                value: Any = (dict(gcd.terms), dict(cff.terms), dict(cfg.terms))
            except Exception as e:
                value = type(e).__name__
            steps, deflated = hg.deflate([f, g])
            results.append((value, steps, [dict(p.terms) for p in deflated],
                            dict(hg.inflate(deflated[0], steps).terms)))
        assert results[0] == results[1], (common, a, b)


def test_monomial_functions(impl: str) -> None:
    r = mod(impl, "rings")
    assert r.monomial_mul((1, 2), (3, 0)) == (4, 2)
    assert r.monomial_div((4, 2), (3, 0)) == (1, 2)
    with pytest.raises(ValueError):
        r.monomial_div((1, 2), (3, 0))
    assert r.monomial_divides((1, 0), (1, 2)) and not r.monomial_divides((2, 0), (1, 2))
    assert r.monomial_lcm((1, 3), (2, 0)) == (2, 3)
    assert r.monomial_gcd((1, 3), (2, 0)) == (1, 0)
