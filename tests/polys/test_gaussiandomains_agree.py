"""The Rust port of ``gaussiandomains`` against the Python module: the
docstring examples, the domain methods of ``ZZ_I`` and ``QQ_I`` on random
elements, and the dense algorithms (gcd, square-free decomposition,
resultants) over them."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed

F = Fraction


def rs(name: str) -> Any:
    return load("sympy_extras_rs.polys." + name)


def test_known(rust: Any) -> None:
    g = rs("domains.gaussiandomains")
    ZZ_I, QQ_I, G = g.ZZ_I, g.QQ_I, g.GaussianInteger
    a, b = G(3, 4), G(1, 2)
    assert ZZ_I.mul(a, b) == G(-5, 10)
    assert ZZ_I.divmod(a, b) == (G(2, 0), G(1, 0))
    assert ZZ_I.gcd(G(4, 2), G(2, 6)) == G(4, 2)
    assert ZZ_I.canonical_unit(G(-2, 3)) == G(0, -1)
    assert QQ_I.invert(g.GaussianRational(F(3), F(4))) == g.GaussianRational(F(3, 25), F(-4, 25))
    assert repr(ZZ_I) == "ZZ_I" and not QQ_I.is_ordered and QQ_I.is_field
    with pytest.raises(ValueError):
        ZZ_I.invert(a)
    with pytest.raises(ZeroDivisionError):
        ZZ_I.divmod(a, G(0, 0))


def pair(py_module: Any, rng: random.Random, rational: bool) -> tuple[Any, Any]:
    if rational:
        x, y = F(rng.randint(-9, 9), rng.randint(1, 4)), F(rng.randint(-9, 9), rng.randint(1, 4))
        return py_module.GaussianRational(x, y), rs("domains.gaussiandomains").GaussianRational(x, y)
    x, y = rng.randint(-30, 30), rng.randint(-30, 30)
    return py_module.GaussianInteger(x, y), rs("domains.gaussiandomains").GaussianInteger(x, y)


def outcome(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except (ValueError, ZeroDivisionError) as e:
        return type(e)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("rational", [False, True], ids=["ZZ_I", "QQ_I"])
def test_domain_agrees(rust: Any, seed: int, rational: bool) -> None:
    from sympy_extras.polys.domains import gaussiandomains as py
    g = rs("domains.gaussiandomains")
    P, R = (py.QQ_I, g.QQ_I) if rational else (py.ZZ_I, g.ZZ_I)
    rng = random.Random(stable_seed("gaussian", seed, rational))
    for _ in range(40):
        (a, ra), (b, rb) = pair(py, rng, rational), pair(py, rng, rational)
        if rng.random() < 0.1:
            b, rb = P.zero, R.zero
        for name in ("add", "sub", "mul", "exquo", "quo", "rem", "divmod", "gcd", "gcdex"):
            assert outcome(getattr(R, name), ra, rb) == outcome(getattr(P, name), a, b), (name, a, b)
        for name in ("neg", "canonical_unit", "invert", "is_unit", "size"):
            assert outcome(getattr(R, name), ra) == outcome(getattr(P, name), a), (name, a)
        if a or b:
            assert R.lcm(ra, rb) == P.lcm(a, b)
        assert R.pow(ra, 3) == P.pow(a, 3)


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("rational", [False, True], ids=["ZZ_I", "QQ_I"])
def test_dense_agrees(rust: Any, seed: int, rational: bool) -> None:
    from sympy_extras.polys import densearith, euclidtools, sqfreetools
    from sympy_extras.polys.domains import gaussiandomains as py
    g = rs("domains.gaussiandomains")
    P, R = (py.QQ_I, g.QQ_I) if rational else (py.ZZ_I, g.ZZ_I)
    et, sq, da = rs("euclidtools"), rs("sqfreetools"), rs("densearith")
    rng = random.Random(stable_seed("gaussian dense", seed, rational))

    def poly(n: int) -> list[Any]:
        f = [pair(py, rng, rational)[0] for _ in range(n + 1)]
        return f if f[0] else [P.one] + f[1:]

    common = poly(rng.randint(0, 2))
    f = densearith.dup_mul(poly(rng.randint(0, 3)), common, P)
    h = densearith.dup_mul(poly(rng.randint(0, 3)), common, P)
    if rng.random() < 0.4:
        f = densearith.dup_mul(f, common, P)
    assert da.dup_mul(f, h, R) == densearith.dup_mul(f, h, P)
    assert outcome(da.dup_div, f, h, R) == outcome(densearith.dup_div, f, h, P)
    assert et.dup_gcd(f, h, R) == euclidtools.dup_gcd(f, h, P)
    assert et.dup_resultant(f, h, R) == euclidtools.dup_resultant(f, h, P)
    assert sq.dup_sqf_list(f, R) == sqfreetools.dup_sqf_list(f, P)
    assert sq.dup_sqf_part(f, R) == sqfreetools.dup_sqf_part(f, P)
