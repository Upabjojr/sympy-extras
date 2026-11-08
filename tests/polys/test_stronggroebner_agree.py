"""The Rust port of ``stronggroebner`` against the Python package: the
docstring examples on both implementations, and the reduced strong Gröbner
bases of random systems over ZZ (with integer generators among them)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_known(impl: str) -> None:
    d, rings, sg = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "stronggroebner")
    R = rings.PolyRing(d.ZZ, ['x', 'y'], rings.lex)
    X, Y = R.gens
    assert [str(g) for g in sg.strong_groebner([X.mul_ground(2), Y.mul_ground(3)], R)] == ['x*y', '2*x', '3*y']
    assert [str(g) for g in sg.strong_groebner([X.mul_ground(2), X.mul_ground(3)], R)] == ['x']
    assert [str(g) for g in sg.strong_groebner([4, X.mul_ground(2) + R.one], R)] == ['1']
    f, g = (X*X).mul_ground(2) + Y, (X*Y).mul_ground(3) + R.one
    G = sg.strong_groebner([f, g], R)
    assert [str(h) for h in G] == ['x*y + 3*y**3 + 1', '2*x - 3*y**2', '9*y**3 + 2']
    assert (str(sg.strong_reduce(X*f - Y*g, G)), str(sg.strong_reduce(X, G))) == ('0', 'x')
    assert str(sg.s_polynomial_zz(f, g)) == '-2*x + 3*y**2'
    assert str(sg.g_polynomial_zz(X.mul_ground(2), Y.mul_ground(3))) == 'x*y'
    assert sg.g_polynomial_zz(X.mul_ground(2), Y.mul_ground(4)) is None
    assert (sg.is_strong_groebner([X.mul_ground(2), Y.mul_ground(3)]),
            sg.is_strong_groebner([X.mul_ground(2), Y.mul_ground(3), X*Y])) == (False, True)
    with pytest.raises(ValueError):
        sg.strong_groebner([], rings.PolyRing(d.QQ, ['x'], rings.lex))


@pytest.mark.parametrize("order", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize("nvars", [1, 2])
@pytest.mark.parametrize("seed", range(8))
def test_strong_groebner_agrees(rust: Any, order: str, nvars: int, seed: int) -> None:
    rng = random.Random(stable_seed(order, nvars, seed))
    names = ['x', 'y'][:nvars]
    for _ in range(3):
        state = rng.getstate()
        results = []
        for impl in ("sympy_extras", "sympy_extras_rs"):
            rng.setstate(state)
            d, rings, sg = mod(impl, "domains"), mod(impl, "rings"), mod(impl, "stronggroebner")
            R = rings.PolyRing(d.ZZ, names, getattr(rings, order))
            system: list[Any] = []
            for _ in range(rng.randint(1, 3)):
                p = R.zero
                for _ in range(rng.randint(1, 3)):
                    m = R.one
                    for x in R.gens:
                        m = m * x**rng.randint(0, 2)
                    p = p + m.mul_ground(rng.choice([-6, -4, -3, -2, 2, 3, 4, 5, 6, 9]))
                system.append(p)
            if rng.random() < 0.3:
                system.append(rng.choice([4, 6, 12]))
            basis = sg.strong_groebner(system, R)
            probe = system[0] if not isinstance(system[0], int) else R.from_int(system[0])
            probe = probe * R.gens[0] + R.gens[-1]
            results.append(([dict(g.terms) for g in basis], [str(g) for g in basis],
                             sg.is_strong_groebner(basis), dict(sg.strong_reduce(probe, basis).terms)))
        assert results[0] == results[1], [str(p) for p in system]
