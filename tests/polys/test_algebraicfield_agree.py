"""The Rust port of ``AlgebraicField`` and of the factorisation over
number fields (Trager's algorithm in ``factortools``) against the Python
modules: the docstring examples, the field arithmetic on random elements,
and the factorisations of random products over five number fields."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed

F = Fraction

#: minimal polynomials: sqrt(2), cbrt(2), i, the golden ratio, sqrt(2) + sqrt(3)
FIELDS = {
    "sqrt2": [1, 0, -2],
    "cbrt2": [1, 0, 0, -2],
    "i": [1, 0, 1],
    "phi": [1, -1, -1],
    "sqrt2sqrt3": [1, 0, -10, 0, 1],
}


def fields(name: str) -> tuple[Any, Any]:
    from sympy_extras.core import Symbol
    from sympy_extras.polys.domains.algebraicfield import AlgebraicField
    rs = load("sympy_extras_rs.polys.domains.algebraicfield")
    minpoly = [F(c) for c in FIELDS[name]]
    theta = Symbol("theta")
    return AlgebraicField(minpoly, theta), rs.AlgebraicField(minpoly, theta)


def test_known(rust: Any) -> None:
    from sympy_extras.core import Integer, sqrt
    rs = load("sympy_extras_rs.polys.domains.algebraicfield")
    K = rs.AlgebraicField([F(1), F(0), F(-2)], sqrt(Integer(2)))
    assert repr(K) == "QQ<sqrt(2)>"
    a = [F(1), F(1)]
    assert (K.mul(a, a), K.invert(a)) == ([F(2), F(3)], [F(1), F(-1)])
    assert (K.norm(a), K.minimal_polynomial(a)) == (F(-1), [F(1), F(-2), F(-1)])
    ft = load("sympy_extras_rs.polys.factortools")
    # x**2 - 2 = (x - sqrt(2)) (x + sqrt(2))
    assert ft.dup_ext_factor([[F(1)], [], [F(-2)]], K) == ([F(1)], [([[F(1)], [F(-1), F(0)]], 1),
                                                               ([[F(1)], [F(1), F(0)]], 1)])
    with pytest.raises(ZeroDivisionError):
        K.invert([])
    with pytest.raises(ValueError):
        rs.AlgebraicField([F(2), F(0)], sqrt(Integer(2)))


def element(rng: random.Random, degree: int) -> list[Fraction]:
    a = [F(rng.randint(-3, 3), rng.choice([1, 1, 2, 3])) for _ in range(rng.randint(0, degree))]
    while a and not a[0]:
        a.pop(0)
    return a


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("name", sorted(FIELDS))
def test_arithmetic_agrees(rust: Any, name: str, seed: int) -> None:
    py, rs = fields(name)
    rng = random.Random(stable_seed("arithmetic", name, seed))
    for _ in range(10):
        a, b = element(rng, py.degree), element(rng, py.degree)
        assert rs.mul(a, b) == py.mul(a, b)
        assert rs.add(a, b) == py.add(a, b) and rs.sub(a, b) == py.sub(a, b)
        assert rs.norm(a) == py.norm(a)
        assert rs.characteristic_polynomial(a) == py.characteristic_polynomial(a)
        assert rs.minimal_polynomial(a) == py.minimal_polynomial(a)
        if a:
            assert rs.invert(a) == py.invert(a)
            assert rs.exquo(b, a) == py.exquo(b, a)
            assert rs.pow(a, -3) == py.pow(py.invert(a), 3)
        assert rs.pow(a, 4) == py.pow(a, 4)
    assert rs.alpha == py.alpha and repr(rs) == py.name


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("name", sorted(FIELDS))
def test_ext_factor_agrees(rust: Any, name: str, seed: int) -> None:
    from sympy_extras.polys import densearith, factortools as py_ft
    py, rs = fields(name)
    ft = load("sympy_extras_rs.polys.factortools")
    rng = random.Random(stable_seed("ext_factor", name, seed))
    f: list[list[Fraction]] = [element(rng, py.degree) or [F(1)]]
    for _ in range(rng.randint(1, 3)):
        kind = rng.random()
        if kind < 0.4:      # a linear factor over the field
            g = [py.one, element(rng, py.degree)]
        elif kind < 0.7:    # a polynomial over QQ, which may split over the field
            g = [[F(c)] if c else [] for c in rng.choice([[1, 0, -2], [1, 0, 1], [1, -1, -1], [1, 0, 0, -2],
                                                          [1, 0, -10, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1]])]
        else:               # a random quadratic over the field
            g = [py.one, element(rng, py.degree), element(rng, py.degree) or [F(1)]]
        while g and not g[0]:
            g.pop(0)
        f = densearith.dup_mul(f, g, py)
        if rng.random() < 0.2:
            f = densearith.dup_mul(f, g, py)  # a repeated factor
    assert ft.dup_ext_norm(f, rs) == py_ft.dup_ext_norm(f, py)
    assert ft.dup_ext_factor(f, rs) == py_ft.dup_ext_factor(f, py)
    assert rs.factor_dense(f) == py.factor_dense(f)
    from sympy_extras.polys.sqfreetools import dup_sqf_part
    g = dup_sqf_part(f, py)
    if len(g) > 1:  # the shifts end on square-free polynomials only
        assert ft.dup_ext_sqf_norm(g, rs) == py_ft.dup_ext_sqf_norm(g, py)
