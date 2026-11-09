"""The Rust port of ``sparsefractionfield`` (and of ``cancel`` of the
sparse polynomial rings) against the Python module: random rational
functions over ``ZZ(x, y)`` and ``QQ(x, y, z)``, the arithmetic, powers,
derivatives and the normal form of the denominator."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed

F = Fraction


def fields(rational: bool, names: list[str]) -> tuple[Any, Any]:
    from sympy_extras.polys.domains import sparsepolynomialring as pr
    from sympy_extras.polys.domains.sparsefractionfield import SparseFractionField
    rs_ring = load("sympy_extras_rs.polys.domains.sparsepolynomialring")
    rs_field = load("sympy_extras_rs.polys.domains.sparsefractionfield")
    if rational:
        return (SparseFractionField(pr.RationalPolynomialRing(names)),
                rs_field.SparseFractionField(rs_ring.RationalPolynomialRing(names)))
    return (SparseFractionField(pr.IntegerPolynomialRing(names)),
            rs_field.SparseFractionField(rs_ring.IntegerPolynomialRing(names)))


def as_terms(a: Any) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """The numerator and denominator of an element of either field."""
    def terms(p: Any) -> dict[Any, Any]:
        t = p.terms
        return dict(t() if callable(t) else t)
    return terms(a.numer), terms(a.denom)


def test_known(rust: Any) -> None:
    py, rs = fields(True, ["x", "y"])
    assert repr(rs) == py.name == "QQ(x,y)"
    x, y = rs.gens
    two = rs.from_int(2)
    a = rs.exquo(rs.sub(rs.mul(x, x), rs.mul(y, y)), rs.add(rs.mul(two, x), rs.mul(two, y)))
    assert (str(a.numer), str(a.denom)) == ("x - y", "2")
    s = rs.add(rs.invert(x), rs.invert(y))
    assert (str(s.numer), str(s.denom)) == ("x + y", "x*y")
    d = rs.diff(rs.exquo(x, rs.add(x, y)), 0)
    assert (str(d.numer), str(d.denom)) == ("y", "x**2 + 2*x*y + y**2")
    with pytest.raises(ZeroDivisionError):
        rs.invert(rs.zero)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("rational", [False, True], ids=["ZZ", "QQ"])
def test_field_agrees(rust: Any, seed: int, rational: bool) -> None:
    rng = random.Random(stable_seed("sparse fractions", seed, rational))
    names = ["x", "y", "z"][:rng.randint(1, 3)]
    py, rs = fields(rational, names)
    n = len(names)

    def poly() -> dict[tuple[int, ...], Any]:
        t: dict[tuple[int, ...], Any] = {}
        for _ in range(rng.randint(1, 3)):
            m = tuple(rng.randint(0, 2) for _ in range(n))
            c = rng.choice([-3, -2, -1, 1, 2, 4])
            t[m] = F(c, rng.choice([1, 2, 3])) if rational else c
        return t

    def element() -> tuple[Any, Any]:
        num, den = poly(), poly()
        common = poly()  # a common factor, cancelled by both
        pn, pd, pc = (py.ring.ring.from_dict(t) for t in (num, den, common))
        rn, rd, rc = (rs_poly(t) for t in (num, den, common))
        return py.new(pn * pc, pd * pc), rs.new(rn * rc, rd * rc)

    def rs_poly(t: dict[tuple[int, ...], Any]) -> Any:
        return rs.from_ring(rs.one.numer.ring.from_dict(t)).numer

    for _ in range(6):
        (a, ra), (b, rb) = element(), element()
        assert as_terms(ra) == as_terms(a)
        for name in ("add", "sub", "mul", "exquo"):
            assert as_terms(getattr(rs, name)(ra, rb)) == as_terms(getattr(py, name)(a, b)), name
        assert as_terms(rs.neg(ra)) == as_terms(py.neg(a))
        assert as_terms(rs.invert(ra)) == as_terms(py.invert(a))
        assert as_terms(rs.pow(ra, 2)) == as_terms(py.pow(a, 2))
        assert as_terms(rs.pow(ra, -1)) == as_terms(py.pow(a, -1))
        assert as_terms(rs.diff(ra, n - 1)) == as_terms(py.diff(a, n - 1))
        assert as_terms(rs.canonical_unit(ra)) == as_terms(py.canonical_unit(a))
        assert (rs.is_negative(ra), rs.is_one(ra), rs.size(ra)) == (py.is_negative(a), py.is_one(a), py.size(a))
