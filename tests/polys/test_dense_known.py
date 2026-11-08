"""The examples of the docstrings of the dense polynomial modules, run
against both implementations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load

F = Fraction


def mod(impl: str, name: str) -> Any:
    return load(impl + ".polys." + name)


def test_domains(impl: str) -> None:
    d = mod(impl, "domains")
    ZZ, QQ, GF = d.ZZ, d.QQ, d.GF
    assert (ZZ.gcd(12, -18), ZZ.gcdex(12, 18), ZZ.quo(-7, 2), ZZ.rem(-7, 2)) == (6, (-1, 1, 6), -3, -1)
    assert ZZ.exquo(12, 4) == 3
    with pytest.raises(ValueError):
        ZZ.exquo(12, 5)
    with pytest.raises(ZeroDivisionError):
        ZZ.exquo(12, 0)
    assert (QQ.add(F(1, 2), F(1, 3)), QQ.invert(F(-2, 3))) == (F(5, 6), F(-3, 2))
    assert (QQ.numer(F(-6, 4)), QQ.denom(F(-6, 4))) == (-3, 2)
    K = GF(7)
    assert (K.add(5, 4), K.mul(3, 5), K.invert(3), K.pow(3, 100)) == (2, 1, 5, 4)
    assert K.from_int(-1) == 6
    assert list(GF(3).elements()) == [0, 1, 2]
    assert (ZZ.name, QQ.name, K.name, K.characteristic, K.is_field, ZZ.is_field) == ('ZZ', 'QQ', 'GF(7)', 7, True, False)
    with pytest.raises(ValueError):
        GF(1)


def test_densebasic(impl: str) -> None:
    d, b = mod(impl, "domains"), mod(impl, "densebasic")
    ZZ, QQ = d.ZZ, d.QQ
    assert b.dup_strip([0, 0, 1, 2], ZZ) == [1, 2]
    assert (b.dup_degree([1, 0, -2]), b.dup_degree([])) == (2, -1)
    assert b.dup_from_dict({3: 1, 0: -1}, ZZ) == [1, 0, 0, -1]
    assert b.dup_to_dict([1, 0, 0, -1], ZZ) == {3: 1, 0: -1}
    assert b.dup_convert([1, -2], ZZ, QQ) == [F(1), F(-2)]
    with pytest.raises(ValueError):
        b.dup_convert([F(1, 2)], QQ, ZZ)


def test_densearith(impl: str) -> None:
    d, a, b = mod(impl, "domains"), mod(impl, "densearith"), mod(impl, "densebasic")
    ZZ, QQ = d.ZZ, d.QQ
    assert a.dup_add([1, 2], [1, -2], ZZ) == [2, 0]
    assert a.dup_mul([1, 1], [1, -1], ZZ) == [1, 0, -1]
    assert a.dup_div(b.dup_from_list([1, 0, -1], QQ), b.dup_from_list([2, 1], QQ), QQ) == ([F(1, 2), F(-1, 4)], [F(-3, 4)])
    assert a.dup_pdiv([1, 0, -1], [2, 1], ZZ) == ([2, -1], [-3])
    assert a.dup_exquo([1, 0, -1], [1, 1], ZZ) == [1, -1]
    assert a.dup_pow([1, 1], 3, ZZ) == [1, 3, 3, 1]
    with pytest.raises(ZeroDivisionError):
        a.dup_div([1], [], ZZ)


def test_densetools(impl: str) -> None:
    d, t = mod(impl, "domains"), mod(impl, "densetools")
    ZZ, QQ = d.ZZ, d.QQ
    assert t.dup_diff([1, 0, -2, 5], 1, ZZ) == [3, 0, -2]
    assert t.dup_eval([1, 0, -2], 3, ZZ) == 7
    assert t.dup_shift([1, 0, 0], 1, ZZ) == [1, 2, 1]
    assert t.dup_compose([1, 0, 1], [1, 1], ZZ) == [1, 2, 2]
    assert (t.dup_content([4, -6, 8], ZZ), t.dup_primitive([4, -6, 8], ZZ)) == (2, (2, [2, -3, 4]))
    assert t.dup_monic([F(2), F(4)], QQ) == [F(1), F(2)]
    assert t.dup_clear_denoms([F(1, 2), F(1, 3)], QQ) == (6, [F(3), F(2)])
    assert t.dup_trunc([7, 8, 9], 5, ZZ) == [2, -2, -1]
    assert t.dup_integrate([F(3), F(0), F(-2)], 1, QQ) == [F(1), F(0), F(-2), F(0)]


def test_euclidtools(impl: str) -> None:
    d, e = mod(impl, "domains"), mod(impl, "euclidtools")
    ZZ, QQ, GF = d.ZZ, d.QQ, d.GF
    assert e.dup_gcd([1, 0, -1], [1, -2, 1], ZZ) == [1, -1]
    assert e.dup_gcd([2, 0, -2], [4, -8, 4], ZZ) == [2, -2]
    assert e.dup_resultant([1, 0, -2], [1, 0, 0, -3], ZZ) == 1
    assert e.dup_discriminant([1, 1, 1], ZZ) == -3
    assert e.dup_gcdex([F(1), F(0), F(-1)], [F(1), F(-2)], QQ) == ([F(1, 3)], [F(-1, 3), F(-2, 3)], [F(1)])
    assert e.dup_invert([1, 1], [1, 0, 1], GF(7)) == [3, 4]
    assert e.dup_prs_resultant([-7, -3], [-2, 6, 1, -6], ZZ) == 1773


def test_sqfreetools(impl: str) -> None:
    d, s = mod(impl, "domains"), mod(impl, "sqfreetools")
    ZZ, GF = d.ZZ, d.GF
    assert s.dup_sqf_list([1, -4, 6, -4, 1], ZZ) == (1, [([1, -1], 4)])
    assert s.dup_sqf_list([2, 0, -4, 0, 2], ZZ) == (2, [([1, 0, -1], 2)])
    assert (s.dup_sqf_part([1, -4, 6, -4, 1], ZZ), s.dup_sqf_p([1, 0, -1], ZZ), s.dup_sqf_p([1, 0, 0], ZZ)) == ([1, -1], True, False)
    assert s.dup_sqf_list([1, 0, 0, 0, 0, 0, 1], GF(3)) == (1, [([1, 0, 1], 3)])
