"""SparseFractionField against SymPy's FracField (the oracle): the same
numerators and denominators (SymPy's normal form: lowest terms, the sign
of the denominator canonical) after the field operations."""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Callable

import pytest
import sympy
from sympy.polys.fields import FracElement, field as sympy_field

from sympy_extras.interop import expr_from_sympy
from sympy_extras.polys.domains.sparsefractionfield import SparseFractionField, SparseFracElement
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing

NAMES = ['x', 'y', 'z']
GENS = sympy.symbols(NAMES)
F: SparseFractionField[Fraction] = SparseFractionField(RationalPolynomialRing(NAMES))
K, *_ = sympy_field(NAMES, sympy.QQ)

Ours = SparseFracElement[Fraction]


def _random_poly(rng: random.Random) -> sympy.Expr:
    terms = []
    for _ in range(rng.randint(1, 4)):
        c = sympy.Rational(rng.randint(-6, 6), rng.choice([1, 1, 2, 3]))
        terms.append(c*sympy.Mul(*[g**rng.randint(0, 2) for g in GENS]))
    p = sympy.Add(*terms)
    return p if p != 0 else sympy.Integer(1)


def _random_pair(rng: random.Random) -> tuple[Ours, FracElement]:
    e = _random_poly(rng)/_random_poly(rng)
    return F.from_expr(expr_from_sympy(e)), K.from_expr(e)


def _same(ours: Ours, theirs: FracElement) -> bool:
    # SymPy's negative powers skip the normalisation of the sign (raw_new):
    # its normal form is that of FracElement.new (issue #25)
    normal = theirs.new(theirs.numer, theirs.denom)
    return (F.ring.to_expr(ours.numer) == expr_from_sympy(normal.numer.as_expr())
            and F.ring.to_expr(ours.denom) == expr_from_sympy(normal.denom.as_expr()))


def test_normal_form_and_operations_match_sympy() -> None:
    rng = random.Random(7)
    binary: list[tuple[str, Callable[[Ours, Ours], Ours], Callable[[FracElement, FracElement], FracElement]]] = [
        ('add', F.add, lambda a, b: a + b), ('sub', F.sub, lambda a, b: a - b),
        ('mul', F.mul, lambda a, b: a*b), ('div', F.exquo, lambda a, b: a/b)]
    for _ in range(40):
        (a, sa), (b, sb) = _random_pair(rng), _random_pair(rng)
        assert _same(a, sa) and _same(b, sb)
        for name, ours, theirs in binary:
            if name == 'div' and F.is_zero(b):
                continue
            assert _same(ours(a, b), theirs(sa, sb)), (name, sa, sb)
        for n in (2, 3, -1, -2):
            if n < 0 and F.is_zero(a):
                continue
            assert _same(F.pow(a, n), sa**n), (sa, n)
        for i, g in enumerate(GENS):
            assert _same(F.diff(a, i), sa.diff(K.gens[i])), (sa, g)


def test_sign_and_content_of_the_denominator() -> None:
    x, y = GENS[0], GENS[1]
    for e in [1/(-2*x - 2*y), (x - y)/(y - x), (2*x + 4)/(6*y - 3), sympy.Rational(3, 2)/(x/3 - y/5)]:
        assert _same(F.from_expr(expr_from_sympy(e)), K.from_expr(e)), e


def test_integer_fraction_field_conversions_match_sympy() -> None:
    """``ZZ(x, y)``: SymPy's normal form over ``ZZ`` (integer numerator and
    denominator), the conversions from ``ZZ``, ``QQ`` and the ring, and
    back to the ring (``to_ring``)."""
    from sympy_extras.polys.domains import QQ, ZZ
    from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing
    from sympy_extras.polys.polyerrors import CoercionFailed
    FZ: SparseFractionField[int] = SparseFractionField(IntegerPolynomialRing(['x', 'y']))
    KZ = sympy.ZZ.frac_field(*GENS[:2])
    rng = random.Random(7)
    for _ in range(40):
        e = _random_poly(rng).subs(GENS[2], 1) / _random_poly(rng).subs(GENS[2], 1)
        if e.has(sympy.zoo, sympy.nan):
            continue
        ours = FZ.from_expr(expr_from_sympy(e))
        theirs = KZ.from_sympy(e)
        assert (FZ.ring.to_expr(ours.numer), FZ.ring.to_expr(ours.denom)) == \
            (expr_from_sympy(theirs.numer.as_expr()), expr_from_sympy(theirs.denom.as_expr())), e
        assert FZ.to_expr(FZ.mul(ours, FZ.invert(ours))) == expr_from_sympy(sympy.Integer(1)) or FZ.is_zero(ours)
    half = FZ.convert(Fraction(1, 2), QQ)
    assert (FZ.ring.to_expr(half.numer), FZ.ring.to_expr(half.denom)) == (expr_from_sympy(sympy.Integer(1)),
                                                                          expr_from_sympy(sympy.Integer(2)))
    assert FZ.is_one(FZ.add(half, half)) and FZ.is_one(FZ.convert(3, ZZ)) is False
    x = FZ.ring.ring.gens[0]
    assert FZ.to_ring(FZ.from_ring(x * x)) == x * x
    assert FZ.to_ring(FZ.exquo(FZ.from_ring(x * x), FZ.from_ring(x))) == x
    for not_polynomial in (half, FZ.invert(FZ.from_ring(x))):
        with pytest.raises(CoercionFailed):
            FZ.to_ring(not_polynomial)
    # over QQ a number as the denominator is a unit
    y = F.ring.ring.gens[1]
    assert F.to_ring(F.new(y, F.ring.from_int(2))) == y.mul_ground(Fraction(1, 2))
