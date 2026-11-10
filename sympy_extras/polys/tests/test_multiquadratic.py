"""The multiquadratic field against SymPy's simplification of the same
expressions in square roots."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest
import sympy

from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.domains.multiquadratic import MultiquadraticField


def _value(F: MultiquadraticField[Fraction], a: dict[int, Fraction]) -> sympy.Expr:
    total: sympy.Expr = sympy.Integer(0)
    for mask, c in a.items():
        term: sympy.Expr = sympy.Rational(c.numerator, c.denominator)
        for i, d in enumerate(F.radicands):
            if mask >> i & 1:
                term = term*sympy.sqrt(d)
        total = total + term
    return total


def test_field_operations() -> None:
    F = MultiquadraticField(QQ, [-1, 2, 3, 5])
    rnd = random.Random(17)

    def element() -> dict[int, Fraction]:
        return {mask: Fraction(rnd.randint(-4, 4), rnd.randint(1, 3)) for mask in rnd.sample(range(16), rnd.randint(1, 5))
                if rnd.random() < 0.9} or F.one

    for _ in range(40):
        a, b, c = element(), element(), element()
        a = {m: v for m, v in a.items() if v}
        b = {m: v for m, v in b.items() if v}
        c = {m: v for m, v in c.items() if v}
        assert sympy.expand(_value(F, F.mul(a, b)) - _value(F, a)*_value(F, b)) == 0
        assert F.mul(a, F.add(b, c)) == F.add(F.mul(a, b), F.mul(a, c))
        assert F.mul(F.mul(a, b), c) == F.mul(a, F.mul(b, c))
        assert F.sub(F.add(a, b), b) == a
        if a:
            assert F.mul(a, F.invert(a)) == F.one and F.exquo(F.mul(a, b), a) == b
    assert F.mul(F.generator(0), F.generator(0)) == F.from_int(-1)
    assert F.pow(F.add(F.generator(1), F.one), 2) == {0: Fraction(3), 2: Fraction(2)}
    with pytest.raises(ZeroDivisionError):
        F.invert(F.zero)
    with pytest.raises(ValueError):
        MultiquadraticField(QQ, [2, 2])
    with pytest.raises(ValueError):
        MultiquadraticField(GF(7), [2])
