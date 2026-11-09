"""The native minimal polynomials against SymPy's ``minimal_polynomial``:
the atoms, and random sums, products and powers of them. The minimal
polynomial is unique once primitive with a positive leading coefficient,
so the coefficient lists must agree exactly."""

from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import AlgebraicNumber, Symbol
from sympy_extras.interop import expr_from_sympy
from sympy_extras.polys.numberfields import minimal_polynomial, minimal_polynomial_coeffs
from sympy_extras.polys.polyerrors import NotAlgebraic

sx = sympy.Symbol('x')


def _sympy_coeffs(e: sympy.Expr) -> list[int]:
    p = sympy.Poly(sympy.minimal_polynomial(e, sx), sx)
    coefficients = [int(c) for c in p.all_coeffs()]
    return [-c for c in coefficients] if coefficients[0] < 0 else coefficients


ATOMS: list[sympy.Expr] = [
    sympy.Rational(3, 7), sympy.Integer(-2), sympy.I, sympy.GoldenRatio, sympy.TribonacciConstant,
    sympy.sqrt(2), sympy.sqrt(3), sympy.sqrt(-5), sympy.cbrt(2), sympy.Integer(5)**sympy.Rational(2, 3),
    sympy.root(3, 4), sympy.sqrt(1 + sympy.sqrt(2)), 1/sympy.sqrt(7),
    sympy.cos(sympy.pi/7), sympy.cos(2*sympy.pi/5), sympy.cos(3*sympy.pi/11), sympy.sin(sympy.pi/9),
    sympy.sin(2*sympy.pi/7), sympy.sin(sympy.pi/12), sympy.tan(sympy.pi/7), sympy.tan(sympy.pi/8),
    sympy.exp(2*sympy.I*sympy.pi/7), sympy.exp(sympy.I*sympy.pi/12), sympy.exp(-3*sympy.I*sympy.pi/10),
]


@pytest.mark.parametrize('e', ATOMS, ids=str)
def test_atoms(e: sympy.Expr) -> None:
    assert minimal_polynomial_coeffs(expr_from_sympy(e)) == _sympy_coeffs(e)


def test_random_compositions() -> None:
    rng = random.Random(20261109)
    simple = [sympy.sqrt(2), sympy.sqrt(3), sympy.sqrt(5), sympy.cbrt(2), sympy.I, sympy.GoldenRatio,
              sympy.Rational(1, 2), sympy.Integer(3), sympy.cos(2*sympy.pi/7), sympy.sqrt(-3)]
    checked = 0
    for _ in range(200):
        e = rng.choice(simple)
        for _ in range(rng.randint(1, 3)):
            other = rng.choice(simple)
            op = rng.randrange(3)
            if op == 0:
                e = e + rng.randint(1, 3)*other
            elif op == 1:
                e = e*other
            else:
                e = e**rng.choice([2, -1, sympy.Rational(1, 2), sympy.Rational(-2, 3)])
        if e == 0 or not e.is_algebraic or e.has(sympy.zoo):
            continue
        got = minimal_polynomial_coeffs(expr_from_sympy(e))
        if len(got) > 17:
            continue   # SymPy is slow on these
        assert got == _sympy_coeffs(e), e
        checked += 1
    assert checked >= 100


def test_algebraic_number_and_errors() -> None:
    a = AlgebraicNumber(expr_from_sympy(sympy.sqrt(2)), [1, 1])
    assert minimal_polynomial_coeffs(a) == [1, -2, -1]
    assert str(minimal_polynomial(a, Symbol('x'))) == 'x**2 - 2*x - 1'
    with pytest.raises(NotAlgebraic):
        minimal_polynomial_coeffs(expr_from_sympy(sympy.pi))
    with pytest.raises(NotAlgebraic):
        minimal_polynomial_coeffs(expr_from_sympy(sympy.cos(1)))
    with pytest.raises(NotImplementedError):
        minimal_polynomial_coeffs(expr_from_sympy(sympy.sqrt(sympy.Symbol('y'))))
    with pytest.raises(ZeroDivisionError):
        minimal_polynomial_coeffs(expr_from_sympy(sympy.Pow(0, -1, evaluate=False)))
