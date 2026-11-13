"""Expr.equals by the native steps of SymPy's (the simplified difference,
its factors, the minimal polynomial of a number), checked against SymPy's
equals."""
from __future__ import annotations

import random

import sympy

from sympy_extras.core.expr import _simplified_zero_test
from sympy_extras.interop import expr_from_sympy


def test_equals_matches_sympy() -> None:
    x, y = sympy.symbols('x y')
    rng = random.Random(7)
    pieces = [(x + 1)**2 - x**2 - 2*x - 1, sympy.sin(x)**2 + sympy.cos(x)**2 - 1, sympy.sqrt(2)*sympy.sqrt(3) - sympy.sqrt(6),
              sympy.sqrt(5 + 2*sympy.sqrt(6)) - sympy.sqrt(2) - sympy.sqrt(3), x - 1, x*y - y*x, sympy.exp(x)*sympy.exp(-x) - 1,
              sympy.Rational(1, 3), (x**2 - 1)/(x - 1) - x - 1, sympy.log(4) - 2*sympy.log(2), sympy.sqrt(2) - 1]
    decided = 0
    for _ in range(40):
        a, b = rng.choice(pieces), rng.choice(pieces)
        e = sympy.Add(a, rng.choice([1, -1, 2]) * b, evaluate=False)
        want = e.equals(0)
        got = _simplified_zero_test(expr_from_sympy(e))
        # the native steps decide part of what SymPy decides, and agree
        if got is not None:
            decided += 1
            assert got == want, (e, got, want)
    assert decided > 20
