"""summation (Sum.doit) against SymPy (the oracle)."""
from __future__ import annotations

import sympy
from sympy import KroneckerDelta, Piecewise, Rational as R, binomial, exp, factorial, oo, sin

from sympy_extras.concrete.summations import summation
from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

i = sympy.Symbol('i', integer=True)
k = sympy.Symbol('k', integer=True)
n = sympy.Symbol('n', integer=True)
m = sympy.Symbol('m', integer=True, positive=True)
x, y = sympy.symbols('x y')
j = sympy.Symbol('j')

CASES = [(i**2, (i, 1, n)), (i**3, (i, 0, n)), (x**i, (i, 0, n)), (2**i, (i, 1, n)), (1/(i*(i + 1)), (i, 1, n)),
         (i, (i, 1, 4)), (1/i**2, (i, 1, oo)), (1/i, (i, 1, n)), (1/i**3, (i, 1, n)), (binomial(n, i), (i, 0, n)),
         (i*binomial(n, i), (i, 0, n)), (x**i/factorial(i), (i, 0, oo)), (i*x**i, (i, 0, n)), (1/(i**2 - 1), (i, 2, n)),
         (sin(i), (i, 0, 3)), (i**2 + 3*i + 1, (i, 0, m)), (y, (i, 1, n)), (i*j, (i, 1, n)),
         (Piecewise((i, i < 5), (0, True)), (i, 0, 10)), (KroneckerDelta(i, k), (i, 0, n)),
         (1/((i + 1)*(i + 2)), (i, 0, oo)), (R(1, 2)**i, (i, 0, oo)), (factorial(i), (i, 0, n)), (i**4, (i, 1, n)),
         (1/i**2, (i, 1, n)), ((-1)**i/i, (i, 1, oo)), (exp(i), (i, 0, n)), (i/(i + 1), (i, 0, 5)), (x, (i, n, n)),
         (i, (i, 5, 3)), (i, (i, 1, 0)), (i**2, (i, -n, n)), (1/(2*i + 1), (i, 0, n)), (j**i, (i, 1, n)),
         (i*(i - 1)*binomial(n, i), (i, 0, n)), (binomial(m, i)*binomial(m, m - i), (i, 0, m)), (i**R(1, 2), (i, 1, n)),
         (1/i**(x + 1), (i, 1, oo)), (1/(i + 1)**2, (i, 0, oo))]


def test_summation_matches_sympy() -> None:
    for f, lim in CASES:
        v, a, b = lim
        var = from_sympy(v)
        assert isinstance(var, Symbol)
        ours = summation(expr_from_sympy(f), (var, expr_from_sympy(sympy.sympify(a)), expr_from_sympy(sympy.sympify(b))))
        assert str(to_sympy(ours, evaluate=False)) == str(sympy.summation(f, lim)), (f, lim)


def test_sum_doit_matches_sympy() -> None:
    for f, lim in CASES[:12]:
        s = sympy.Sum(f, lim)
        assert str(to_sympy(expr_from_sympy(s).doit(), evaluate=False)) == str(s.doit()), s
