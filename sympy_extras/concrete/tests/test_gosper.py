"""Gosper's algorithm against SymPy (the oracle): the term and the
definite sums of hypergeometric terms."""
from __future__ import annotations

from typing import Optional

import sympy
from sympy import binomial, factorial, rf
from sympy.concrete.gosper import gosper_sum as sympy_gosper_sum, gosper_term as sympy_gosper_term

from sympy_extras.concrete.gosper import gosper_sum, gosper_term
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

i = sympy.Symbol('i', integer=True)
n = sympy.Symbol('n', integer=True)
m = sympy.Symbol('m', integer=True)
a, b, x = sympy.symbols('a b x')
TERMS = [i*factorial(i), 1/(i*(i + 1)), binomial(m, i)*(-1)**i, i, i**2, 2**i, x**i, i*2**i, factorial(i), 1/i,
         binomial(n, i), i/((i + 1)*(i + 2)), factorial(2*i)/factorial(i)**2, rf(a, i)/factorial(i),
         1/(i**2 + 3*i + 2), (4*i + 1)*factorial(i)/factorial(2*i + 1), i**3, binomial(i, 2), x**i*i**2,
         rf(a, i)*rf(b, i)/(factorial(i)*rf(a + b, i)), (i**2 - 1)/(i**2 + 1), 1/((2*i - 1)*(2*i + 1)),
         factorial(i + 1)/factorial(i - 1), a**i/factorial(i), 1/(i*(i + 2))]


def _s(e: Optional[Expr]) -> str:
    return 'None' if e is None else str(to_sympy(e, evaluate=False))


def test_gosper_matches_sympy() -> None:
    oi = from_sympy(i)
    assert isinstance(oi, Symbol)
    for t in TERMS:
        ot = expr_from_sympy(t)
        assert _s(gosper_term(ot, oi)) == str(sympy_gosper_term(t, i)), t
        for lo in (0, 1):
            got = gosper_sum(ot, (oi, expr_from_sympy(sympy.Integer(lo)), expr_from_sympy(n)))
            assert _s(got) == str(sympy_gosper_sum(t, (i, lo, n))), (t, lo)
