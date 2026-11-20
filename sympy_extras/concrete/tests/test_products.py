"""product (Product.doit) against SymPy (the oracle)."""
from __future__ import annotations

import sympy
from sympy import KroneckerDelta, Product as SympyProduct, Rational as R, exp, factorial, oo, sin

from sympy_extras.concrete.products import product
from sympy_extras.interop import to_sympy
from sympy_extras.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

k = sympy.Symbol('k', integer=True)
n = sympy.Symbol('n', integer=True)
m = sympy.Symbol('m', integer=True, positive=True)
x, y = sympy.symbols('x y')
j = sympy.Symbol('j', integer=True)
CASES = [(k, (k, 1, n)), (x**k, (k, 1, n)), (k**2 + k, (k, 1, 3)), (2, (k, 1, n)), (k + 1, (k, 0, n)), (k**2, (k, 1, n)),
         (2*k, (k, 1, n)), (x*k, (k, 1, m)), (1 + 1/k, (k, 1, n)), (k/(k + 1), (k, 1, n)), (exp(k), (k, 1, n)),
         (k**2 - 1, (k, 2, n)), (k**3 + 2, (k, 1, n)), (sin(k), (k, 1, 4)), (SympyProduct(j, (j, 1, k)), (k, 1, 3)),
         (k, (k, 5, 3)), (k, (k, 1, 0)), (y, (k, 1, n)), (KroneckerDelta(k, j), (k, 1, n)), (k*(k + 1), (k, 1, n)),
         (2**k*k, (k, 1, n)), (x**(2*k + 1), (k, 0, n)), (k**R(1, 2), (k, 1, n)), (1 - 1/(4*k**2), (k, 1, oo)),
         (factorial(k), (k, 1, 3)), (k + x, (k, 1, n)), ((k + 1)**2/k**2, (k, 1, n))]


def test_product_matches_sympy() -> None:
    for f, (v, lo, hi) in CASES:
        var = from_sympy(v)
        assert isinstance(var, Symbol)
        limits = (var, expr_from_sympy(sympy.sympify(lo)), expr_from_sympy(sympy.sympify(hi)))
        got = product(expr_from_sympy(f), limits)
        assert str(to_sympy(got, evaluate=False)) == str(sympy.product(f, (v, lo, hi))), (f, v, lo, hi)


def test_product_doit_matches_sympy() -> None:
    for f, lim in CASES[:12]:
        p = SympyProduct(f, lim)
        assert str(to_sympy(expr_from_sympy(p).doit(), evaluate=False)) == str(p.doit()), p


def test_product_of_a_polynomial_with_some_roots_not_found() -> None:
    # SymPy's Product._eval_product took the product of quo(poly, Q.as_poly(k))
    # with Q = prod((n - r)**m), a constant in k: the whole polynomial again,
    # so the native port recursed without end on k**5 - k - 1 (no root
    # found) and SymPy fails with an AttributeError on k*(k**5 - k - 1)
    from sympy_extras.concrete.expr_with_limits import Product
    from sympy_extras.core.numbers import Integer
    nk, nn = Symbol('k', integer=True), Symbol('n', integer=True)
    for f in [nk**5 - nk - 1, nk*(nk**5 - nk - 1), 3*(nk + 2)**2*(nk**5 - nk - 1),
              (nk - 1)*(nk**3 - 2)*(nk**5 - nk - 1)]:
        got = product(f, (nk, Integer(1), nn))
        assert got.has(Product), f
        for top in range(2, 5):
            want = product(f, (nk, Integer(1), Integer(top)))
            assert (got.subs(nn, Integer(top)).doit() - want).expand() == 0, (f, top)
