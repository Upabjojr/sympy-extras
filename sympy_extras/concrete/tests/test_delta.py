"""Sums and products with Kronecker deltas against SymPy (the oracle)."""
from __future__ import annotations

import sympy
from sympy import KroneckerDelta as KD, oo
from sympy.concrete.delta import deltaproduct as sympy_deltaproduct, deltasummation as sympy_deltasummation

from sympy_extras.concrete.delta import deltaproduct, deltasummation
from sympy_extras.concrete.products import product
from sympy_extras.concrete.summations import summation
from sympy_extras.interop import to_sympy
from sympy_extras.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

i, j, k, n = sympy.symbols('i j k n', integer=True)
x = sympy.Symbol('x')
FS = [KD(i, k), k*KD(i, k), KD(i, 2*k), KD(i, k)*KD(j, k), x**k*KD(k, 3), KD(i, k) + k, (KD(i, k) + 1)*k,
      KD(i, k + 1)*k**2, KD(k, 2) + KD(k, 3), KD(i, j)*k]
LIMITS = [(k, 0, n), (k, 1, 3), (k, 1, n), (k, -oo, oo), (k, 0, 5)]


def test_delta_sums_and_products_match_sympy() -> None:
    for f in FS:
        for v, lo, hi in LIMITS:
            slo, shi = sympy.sympify(lo), sympy.sympify(hi)
            var = from_sympy(v)
            assert isinstance(var, Symbol)
            limits = (var, expr_from_sympy(slo), expr_from_sympy(shi))
            of = expr_from_sympy(f)
            assert str(to_sympy(deltasummation(of, limits), evaluate=False)) == str(sympy_deltasummation(f, (v, slo, shi)))
            assert str(to_sympy(summation(of, limits), evaluate=False)) == str(sympy.summation(f, (v, slo, shi)))
            if lo == -oo:
                continue
            assert str(to_sympy(deltaproduct(of, limits), evaluate=False)) == str(sympy_deltaproduct(f, (v, slo, shi)))
            assert str(to_sympy(product(of, limits), evaluate=False)) == str(sympy.product(f, (v, slo, shi)))


def test_canonical_orders_by_nodes_as_sympy() -> None:
    from sympy_extras.core.relational import Relational
    for r in [i/2 <= n, n >= i/2, x + 1 < 2*n, -x <= n, i < x**2 + 1]:
        ours = from_sympy(r)
        assert isinstance(ours, Relational)
        assert str(to_sympy(ours.canonical, evaluate=False)) == str(r.canonical), r
