"""Sum and Product on the typed core against SymPy (the oracle): the
construction, the bound variables (free symbols, substitution), the
evaluation and factor_terms."""
from __future__ import annotations

import sympy

from sympy_extras.concrete.expr_with_limits import Product, Sum, product_, sum_
from sympy_extras.core import to_sympy
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.expr import Expr
from sympy_extras.core.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

i, j, k, n, m, x = sympy.symbols('i j k n m x')


def _s(e: Expr) -> str:
    return str(to_sympy(e, evaluate=False))


def _sym(s: sympy.Symbol) -> Symbol:
    r = from_sympy(s)
    assert isinstance(r, Symbol)
    return r


CASES = [sympy.Sum(x**i, (i, 0, n)), sympy.Sum(i*j, (i, 0, j), (j, 0, n)), sympy.Product(i, (i, 1, n)),
         sympy.Sum(1/i**2, (i, 1, sympy.oo)), sympy.Product(x + i, (i, 1, m)), sympy.Sum(sympy.binomial(n, i), (i, 0, n)),
         sympy.Sum(sympy.Piecewise((i, i < n), (0, True)), (i, 0, m)), sympy.Sum(i*x + x, (i, 0, n), (j, 0, m))]


def test_construction_and_bound_variables_match_sympy() -> None:
    for e in CASES:
        ours = expr_from_sympy(e)
        assert isinstance(ours, (Sum, Product)) and _s(ours) == str(e)
        assert {str(s) for s in ours.free_symbols} == {str(s) for s in e.free_symbols}, e
        for old, new in [(i, k), (n, 3), (x, 2*k), (j, n), (m, n + 1)]:
            assert _s(ours.subs(_sym(old), expr_from_sympy(new))) == str(e.subs(old, new)), (e, old, new)
    f = expr_from_sympy(sympy.Sum(i*j, (i, 0, j)))
    assert _s(sum_(f, (_sym(j), 0, _sym(n)))) == str(sympy.Sum(sympy.Sum(i*j, (i, 0, j)), (j, 0, n)))
    assert _s(product_(expr_from_sympy(i), (_sym(i), 1, _sym(n)))) == 'Product(i, (i, 1, n))'


def test_doit_and_factor_terms_match_sympy() -> None:
    for e in CASES:
        ours = expr_from_sympy(e)
        assert _s(ours.doit()) == str(e.doit()), e
        assert _s(factor_terms(ours)) == str(sympy.factor_terms(e)), e
    for e in [2*sympy.Sum(x*i + x, (i, 0, n)) + 2*x, sympy.Sum(x + i, (i, 0, n))]:
        assert _s(factor_terms(expr_from_sympy(e))) == str(sympy.factor_terms(e)), e


def test_simplify_of_sums_matches_sympy() -> None:
    from sympy_extras.simplify.simplify import simplify
    for e in [sympy.Sum(x*i, (i, 0, n)) + sympy.Sum(x*i, (i, n + 1, m)), sympy.Sum(i, (i, 0, n)) - sympy.Sum(i, (i, 0, n)),
              2*sympy.Sum(sympy.sin(x)**2 + sympy.cos(x)**2, (i, 0, n)),
              sympy.Product(i, (i, 1, n))*sympy.Product(i, (i, n + 1, m)), sympy.Sum(i**2, (i, 0, n)).doit()]:
        assert _s(simplify(expr_from_sympy(e))) == str(sympy.simplify(e)), e
