"""The evaluation of products (SymPy's ``Product.doit``, ``product``,
``Product._eval_product``): definite products of few factors factor by
factor, polynomials by their roots and rising factorials, products of
factors and powers piece by piece (the exponents summed).

Ported from SymPy's ``sympy.concrete.products`` (the Kronecker delta
products in :mod:`~sympy_extras.concrete.delta`). A polynomial factor
whose coefficients our Poly does not take goes through SymPy.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.concrete.products import product
>>> k, n, x = Symbol('k', integer=True), Symbol('n', integer=True), Symbol('x')
>>> product(k, (k, 1, n)), product(x**k, (k, 1, n)), product(k**2 + k, (k, 1, 3))
(factorial(n), x**(n**2/2 + n/2), 144)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/concrete/products.py (Product.doit,
# Product._eval_product, Product._eval_product_direct, product)
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, NaN
from sympy_extras.core.power import Pow

from .expr_with_limits import LimitLike, Limits, Product, product_

__all__ = ['product', 'product_doit', 'eval_product']

ONE = Integer(1)


class _ThroughSymPy(Exception):
    """A case the native code does not take: the product goes to SymPy."""


def _e(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


def _through(name: str, f: Expr, limits: Limits) -> Optional[Expr]:
    import importlib
    from sympy_extras.core.interop import expr_from_sympy, to_sympy
    module, function = name.rsplit('.', 1)
    fn = getattr(importlib.import_module('sympy.concrete.' + module), function)
    k, a, n = limits
    r = fn(to_sympy(f), (to_sympy(k), to_sympy(a), to_sympy(n)))
    return None if r is None else expr_from_sympy(r)


# PORTED FROM SYMPY: sympy/concrete/products.py:Product._eval_product_direct
def _eval_product_direct(term: Expr, limits: Limits) -> Expr:
    k, a, n = limits
    count = n - a + 1
    assert isinstance(count, Integer)
    return mul(*[_e(term.subs(k, a + j)) for j in range(count.p)])


def _doit(term: Expr, limits: Limits) -> Expr:
    p = product_(term, limits)
    return product_doit(p) if isinstance(p, Product) else p


# PORTED FROM SYMPY: sympy/concrete/products.py:Product._eval_product
def eval_product(term: Expr, limits: Limits) -> Optional[Expr]:
    """The product of ``term`` over one limit, ``None`` when no method
    works."""
    from sympy_extras.core.exprtools import factor_terms
    from sympy_extras.functions.combinatorial.factorials import rf
    from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode
    from sympy_extras.polys.polyerrors import CoercionFailed
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.polytools import Poly
    from .summations import summation
    k, a, n = limits
    if k not in term.free_symbols:
        if (term - 1).is_zero is True:
            return ONE
        return term**(n - a + 1)
    if a == n:
        return _e(term.subs(k, a))
    if term.has(KroneckerDeltaNode):
        delta = _through_delta(term, limits)
        if delta is not None:
            return delta
    dif = n - a
    definite = isinstance(dif, Integer)
    if isinstance(dif, Integer) and dif.p < 100:
        return _eval_product_direct(term, limits)
    if term.is_polynomial(k):
        try:
            poly = Poly(term, k)
        except (CoercionFailed, NotImplementedError) as exc:
            raise _ThroughSymPy() from exc
        A: Expr = ONE
        B: Expr = ONE
        Q: Expr = ONE
        found = roots(poly)
        assert isinstance(found, dict)
        M = 0
        for r, m in found.items():
            M += m
            A = A * rf(a - r, n - a + 1)**m
            Q = Q * (n - r)**m
        if M < poly.degree():
            from sympy_extras.core.add import add
            from sympy_extras.polys.polytools import cancel
            # SymPy's quo(poly, Q.as_poly(k)): Q, in n, is a constant in k,
            # each coefficient divided (an element of the fraction field)
            arg = add(*[cancel(c / Q) * k**m[0] for m, c in poly.terms()])
            B = _doit(arg, (k, a, n))
        return poly.LC()**(n - a + 1) * A * B
    if isinstance(term, Add):
        factored = factor_terms(term, fraction=True)
        if isinstance(factored, Mul):
            return eval_product(factored, (k, a, n))
    elif isinstance(term, Mul):
        factors = [f for f in term.args if isinstance(f, Expr)]
        without_k = mul(*[f for f in factors if not f.has(k)])
        with_k = [f for f in factors if f.has(k)]
        if len(with_k) >= 2:
            exclude: list[Expr] = []
            include: list[Expr] = []
            for t in with_k:
                p = eval_product(t, (k, a, n))
                if p is not None:
                    exclude.append(p)
                else:
                    include.append(t)
            if not exclude:
                return None
            arg = mul(*include)
            return without_k**(n - a + 1) * mul(*exclude) * _doit(arg, (k, a, n))
        p1 = eval_product(with_k[0], (k, a, n))
        if p1 is None:
            p1 = _doit(with_k[0], (k, a, n))
        return without_k**(n - a + 1) * p1
    elif isinstance(term, Pow):
        base, exponent = _e(term.args[0]), _e(term.args[1])
        if not base.has(k):
            s = summation(exponent, (k, a, n))
            return base**s
        if not exponent.has(k):
            p2 = eval_product(base, (k, a, n))
            if p2 is not None:
                return p2**exponent
    elif isinstance(term, Product):
        evaluated = product_doit(term)
        f = eval_product(evaluated, limits)
        if f is None:
            return product_(evaluated, limits)
        return f
    if definite:
        return _eval_product_direct(term, limits)
    return None


def _through_delta(term: Expr, limits: Limits) -> Optional[Expr]:
    """SymPy's ``deltaproduct`` when the delta is simple."""
    from .delta import _has_simple_delta, deltaproduct
    if not _has_simple_delta(term, limits[0]):
        return None
    return deltaproduct(term, limits)


# PORTED FROM SYMPY: sympy/concrete/products.py:Product.doit
def product_doit(p: Product, deep: bool = True) -> Expr:
    """The product evaluated: each limit by :func:`eval_product`, the
    product kept where no method works."""
    from sympy_extras.simplify.powsimp import powsimp
    from .summations import _dummy_with_inherited_properties
    reps: dict[Basic, Basic] = {}
    for xab in p.limits:
        d = _dummy_with_inherited_properties(xab)
        if d is not None:
            reps[xab[0]] = d
    if reps:
        undo = {v: k for k, v in reps.items()}
        replaced = p.xreplace(reps)
        assert isinstance(replaced, Product)
        return _e(product_doit(replaced, deep).xreplace(undo))
    try:
        f = p.function
        for index, limit in enumerate(p.limits):
            i, a, b = limit
            dif = b - a
            if dif.is_integer is True and dif.is_negative is True:
                a, b = b + 1, a - 1
                f = 1 / f
            g = eval_product(f, (i, a, b))
            if g is None or isinstance(g, NaN):
                return product_(powsimp(f), *p.limits[index:])
            f = g
    except _ThroughSymPy:
        from sympy_extras.core.interop import expr_through_sympy
        return expr_through_sympy(p, lambda s: s.doit(deep=deep))
    if deep:
        return f.doit()
    return powsimp(f)


def product(f: ExprLike, *limits: LimitLike) -> Expr:
    """The product of ``f`` over the limits (SymPy's ``product``:
    ``Product(f, *limits).doit(deep=False)``)."""
    p = product_(to_expr(f), *limits)
    if not isinstance(p, Product):
        return p
    return product_doit(p, deep=False)
