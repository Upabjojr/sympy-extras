"""Conversions between the expressions of the typed core
(:mod:`sympy_extras.core`) and the polynomials of the typed layer, without
SymPy.

``dict_from_expr`` walks the expression tree (``Add``, ``Mul``, ``Pow``
with a nonnegative integer exponent, the generators as ``Symbol``) and
multiplies out with the ring arithmetic, so that no expansion of the
expression is needed; a subexpression free of the generators is a
coefficient, converted by the domain (``Domain.from_expr``).
``expr_from_dict`` builds the expression back with the evaluating
functions ``add``, ``mul``, ``pow`` of the core.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.core import Symbol, Rational
>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.polyutils import dict_from_expr, expr_from_dict
>>> x, y = Symbol('x'), Symbol('y')
>>> terms = dict_from_expr((x + y)**2 - Rational(1, 2)*x, ['x', 'y'], QQ)
>>> terms == {(2, 0): 1, (1, 1): 2, (0, 2): 1, (1, 0): Fraction(-1, 2)}
True
>>> expr_from_dict(terms, ['x', 'y'], QQ)
x**2 + 2*x*y - x/2 + y**2
"""
from __future__ import annotations

from typing import Sequence, TypeVar

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol

from .domains.domain import Domain
from .rings import Monomial

__all__ = ['dict_from_expr', 'expr_from_dict']

T = TypeVar('T')

Terms = dict[Monomial, T]


def _add_terms(a: Terms[T], b: Terms[T], K: Domain[T]) -> Terms[T]:
    result = dict(a)
    for m, c in b.items():
        if m in result:
            s = K.add(result[m], c)
            if K.is_zero(s):
                del result[m]
            else:
                result[m] = s
        elif not K.is_zero(c):
            result[m] = c
    return result


def _mul_terms(a: Terms[T], b: Terms[T], K: Domain[T]) -> Terms[T]:
    result: Terms[T] = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = tuple(i + j for i, j in zip(m1, m2))
            product = K.mul(c1, c2)
            if m in result:
                s = K.add(result[m], product)
                if K.is_zero(s):
                    del result[m]
                else:
                    result[m] = s
            elif not K.is_zero(product):
                result[m] = product
    return result


def _pow_terms(a: Terms[T], n: int, one: Terms[T], K: Domain[T]) -> Terms[T]:
    result = one
    base = a
    while n:
        if n & 1:
            result = _mul_terms(result, base, K)
        n >>= 1
        if n:
            base = _mul_terms(base, base, K)
    return result


def dict_from_expr(expr: Expr, gens: Sequence[str], K: Domain[T]) -> Terms[T]:
    """The terms ``{exponents: coefficient}`` of a core expression which
    is a polynomial in the generators named ``gens`` with coefficients in
    ``K``; ``ValueError`` when it is not one (a negative or symbolic power
    of a generator, a generator inside a function)."""
    names = list(gens)
    index = {name: i for i, name in enumerate(names)}
    zero_monomial: Monomial = tuple(0 for _ in names)
    one: Terms[T] = {zero_monomial: K.one}

    def walk(e: Expr) -> Terms[T]:
        if isinstance(e, Symbol) and e.name in index:
            return {tuple(1 if j == index[e.name] else 0 for j in range(len(names))): K.one}
        if not any(s.name in index for s in e.free_symbols if isinstance(s, Symbol)):
            c = K.from_expr(e)
            return {} if K.is_zero(c) else {zero_monomial: c}
        if isinstance(e, Add):
            total: Terms[T] = {}
            for a in e.args:
                total = _add_terms(total, walk(a), K)
            return total
        if isinstance(e, Mul):
            product = one
            for a in e.args:
                product = _mul_terms(product, walk(a), K)
            return product
        if isinstance(e, Pow):
            exponent = e.exp
            if isinstance(exponent, Integer) and exponent.p >= 0:
                return _pow_terms(walk(e.base), exponent.p, one, K)
        raise ValueError('%s is not a polynomial in %s' % (e, ', '.join(names)))

    return walk(expr)


def expr_from_dict(terms: Terms[T], gens: Sequence[str], K: Domain[T]) -> Expr:
    """The core expression of the terms, generators named ``gens``."""
    symbols = [Symbol(name) for name in gens]
    parts: list[Expr] = []
    for m, c in terms.items():
        factors: list[Expr] = [K.to_expr(c)]
        for s, e in zip(symbols, m):
            if e:
                factors.append(pow(s, Integer(e)) if e > 1 else s)
        parts.append(mul(*factors))
    return add(*parts) if parts else Integer(0)
