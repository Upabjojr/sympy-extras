"""``ratsimp``: a rational function as a polynomial plus a proper
fraction.

Ported from SymPy's ``sympy.simplify.ratsimp``: the expression is
cancelled, its numerator divided by its denominator (SymPy's
``reduced(f, [g], field=True, expand=False)``: the multivariate division
over ``QQ`` in the lexicographic order of the generators), and the
remainder over the denominator cancelled.

>>> from sympy_extras.core import Symbol
>>> x, y = Symbol('x'), Symbol('y')
>>> ratsimp(1/x + 1/y), ratsimp((x**2 + 1)/(x + 1))
((x + y)/(x*y), x - 1 + 2/(x + 1))

The division is over ``QQ``, ``QQ_I`` (Gaussian coefficients) and, for
the other coefficients, ``EX``. Not ported yet: ``ratsimpmodprime``.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/ratsimp.py (ratsimp), sympy/polys/rings.py
# (PolyElement.div, one divisor over a field) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the division over the typed domains QQ, QQ_I and EX.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import TypeVar

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.polys.domains.domain import Domain

__all__ = ['ratsimp']

Monomial = tuple[int, ...]
T = TypeVar('T')


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.div (one divisor, a field)
def _divide(f: dict[Monomial, T], g: dict[Monomial, T], K: Domain[T]) -> tuple[dict[Monomial, T], dict[Monomial, T]]:
    """The quotient and the remainder of ``f`` by ``g`` over the field
    ``K`` (lexicographic order): leading terms divided while ``g``'s
    leading monomial divides them, moved to the remainder otherwise."""
    lg = max(g)
    cg = g[lg]
    p = dict(f)
    q: dict[Monomial, T] = {}
    r: dict[Monomial, T] = {}
    while p:
        lp = max(p)
        cp = p[lp]
        if all(a >= b for a, b in zip(lp, lg)):
            m = tuple(a - b for a, b in zip(lp, lg))
            c = K.quo(cp, cg)
            q[m] = K.add(q.get(m, K.zero), c)
            if K.is_zero(q[m]):
                del q[m]
            for mg, coefficient in g.items():
                k = tuple(a + b for a, b in zip(m, mg))
                value = K.sub(p.get(k, K.zero), K.mul(c, coefficient))
                if not K.is_zero(value):
                    p[k] = value
                else:
                    p.pop(k, None)
        else:
            r[lp] = K.add(r.get(lp, K.zero), cp)
            del p[lp]
    return q, r


def _expr_of(terms: dict[Monomial, T], gens: tuple[Expr, ...], K: Domain[T]) -> Expr:
    return add(*[mul(K.to_expr(c), *[pow(g, Integer(k)) for g, k in zip(gens, m) if k])
                 for m, c in sorted(terms.items(), key=lambda t: t[0], reverse=True)])


def _quotient_and_remainder(gens: tuple[Expr, ...], f: dict[Monomial, T], g: dict[Monomial, T],
                            K: Domain[T]) -> tuple[Expr, Expr]:
    q, r = _divide(f, g, K)
    return _expr_of(q, gens, K), _expr_of(r, gens, K)


# PORTED FROM SYMPY: sympy/simplify/ratsimp.py:ratsimp
def ratsimp(expr: ExprLike) -> Expr:
    """The cancelled expression as the quotient of its numerator by its
    denominator plus the cancelled remainder over the denominator."""
    from sympy_extras.polys.domains.expressiondomain import EX
    from sympy_extras.polys.domains.gaussiandomains import QQ_I
    from sympy_extras.polys.domains.rationalfield import QQ
    from sympy_extras.polys.polyerrors import CoercionFailed
    from sympy_extras.polys.polytools import _joint_rings, _joint_rings_ex, _joint_rings_gaussian, cancel
    f, g = cancel(to_expr(expr)).as_numer_denom()
    gens: tuple[Expr, ...]
    try:
        gens, (pf, pg) = _joint_rings([f, g], expand=False)
        if gens:
            q, r = _quotient_and_remainder(gens, pf, pg, QQ)
    except CoercionFailed:
        try:
            gens, (gf, gg) = _joint_rings_gaussian([f, g], expand=False)
            if gens:
                q, r = _quotient_and_remainder(gens, gf, gg, QQ_I)
        except CoercionFailed:
            gens, (ef, eg) = _joint_rings_ex([f, g])
            if gens:
                q, r = _quotient_and_remainder(gens, ef, eg, EX)
    if not gens:
        # SymPy's ComputationFailed (no generators)
        return mul(f, pow(g, Integer(-1)))
    return add(q, cancel(mul(r, pow(g, Integer(-1)))))
