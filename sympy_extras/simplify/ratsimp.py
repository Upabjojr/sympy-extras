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

Not ported yet: ``ratsimpmodprime``. The domains ``ZZ_I``, ``RR`` and
``EX`` go through SymPy.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/ratsimp.py (ratsimp), sympy/polys/rings.py
# (PolyElement.div, one divisor over a field) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the division on dictionaries of Fraction coefficients.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, number_from_fraction
from sympy_extras.core.power import pow

__all__ = ['ratsimp']

Monomial = tuple[int, ...]
Terms = dict[Monomial, Fraction]


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.div (one divisor, a field)
def _divide(f: Terms, g: Terms) -> tuple[Terms, Terms]:
    """The quotient and the remainder of ``f`` by ``g`` (lexicographic
    order): leading terms divided while ``g``'s leading monomial divides
    them, moved to the remainder otherwise."""
    lg = max(g)
    cg = g[lg]
    p = dict(f)
    q: Terms = {}
    r: Terms = {}
    while p:
        lp = max(p)
        cp = p[lp]
        if all(a >= b for a, b in zip(lp, lg)):
            m = tuple(a - b for a, b in zip(lp, lg))
            c = cp / cg
            q[m] = q.get(m, Fraction(0)) + c
            if not q[m]:
                del q[m]
            for mg, coefficient in g.items():
                k = tuple(a + b for a, b in zip(m, mg))
                value = p.get(k, Fraction(0)) - c * coefficient
                if value:
                    p[k] = value
                else:
                    p.pop(k, None)
        else:
            r[lp] = r.get(lp, Fraction(0)) + cp
            del p[lp]
    return q, r


def _expr_of(terms: Terms, gens: tuple[Expr, ...]) -> Expr:
    return add(*[mul(number_from_fraction(c), *[pow(g, Integer(k)) for g, k in zip(gens, m) if k])
                 for m, c in sorted(terms.items(), reverse=True)])


# PORTED FROM SYMPY: sympy/simplify/ratsimp.py:ratsimp
def ratsimp(expr: ExprLike) -> Expr:
    """The cancelled expression as the quotient of its numerator by its
    denominator plus the cancelled remainder over the denominator."""
    from sympy_extras.polys.polyerrors import CoercionFailed
    from sympy_extras.polys.polytools import _joint_rings, cancel
    f, g = cancel(to_expr(expr)).as_numer_denom()
    try:
        gens, (pf, pg) = _joint_rings([f, g], expand=False)
    except CoercionFailed:
        import sympy
        from sympy_extras.interop import expr_through_sympy
        return expr_through_sympy(to_expr(expr), sympy.ratsimp)
    if not gens:
        # SymPy's ComputationFailed (no generators)
        return mul(f, pow(g, Integer(-1)))
    q, r = _divide(pf, pg)
    return add(_expr_of(q, gens), cancel(mul(_expr_of(r, gens), pow(g, Integer(-1)))))
