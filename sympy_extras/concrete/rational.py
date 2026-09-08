"""Indefinite summation of rational functions: Abramov's decomposition
`f = \\Delta g + h` with a remainder `h` of minimal denominator.

For a rational function `f(k)` the denominator is factored and its
irreducible factors are grouped into *shift classes* (`q_2(k) = q_1(k +
j)` for an integer `j`). A partial fraction term `c(k)/q(k + j)^m` with
`j > 0` differs from `c(k - j)/q(k)^m` by a difference `\\Delta g` with
`g = \\sum_{i=0}^{j-1} c(k - j + i)/q(k + i)^m`, so every class is moved to
a single representative; the polynomial part is summed by Faulhaber's
formula. What remains, `h`, has a denominator whose factors are pairwise
shift-inequivalent, and `f` has a rational indefinite sum exactly when
`h = 0` (Abramov). When `h \\ne 0` the sum of `h` is not rational: SymPy
writes it with polygamma functions when the poles are simple.

SymPy's ``summation`` sums rational functions through partial fractions
and polygamma functions but does not separate the rational part; Karr's
algorithm (:mod:`sympy_extras.concrete.karr`) decides rational summability
in a ΠΣ-field without producing the remainder.

References
==========

.. [Abramov] S. A. Abramov, The rational component of the solution of a
   first-order linear recurrence relation with a rational right side,
   USSR Computational Mathematics and Mathematical Physics 15 (1975).
.. [Paule] P. Paule, Greatest factorial factorization and symbolic
   summation, Journal of Symbolic Computation 20 (1995).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.concrete.summations import Sum, summation
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor_list, div

from sympy_extras._typing import as_expr

__all__ = ['abramov_decomposition', 'rational_sum', 'RationalDecomposition']


class RationalDecomposition:
    """``f = g(k + 1) - g(k) + h`` with ``h`` of minimal denominator.

    Attributes
    ==========

    g : Expr
        The rational part of the sum.
    h : Expr
        The remainder, ``0`` exactly when ``f`` has a rational sum.
    """

    def __init__(self, g: Expr, h: Expr) -> None:
        self.g = g
        self.h = h

    @property
    def summable(self) -> bool:
        return self.h == 0

    def __repr__(self) -> str:
        return "RationalDecomposition(%s, %s)" % (self.g, self.h)


def _shift(q1: Poly, q2: Poly, k: Symbol) -> Optional[int]:
    """``j`` with ``q2(k) == q1(k + j)`` for monic polynomials of the same
    degree, or ``None``."""
    if q1.degree() != q2.degree() or q1.degree() == 0:
        return None
    d = q1.degree()
    c1 = as_expr(q1.all_coeffs()[1]) if d >= 1 else S.Zero
    c2 = as_expr(q2.all_coeffs()[1]) if d >= 1 else S.Zero
    # the coefficient of k**(d-1) of q1(k + j) is c1 + d*j
    j = as_expr(cancel((c2 - c1)/d))
    if not isinstance(j, Integer):
        return None
    if as_expr(q1.as_expr().subs(k, k + j) - q2.as_expr()).expand() == 0:
        return int(j)
    return None


def abramov_decomposition(f: Union[Expr, int], k: Symbol) -> RationalDecomposition:
    """Abramov's decomposition of a rational function of ``k``.

    Examples
    ========

    >>> from sympy.abc import k
    >>> from sympy_extras.concrete.rational import abramov_decomposition
    >>> abramov_decomposition(1/(k*(k + 1)), k)
    RationalDecomposition(-1/k, 0)
    >>> abramov_decomposition(1/k, k)
    RationalDecomposition(0, 1/k)
    >>> abramov_decomposition((2*k + 1)/(k**2*(k + 1)**2), k)
    RationalDecomposition(-1/k**2, 0)
    """
    expression = as_expr(cancel(sympify(f)))
    if not expression.is_rational_function(k):
        raise ValueError("%s is not a rational function of %s" % (expression, k))
    numerator, denominator = expression.as_numer_denom()
    try:
        num, den = Poly(numerator, k), Poly(denominator, k)
    except PolynomialError:
        raise ValueError("%s is not a rational function of %s" % (expression, k))
    quotient, remainder = div(num, den)
    g: Expr = S.Zero
    # the polynomial part: sum_{i=0}^{k-1} quotient(i)
    if not quotient.is_zero:
        i = Dummy('i', integer=True)
        summed = summation(quotient.as_expr().subs(k, i), (i, 0, k - 1))
        if isinstance(summed, Sum):
            raise ValueError("the polynomial part could not be summed")
        g = as_expr(g + summed)
    if remainder.is_zero:
        return RationalDecomposition(as_expr(cancel(g)), S.Zero)
    # the shift classes of the irreducible factors of the denominator
    lead, factors = factor_list(den.as_expr(), k)
    monic: list[tuple[Poly, int]] = [(Poly(fac, k).monic(), int(m)) for fac, m in factors]
    classes: list[list[tuple[Poly, int]]] = []   # (factor, shift from the representative)
    for factor_poly, _ in monic:
        placed = False
        for group in classes:
            j = _shift(group[0][0], factor_poly, k)
            if j is not None:
                group.append((factor_poly, j))
                placed = True
                break
        if not placed:
            classes.append([(factor_poly, 0)])
    # partial fractions of the proper part
    proper = as_expr(remainder.as_expr()/den.as_expr())
    terms = _partial_terms(proper, k)
    h_terms: list[Expr] = []
    for c, q, m in terms:
        q_poly = Poly(q, k).monic()
        scale = as_expr(Poly(q, k).LC())
        c_ = as_expr(c/scale**m)
        # the class and the shift of this factor
        representative: Optional[Poly] = None
        shift = 0
        for group in classes:
            for member, j in group:
                if as_expr(member.as_expr() - q_poly.as_expr()).expand() == 0:
                    representative = group[0][0]
                    shift = j
                    break
            if representative is not None:
                break
        if representative is None:
            h_terms.append(as_expr(c_/q_poly.as_expr()**m))
            continue
        # the leftmost member of the class is the target denominator
        least = min(j for _, j in group)
        offset = shift - least
        target = as_expr(representative.as_expr().subs(k, k + least))
        if offset == 0:
            h_terms.append(as_expr(c_/target**m))
            continue
        # c(k)/q(k + offset)^m = c(k - offset)/q(k)^m + Delta(sum_{i=0}^{offset-1} c(k - offset + i)/q(k + i)^m)
        h_terms.append(as_expr(c_.subs(k, k - offset)/target**m))
        g = as_expr(g + Add(*[c_.subs(k, k - offset + i)/target.subs(k, k + i)**m for i in range(offset)]))
    h = as_expr(cancel(Add(*h_terms))) if h_terms else S.Zero
    return RationalDecomposition(as_expr(cancel(g)), h)


def _partial_terms(proper: Expr, k: Symbol) -> list[tuple[Expr, Expr, int]]:
    """``(c(k), q(k), m)`` for the terms ``c/q**m`` of the partial fraction
    expansion of a proper rational function over the rationals."""
    from sympy.polys.partfrac import apart
    from sympy.core.power import Pow
    from sympy.core.mul import Mul
    result: list[tuple[Expr, Expr, int]] = []
    expanded = as_expr(apart(proper, k, full=False))
    for term in Add.make_args(expanded):
        c: Expr = S.One
        q: Optional[Expr] = None
        m = 1
        for factor in Mul.make_args(term):
            factor_ = as_expr(factor)
            if isinstance(factor_, Pow) and factor_.exp.is_Integer and int(factor_.exp) < 0 and factor_.base.has(k):
                q = as_expr(factor_.base)
                m = -int(factor_.exp)
            elif factor_.has(k) and not factor_.is_polynomial(k):
                inner = factor_.as_numer_denom()
                q = as_expr(inner[1])
                c = as_expr(c*inner[0])
            else:
                c = as_expr(c*factor_)
        if q is None:
            continue
        result.append((as_expr(c), q, m))
    return result


def rational_sum(f: Union[Expr, int], limits: Sequence[Union[Symbol, Expr, int]]) -> Expr:
    """``sum_{k=lo}^{hi} f`` for a rational function: the rational part in
    closed form and the remainder summed by SymPy (polygamma functions)
    or left as a ``Sum``.

    Examples
    ========

    >>> from sympy.abc import k, n
    >>> from sympy_extras.concrete.rational import rational_sum
    >>> rational_sum(1/(k*(k + 1)), (k, 1, n))
    n/(n + 1)
    >>> rational_sum(1/(k*(k + 2)), (k, 1, n))
    n*(3*n + 5)/(4*(n + 1)*(n + 2))
    """
    k_, lo, hi = limits
    k = k_ if isinstance(k_, Symbol) else Symbol(str(k_))
    decomposition = abramov_decomposition(f, k)
    g, h = decomposition.g, decomposition.h
    total = as_expr(g.subs(k, sympify(hi) + 1) - g.subs(k, sympify(lo)))
    if h != 0:
        total = as_expr(total + summation(h, (k, sympify(lo), sympify(hi))))
    from sympy.polys.polytools import factor
    return as_expr(factor(cancel(total))) if total.is_rational_function() else total
