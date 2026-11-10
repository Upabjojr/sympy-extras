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

The module works on the expressions of the core
(:mod:`sympy_extras.core`) and imports no SymPy: the polynomials, the
factorisation, the partial fractions and the summation of the remainder
are those of the core; only the isolated values of the parameters go
through SymPy at the call (:func:`~sympy_extras.concrete.karr._with_special_values`).

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

from typing import Optional, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.polys.partfrac import apart
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly, cancel, factor, factor_list

from .expr_with_limits import Sum
from .karr import SumPoint, _with_special_values
from .summations import summation

__all__ = ['abramov_decomposition', 'rational_sum', 'RationalDecomposition']

ZERO = Integer(0)


def _e(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, not %s' % (b,))
    return b


class RationalDecomposition:
    """``f = g(k + 1) - g(k) + h`` with ``h`` of minimal denominator
    (immutable).

    Attributes
    ==========

    g : Expr
        The rational part of the sum.
    h : Expr
        The remainder, ``0`` exactly when ``f`` has a rational sum.
    """

    __slots__ = ('_g', '_h')

    def __init__(self, g: Expr, h: Expr) -> None:
        self._g = g
        self._h = h

    @property
    def g(self) -> Expr:
        return self._g

    @property
    def h(self) -> Expr:
        return self._h

    @property
    def summable(self) -> bool:
        return self._h == ZERO

    def __repr__(self) -> str:
        return "RationalDecomposition(%s, %s)" % (self._g, self._h)


def _shift(q1: Poly, q2: Poly, k: Symbol) -> Optional[int]:
    """``j`` with ``q2(k) == q1(k + j)`` for monic polynomials of the same
    degree, or ``None``."""
    if q1.degree() != q2.degree() or q1.degree() == 0:
        return None
    d = q1.degree()
    c1 = q1.all_coeffs()[1]
    c2 = q2.all_coeffs()[1]
    # the coefficient of k**(d-1) of q1(k + j) is c1 + d*j
    j = cancel(mul(add(c2, -c1), pow(Integer(d), Integer(-1))))
    if not isinstance(j, Integer):
        return None
    if add(_e(q1.as_expr().subs(k, add(k, j))), -q2.as_expr()).expand() == ZERO:
        return j.p
    return None


def abramov_decomposition(f: ExprLike, k: Symbol) -> RationalDecomposition:
    """Abramov's decomposition of a rational function of ``k``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.concrete.rational import abramov_decomposition
    >>> k = Symbol('k')
    >>> abramov_decomposition(1/(k*(k + 1)), k)
    RationalDecomposition(-1/k, 0)
    >>> abramov_decomposition(1/k, k)
    RationalDecomposition(0, 1/k)
    >>> abramov_decomposition((2*k + 1)/(k**2*(k + 1)**2), k)
    RationalDecomposition(-1/k**2, 0)
    """
    expression = cancel(to_expr(f))
    if expression.is_rational_function(k) is not True:
        raise ValueError("%s is not a rational function of %s" % (expression, k))
    numerator, denominator = expression.as_numer_denom()
    try:
        num, den = Poly(numerator, k), Poly(denominator, k)
    except BasePolynomialError:
        raise ValueError("%s is not a rational function of %s" % (expression, k))
    quotient, remainder = num.div(den)
    g: Expr = ZERO
    # the polynomial part: sum_{i=0}^{k-1} quotient(i)
    if not quotient.is_zero:
        i = Dummy('i', integer=True)
        summed = summation(_e(quotient.as_expr().subs(k, i)), (i, ZERO, add(k, Integer(-1))))
        if isinstance(summed, Sum):
            raise ValueError("the polynomial part could not be summed")
        g = add(g, summed)
    if remainder.is_zero:
        return RationalDecomposition(cancel(g), ZERO)
    # the shift classes of the irreducible factors of the denominator
    _, factors = factor_list(den.as_expr(), k)
    monic: list[Poly] = [Poly(fac, k).monic() for fac, _ in factors]
    classes: list[list[tuple[Poly, int]]] = []   # (factor, shift from the representative)
    for factor_poly in monic:
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
    proper = mul(remainder.as_expr(), pow(den.as_expr(), Integer(-1)))
    h_terms: list[Expr] = []
    for c, q, m in _partial_terms(proper, k):
        q_poly = Poly(q, k).monic()
        scale = Poly(q, k).LC()
        c_ = mul(c, pow(scale, Integer(-m)))
        # the class and the shift of this factor
        found: Optional[tuple[list[tuple[Poly, int]], int]] = None
        for group in classes:
            for member, j in group:
                if add(member.as_expr(), -q_poly.as_expr()).expand() == ZERO:
                    found = (group, j)
                    break
            if found is not None:
                break
        if found is None:
            h_terms.append(mul(c_, pow(q_poly.as_expr(), Integer(-m))))
            continue
        group, shift = found
        representative = group[0][0]
        # the leftmost member of the class is the target denominator
        least = min(j for _, j in group)
        offset = shift - least
        target = _e(representative.as_expr().subs(k, add(k, Integer(least))))
        if offset == 0:
            h_terms.append(mul(c_, pow(target, Integer(-m))))
            continue
        # c(k)/q(k + offset)^m = c(k - offset)/q(k)^m + Delta(sum_{i=0}^{offset-1} c(k - offset + i)/q(k + i)^m)
        h_terms.append(mul(_e(c_.subs(k, add(k, Integer(-offset)))), pow(target, Integer(-m))))
        g = add(g, *[mul(_e(c_.subs(k, add(k, Integer(i - offset)))),
                         pow(_e(target.subs(k, add(k, Integer(i)))), Integer(-m))) for i in range(offset)])
    h = cancel(add(*h_terms)) if h_terms else ZERO
    return RationalDecomposition(cancel(g), h)


def _partial_terms(proper: Expr, k: Symbol) -> list[tuple[Expr, Expr, int]]:
    """``(c(k), q(k), m)`` for the terms ``c/q**m`` of the partial fraction
    expansion of a proper rational function over the rationals."""
    result: list[tuple[Expr, Expr, int]] = []
    expanded = apart(proper, k, full=False)
    for term in (expanded.args if isinstance(expanded, Add) else (expanded,)):
        c: Expr = Integer(1)
        q: Optional[Expr] = None
        m = 1
        for factor_ in (term.args if isinstance(term, Mul) else (term,)):
            factor_e = _e(factor_)
            if (isinstance(factor_e, Pow) and isinstance(factor_e.exp, Integer) and factor_e.exp.p < 0
                    and factor_e.base.has(k)):
                q = factor_e.base
                m = -factor_e.exp.p
            elif factor_e.has(k) and factor_e.is_polynomial(k) is not True:
                inner = factor_e.as_numer_denom()
                q = inner[1]
                c = mul(c, inner[0])
            else:
                c = mul(c, factor_e)
        if q is None:
            continue
        result.append((c, q, m))
    return result


def rational_sum(f: ExprLike, limits: Sequence[ExprLike], special_values: bool = True) -> Expr:
    """``sum_{k=lo}^{hi} f`` for a rational function: the rational part in
    closed form and the remainder summed by the summation of the core
    (polygamma functions) or left as a ``Sum``.

    With ``special_values`` (the default) the isolated values of the
    parameters of ``f`` at which the sum is undefined get cases of their
    own, the sum computed again there
    (:func:`~sympy_extras.concrete.karr_sympy.with_sum_special_values`). Where
    two poles of ``f`` merge (``k = -a`` and ``k = -b`` at ``a = b``) the
    partial fractions of the generic sum have the difference of the
    poles, a factor of the resultant of the factors of the denominator,
    in their denominators, so that these points are found among the
    zeros of the denominators of the value; the generic value, a
    continuous function of the parameters wherever it is defined, needs
    no other case. A point where a pole of ``f`` falls on an integer of
    the range (``a = 0`` below, the term ``1/0`` at ``k = 0``) gets none.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.concrete.rational import rational_sum
    >>> k, n, a, b = symbols('k n a b')
    >>> rational_sum(1/(k*(k + 1)), (k, 1, n))
    n/(n + 1)
    >>> rational_sum(1/(k*(k + 2)), (k, 1, n))
    n*(3*n + 5)/(4*(n + 1)*(n + 2))
    >>> s = rational_sum(1/((k + a)*(k + b)), (k, 0, n))
    >>> s.args[0].cond
    Eq(a, b)
    >>> s.subs({a: 1, b: 1, n: 2}).evalf(6)     # 1 + 1/4 + 1/9 = 49/36
    1.36111
    """
    if len(limits) != 3:
        raise ValueError('the limits are (k, lower, upper)')
    k_, lo, hi = (to_expr(e) for e in limits)
    if not isinstance(k_, Symbol):
        raise TypeError('the index of summation must be a symbol, not %s' % (k_,))
    k = k_
    f_ = to_expr(f)
    decomposition = abramov_decomposition(f_, k)
    g, h = decomposition.g, decomposition.h
    total = add(_e(g.subs(k, add(hi, Integer(1)))), -_e(g.subs(k, lo)))
    if h != ZERO:
        total = add(total, summation(h, (k, lo, hi)))
    value = factor(cancel(total)) if total.is_rational_function() is True else total
    if not special_values:
        return value

    def at_point(point: SumPoint) -> Optional[Expr]:
        return rational_sum(f_.xreplace(point), (k, lo, hi))

    return _with_special_values(value, f_, [k], [lo, hi], at_point)
