"""Limits of sequences (SymPy's ``sympy.series.limitseq``), on the typed
core: :func:`limit_seq`, :func:`difference_delta`, :func:`dominant`.

The algorithm follows Kauers [Kauers]_ as SymPy implements it: the limit
of the function of a real variable first; then the quotient of the
differences of numerator and denominator (Stolz–Cesàro), reduced to the
dominant terms, a few times over. A sequence with a power of a negative
number or a sine or cosine in the index is split into its even and odd
terms; two different limits give the accumulation bounds between them.

This module is a typed port of SymPy's ``sympy/series/limitseq.py``; the
differences of sums, products and powers (SymPy's ``_eval_difference_delta``
methods of ``Add``, ``Mul``, ``Pow`` and ``Sum``) are spelled out in
:func:`difference_delta`. Nothing of SymPy is loaded.

Examples
========

>>> from sympy_extras.core import Symbol, oo
>>> from sympy_extras.functions import factorial, harmonic, log
>>> from sympy_extras.series.limitseq import limit_seq, difference_delta, dominant
>>> n = Symbol('n')
>>> limit_seq((5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5), n)
5/3
>>> limit_seq((-1)**n, n), limit_seq(2**n/factorial(n), n)
(AccumBounds(-1, 1), 0)
>>> difference_delta(n*(n + 1), n), dominant(5*n**3 + 4*n**2 + n + 1, n)
(2*n + 2, 5*n**3)

References
==========

.. [Kauers] M. Kauers, Computing limits of sequences, ISSAC 2009.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/limitseq.py, sympy/core/add.py, sympy/core/mul.py,
# sympy/core/power.py, sympy/concrete/summations.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; the _eval_difference_delta methods as one function;
# the rewriting of fibonacci in GoldenRatio written down; no Max/Min classes.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.accumbounds import accum_bounds
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.limits import Limit
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import GoldenRatio, Infinity, Integer, NegativeInfinity
from sympy_extras.core.power import Pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import Factorial, Subfactorial
from sympy_extras.functions.combinatorial.numbers import Fibonacci
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.elementary.miscellaneous import max_, min_
from sympy_extras.functions.elementary.trigonometric import Cos, Sin
from sympy_extras.functions.special.gamma_functions import Gamma

__all__ = ['difference_delta', 'dominant', 'limit_seq']


def _has_sum(e: Basic) -> bool:
    from sympy_extras.concrete.expr_with_limits import Sum
    return e.has(Sum)


# PORTED FROM SYMPY: sympy/series/limitseq.py:difference_delta
def difference_delta(expr: ExprLike, n: Symbol, step: ExprLike = 1) -> Expr:
    """The difference operator: ``expr(n + step) - expr(n)``, termwise on
    sums, by the product rule of differences on products, as a multiple
    of ``expr`` for a power with the index in the exponent only, and over
    the new terms for a sum whose upper limit depends on ``n``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.series.limitseq import difference_delta
    >>> n = Symbol('n')
    >>> difference_delta(n*(n + 1), n), difference_delta(n*(n + 1), n, 2)
    (2*n + 2, 4*n + 6)
    """
    e = to_expr(expr)
    h = to_expr(step)
    if h.is_number is False or h.is_finite is False:
        raise ValueError("Step should be a finite number.")
    found = _difference_delta(e, n, h)
    if found is not None and found != 0:
        return found
    return as_expr(e.subs(n, n + h)) - e


def _difference_delta(e: Expr, n: Symbol, h: Expr) -> Optional[Expr]:
    """SymPy's ``_eval_difference_delta`` of the node, ``None`` when it has
    none."""
    from sympy_extras.concrete.expr_with_limits import Sum, sum_
    if isinstance(e, Add):
        # PORTED FROM SYMPY: sympy/core/add.py:Add._eval_difference_delta
        return add(*[difference_delta(a, n, h) for a in e.args if isinstance(a, Expr)])
    if isinstance(e, Mul):
        # PORTED FROM SYMPY: sympy/core/mul.py:Mul._eval_difference_delta
        factors = [a for a in e.args if isinstance(a, Expr)]
        first, rest = factors[0], mul(*factors[1:])
        return as_expr(first.subs(n, n + h))*difference_delta(rest, n, h) + difference_delta(first, n, h)*rest
    if isinstance(e, Pow):
        # PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_difference_delta
        b, x = e.base, e.exp
        if x.has(n) and not b.has(n):
            shifted = as_expr(x.subs(n, n + h))
            return (b**(shifted - x) - 1)*e
        return None
    if isinstance(e, Sum):
        # PORTED FROM SYMPY: sympy/concrete/summations.py:Sum._eval_difference_delta
        limits = e.limits
        k, _, upper = limits[-1]
        new_upper = as_expr(upper.subs(n, n + h))
        f = e.function if len(limits) == 1 else sum_(e.function, *limits[:-1])
        return as_expr(sum_(f, (k, upper + 1, new_upper)).doit())
    return None


# PORTED FROM SYMPY: sympy/series/limitseq.py:dominant
def dominant(expr: ExprLike, n: Symbol) -> Optional[Expr]:
    """The term of a sum which dominates every other one (``a`` dominates
    ``b`` when ``a/b`` tends to infinity), ``None`` when there is no
    single one.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.series.limitseq import dominant
    >>> n = Symbol('n')
    >>> dominant(5*n**3 + 4*n**2 + n + 1, n), dominant(2**n + n**5 + 2**n, n)
    (5*n**3, 2*2**n)
    >>> dominant(n**2 + 2*n + 1 + n**3*(1 - 1/n**3) - n**3, n) is None
    False
    """
    terms = Add.make_args(to_expr(expr).expand(func=True))
    term0 = terms[-1]
    comparable = [term0]
    for t in terms[:-1]:
        r = term0/t
        e = r.gammasimp()
        if e == r:
            e = r.factor()
        lim = limit_seq(e, n)
        if lim is None:
            return None
        if lim.is_zero is True:
            term0 = t
            comparable = [term0]
        elif not isinstance(lim, (Infinity, NegativeInfinity)):
            comparable.append(t)
    if len(comparable) > 1:
        return None
    return term0


# PORTED FROM SYMPY: sympy/series/limitseq.py:_limit_inf
def _limit_inf(expr: Expr, n: Symbol) -> Optional[Expr]:
    try:
        return Limit(expr, n, Infinity()).doit(deep=False)
    except (NotImplementedError, PoleError):
        return None


# PORTED FROM SYMPY: sympy/series/limitseq.py:_limit_seq
def _limit_seq(expr: Expr, n: Symbol, trials: int) -> Optional[Expr]:
    for _ in range(trials):
        if not _has_sum(expr):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()
        if not den.has(n) or not num.has(n):
            result = _limit_inf(expr.doit(), n)
            if result is not None:
                return result
            return None

        num, den = difference_delta(num.expand(), n), difference_delta(den.expand(), n)
        expr = (num/den).gammasimp()

        if not _has_sum(expr):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()

        dominant_num = dominant(num, n)
        if dominant_num is None:
            return None

        dominant_den = dominant(den, n)
        if dominant_den is None:
            return None

        expr = (dominant_num/dominant_den).gammasimp()
    return None


def _fibonacci_as_golden_ratio(e: Expr) -> Expr:
    """SymPy's ``rewrite(fibonacci, GoldenRatio)``: Binet's formula."""
    rules: dict[Basic, Basic] = {}
    for node in e.atoms(Fibonacci):
        if isinstance(node, Fibonacci) and len(node.args) == 1:
            k = as_expr(node.args[0])
            rules[node] = (GoldenRatio**k - 1/(-GoldenRatio)**k)/(2*GoldenRatio - 1)
    return as_expr(e._xreplace_evaluated(rules)) if rules else e


# PORTED FROM SYMPY: sympy/series/limitseq.py:limit_seq
def limit_seq(expr: ExprLike, n: Optional[Symbol] = None, trials: int = 5) -> Optional[Expr]:
    """The limit of the sequence ``expr`` as the integer ``n`` tends to
    positive infinity, ``None`` when it is not found (the algorithm is
    designed for rational functions, indefinite sums and products, and
    terms of alternating sign).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.functions import cos, harmonic, log
    >>> from sympy_extras.series.limitseq import limit_seq
    >>> n = Symbol('n')
    >>> limit_seq((5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5), n)
    5/3
    >>> limit_seq((-1)**n/n, n), limit_seq(cos(n)/n, n)
    (0, 0)
    >>> limit_seq(n**3/2**n, n)
    0
    """
    e = to_expr(expr)
    if n is None:
        free = [s for s in e.free_symbols if isinstance(s, Symbol)]
        if len(free) == 1:
            n = free[0]
        elif not free:
            return e
        else:
            raise ValueError("Expression has more than one variable. Please specify a variable.")
    elif n not in e.free_symbols:
        return e

    e = _fibonacci_as_golden_ratio(e)
    e = as_expr(e.rewrite(Factorial, Subfactorial, Gamma))
    n_ = Dummy("n", integer=True, positive=True)
    n1 = Dummy("n", odd=True, positive=True)
    n2 = Dummy("n", even=True, positive=True)

    # a negative number raised to a power in n, or a trigonometric
    # function: the even and the odd terms separately
    powers = [(p.base, p.exp) for p in e.atoms(Pow) if isinstance(p, Pow)]
    first: Optional[Expr]
    if any(b.is_negative is True and x.has(n) for b, x in powers) or e.has(Cos, Sin):
        first = _limit_seq(as_expr(e._xreplace_evaluated({n: n1})), n1, trials)
        if first is not None:
            second = _limit_seq(as_expr(e._xreplace_evaluated({n: n2})), n2, trials)
            if second is None or first != second:
                if second is not None and first.is_comparable and second.is_comparable:
                    return accum_bounds(min_(first, second), max_(first, second))
                return None
    else:
        first = _limit_seq(as_expr(e._xreplace_evaluated({n: n_})), n_, trials)
    if first is not None:
        return first
    if isinstance(e, Add):
        limits = [limit_seq(term, n, trials) for term in e.args if isinstance(term, Expr)]
        found: list[Expr] = []
        for result in limits:
            if result is None:
                return None
            found.append(result)
        return add(*found)
    # the absolute value may be easier (not with a sum): when it tends to
    # 0, so does the sequence
    if not _has_sum(e):
        lim = _limit_seq(abs_(as_expr(e._xreplace_evaluated({n: n_}))), n_, trials)
        if lim is not None and lim.is_zero is True:
            return Integer(0)
    return None
