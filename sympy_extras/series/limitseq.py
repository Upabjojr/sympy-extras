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

from sympy_extras.core.accumbounds import AccumBounds, accum_bounds
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.limits import Limit
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import GoldenRatio, Infinity, Integer, NegativeInfinity, Rational
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
        # SymPy takes the new terms of the last limit only, with the summand at n: wrong when the
        # summand (an inner sum to n) or the lower limit depends on n too, as for
        # Sum(1/k, (k, 1, n), (m, 1, n)), which is n*harmonic(n) (sympy/sympy#31275, V213)
        limits = e.limits
        k, lower, upper = limits[-1]
        if k == n or lower.has(n):
            return None
        new_upper = as_expr(upper.subs(n, n + h))
        f = e.function if len(limits) == 1 else sum_(e.function, *limits[:-1])
        new_terms = as_expr(sum_(as_expr(f.subs(n, n + h)), (k, upper + 1, new_upper)).doit())
        if not f.has(n):
            return new_terms
        return new_terms + as_expr(sum_(difference_delta(f, n, h), (k, lower, upper)).doit())
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
    return _dominant(to_expr(expr), n, 0)


#: how deep :func:`dominant` and the limit of a sequence may call each other
_MAX_DEPTH = 3


def _dominant(expr: Expr, n: Symbol, depth: int) -> Optional[Expr]:
    terms = Add.make_args(expr.expand(func=True))
    term0 = terms[-1]
    comparable = [term0]
    for t in terms[:-1]:
        r = term0/t
        e = r.gammasimp(generic=True)
        if e == r:
            e = r.factor()
        lim = _limit_seq_of(e, n, 5, depth + 1)
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


def _asymptotic_factors(expr: Expr, n: Symbol, depth: int) -> Optional[Expr]:
    """``expr`` with every factor ``Sum(f(k), (k, a, n))`` replaced by an
    asymptotically equivalent ``c*f(n)``, ``None`` when a sum is not a
    factor (or a power of one with a constant exponent) or has no such
    equivalent.

    When the quotient of consecutive terms ``f(n + 1)/f(n)`` tends to
    ``r`` with ``|r| > 1``, the quotient of the sum and its last term,
    ``sum(f(n - j)/f(n), j = 0..n - a)``, tends to the geometric series
    ``r/(r - 1)`` (to 1 for an infinite ``r``): the theorem of Stolz and
    Cesàro for a sum of geometric growth. Equivalent factors may replace
    each other in a product, not in a sum.
    """
    from sympy_extras.concrete.expr_with_limits import Sum
    if not _has_sum(expr):
        return expr
    if isinstance(expr, Mul):
        factors: list[Expr] = []
        for a in expr.args:
            if not isinstance(a, Expr):
                return None
            factor = _asymptotic_factors(a, n, depth)
            if factor is None:
                return None
            factors.append(factor)
        return mul(*factors)
    if isinstance(expr, Pow):
        if expr.exp.has(n) or _has_sum(expr.exp):
            return None
        base = _asymptotic_factors(expr.base, n, depth)
        return None if base is None else base**expr.exp
    if isinstance(expr, Sum) and len(expr.limits) == 1:
        k, lower, upper = expr.limits[0]
        f = expr.function
        if upper != n or lower.has(n) or f.has(n) or _has_sum(f):
            return None
        last = as_expr(f.subs(k, n))
        ratio = _limit_inf((as_expr(f.subs(k, n + 1))/last).gammasimp(generic=True), n)
        if ratio is None:
            return None
        if isinstance(ratio, (Infinity, NegativeInfinity)):
            return last
        if not ratio.is_number or (abs_(ratio) - 1).is_positive is not True:
            return None
        c = ratio/(ratio - 1)
        return c*last
    return None


def _is_exp_log(e: Expr, n: Symbol) -> bool:
    """Whether ``e`` is built from ``n`` and constants by sums, products,
    powers with a constant exponent or a constant positive base,
    exponentials and logarithms: such functions form a Hardy field (every
    one of them, and its difference ``e(n + 1) - e(n)``, has a constant
    sign from some point on)."""
    from sympy_extras.functions.elementary.exponential import Exp, Log
    if not e.has(n):
        return not e.free_symbols and e.is_real is True
    if e == n:
        return True
    if isinstance(e, (Add, Mul)):
        return all(isinstance(a, Expr) and _is_exp_log(a, n) for a in e.args)
    if isinstance(e, Pow):
        if not e.exp.has(n):
            return e.exp.is_real is True and not e.exp.free_symbols and _is_exp_log(e.base, n)
        return not e.base.has(n) and e.base.is_positive is True and not e.base.free_symbols and _is_exp_log(e.exp, n)
    if isinstance(e, (Exp, Log)):
        argument = e.args[0]
        return isinstance(argument, Expr) and _is_exp_log(argument, n)
    return False


def _stolz_cesaro(expr: Expr, n: Symbol) -> Optional[Expr]:
    """The limit of ``g(n)*Sum(f(k), (k, a, n))**p`` (``p`` a rational
    number; positive terms ``f`` unless ``p`` is 1 or -1) by the theorem
    of Stolz and Cesàro on the sum against ``b(n) = g(n)**(-1/p)``: when
    ``b`` is strictly monotone from some point on and unbounded, and
    ``f(n + 1)/(b(n + 1) - b(n))`` has the limit ``L``, finite or not,
    the quotient of the sum and ``b`` has the limit ``L``, and the
    expression is its power ``p``. ``b`` is asked to be a function of
    exponentials and logarithms which tends to an infinity: it is then
    strictly monotone from some point on (:func:`_is_exp_log`). ``None``
    when this does not apply.

    The differences of the numerator and of the denominator of
    ``log(n)*Sum(1/log(k + 1), (k, 1, n))/n`` as they stand, ``log(n)*Sum``
    and ``n``, have the sum again: the limit, 1, was not found."""
    from sympy_extras.concrete.expr_with_limits import Sum
    sums: list[Expr] = []
    others: list[Expr] = []
    power: Rational = Integer(1)
    for factor in Mul.make_args(expr):
        base, exponent = factor.as_base_exp()
        if isinstance(base, Sum) and isinstance(exponent, Rational) and exponent.p != 0:
            sums.append(base)
            power = exponent
        elif _has_sum(factor):
            return None
        else:
            others.append(factor)
    if len(sums) != 1:
        return None
    total = sums[0]
    if not isinstance(total, Sum) or len(total.limits) != 1:
        return None
    k, lower, upper = total.limits[0]
    f = total.function
    if upper != n or lower.has(n) or f.has(n) or _has_sum(f):
        return None
    # the sum against b: g*Sum**p is (Sum/b)**p with b = g**(-1/p)
    rest = mul(*others)
    plain = power == 1 or power == -1
    b = rest**(-1/power)
    if not _is_exp_log(b, n):
        return None
    at_infinity = _limit_inf(b, n)
    if not isinstance(at_infinity, (Infinity, NegativeInfinity)):
        return None
    if not plain and not (isinstance(at_infinity, Infinity) and f.is_positive is True and lower.is_integer is True):
        # another power is that of a positive quotient
        return None
    quotient = as_expr(f.subs(k, n + 1))/(as_expr(b.subs(n, n + 1)) - b)
    found = _limit_inf(quotient, n)
    if found is None or isinstance(found, Limit) or found.has(n) or isinstance(found, AccumBounds):
        return None
    if power == 1:
        return found
    if isinstance(found, (Infinity, NegativeInfinity)):
        if power.p < 0:
            return Integer(0)
        return found if plain else None if isinstance(found, NegativeInfinity) else found
    if found.is_number and found.is_finite is True and (found.is_positive is True or (plain and found.is_zero is False)):
        return found**power
    if not plain and found.is_zero is True:
        # a positive quotient which tends to 0
        return Integer(0) if power.p > 0 else Infinity()
    return None


# PORTED FROM SYMPY: sympy/series/limitseq.py:_limit_seq
def _limit_seq(expr: Expr, n: Symbol, trials: int, depth: int) -> Optional[Expr]:
    if _has_sum(expr):
        # not SymPy's: the sums which are factors by their last terms
        equivalent = _asymptotic_factors(expr, n, depth)
        if equivalent is not None and not _has_sum(equivalent):
            result = _limit_inf(equivalent.gammasimp(generic=True), n)
            if result is not None:
                return result
        # and the sums with a closed form
        closed = expr.doit()
        if not _has_sum(closed):
            result = _limit_inf(closed, n)
            if result is not None:
                return result
        # and a sum which is the only one, a factor, against the rest
        result = _stolz_cesaro(expr, n)
        if result is not None:
            return result
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
        expr = (num/den).gammasimp(generic=True)

        if not _has_sum(expr):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()

        dominant_num = _dominant(num, n, depth)
        if dominant_num is None:
            return None

        dominant_den = _dominant(den, n, depth)
        if dominant_den is None:
            return None

        expr = (dominant_num/dominant_den).gammasimp(generic=True)
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
    return _limit_seq_of(to_expr(expr), n, trials, 0)


def _limit_seq_of(e: Expr, n: Optional[Symbol], trials: int, depth: int) -> Optional[Expr]:
    """:func:`limit_seq` at a depth of the mutual recursion with
    :func:`dominant`.

    The comparison of two terms with sums is the limit of a quotient with
    the same sums, whose differences have them again: SymPy's functions
    call each other without end on ``Sum(a)/(Sum(b)*Sum(c))`` with
    growing expressions (sympy/sympy#31275, V214). Beyond ``_MAX_DEPTH``
    the limit is not found.
    """
    if depth > _MAX_DEPTH:
        return None
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
        first = _limit_seq(as_expr(e._xreplace_evaluated({n: n1})), n1, trials, depth)
        if first is not None:
            second = _limit_seq(as_expr(e._xreplace_evaluated({n: n2})), n2, trials, depth)
            if second is None or first != second:
                if second is not None and first.is_comparable and second.is_comparable:
                    return accum_bounds(min_(first, second), max_(first, second))
                return None
    else:
        first = _limit_seq(as_expr(e._xreplace_evaluated({n: n_})), n_, trials, depth)
    if first is not None:
        return first
    if isinstance(e, Add):
        limits = [_limit_seq_of(term, n, trials, depth) for term in e.args if isinstance(term, Expr)]
        found: list[Expr] = []
        for result in limits:
            if result is None:
                return None
            found.append(result)
        return add(*found)
    # the absolute value may be easier (not with a sum): when it tends to
    # 0, so does the sequence
    if not _has_sum(e):
        lim = _limit_seq(abs_(as_expr(e._xreplace_evaluated({n: n_}))), n_, trials, depth)
        if lim is not None and lim.is_zero is True:
            return Integer(0)
    return None
