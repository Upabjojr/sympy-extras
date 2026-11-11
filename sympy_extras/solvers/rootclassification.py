r"""The number of distinct real roots of a polynomial with parameters, as
conditions on the parameters.

For $p(x) = a_0 x^n + \dots + a_n$ with $a_0 \ne 0$ and roots $\alpha_i$,
Hermite's quadratic form has the Hankel matrix $H = (s_{i+j})_{0 \le i,
j < n}$ of the power sums $s_k = \sum_i \alpha_i^k$ (rational functions
of the coefficients, by Newton's identities). Its rank is the number of
distinct complex roots and its signature the number of distinct *real*
roots [1]_. For a Hankel matrix of rank $r$ the signature is read off
the leading principal minors $D_0 = 1, D_1, \dots, D_r$ by the rule of
Frobenius [2]_: each pair of consecutive nonzero minors $D_h$, $D_{h + g
+ 1}$ separated by $g$ zeros contributes

.. math:: (-1)^{g/2} \operatorname{sign}(D_h D_{h+g+1}) \ \text{for even } g,
          \qquad 0 \ \text{for odd } g.

The minors are, up to positive factors, the principal subresultant
coefficients of $p$ and $p'$ (the last one is the discriminant): the
*discrimination sequence* of the polynomial [3]_. The number of real
roots therefore depends only on the signs of these $n$ polynomials in
the coefficients. :func:`real_root_classification` goes through the sign
patterns, computes the count of each by the rule, and simplifies the
conditions of each count by quantifier-free simplification over the
reals (:func:`sympy_extras.assumptions.resolve.resolve`), which also
discards the patterns which do not occur. Where the leading coefficient
vanishes the polynomial of lower degree is classified in the same way.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.solvers.rootclassification import real_root_classification, discrimination_sequence
>>> x, b, c, p, q = (Symbol(s) for s in 'xbcpq')
>>> real_root_classification(x**2 + b*x + c, x)
[(b**2 - 4*c < 0, 0), (Eq(b**2 - 4*c, 0), 1), (b**2 - 4*c > 0, 2)]
>>> discrimination_sequence(x**3 + p*x + q, x)
[3, -6*p, -4*p**3 - 27*q**2]
>>> for condition, count in real_root_classification(x**3 + p*x + q, x):
...     print(count, condition)
1 (p >= 0) | (4*p**3 + 27*q**2 > 0)
2 Ne(p, 0) & Eq(4*p**3 + 27*q**2, 0)
3 4*p**3 + 27*q**2 < 0

References
==========

.. [1] C. Hermite, *Remarques sur le théorème de Sturm*, Comptes Rendus
       de l'Académie des Sciences 36 (1853), 294-297; S. Basu, R. Pollack,
       M.-F. Roy, *Algorithms in Real Algebraic Geometry*, 2nd ed.,
       Springer, 2006, Section 4.3 (Hermite's quadratic form).
.. [2] F. R. Gantmacher, *The Theory of Matrices*, vol. 1, Chelsea, 1959,
       Chapter X, Section 10 (Hankel forms: the rule of Frobenius).
.. [3] L. Yang, X. Hou, Z. Zeng, *A complete discrimination system for
       polynomials*, Science in China E 39 (1996), 628-646.
"""
from __future__ import annotations

from itertools import product
from typing import Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import And, Boolean, Or, false, true
from sympy_extras.core.numbers import Integer
from sympy_extras.core.relational import Eq, Gt, Lt
from sympy_extras.core.symbol import Symbol
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.matrices.eigen import _coefficients
from sympy_extras.matrices.linalg import det, normal
from sympy_extras.simplify.radsimp import fraction

__all__ = ['discrimination_sequence', 'signature_from_signs', 'real_root_classification']


def _coefficient_list(p: ExprLike, x: Symbol) -> list[Expr]:
    numerator, denominator = fraction(normal(to_expr(p)))
    if denominator.has(x):
        raise ValueError('%s is not a polynomial in %s' % (p, x))
    expanded = expand(numerator)
    return [Integer(0)] if expanded == 0 else _coefficients(expanded, x)


def _minors(coefficients: Sequence[Expr]) -> list[Expr]:
    """Polynomials with the signs of the leading principal minors ``D_1,
    ..., D_n`` of Hermite's matrix (for a nonzero leading coefficient)."""
    n = len(coefficients) - 1
    leading = coefficients[0]
    c = [normal(a/leading) for a in coefficients]
    # Newton's identities for the monic polynomial x**n + c1 x**(n-1) + ...
    sums: list[Expr] = [Integer(n)]
    for k in range(1, 2*n - 1):
        total: list[Expr] = []
        for i in range(1, min(k, n) + 1):
            if i == k:
                total.append(-k*c[i])
            else:
                total.append(-c[i]*sums[k - i])
        sums.append(normal(add(*total)))
    result: list[Expr] = []
    for k in range(1, n + 1):
        minor = det(ExprMatrix(k, k, [sums[i + j] for i in range(k) for j in range(k)]))
        numerator, denominator = fraction(normal(minor))
        # the sign of a quotient is that of the product; squares of the leading coefficient are dropped
        result.append(_without_even_powers(expand(numerator*denominator), leading))
    return result


def _without_even_powers(e: Expr, leading: Expr) -> Expr:
    """``e`` divided by the largest even power of ``leading`` which
    divides it (a positive factor where the leading coefficient is not
    zero)."""
    if leading.is_number or e == 0:
        return e
    square = expand(leading**2)
    current = e
    for _ in range(64):
        quotient = normal(current/square)
        if fraction(quotient)[1] != 1:
            break
        current = expand(quotient)
    return current


def discrimination_sequence(p: ExprLike, x: Symbol) -> list[Expr]:
    """Polynomials in the coefficients of ``p`` (of degree ``n`` in
    ``x``) with the signs of the leading principal minors ``D_1, ...,
    D_n`` of Hermite's form: the number of distinct real roots is a
    function of their signs (:func:`signature_from_signs`), and the last
    one has the sign of the discriminant.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.rootclassification import discrimination_sequence
    >>> x, a, b, c = (Symbol(s) for s in 'xabc')
    >>> discrimination_sequence(a*x**2 + b*x + c, x)
    [2, -4*a*c + b**2]
    """
    coefficients = _coefficient_list(p, x)
    if len(coefficients) < 2:
        return []
    return _minors(coefficients)


def signature_from_signs(signs: Sequence[int]) -> int:
    """The number of distinct real roots from the signs (``-1``, ``0``,
    ``1``) of the discrimination sequence ``D_1, ..., D_n``, by the rule
    of Frobenius for Hankel forms.

    >>> from sympy_extras.solvers.rootclassification import signature_from_signs
    >>> signature_from_signs([1, 1, 1]), signature_from_signs([1, 1, -1]), signature_from_signs([1, 1, 0]), signature_from_signs([1, 0, 0])
    (3, 1, 2, 1)
    """
    total = 0
    previous = 1                    # D_0 = 1
    gap = 0
    for sign in signs:
        if sign == 0:
            gap += 1
            continue
        if gap % 2 == 0:
            total += (-1)**(gap//2)*previous*sign
        previous, gap = sign, 0
    return total


def _condition(quantity: Expr, sign: int) -> Optional[Boolean]:
    """The condition ``sign(quantity) = sign``, or ``None`` when it is
    false for a number."""
    if quantity.is_number:
        actual = 1 if quantity.is_positive else (-1 if quantity.is_negative else 0)
        return true if actual == sign else None
    if sign > 0:
        return Gt(quantity, 0)
    if sign < 0:
        return Lt(quantity, 0)
    return Eq(quantity, 0)


def _simplified(formula: Boolean) -> Boolean:
    from sympy_extras.assumptions.resolve import resolve
    try:
        return resolve(formula)
    except (ValueError, NotImplementedError):
        return formula


def real_root_classification(p: ExprLike, x: Symbol, max_degree: int = 6) -> list[tuple[Boolean, Optional[int]]]:
    """The number of distinct real roots of the polynomial ``p`` in
    ``x`` for every real value of its other symbols: pairs of a condition
    on the parameters and of the count, in increasing order of the
    counts; the count is ``None`` where the polynomial vanishes
    identically. The conditions are exhaustive and exclusive.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.rootclassification import real_root_classification
    >>> x, a = Symbol('x'), Symbol('a')
    >>> real_root_classification(a*x**2 + x + 1, x)
    [(a > 1/4, 0), (Eq(a, 0) | Eq(a, 1/4), 1), ((a < 0) | ((a > 0) & (a < 1/4)), 2)]
    >>> real_root_classification(x**4 - 5*x**2 + 4, x), real_root_classification(a*x, x)
    ([(True, 4)], [(Eq(a, 0), None), ((a > 0) | (a < 0), 1)])
    """
    coefficients = _coefficient_list(p, x)
    if len(coefficients) - 1 > max_degree and any(c.free_symbols for c in coefficients):
        raise NotImplementedError('the degree %d exceeds max_degree = %d: the number of sign patterns is 3**(n - 1)'
                                  % (len(coefficients) - 1, max_degree))
    by_count: dict[Optional[int], list[Boolean]] = {}
    vanishing: list[Boolean] = []                     # the leading coefficients which are zero so far
    remaining = list(coefficients)
    while True:
        leading = remaining[0]
        degree = len(remaining) - 1
        if leading.is_number and leading == 0 and degree > 0:
            remaining = remaining[1:]
            continue
        here: list[Boolean] = list(vanishing)
        if not leading.is_number:
            here.append(Or(Gt(leading, 0), Lt(leading, 0)))
        if degree == 0:
            if leading.is_number:
                by_count.setdefault(None if leading == 0 else 0, []).append(And(*here) if here else true)
            else:
                by_count.setdefault(0, []).append(And(*here))
                by_count.setdefault(None, []).append(And(*(vanishing + [Eq(leading, 0)])))
            break
        minors = _minors(remaining)
        if all(minor.is_number for minor in minors):
            # no parameter: one pattern
            actual = [1 if minor.is_positive else (-1 if minor.is_negative else 0) for minor in minors]
            by_count.setdefault(signature_from_signs(actual), []).append(And(*here) if here else true)
            patterns: list[tuple[int, ...]] = []
        else:
            patterns = list(product((1, 0, -1), repeat=degree))
        for signs in patterns:
            conditions = list(here)
            feasible = True
            for minor, sign in zip(minors, signs):
                condition = _condition(minor, sign)
                if condition is None:
                    feasible = False
                    break
                if condition is not true:
                    conditions.append(condition)
            # a Hankel form of rank r has D_r != 0 and no nonzero minor after it: patterns with a nonzero minor
            # after the last one are those of a larger rank and are listed there
            if not feasible:
                continue
            count = signature_from_signs(signs)
            by_count.setdefault(count, []).append(And(*conditions) if conditions else true)
        if leading.is_number:
            break
        vanishing.append(Eq(leading, 0))
        remaining = remaining[1:]
    results: list[tuple[Boolean, Optional[int]]] = []
    for key in sorted(by_count, key=lambda k: -1 if k is None else k):
        alternatives = by_count[key]
        formula: Boolean = alternatives[0] if len(alternatives) == 1 else Or(*alternatives)
        simplified = _simplified(formula)
        if simplified is not false:
            results.append((simplified, key))
    return results


