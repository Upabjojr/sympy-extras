"""The ``rewrite`` rules of the orthogonal polynomials (SymPy's
``_eval_rewrite_as_Sum``, ``_eval_rewrite_as_polynomial``, and
``hermite``/``hermite_prob`` in terms of each other), as a typed table
for :func:`sympy_extras.core.rewriting.rewrite`. Ported from SymPy's
``sympy.functions.special.polynomials``.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/polynomials.py (the _eval_rewrite_as_Sum,
# _eval_rewrite_as_polynomial, _eval_rewrite_as_hermite_prob and _eval_rewrite_as_hermite
# methods) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by (class, target) in place of methods; the phase (-1)**m of
# assoc_legendre's sum restored (a SymPy bug).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.rewriting import RewriteHints, Rule, RuleKey

__all__ = ['rules']

ONE = Integer(1)
MINUS_ONE = Integer(-1)
ZERO = Integer(0)


def _check_nonnegative_integer(n: Expr) -> None:
    if n.is_negative is True or n.is_integer is False:
        raise ValueError('Error: n should be a non-negative integer.')


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:jacobi._eval_rewrite_as_Sum
def _jacobi_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial, rf
    n, a, b, x = args
    _check_nonnegative_integer(n)
    k = Dummy('k')
    kern = (rf(-n, k) * rf(a + b + n + 1, k) * rf(a + k + 1, n - k) / factorial(k) * ((1 - x) / 2)**k)
    return 1 / factorial(n) * sum_(kern, (k, ZERO, n))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:gegenbauer._eval_rewrite_as_Sum
def _gegenbauer_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial, rf
    from sympy_extras.functions.elementary.integers import floor
    n, a, x = args
    k = Dummy('k')
    kern = (MINUS_ONE**k * rf(a, n - k) * (2 * x)**(n - 2 * k) / (factorial(k) * factorial(n - 2 * k)))
    return sum_(kern, (k, ZERO, floor(n / 2)))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:chebyshevt._eval_rewrite_as_Sum
def _chebyshevt_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import binomial
    from sympy_extras.functions.elementary.integers import floor
    n, x = args
    k = Dummy('k')
    kern = binomial(n, 2 * k) * (x**2 - 1)**k * x**(n - 2 * k)
    return sum_(kern, (k, ZERO, floor(n / 2)))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:chebyshevu._eval_rewrite_as_Sum
def _chebyshevu_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.integers import floor
    n, x = args
    k = Dummy('k')
    kern = MINUS_ONE**k * factorial(n - k) * (2 * x)**(n - 2 * k) / (factorial(k) * factorial(n - 2 * k))
    return sum_(kern, (k, ZERO, floor(n / 2)))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:legendre._eval_rewrite_as_Sum
def _legendre_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import binomial
    n, x = args
    k = Dummy('k')
    kern = MINUS_ONE**k * binomial(n, k)**2 * ((1 + x) / 2)**(n - k) * ((1 - x) / 2)**k
    return sum_(kern, (k, ZERO, n))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:assoc_legendre._eval_rewrite_as_Sum
def _assoc_legendre_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.integers import floor
    n, m, x = args
    k = Dummy('k')
    kern = (factorial(2 * n - 2 * k) / (Integer(2)**n * factorial(n - k) * factorial(k) * factorial(n - 2 * k - m))
            * MINUS_ONE**k * x**(n - m - 2 * k))
    # SymPy's rule omits the Condon-Shortley phase (-1)**m which its
    # assoc_legendre has (assoc_legendre(1, 1, x) is -sqrt(1 - x**2), its
    # Sum sqrt(1 - x**2)): restored here
    return MINUS_ONE**m * (1 - x**2)**(m / 2) * sum_(kern, (k, ZERO, floor((n - m) * Rational(1, 2))))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:hermite._eval_rewrite_as_Sum
def _hermite_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.integers import floor
    n, x = args
    k = Dummy('k')
    kern = MINUS_ONE**k / (factorial(k) * factorial(n - 2 * k)) * (2 * x)**(n - 2 * k)
    return factorial(n) * sum_(kern, (k, ZERO, floor(n / 2)))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:hermite._eval_rewrite_as_hermite_prob
def _hermite_as_prob(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.special.polynomials import hermite_prob
    n, x = args
    return sqrt(Integer(2))**n * hermite_prob(n, x * sqrt(Integer(2)))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:hermite_prob._eval_rewrite_as_Sum
def _hermite_prob_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.integers import floor
    n, x = args
    k = Dummy('k')
    kern = Rational(-1, 2)**k * x**(n - 2 * k) / (factorial(k) * factorial(n - 2 * k))
    return factorial(n) * sum_(kern, (k, ZERO, floor(n / 2)))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:hermite_prob._eval_rewrite_as_hermite
def _prob_as_hermite(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.special.polynomials import hermite
    n, x = args
    return sqrt(Integer(2))**(-n) * hermite(n, x / sqrt(Integer(2)))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:laguerre._eval_rewrite_as_Sum
def _laguerre_sum_of(n: Expr, x: Expr) -> Expr:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial, rf
    from sympy_extras.functions.elementary.exponential import exp
    if n.is_negative is True:
        return exp(x) * _laguerre_sum_of(-n - 1, -x)
    if n.is_integer is False:
        raise ValueError('Error: n should be an integer.')
    k = Dummy('k')
    kern = rf(-n, k) / factorial(k)**2 * x**k
    return sum_(kern, (k, ZERO, n))


def _laguerre_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    n, x = args
    return _laguerre_sum_of(n, x)


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:assoc_laguerre._eval_rewrite_as_Sum
def _assoc_laguerre_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial, rf
    from sympy_extras.functions.special.gamma_functions import gamma
    n, alpha, x = args
    _check_nonnegative_integer(n)
    k = Dummy('k')
    kern = rf(-n, k) / (gamma(k + alpha + 1) * factorial(k)) * x**k
    return gamma(n + alpha + 1) / factorial(n) * sum_(kern, (k, ZERO, n))


def rules() -> dict[RuleKey, Rule]:
    """The rules keyed by (the class of the node, the target class or
    SymPy name); ``polynomial`` is SymPy's old name of ``Sum``."""
    from . import polynomials as p
    sums: dict[type, Rule] = {
        p.Jacobi: _jacobi_sum, p.Gegenbauer: _gegenbauer_sum, p.ChebyshevT: _chebyshevt_sum,
        p.ChebyshevU: _chebyshevu_sum, p.Legendre: _legendre_sum, p.AssocLegendre: _assoc_legendre_sum,
        p.Hermite: _hermite_sum, p.HermiteProb: _hermite_prob_sum, p.Laguerre: _laguerre_sum,
        p.AssocLaguerre: _assoc_laguerre_sum}
    table: dict[RuleKey, Rule] = {}
    for cls, rule in sums.items():
        table[(cls, 'Sum')] = rule
        table[(cls, 'polynomial')] = rule
    table[(p.Hermite, p.HermiteProb)] = _hermite_as_prob
    table[(p.HermiteProb, p.Hermite)] = _prob_as_hermite
    return table
