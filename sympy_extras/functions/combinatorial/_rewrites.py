"""The ``rewrite`` rules of the combinatorial functions (and ``beta``) in
terms of ``gamma`` (SymPy's ``_eval_rewrite_as_gamma``), and of ``gamma``,
``binomial`` and the Pochhammer symbols in terms of ``factorial`` (SymPy's
``_eval_rewrite_as_factorial``), as a typed table for
:func:`sympy_extras.core.rewriting.rewrite`.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py, sympy/functions/combinatorial/numbers.py,
# sympy/functions/special/beta_functions.py, sympy/functions/special/gamma_functions.py (the
# _eval_rewrite_as_gamma and _eval_rewrite_as_factorial methods) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by (class, target) in place of methods.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import true
from sympy_extras.core.mul import mul, neg
from sympy_extras.core.numbers import I, Integer, Rational, pi
from sympy_extras.core.power import pow
from sympy_extras.core.rewriting import RewriteHints

__all__ = ['rules']

Rule = Callable[[Expr, tuple[Expr, ...], RewriteHints], Optional[Expr]]

ONE = Integer(1)
MINUS_ONE = Integer(-1)


def _inv(e: Expr) -> Expr:
    return pow(e, MINUS_ONE)


def _factorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions import gamma
    return gamma(add(args[0], ONE))


def _subfactorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions import exp, gamma, lowergamma
    a = args[0]
    return mul(add(mul(pow(MINUS_ONE, add(a, ONE)), exp(mul(mul(neg(I), pi), a)), lowergamma(add(a, ONE), MINUS_ONE)),
                   gamma(add(a, ONE))), exp(MINUS_ONE))


def _factorial2(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.mod import mod
    from sympy_extras.core.relational import eq
    from sympy_extras.functions import gamma, sqrt
    from sympy_extras.functions.elementary.piecewise import piecewise
    n = args[0]
    half = mul(n, Rational(1, 2))
    parity = mod(n, Integer(2))
    return mul(pow(Integer(2), half), gamma(add(half, ONE)),
               piecewise((ONE, eq(parity, Integer(0))), (sqrt(mul(Integer(2), _inv(pi))), eq(parity, ONE))))


def _rising(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import gt, le
    from sympy_extras.functions import gamma
    from sympy_extras.functions.elementary.piecewise import piecewise
    x, k = args
    positive = mul(gamma(add(x, k)), _inv(gamma(x)))
    other = mul(mul(pow(MINUS_ONE, k), gamma(add(ONE, neg(x)))), _inv(gamma(add(neg(k), neg(x), ONE))))
    if not hints.piecewise:
        return other if le(x, Integer(0)) == true else positive
    return piecewise((positive, gt(x, Integer(0))), (other, true))


def _falling(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import ge, lt
    from sympy_extras.functions import gamma
    from sympy_extras.functions.elementary.piecewise import piecewise
    x, k = args
    nonnegative = mul(gamma(add(x, ONE)), _inv(gamma(add(x, neg(k), ONE))))
    other = mul(mul(pow(MINUS_ONE, k), gamma(add(k, neg(x)))), _inv(gamma(neg(x))))
    if not hints.piecewise:
        return other if lt(x, Integer(0)) == true else nonnegative
    return piecewise((nonnegative, ge(x, Integer(0))), (other, true))


def _binomial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions import gamma
    n, k = args
    return mul(gamma(add(n, ONE)), _inv(mul(gamma(add(k, ONE)), gamma(add(n, neg(k), ONE)))))


def _catalan(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions import gamma
    n = args[0]
    half = Rational(1, 2)
    return mul(pow(Integer(4), n), gamma(add(n, half)), _inv(mul(gamma(half), gamma(add(n, Integer(2))))))


def _beta(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    # SymPy's beta._eval_expand_func, on the arguments of the node itself
    from sympy_extras.functions import gamma
    x, y = [a for a in node.args if isinstance(a, Expr)]
    return mul(gamma(x), gamma(y), _inv(gamma(add(x, y))))


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma._eval_rewrite_as_factorial
def _gamma_as_factorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions import factorial
    return factorial(add(args[0], MINUS_ONE))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial._eval_rewrite_as_factorial
def _binomial_as_factorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions import factorial
    n, k = args
    return mul(factorial(n), _inv(mul(factorial(k), factorial(add(n, neg(k))))))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:RisingFactorial._eval_rewrite_as_factorial
def _rising_as_factorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import gt
    from sympy_extras.functions import factorial
    from sympy_extras.functions.elementary.piecewise import piecewise
    x, k = args
    if not (x.is_integer and k.is_integer):
        return None
    return piecewise((mul(factorial(add(k, x, MINUS_ONE)), _inv(factorial(add(x, MINUS_ONE)))), gt(x, Integer(0))),
                     (mul(pow(MINUS_ONE, k), factorial(neg(x)), _inv(factorial(add(neg(k), neg(x))))), true))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:FallingFactorial._eval_rewrite_as_factorial
def _falling_as_factorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import ge
    from sympy_extras.functions import factorial
    from sympy_extras.functions.elementary.piecewise import piecewise
    x, k = args
    if not (x.is_integer and k.is_integer):
        return None
    return piecewise((mul(factorial(x), _inv(factorial(add(neg(k), x)))), ge(x, Integer(0))),
                     (mul(pow(MINUS_ONE, k), factorial(add(k, neg(x), MINUS_ONE)), _inv(factorial(add(neg(x), MINUS_ONE)))),
                      true))


def rules() -> dict[tuple[type, type], Rule]:
    """The rules keyed by (the class of the node, the target class)."""
    from sympy_extras.functions.combinatorial import factorials as f, numbers as nb
    from sympy_extras.functions.special.beta_functions import Beta
    from sympy_extras.functions.special.gamma_functions import Gamma
    table: dict[tuple[type, type], Rule] = {
        (f.Factorial, Gamma): _factorial, (f.Subfactorial, Gamma): _subfactorial,
        (f.Factorial2, Gamma): _factorial2, (f.RisingFactorial, Gamma): _rising,
        (f.FallingFactorial, Gamma): _falling, (f.Binomial, Gamma): _binomial, (nb.Catalan, Gamma): _catalan,
        (Beta, Gamma): _beta,
        (Gamma, f.Factorial): _gamma_as_factorial, (f.Binomial, f.Factorial): _binomial_as_factorial,
        (f.RisingFactorial, f.Factorial): _rising_as_factorial,
        (f.FallingFactorial, f.Factorial): _falling_as_factorial}
    return table
