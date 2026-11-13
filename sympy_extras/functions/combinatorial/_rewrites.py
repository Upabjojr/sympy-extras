"""The ``rewrite`` rules of the combinatorial functions (and ``beta``) in
terms of ``gamma`` (SymPy's ``_eval_rewrite_as_gamma``), and of ``gamma``,
``binomial`` and the Pochhammer symbols in terms of ``factorial`` (SymPy's
``_eval_rewrite_as_factorial``), and of the Pochhammer symbols and
``binomial`` in terms of each other, and the other rules of the
combinatorial numbers (sums, products, ``zeta``, ``polygamma``, closed
forms with radicals), as a typed table for
:func:`sympy_extras.core.rewriting.rewrite`. Ported from SymPy's
``sympy.functions.combinatorial`` modules.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py, sympy/functions/combinatorial/numbers.py,
# sympy/functions/special/beta_functions.py, sympy/functions/special/gamma_functions.py (the
# _eval_rewrite_as_gamma and _eval_rewrite_as_factorial methods; the _eval_rewrite_as_binomial,
# _eval_rewrite_as_FallingFactorial, _eval_rewrite_as_RisingFactorial methods; the _eval_rewrite_as_Product,
# _eval_rewrite_as_Sum, _eval_rewrite_as_hyper, _eval_rewrite_as_sqrt, _eval_rewrite_as_GoldenRatio,
# _eval_rewrite_as_TribonacciConstant, _eval_rewrite_as_zeta, _eval_rewrite_as_polylog,
# _eval_rewrite_as_polygamma, _eval_rewrite_as_digamma, _eval_rewrite_as_trigamma,
# _eval_rewrite_as_genocchi, _eval_rewrite_as_bernoulli, _eval_rewrite_as_dirichlet_eta and
# _eval_rewrite_as_uppergamma methods of the combinatorial functions) (SymPy 1.14)
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
from sympy_extras.core.rewriting import RewriteHints, RuleKey

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


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:RisingFactorial._eval_rewrite_as_FallingFactorial
def _rising_as_falling(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import ff
    x, k = args
    return ff(add(x, k, MINUS_ONE), k)


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:RisingFactorial._eval_rewrite_as_binomial
def _rising_as_binomial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import binomial, factorial
    x, k = args
    if not k.is_integer:
        return None
    return mul(factorial(k), binomial(add(x, k, MINUS_ONE), k))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:FallingFactorial._eval_rewrite_as_RisingFactorial
def _falling_as_rising(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import rf
    x, k = args
    return rf(add(x, neg(k), ONE), k)


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:FallingFactorial._eval_rewrite_as_binomial
def _falling_as_binomial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import binomial, factorial
    x, k = args
    if not k.is_integer:
        return None
    return mul(factorial(k), binomial(x, k))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial._eval_rewrite_as_FallingFactorial
def _binomial_as_falling(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import factorial, ff
    n, k = args
    if not k.is_integer:
        return None
    return mul(ff(n, k), _inv(factorial(k)))


def rules() -> dict[RuleKey, Rule]:
    """The rules keyed by (the class of the node, the target class or
    SymPy name)."""
    from sympy_extras.functions.combinatorial import factorials as f, numbers as nb
    from sympy_extras.functions.special.beta_functions import Beta
    from sympy_extras.functions.special.gamma_functions import Gamma
    table: dict[RuleKey, Rule] = {
        (f.Factorial, Gamma): _factorial, (f.Subfactorial, Gamma): _subfactorial,
        (f.Factorial2, Gamma): _factorial2, (f.RisingFactorial, Gamma): _rising,
        (f.FallingFactorial, Gamma): _falling, (f.Binomial, Gamma): _binomial, (nb.Catalan, Gamma): _catalan,
        (Beta, Gamma): _beta,
        (Gamma, f.Factorial): _gamma_as_factorial, (f.Binomial, f.Factorial): _binomial_as_factorial,
        (f.RisingFactorial, f.Factorial): _rising_as_factorial,
        (f.FallingFactorial, f.Factorial): _falling_as_factorial,
        (f.RisingFactorial, f.FallingFactorial): _rising_as_falling, (f.RisingFactorial, f.Binomial): _rising_as_binomial,
        (f.FallingFactorial, f.RisingFactorial): _falling_as_rising,
        (f.FallingFactorial, f.Binomial): _falling_as_binomial, (f.Binomial, f.FallingFactorial): _binomial_as_falling}
    table.update(_number_rules())
    return table


# the other rules of the combinatorial functions

def _second(args: tuple[Expr, ...]) -> Optional[Expr]:
    return args[1] if len(args) > 1 else None


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:factorial._eval_rewrite_as_Product
def _factorial_as_product(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import product_
    from sympy_extras.core.symbol import Dummy
    n = args[0]
    if n.is_nonnegative is True and n.is_integer is True:
        i = Dummy('i', integer=True)
        return product_(i, (i, ONE, n))
    return None


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:subfactorial._eval_rewrite_as_factorial
def _subfactorial_as_factorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.summations import summation
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial
    arg = args[0]
    i = Dummy('i')
    return mul(factorial(arg), summation(mul(pow(MINUS_ONE, i), _inv(factorial(i))), (i, Integer(0), arg)))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:subfactorial._eval_rewrite_as_uppergamma
def _subfactorial_as_uppergamma(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.numbers import E
    from sympy_extras.functions.special.gamma_functions import uppergamma
    return mul(uppergamma(add(args[0], ONE), MINUS_ONE), _inv(E))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:fibonacci._eval_rewrite_as_sqrt
def _fibonacci_as_sqrt(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    if len(args) != 1:
        return None  # SymPy's rule takes no polynomial variable (a TypeError there)
    n = args[0]
    five = sqrt(Integer(5))
    return (Integer(2)**(-n) * five * ((1 + five)**n - (-five + 1)**n)) / 5


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:fibonacci._eval_rewrite_as_GoldenRatio
def _fibonacci_as_golden_ratio(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.numbers import GoldenRatio
    if len(args) != 1:
        return None
    n = args[0]
    return (GoldenRatio**n - 1 / (-GoldenRatio)**n) / (2 * GoldenRatio - 1)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:lucas._eval_rewrite_as_sqrt
def _lucas_as_sqrt(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    n = args[0]
    five = sqrt(Integer(5))
    return Integer(2)**(-n) * ((1 + five)**n + (-five + 1)**n)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:tribonacci._eval_rewrite_as_sqrt
def _tribonacci_as_sqrt(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.miscellaneous import cbrt
    if len(args) != 1:
        return None
    n = args[0]
    w = (-1 + I * sqrt(Integer(3))) / 2
    r33 = sqrt(Integer(33))
    c1, c2 = cbrt(19 + 3 * r33), cbrt(19 - 3 * r33)
    a = (1 + c1 + c2) / 3
    b = (1 + w * c1 + w**2 * c2) / 3
    c = (1 + w**2 * c1 + w * c2) / 3
    return (a**(n + 1) / ((a - b) * (a - c)) + b**(n + 1) / ((b - a) * (b - c))
            + c**(n + 1) / ((c - a) * (c - b)))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:tribonacci._eval_rewrite_as_TribonacciConstant
def _tribonacci_as_constant(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.numbers import TribonacciConstant
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.integers import floor
    from sympy_extras.functions.elementary.miscellaneous import cbrt
    if len(args) != 1:
        return None
    n = args[0]
    b = cbrt(586 + 102 * sqrt(Integer(33)))
    tn = 3 * b * TribonacciConstant**n / (b**2 - 2 * b + 4)
    return floor(tn + Rational(1, 2))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:bernoulli._eval_rewrite_as_zeta
def _bernoulli_as_zeta(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.special.zeta_functions import zeta
    n = args[0]
    x = args[1] if len(args) > 1 else ONE
    return piecewise((ONE, eq(n, Integer(0))), (-n * zeta(1 - n, x), true))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:bell._eval_rewrite_as_Sum
def _bell_as_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.numbers import E, oo
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import factorial
    n = args[0]
    if len(args) > 1 or n.is_nonnegative is not True:
        return node  # SymPy returns the node itself
    k = Dummy('k', integer=True, nonnegative=True)
    return 1 / E * sum_(k**n / factorial(k), (k, Integer(0), oo))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_rewrite_as_polygamma
def _harmonic_as_polygamma(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.numbers import EulerGamma
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.special.gamma_functions import gamma, polygamma
    n = args[0]
    m = args[1] if len(args) > 1 else ONE
    if m.is_integer is True and m.is_positive is True:
        return piecewise((polygamma(Integer(0), n + 1) + EulerGamma, eq(m, ONE)),
                         (MINUS_ONE**m * (polygamma(m - 1, ONE) - polygamma(m - 1, n + 1)) / gamma(m), true))
    return None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_rewrite_as_digamma,
# harmonic._eval_rewrite_as_trigamma (self.rewrite(polygamma))
def _harmonic_via_polygamma(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.rewriting import rewrite
    from sympy_extras.functions.special.gamma_functions import PolyGamma
    rewritten = rewrite(node, (), PolyGamma, hints)
    assert isinstance(rewritten, Expr)
    return rewritten


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_rewrite_as_Sum
def _harmonic_as_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    n = args[0]
    m = args[1] if len(args) > 1 else ONE
    k = Dummy('k', integer=True)
    return sum_(k**(-m), (k, ONE, n))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_rewrite_as_zeta
def _harmonic_as_zeta(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.numbers import EulerGamma
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.special.gamma_functions import digamma
    from sympy_extras.functions.special.zeta_functions import zeta
    n = args[0]
    m = args[1] if len(args) > 1 else ONE
    return piecewise((digamma(n + 1) + EulerGamma, eq(m, ONE)), (zeta(m) - zeta(m, n + 1), true))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:euler._eval_rewrite_as_Sum
def _euler_as_sum(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.combinatorial.factorials import binomial
    from sympy_extras.functions.combinatorial.numbers import euler
    n = args[0]
    x = _second(args)
    if x is None and n.is_even is True:
        k = Dummy('k', integer=True)
        j = Dummy('j', integer=True)
        h = n / 2
        inner = sum_(binomial(k, j) * (MINUS_ONE**j * (k - 2 * j)**(2 * h + 1)) / (Integer(2)**k * I**k * k),
                     (j, Integer(0), k))
        return I * sum_(inner, (k, ONE, 2 * h + 1))
    if x is not None and not (x.is_Number and x == 0):
        k = Dummy('k', integer=True)
        return sum_(binomial(n, k) * euler(k) / Integer(2)**k * (x - Rational(1, 2))**(n - k), (k, Integer(0), n))
    return None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:euler._eval_rewrite_as_genocchi
def _euler_as_genocchi(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.combinatorial.numbers import genocchi
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.functions.special.gamma_functions import digamma
    n = args[0]
    x = _second(args)
    if x is None:
        return piecewise((pi / 2, eq(n, MINUS_ONE)), (-Integer(2)**n * genocchi(n + 1, Rational(1, 2)) / (n + 1), true))
    return piecewise((digamma((x + 1) / 2) - digamma(x / 2), eq(n, MINUS_ONE)), (-genocchi(n + 1, x) / (n + 1), true))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:catalan._eval_rewrite_as_binomial
def _catalan_as_binomial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import binomial
    n = args[0]
    return binomial(2 * n, n) / (n + 1)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:catalan._eval_rewrite_as_factorial
def _catalan_as_factorial(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import factorial
    n = args[0]
    return factorial(2 * n) / (factorial(n + 1) * factorial(n))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:catalan._eval_rewrite_as_hyper
def _catalan_as_hyper(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.special.hyper import hyper
    n = args[0]
    return hyper([1 - n, -n], [Integer(2)], ONE)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:catalan._eval_rewrite_as_Product
def _catalan_as_product(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.concrete.expr_with_limits import product_
    from sympy_extras.core.symbol import Dummy
    n = args[0]
    if not (n.is_integer is True and n.is_nonnegative is True):
        return node  # SymPy returns the node itself
    k = Dummy('k', integer=True, positive=True)
    return product_((n + k) / k, (k, Integer(2), n))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:genocchi._eval_rewrite_as_bernoulli
def _genocchi_as_bernoulli(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.numbers import bernoulli
    n = args[0]
    x = args[1] if len(args) > 1 else ONE
    if x == 1 and n.is_integer is True and n.is_nonnegative is True:
        return 2 * (1 - Integer(2)**n) * bernoulli(n)
    return 2 * (bernoulli(n, x) - Integer(2)**n * bernoulli(n, (x + 1) / 2))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:genocchi._eval_rewrite_as_dirichlet_eta
def _genocchi_as_eta(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.special.zeta_functions import dirichlet_eta
    n = args[0]
    x = args[1] if len(args) > 1 else ONE
    return -2 * n * dirichlet_eta(1 - n, x)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:andre._eval_rewrite_as_zeta
def _andre_as_zeta(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.trigonometric import cos
    from sympy_extras.functions.special.gamma_functions import gamma
    from sympy_extras.functions.special.zeta_functions import zeta
    s = args[0]
    return (2 * gamma(s + 1) / (2 * pi)**(s + 1)
            * (zeta(s + 1, Rational(1, 4)) - cos(pi * s) * zeta(s + 1, Rational(3, 4))))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:andre._eval_rewrite_as_polylog
def _andre_as_polylog(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.special.zeta_functions import polylog
    s = args[0]
    return (-I)**(s + 1) * polylog(-s, I) + I**(s + 1) * polylog(-s, -I)


def _number_rules() -> dict[RuleKey, Rule]:
    from sympy_extras.functions.combinatorial import factorials as f, numbers as nb
    from sympy_extras.functions.special.gamma_functions import DiGamma, PolyGamma, TriGamma, UpperGamma
    from sympy_extras.functions.special.zeta_functions import DirichletEta, PolyLog, Zeta
    return {
        (f.Factorial, 'Product'): _factorial_as_product,
        (f.Subfactorial, f.Factorial): _subfactorial_as_factorial,
        (f.Subfactorial, UpperGamma): _subfactorial_as_uppergamma,
        (nb.Fibonacci, 'sqrt'): _fibonacci_as_sqrt, (nb.Fibonacci, 'GoldenRatio'): _fibonacci_as_golden_ratio,
        (nb.Lucas, 'sqrt'): _lucas_as_sqrt,
        (nb.Tribonacci, 'sqrt'): _tribonacci_as_sqrt, (nb.Tribonacci, 'TribonacciConstant'): _tribonacci_as_constant,
        (nb.Bernoulli, Zeta): _bernoulli_as_zeta, (nb.Bell, 'Sum'): _bell_as_sum,
        (nb.Harmonic, PolyGamma): _harmonic_as_polygamma, (nb.Harmonic, DiGamma): _harmonic_via_polygamma,
        (nb.Harmonic, TriGamma): _harmonic_via_polygamma, (nb.Harmonic, 'Sum'): _harmonic_as_sum,
        (nb.Harmonic, Zeta): _harmonic_as_zeta,
        (nb.Euler, 'Sum'): _euler_as_sum, (nb.Euler, nb.Genocchi): _euler_as_genocchi,
        (nb.Catalan, f.Binomial): _catalan_as_binomial, (nb.Catalan, f.Factorial): _catalan_as_factorial,
        (nb.Catalan, 'hyper'): _catalan_as_hyper, (nb.Catalan, 'Product'): _catalan_as_product,
        (nb.Genocchi, nb.Bernoulli): _genocchi_as_bernoulli, (nb.Genocchi, DirichletEta): _genocchi_as_eta,
        (nb.Andre, Zeta): _andre_as_zeta, (nb.Andre, PolyLog): _andre_as_polylog}
