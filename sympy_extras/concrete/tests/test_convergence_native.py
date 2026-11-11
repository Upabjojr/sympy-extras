"""The native convergence tests on objects of the core (the tests on SymPy
objects, through the face, are in test_convergence.py)."""
from __future__ import annotations

from sympy_extras.core import Rational, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.core.symbol import Symbol
from sympy_extras.core.logic import Boolean, and_, false, true
from sympy_extras.core.relational import eq
from sympy_extras.functions import Factorial, binomial, cos, exp, factorial, gamma, log, sin
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.concrete.convergence import (_coefficients, _majorant, _real_polynomial, is_convergent,
                                               product_convergence, sum_convergence)

n, x, p = Symbol('n'), Symbol('x'), Symbol('p')


def _at(condition: Boolean, values: dict[str, Rational]) -> Basic:
    """The condition at a point of the parameters."""
    rule: dict[Basic, Basic] = {s: values[s.name] for s in condition.free_symbols
                                if isinstance(s, Symbol) and s.name in values}
    return condition._xreplace_evaluated(rule)


def test_numerical_series() -> None:
    # textbook verdicts (Knopp, chapter IX)
    assert sum_convergence(factorial(n)/n**n, n) is true           # ratio test, limit 1/e
    assert sum_convergence(binomial(2*n, n)/4**n, n) is false      # Raabe, rho = 1/2
    assert sum_convergence(1/(n**2 + 1), n) is true
    assert sum_convergence(1/n, n) is false
    assert sum_convergence(1/sqrt(n), n) is false
    assert sum_convergence(2**n/factorial(n), n) is true
    assert sum_convergence(n**3*exp(-n), n) is true
    assert sum_convergence(1, n) is false
    assert sum_convergence(0, n) is true


def test_alternating_and_oscillating() -> None:
    assert sum_convergence((-1)**n/n, n) is true
    assert sum_convergence((-1)**n/log(n), n, lower=2) is true
    assert sum_convergence((-1)**n, n) is false
    assert sum_convergence(cos(n*pi)/n, n) is true
    # Dirichlet's test, where SymPy's is_convergent says False
    assert sum_convergence(sin(n)/n, n) is true
    assert sum_convergence(sin(n), n) is false
    assert sum_convergence(n*sin(n), n) is false


def test_parameters() -> None:
    # SumConvergence[x^n, n] is Abs[x] < 1, SumConvergence[x^n/n, n] adds x == -1
    assert sum_convergence(x**n, n) == and_(x > -1, x < 1)
    assert sum_convergence(x**n/n, n) == and_(x >= -1, x < 1)
    assert sum_convergence(1/n**p, n) == (p > 1)
    assert sum_convergence(1/n**p, n, p > 2) is true
    assert sum_convergence(1/n**p, n, p < 1) is false
    assert sum_convergence(x**n/factorial(n), n) is true
    condition = sum_convergence((-1)**n*n**p, n)
    assert condition is not None
    assert _at(condition, {'p': Rational(-1, 2)}) is true and _at(condition, {'p': Rational(0)}) is false


def test_is_convergent() -> None:
    assert is_convergent(x**n, n, and_(x > 0, x < 1)) is True
    assert is_convergent(x**n, n, x > 1) is False
    assert is_convergent(x**n, n) is None
    assert is_convergent(1/n**2, n) is True


def test_products() -> None:
    assert product_convergence(1 + 1/n**2, n) is true
    assert product_convergence(1 + x/n**2, n) is true
    assert product_convergence(1 + 1/n, n) is false
    assert product_convergence(2, n) is false
    assert product_convergence(1, n) is true
    assert product_convergence(1 + x**n, n) == and_(x > -1, x < 1)
    assert product_convergence(1 + abs_(x)/n, n) == eq(x, 0)
    # sympy-extras#39: prod (1 + (-1)**n/n**a) converges exactly for a > 1/2
    condition = product_convergence(1 + (-1)**n/n**p, n)
    assert condition is not None
    assert _at(condition, {'p': Rational(3, 4)}) is true and _at(condition, {'p': Rational(1, 3)}) is false


def test_polynomial_coefficients() -> None:
    # SymPy's Poly(pi*n + 1, n) has the domain EX, which the typed Poly
    # does not have: the coefficients are collected from the expansion
    assert _coefficients(pi*n + 1, n) == {1: pi, 0: 1}
    assert _coefficients((n + x)**2, n) == {2: 1, 1: 2*x, 0: x**2}
    assert _coefficients(sqrt(n) + 1, n) is None
    assert _coefficients(exp(n), n) is None
    assert _real_polynomial(sqrt(2)*n**2 + 3, n)
    assert not _real_polynomial(x*n, n)
    assert _majorant(((-1)**n - 1)/n**3, n) == 2/abs_(n**3)


def test_rewrite_as_factorial() -> None:
    # the rules of SymPy's _eval_rewrite_as_factorial, used to see the
    # sign of a term
    assert binomial(2*n, n).rewrite(Factorial) == factorial(2*n)/factorial(n)**2
    assert gamma(n + 1).rewrite(Factorial) == factorial(n)
