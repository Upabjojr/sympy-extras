"""Regression tests for the findings of sympy/sympy#31275 in the
combinatorial functions which the ports shared with SymPy (V72, V74): the
rewrites which hold for the nonnegative integers only."""
from __future__ import annotations

import mpmath

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.relational import ge
from sympy_extras.functions import Binomial, Factorial, FallingFactorial, Piecewise, RisingFactorial, catalan, ff, rf
from sympy_extras.functions.combinatorial.numbers import Catalan

x = Symbol('x')
k = Symbol('k', integer=True)
m = Symbol('m', integer=True, nonnegative=True)
neg = Symbol('j', integer=True, negative=True)


def test_factorial_power_as_binomial_is_guarded() -> None:
    # V72: factorial(k)*binomial(x + k - 1, k) is zoo*0 = nan at k = -2, where rf(x, -2) = 1/((x - 1)*(x - 2))
    for node, bare in ((RisingFactorial(x, k), Factorial(k)), (FallingFactorial(x, k), Factorial(k))):
        rewritten = node.rewrite(Binomial)
        assert isinstance(rewritten, Piecewise) and rewritten.args[0].cond == ge(k, Integer(0)), rewritten
        assert rewritten.args[1].expr == node and rewritten.args[0].expr.has(bare)
    point = Rational(1, 3)
    for f, reference in ((rf, mpmath.rf), (ff, mpmath.ff)):
        for order in (-3, -2, 0, 1, 4):
            value = f(x, k).rewrite(Binomial).subs(k, Integer(order)).subs(x, point).doit()
            assert abs(complex(value.evalf(20)) - complex(reference(mpmath.mpf(1)/3, order))) < 1e-12, (f, order)
    # the bare formula where it holds, the node where it does not
    assert not isinstance(RisingFactorial(x, m).rewrite(Binomial), Piecewise)
    assert RisingFactorial(x, m).rewrite(Binomial) != RisingFactorial(x, m)
    assert RisingFactorial(x, neg).rewrite(Binomial) == RisingFactorial(x, neg)
    assert FallingFactorial(x, neg).rewrite(Binomial) == FallingFactorial(x, neg)


def test_catalan_as_factorial_is_guarded() -> None:
    # V74: factorial(2*n)/(factorial(n + 1)*factorial(n)) is nan at n = -1, -2, where catalan is -1/2, 0
    for target in (Factorial, Binomial):
        rewritten = Catalan(k).rewrite(target)
        assert isinstance(rewritten, Piecewise) and rewritten.args[0].cond == ge(k, Integer(0)), rewritten
        values = [rewritten.subs(k, Integer(order)).doit() for order in (-3, -2, -1, 0, 1, 2, 5)]
        assert values == [catalan(Integer(order)) for order in (-3, -2, -1, 0, 1, 2, 5)], target
        assert values == [0, 0, Rational(-1, 2), 1, 1, 2, 42]
        assert not isinstance(Catalan(m).rewrite(target), Piecewise) and Catalan(m).rewrite(target) != Catalan(m)
        # an order which is not an integer: the formula holds
        assert not isinstance(Catalan(x).rewrite(target), Piecewise)
