"""summation (Sum.doit) against SymPy (the oracle)."""
from __future__ import annotations

import sympy
from sympy import KroneckerDelta, Piecewise, Rational as R, binomial, exp, factorial, oo, sin

from sympy_extras.concrete.summations import summation
from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

i = sympy.Symbol('i', integer=True)
k = sympy.Symbol('k', integer=True)
n = sympy.Symbol('n', integer=True)
m = sympy.Symbol('m', integer=True, positive=True)
x, y = sympy.symbols('x y')
j = sympy.Symbol('j')

CASES = [(i**2, (i, 1, n)), (i**3, (i, 0, n)), (x**i, (i, 0, n)), (2**i, (i, 1, n)), (1/(i*(i + 1)), (i, 1, n)),
         (i, (i, 1, 4)), (1/i**2, (i, 1, oo)), (1/i, (i, 1, n)), (1/i**3, (i, 1, n)), (binomial(n, i), (i, 0, n)),
         (i*binomial(n, i), (i, 0, n)), (x**i/factorial(i), (i, 0, oo)), (i*x**i, (i, 0, n)), (1/(i**2 - 1), (i, 2, n)),
         (sin(i), (i, 0, 3)), (i**2 + 3*i + 1, (i, 0, m)), (y, (i, 1, n)), (i*j, (i, 1, n)),
         (Piecewise((i, i < 5), (0, True)), (i, 0, 10)), (KroneckerDelta(i, k), (i, 0, n)),
         (1/((i + 1)*(i + 2)), (i, 0, oo)), (R(1, 2)**i, (i, 0, oo)), (factorial(i), (i, 0, n)), (i**4, (i, 1, n)),
         (1/i**2, (i, 1, n)), ((-1)**i/i, (i, 1, oo)), (exp(i), (i, 0, n)), (i/(i + 1), (i, 0, 5)), (x, (i, n, n)),
         (i, (i, 5, 3)), (i, (i, 1, 0)), (i**2, (i, -n, n)), (1/(2*i + 1), (i, 0, n)), (j**i, (i, 1, n)),
         (i*(i - 1)*binomial(n, i), (i, 0, n)), (binomial(m, i)*binomial(m, m - i), (i, 0, m)), (i**R(1, 2), (i, 1, n)),
         (1/i**(x + 1), (i, 1, oo)), (1/(i + 1)**2, (i, 0, oo))]


def test_summation_matches_sympy() -> None:
    for f, lim in CASES:
        v, a, b = lim
        var = from_sympy(v)
        assert isinstance(var, Symbol)
        ours = summation(expr_from_sympy(f), (var, expr_from_sympy(sympy.sympify(a)), expr_from_sympy(sympy.sympify(b))))
        assert str(to_sympy(ours, evaluate=False)) == str(sympy.summation(f, lim)), (f, lim)


def test_sum_doit_matches_sympy() -> None:
    for f, lim in CASES[:12]:
        s = sympy.Sum(f, lim)
        assert str(to_sympy(expr_from_sympy(s).doit(), evaluate=False)) == str(s.doit()), s


def test_hypergeometric_sums_match_sympy() -> None:
    # eval_sum_hyper: the sum as an expanded hypergeometric function on its
    # condition of convergence
    from sympy import rf
    from sympy.concrete.summations import eval_sum_hyper as sympy_eval_sum_hyper
    from sympy_extras.concrete.summations import eval_sum_hyper
    a = sympy.Symbol('a')
    cases = [(x**i/factorial(i), (i, 0, oo)), (R(1, 2)**i, (i, 0, oo)), ((-1)**i/(2*i + 1), (i, 0, oo)),
             (x**i*i, (i, 0, oo)), (x**(2*i)/factorial(2*i), (i, 0, oo)), (rf(a, i)*x**i/factorial(i), (i, 0, oo)),
             (binomial(n, i)**2, (i, 0, n)), (1/factorial(i), (i, 1, oo)), (x**i/factorial(i), (i, 2, oo)),
             (x**i/(i + 1), (i, 0, oo)), (binomial(2*i, i)*x**i, (i, 0, oo)), (2**i/factorial(i), (i, -oo, 0)),
             (x**i/factorial(i), (i, 0, n)), (1/2**i, (i, 3, oo)), (x**i, (i, -oo, oo)), (1/factorial(i)**2, (i, 0, oo))]
    for f, (v, lo, hi) in cases:
        var = from_sympy(v)
        assert isinstance(var, Symbol)
        limits = (var, expr_from_sympy(sympy.sympify(lo)), expr_from_sympy(sympy.sympify(hi)))
        ours = eval_sum_hyper(expr_from_sympy(f), limits)
        theirs = sympy_eval_sum_hyper(f, (v, lo, hi))
        assert (None if ours is None else str(to_sympy(ours, evaluate=False))) == (None if theirs is None else str(theirs))
        assert str(to_sympy(summation(expr_from_sympy(f), limits), evaluate=False)) == str(sympy.summation(f, (v, lo, hi)))


def test_convergence_statement_matches_sympy() -> None:
    from sympy_extras.functions.special.hyper import Hyper
    a, z = sympy.symbols('a z')
    for h in [sympy.hyper([1, 1], [2], z), sympy.hyper([a], [], z), sympy.hyper([a, 1], [3], z), sympy.hyper([], [1], z),
              sympy.hyper([1, 2, 3], [], z), sympy.hyper([1, 2], [3], 2*z), sympy.hyper([-n, -n, 1], [1, 1], 1)]:
        ours = expr_from_sympy(h)
        if not isinstance(ours, Hyper):
            continue
        c = ours.convergence_statement
        assert (str(c) if isinstance(c, bool) else str(to_sympy(c, evaluate=False))) == str(h.convergence_statement), h
