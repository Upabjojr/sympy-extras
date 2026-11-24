"""Regression tests of the second pass over the reports against SymPy
(sympy/sympy#31275, #31017) in the special functions: the reflection of
the orthogonal polynomials and the leading term of ``hyper`` in its
parameters, each checked against mpmath."""
from __future__ import annotations

from typing import Callable

import mpmath

from sympy_extras.core import Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.limits import limit
from sympy_extras.functions import chebyshevt, chebyshevu, gegenbauer, hermite, hyper, jacobi, legendre
from sympy_extras.functions.special.polynomials import hermite_prob

x = Symbol('x')
n = Symbol('n')
k = Symbol('k', integer=True)


def test_reflection_of_the_polynomials_for_an_integer_degree_only() -> None:
    # legendre(n, -x) was (-1)**n*legendre(n, x) for any symbol n (SymPy's too): the functions of a
    # noninteger degree have no parity, legendre(1/2, -3/10) is 0.3398... and (-1)**(1/2)*legendre(1/2, 3/10)
    # is 0.7009...*I
    two: list[Callable[[Expr, Expr], Expr]] = [legendre, chebyshevt, chebyshevu, hermite, hermite_prob]
    for f in two:
        assert f(n, -x).args == (n, -x), f
        assert f(k, -x) == (-1)**k*f(k, x), f
        value = complex(f(n, -x).subs(n, Rational(1, 2)).subs(x, Rational(3, 10)).evalf(20))
        wrong = complex(((-1)**n*f(n, x)).subs(n, Rational(1, 2)).subs(x, Rational(3, 10)).evalf(20))
        assert abs(value.imag) < 1e-15 and abs(value - wrong) > 0.1, f
    a, b = Symbol('a'), Symbol('b')
    assert jacobi(n, a, b, -x).args == (n, a, b, -x) and jacobi(k, a, b, -x) == (-1)**k*jacobi(k, b, a, x)
    assert gegenbauer(n, a, -x).args == (n, a, -x) and gegenbauer(k, a, -x) == (-1)**k*gegenbauer(k, a, x)
    assert abs(complex(legendre(n, -x).subs(n, Rational(1, 2)).subs(x, Rational(3, 10)).evalf(20))
               - complex(mpmath.legendre(0.5, -0.3))) < 1e-14
    assert abs(mpmath.jacobi(0.5, 0.2, 0.7, -0.3) - (-1)**0.5*mpmath.jacobi(0.5, 0.7, 0.2, 0.3)) > 0.1
    assert abs(mpmath.gegenbauer(0.5, 0.7, -0.3) - (-1)**0.5*mpmath.gegenbauer(0.5, 0.7, 0.3)) > 0.1
    # for the integers the identity holds
    for degree in range(5):
        assert legendre(k, -x).subs(k, degree).doit().expand() == legendre(degree, -x).expand()


def test_limit_of_hyper_in_its_parameters() -> None:
    # (sympy/sympy#31017 C5) hyper([x], [2], 1).as_leading_term(x) was the function itself, and its
    # limit at 0 raised NotImplementedError (the mrv of a hyper); SymPy raises AttributeError
    assert hyper([x], [2], 1).as_leading_term(x) == 1
    assert limit(hyper([x], [2], 1), x, 0) == 1 and limit(hyper([x], [2], 1), x, 0, '-') == 1
    assert limit(x*hyper([x], [2], 1), x, 0) == 0
    at_one = limit(hyper([x], [2], 1), x, 1)
    assert not at_one.has(x) and abs(complex(at_one.evalf(20)) - complex(mpmath.hyp1f1(1, 2, 1))) < 1e-15
    assert abs(mpmath.hyp1f1(mpmath.mpf(1) + mpmath.mpf(10)**-12, 2, 1) - mpmath.hyp1f1(1, 2, 1)) < 1e-11
    inside = limit(hyper([1, x], [3], Rational(1, 2)), x, 1)
    assert abs(complex(inside.evalf(20)) - complex(mpmath.hyp2f1(1, 1, 3, 0.5))) < 1e-15
    # on the circle of convergence at 1 with re(sum(bq) - sum(ap)) > 0 (Gauss)
    gauss = limit(hyper([x, 1], [3], x), x, 1, '-')
    assert abs(complex(gauss.evalf(20)) - 2) < 1e-15
    assert abs(mpmath.hyp2f1(1 - mpmath.mpf(10)**-7, 1, 3, 1 - mpmath.mpf(10)**-7) - 2) < 1e-5
    # no value where the function is not continuous in the parameters: the circle of convergence
    # without the condition (a lower parameter at a nonpositive integer was in this list: it has a
    # pole there, test_limit_of_hyper_at_a_pole_of_a_lower_parameter)
    for e in [hyper([x + 1, 1], [2], 1 - x)]:
        try:
            e.as_leading_term(x)
        except NotImplementedError:
            continue
        raise AssertionError(e)


def test_limit_of_hyper_at_a_pole_of_a_lower_parameter() -> None:
    # limit(hyper([1], [x], 1/2), x, 0) raised NotImplementedError("Don't know how to calculate the mrv
    # of 'hyper((1,), (1/_p,), 1/2)'"): the function has a simple pole where a lower parameter is a
    # nonpositive integer -m, with the residue (-1)**m*(a)_(m+1)*z**(m+1)/(m!*(m+1)!)*hyper(a+m+1; m+2; z)
    from sympy_extras.core import oo
    from sympy_extras.core.limits import Limit
    from sympy_extras.functions import exp, gamma
    half = Rational(1, 2)
    f = hyper([1], [x], half)
    assert limit(f, x, 0) == oo and limit(f, x, 0, '-') == -oo
    assert limit(f, x, -1) == -oo and limit(f, x, -1, '-') == oo
    assert limit(x*f, x, 0) == exp(half)/2 == limit(f/gamma(x), x, 0)
    tiny = mpmath.mpf(10)**-10
    assert abs(tiny*mpmath.hyp1f1(1, tiny, 0.5) - mpmath.exp(0.5)/2) < 1e-8
    with mpmath.workdps(40):
        assert abs(-tiny*mpmath.hyp1f1(1, -1 - tiny, 0.5) + mpmath.exp(0.5)/4) < 1e-8
    assert limit((x + 1)*f, x, -1) == -exp(half)/4
    assert limit(hyper([1], [x**2], half), x, 0, '-') == oo and limit(hyper([1], [1/x], half), x, oo) == oo
    assert limit(x*hyper([1, 1], [x], half), x, 0) == 2
    assert abs(tiny*mpmath.hyp2f1(1, 1, tiny, 0.5) - 2) < 1e-8
    # another lower parameter, m = 3
    residue = limit(x*hyper([1], [x - 3, Rational(1, 3)], 2), x, 0)
    with mpmath.workdps(40):
        numeric = tiny*mpmath.hyper([1], [tiny - 3, mpmath.mpf(1)/3], 2)
    assert not residue.has(x) and abs(complex(residue.evalf(20)) - complex(numeric)) < 1e-7
    # a polynomial (an upper parameter -j) has no pole for j <= m, and one for j > m
    assert limit(hyper([-1], [x], half), x, -1) == Rational(3, 2) == limit(hyper([-1], [x], half), x, -1, '-')
    assert limit(hyper([-1], [x], half), x, -2) == Rational(5, 4)
    assert limit(hyper([-2], [x], half), x, -1) == -oo and limit(hyper([-1], [x], half), x, 0) == -oo
    with mpmath.workdps(40):
        assert abs(mpmath.hyp1f1(-1, -1 + tiny, 0.5) - 1.5) < 1e-8
    # what is not of these kinds is not found, and it does not raise: the argument outside the circle of
    # convergence, an upper parameter tending to a nonpositive integer too, an argument tending to 0
    # with the lower parameter (hyper([1], [x], x) tends to 2: its leading term was 1)
    for e in [hyper([1, 1], [x], 2), hyper([x], [2*x], half), hyper([1], [x], x), hyper([1/x], [1], x)]:
        assert isinstance(limit(e, x, 0), Limit), e
    for e in [hyper([1], [x], x), hyper([1/x], [1], x)]:
        try:
            e.as_leading_term(x)
        except NotImplementedError:
            continue
        raise AssertionError(e)
    assert abs(mpmath.hyp1f1(1, tiny, tiny) - 2) < 1e-8
