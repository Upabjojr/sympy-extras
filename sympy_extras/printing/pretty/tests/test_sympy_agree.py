"""The pretty printer against SymPy's, where SymPy is right.

The corpus is printed by both printers, in Unicode and in ASCII, and the
pictures must be the same character by character. The cases where the
two differ on purpose (the defects of sympy/sympy#30844) are not here
but in ``test_sympy_issue_30844.py``.
"""
from __future__ import annotations

from typing import Union as TUnion

import sympy

from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import (And, E, Eq, Equivalent, EulerGamma, Float, GoldenRatio, I, Implies, Integer, Ne, Not,
    Or, Rational, Symbol, Xor, false, nan, oo, pi, sqrt, true, zoo)
from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Dict, Tuple
from sympy_extras.core.expr import Expr, to_expr
from sympy_extras.core.function import Derivative, Function, Lambda, Subs
from sympy_extras.core.limits import Limit
from sympy_extras.core.mod import Mod
from sympy_extras.core.numbers import Catalan
from sympy_extras.core.order import Order
from sympy_extras.core.symbol import Symbol as SymbolClass
from sympy_extras.functions.combinatorial.factorials import binomial, factorial, factorial2, subfactorial
from sympy_extras.functions.combinatorial.numbers import (bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas,
    tribonacci)
from sympy_extras.functions.elementary.complexes import Abs, arg, atan2, conjugate, im, re, sign
from sympy_extras.functions.elementary.exponential import LambertW, exp, log
from sympy_extras.functions.elementary.hyperbolic import cosh, sinh
from sympy_extras.functions.elementary.integers import ceiling, floor
from sympy_extras.functions.elementary.miscellaneous import Max
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import acos, cos, sin, tan
from sympy_extras.functions.special.bessel import (airyai, airyaiprime, airybi, airybiprime, besseli, besselj,
    besselk, bessely)
from sympy_extras.functions.special.beta_functions import beta
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside
from sympy_extras.functions.special.elliptic_integrals import elliptic_e, elliptic_f, elliptic_k, elliptic_pi
from sympy_extras.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfc, erfi, expint, fresnelc,
    fresnels, li)
from sympy_extras.functions.special.gamma_functions import digamma, gamma, loggamma, lowergamma, polygamma, uppergamma
from sympy_extras.functions.special.hyper import hyper, meijerg
from sympy_extras.functions.special.mathieu_functions import mathieuc, mathieus
from sympy_extras.functions.special.polynomials import assoc_legendre, chebyshevt, hermite, jacobi, legendre
from sympy_extras.functions.special.singularity_functions import SingularityFunction
from sympy_extras.functions.special.tensor_functions import KroneckerDelta
from sympy_extras.functions.special.zeta_functions import dirichlet_eta, lerchphi, polylog, zeta
from sympy_extras.integrals.integral import Integral
from sympy_extras.interop import to_sympy
from sympy_extras.printing.pretty import pretty
from sympy_extras.sets.conditionset import ConditionSet
from sympy_extras.sets.fancysets import Complexes, ImageSet, Integers, Naturals, Naturals0, Rationals
from sympy_extras.sets.sets import (Complement, Contains, EmptySet, FiniteSet, Intersection, Interval, ProductSet,
    Reals, Union, UniversalSet)

x, y, z, n, k, a, b, t = (Symbol(name) for name in 'xyznkabt')
alpha, theta = Symbol('alpha'), Symbol('theta')
f, g = Function('f'), Function('g')
half = Rational(1, 2)

Bounds = tuple[SymbolClass, TUnion[Expr, int], TUnion[Expr, int]]


def _integral(function: Expr, limit: TUnion[SymbolClass, Bounds]) -> Integral:
    if isinstance(limit, tuple):
        return Integral(function, limit[0], to_expr(limit[1]), to_expr(limit[2]))
    return Integral(function, limit)


def _corpus() -> list[Basic]:
    return [
        x, alpha, Symbol('x_1'), Symbol('x_1^2'), Symbol('xdot'), Symbol('alpha_beta'), Symbol('x12'),
        Integer(0), Integer(-3), Rational(1, 2), Rational(-1, 2), Rational(12, 13), Rational(-12, 13), Float(2.5), Float(-2.5),
        oo, -oo, zoo, nan, pi, E, I, GoldenRatio, EulerGamma, Catalan,
        x + y, x - y, -x, -x - y, 1 - x, x**2 + 2*x + 1, x + half, x - half, half - x, x/2 + 1/y, x/y + half,
        2*x, -2*x, x*y, x/2, -x/2, 2*x/3, -2*x/3, x/y, -x/y, (x + 1)/y, -(x + 1)/y, (x + 1)/(y - 1), 1/x, -1/x, 1/(x*y),
        x*y/(a*b), 2*x*y/(3*a*b), (x + y)*(x - y), 2*(x + y), -(x + y), (x + y)/2, x*(y + 1)/z,
        x**2, x**y, x**-2, x**half, x**-half, x**Rational(1, 3), x**Rational(2, 3), x**Rational(-3, 2), (x + y)**2, (x*y)**2,
        (x + y)**half, sqrt(x + y)/z, sqrt(2), sqrt(2)*x, x**(1/n), x**(y + 1), x**(y/2), 2**x, 2**(x**2), (x**2)**y, x**(1/y),
        sqrt(x/y), sqrt(1 + sqrt(x)), 1/sqrt(x), (1/x)**y, pi*x, pi**2, E**x, I*x, 2*I, 1 + I, x**(2*y),
        Float(2.5)*x, Float(-2.5)*x, x**Float(0.5),
        sin(x), sin(x)**2, sin(x)**y, sin(x)**(y/2), sin(x + y), sin(x)/cos(x), sin(2*x)*cos(x), sin(x)**-1, f(x), f(x, y), f(x)**2, g(f(x)),
        exp(x), exp(-x), exp(x**2), exp(x/2), exp(x + y), exp(x)**2, log(x), log(x)**2, log(x + 1)/x, exp(2*x)/(exp(x) + 1),
        tan(x), acos(x), sinh(x), cosh(x)**2, atan2(y, x),
        Abs(x), Abs(x/y), Abs(x)**2, conjugate(x), conjugate(x + y), re(x), im(x), sign(x), arg(x),
        floor(x), ceiling(x/2), floor(x)**2, Max(x, y),
        factorial(x), factorial(x + 1), factorial(x)**2, factorial(factorial(x)), factorial2(x), subfactorial(x), binomial(n, k),
        binomial(n + 1, k/2), factorial(n)/factorial(k), 2*factorial(n), -factorial(n),
        gamma(x), gamma(x)**2, lowergamma(a, x), uppergamma(a, x), beta(x, y), digamma(x), polygamma(n, x), loggamma(x),
        besselj(n, x), bessely(n, x), besseli(n, x), besselk(n, x), airyai(x), airybi(x), airyaiprime(x), airybiprime(x),
        erf(x), erfc(x), erfi(x), Ei(x), Si(x), Ci(x), Shi(x), Chi(x), li(x), expint(n, x), expint(2, x), fresnels(x), fresnelc(x),
        DiracDelta(x), DiracDelta(x, 2), Heaviside(x), LambertW(x),
        elliptic_k(x), elliptic_e(x), elliptic_e(x, y), elliptic_f(x, y), elliptic_pi(x, y), elliptic_pi(x, y, z), elliptic_f(x/2, y),
        zeta(x), zeta(x, a), dirichlet_eta(x), polylog(2, x), polylog(n, x), lerchphi(x, y, z),
        hyper((a, b), (n,), x), hyper((1,), (), x), hyper((a,), (b, n), x/2), meijerg((a,), (b,), (n,), (k,), x), meijerg((1,), (), (), (2,), x/2),
        chebyshevt(n, x), legendre(n, x), hermite(n, x), jacobi(n, a, b, x), assoc_legendre(n, k, x),
        bernoulli(n), bernoulli(n, x), euler(n), fibonacci(n), fibonacci(n, x), lucas(n), bell(n), catalan(n), harmonic(n), tribonacci(n),
        KroneckerDelta(n, k), SingularityFunction(x, a, n), SingularityFunction(x, 1, 2),
        mathieuc(a, b, x), mathieus(a, b, x),
        Eq(x, y), Ne(x, y), x < y, x <= y, x > y, x >= y, Eq(x + 1, y**2), Eq(x/2, 1/y), x + 1 < y,
        And(x < y, y < z), Or(x < y, y < z), And(a, b), Or(a, b), Not(a), And(a, Or(b, t)), Or(a, And(b, t)), Not(And(a, b)),
        Implies(a, b), Equivalent(a, b), Xor(a, b), Not(Not(a)), And(Not(a), b), Not(Implies(a, b)), Not(Equivalent(a, b)),
        Implies(And(a, b), t), And(Implies(a, b), t), true, false,
        Derivative(f(x), x), Derivative(f(x), x, x), Derivative(f(x, y), x, y), Derivative(f(x, y), x, x, y), Derivative(sin(x)/x, x),
        Derivative(f(x), x)**2, 2*Derivative(f(x), x), Derivative(f(x), x) + 1,
        _integral(x, x), _integral(x**2, (x, 0, 1)), _integral(sin(x)/x, (x, 0, oo)), _integral(x + 1, x), _integral(1/x, x), _integral(x, x)**2,
        2*_integral(x, x), _integral(x, x) + 1, _integral(f(x), (x, a, b)), _integral(_integral(f(x, y), x), y), _integral(x*y, (x, 0, 1)),
        _integral(exp(-x**2), (x, -oo, oo)), _integral(-x, x), _integral(f(theta), theta),
        Sum(k, (k, 1, n)), Sum(1/k, (k, 1, oo)), Sum(k**2 + 1, (k, 1, n)), Sum(x**k/factorial(k), (k, 0, oo)), Sum(k, (k, 1, n))**2,
        2*Sum(k, (k, 1, n)), Sum(k, (k, 1, n)) + 1, Sum(Sum(k*n, (k, 1, n)), (n, 1, a)), Sum(k, (k, 1, n), (n, 1, a)), Sum(-k, (k, 1, n)),
        Sum(k, (k, 1, n))/2, x*Sum(1/k, (k, 1, n)),
        Product(k, (k, 1, n)), Product(1/k, (k, 1, oo)), Product(k**2, (k, 1, n))*2, Product(k, (k, 1, n)) + 1, Product(k, (k, 1, n), (n, 1, a)),
        Limit(f(x), x, 0), Limit(sin(x)/x, x, 0), Limit(f(x), x, oo), Limit(f(x), x, 0, '-'), Limit(f(x), x, 0, '+-'), Limit(x**2, x, 1),
        2*Limit(f(x), x, 0), Limit(x + 1, x, 0),
        Subs(f(x), x, 0), Subs(Derivative(f(x), x), x, 1), Subs(f(x, y), (x, y), (0, 1)), Subs(x/y, x, 1),
        Order(x), Order(x**2), x + Order(x**2), Order(1/x, (x, oo)), Order(x*y, x, y),
        Lambda(x, x**2), Lambda((x, y), x + y),
        Piecewise((x, x < 1), (x**2, True)), Piecewise((x, x < 1), (1/x, x < 2), (0, True)), 2*Piecewise((x, x < 1), (0, True)),
        Piecewise((x, x < 1), (0, True)) + 1, Piecewise((x, x < 1), (0, True))**2,
        Tuple(x, y), Tuple(x), Tuple(), Tuple(x/2, y**2), Dict({x: 1, y: 2}), AccumBounds(0, 1),
        Interval(0, 1), Interval(0, 1, True, False), Interval(0, oo), Interval(x, x), FiniteSet(1, 2, 3), FiniteSet(x, y/2), EmptySet,
        Union(Interval(0, 1), Interval(2, 3)), Intersection(Interval(0, 1), FiniteSet(x)), Complement(Interval(0, 1), FiniteSet(x)),
        ProductSet(Interval(0, 1), Interval(2, 3)), ProductSet(Interval(0, 1), Interval(0, 1)), Reals, Naturals, Naturals0, Integers, Rationals, Complexes, UniversalSet,
        Union(Interval(0, 1), FiniteSet(x), Interval(3, 4)),
        ImageSet(Tuple(x), x**2, Naturals), ImageSet(Tuple(x), 1/x, Integers),
        Mod(x, y), Mod(x, 2) + 1, Mod(x + 1, y), 2*Mod(x, y), Mod(x, y)**2,
        x*y*z + x*y + 1, (x + 1)*(y + 1)*(z + 1), x**3 - 3*x**2 + 3*x - 1, a*x**2 + b*x + n, -a*x**2 - b*x, (a + b)**n/(a - b), 1/(1 + 1/x),
        1/(x + 1/(y + 1/z)), x/(y/z), (x/y)/z, x**(y**z), (x**y)**z, ((x + 1)/(x - 1))**2, sqrt((x + 1)/(x - 1)), exp(sin(x))*cos(x),
        x**2/2 + x/3 + Rational(1, 4), Rational(3, 4)*x**2, x + Rational(22, 7), Rational(22, 7)*x, -Rational(22, 7)*x,
    ]


def _unicode_only() -> list[Basic]:
    # (in ASCII SymPy prints Contains with the string printer, arguments included)
    return [Contains(x, Interval(0, 1)), Contains(x, Reals), Contains(x + 1, Union(Interval(0, 1), FiniteSet(3)))]


def _ascii_only() -> list[Basic]:
    # (in Unicode SymPy always brackets the condition; here a relation needs none)
    return [ConditionSet(x, x**2 > 1, Reals), ConditionSet(x, Eq(sin(x), 0), UniversalSet)]


def _differences(expressions: list[Basic], unicode: bool) -> list[str]:
    found: list[str] = []
    for e in expressions:
        mine = pretty(e, use_unicode=unicode, wrap_line=False)
        theirs = sympy.pretty(to_sympy(e, evaluate=False), use_unicode=unicode, wrap_line=False)
        if mine != theirs:
            found.append('%r (unicode=%s):\n%s\nSymPy:\n%s' % (e, unicode, mine, theirs))
    return found


def test_the_corpus_is_printed_as_by_sympy_in_unicode() -> None:
    assert _differences(_corpus() + _unicode_only(), True) == []


def test_the_corpus_is_printed_as_by_sympy_in_ascii() -> None:
    assert _differences(_corpus() + _ascii_only(), False) == []


def test_the_corpus_is_large() -> None:
    assert len(_corpus()) > 350


def test_ascii_output_is_ascii() -> None:
    # SymPy leaks non-ASCII characters in ASCII mode for some classes (sympy/sympy#30844)
    for e in _corpus() + _unicode_only() + _ascii_only():
        text = pretty(e, use_unicode=False, wrap_line=False)
        assert text.isascii(), (e, text)


def test_rows_have_the_same_width() -> None:
    for e in _corpus():
        for unicode in (True, False):
            rows = pretty(e, use_unicode=unicode, wrap_line=False).split('\n')
            assert len({len(row) for row in rows}) == 1, (e, rows)
