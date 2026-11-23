"""The expressions the printer tests share: a corpus on which SymPy's
LaTeX printer is right (``SAME``: the output must be SymPy's), and the
expressions on which this package deliberately prints something else
(``DIFFERENT``, with the expected output and the reason)."""
from __future__ import annotations

from sympy_extras.core import (Catalan as CatalanConstant, Derivative, Dict, E, Eq, Equivalent, EulerGamma, Float,
    Function, GoldenRatio, I, Implies, Integer, Lambda, Mod, Ne, Not, Or, And, Order, Rational, Subs, Tuple, Xor,
    false, nan, oo, pi, sqrt, Symbol, true, zoo)
from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.basic import Basic
from sympy_extras.core.limits import Limit
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial,
    factorial2, subfactorial)
from sympy_extras.functions.combinatorial.numbers import (bell, bernoulli, catalan, divisor_sigma, euler, fibonacci,
    genocchi, harmonic, lucas, mobius, primenu, primeomega, reduced_totient, totient, tribonacci)
from sympy_extras.functions.elementary.complexes import Abs, arg, atan2, conjugate, im, re, sign
from sympy_extras.functions.elementary.exponential import LambertW, exp, log
from sympy_extras.functions.elementary.hyperbolic import asinh, cosh, sinh
from sympy_extras.functions.elementary.integers import ceiling, floor, frac
from sympy_extras.functions.elementary.miscellaneous import Max, Min
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import asin, cos, sin, tan
from sympy_extras.functions.special.bessel import (airyai, airyaiprime, airybi, besseli, besselj, besselk, bessely,
    hankel1, jn, yn)
from sympy_extras.functions.special.beta_functions import beta
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside
from sympy_extras.functions.special.elliptic_integrals import elliptic_e, elliptic_f, elliptic_k, elliptic_pi
from sympy_extras.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erf2, erfc, erfi, erfinv,
    expint, fresnelc, fresnels, li)
from sympy_extras.functions.special.gamma_functions import digamma, gamma, loggamma, lowergamma, polygamma, uppergamma
from sympy_extras.functions.special.hyper import hyper, meijerg
from sympy_extras.functions.special.mathieu_functions import mathieuc, mathieucprime, mathieus
from sympy_extras.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevu,
    gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy_extras.functions.special.singularity_functions import SingularityFunction
from sympy_extras.functions.special.spherical_harmonics import Ynm
from sympy_extras.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy_extras.functions.special.zeta_functions import dirichlet_eta, lerchphi, polylog, zeta
from sympy_extras.integrals.integral import Integral
from sympy_extras.sets.conditionset import ConditionSet
from sympy_extras.sets.fancysets import Complexes, ImageSet, Integers, Naturals, Naturals0, Rationals
from sympy_extras.sets.powerset import PowerSet
from sympy_extras.sets.sets import (Complement, Contains, EmptySet, FiniteSet, Intersection, Interval, ProductSet,
    Reals, Union)

x, y, z, n, k, a, b, m = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('n'), Symbol('k'), Symbol('a'), Symbol('b'), Symbol('m')
f = Function('f')
g = Function('g')

#: SymPy prints these as this package does
SAME: list[Basic] = [
    # sums, products, quotients, powers
    x + y, x - y, -x, -x - y, 2*x, -2*x, x/2, -x/2, x*y/2, 3*x/(2*y), x**2, x**-1, x**-2, 1/x, 1/(x + y), (x + y)/z,
    (x + y)**2, (x + y)*z, x*(y + z), sqrt(x), sqrt(x + 1), x**Rational(1, 3), x**Rational(-1, 2),
    x**Rational(2, 3), x**Rational(-2, 3), 2**x, (-2)**x, (-x)**2, Rational(1, 2)**x, Rational(-1, 2)**x,
    x**y**z, (x**y)**z, x**(y + z), x**(-y), x**(1/y), 1/x**y, x*y*z, 2*x*y, x*y**2, x**2*y, 2*sqrt(2), sqrt(2)*x,
    x*sqrt(y), 3*x**2*y/(4*z**3), -3*x**2*y/(4*z**3), (x + 1)*(y + 2)/(z + 3), x + y + z, x**2 + x + 1,
    x**2 - x - 1, 1 - x, -1 + x, x*y + x + y, 1 + x + x**2/2 + x**3/6, (x - 1)**2, (1 - x)*(1 + x), 2*(x + y),
    -(x + y)*z, -2*(x + y), (x + y)/2, -(x + y)/2, x/(y*z), x/y/z, x/(y/z), 1/(2*x), 1/(x*y), x*y**-1*z**-2,
    x**n*y**-n, a*b/(x*y), a/(b*x*y**2), 2*a/(3*b), -a/b, -a*b, -a/(2*b), a*(b + 1)/(x*(y + 1)), -(a + b)/(x + y),
    (a - b)/(a + b), (a + b)**2/(a - b)**3, x**-y, 2**-x, 2**(-x)*y, (x + y)**-2, 1/sqrt(x), x/sqrt(y), sqrt(x)/y,
    1/(x*sqrt(y)), sqrt(2)/2, 1/sqrt(2), x**Rational(3, 2), x**Rational(-3, 2),
    # numbers and Symbol
    pi, E, I, oo, -oo, pi*x, 2*pi*I, x + I*y, E**x, pi**2, Rational(3, 4), Rational(-3, 4), Integer(-5), Integer(7),
    Float(2.5), Float(-2.5), Float(1e20), Float(1e-20), 2.5*x, x**2.5, Float(1.0), Float(3.14159, 6), I*x, -I, 2*I,
    x + I, (1 + I)*x, pi/2, -pi/2, 2*pi, pi*I/2, E**(I*pi), zoo, nan, GoldenRatio, EulerGamma, CatalanConstant,
    Symbol('alpha'), Symbol('x_1'), Symbol('x1'), Symbol('x_1^2'), Symbol('xdot'), Symbol('Gamma'),
    Symbol('x_alpha'), Symbol('x^2')**2, Symbol('xhat_i'),
    # elementary functions
    exp(x), exp(-x), exp(x)**2, exp(x + y), x*exp(-2), exp(2*x)/y, sin(x), sin(x)**2, sin(x)**-1, sin(x)**y,
    sin(2*x), sin(x + y)*cos(y), log(x), log(x)**2, x*log(x), tan(x)/x, sinh(x)*cosh(x), asin(x), asinh(x)**2,
    sin(x)/cos(x), sin(x)**2 + cos(x)**2, 1/sin(x), 1/sin(x)**2, sin(x)**2/cos(x)**3, log(x)/x, exp(x)/x, x*exp(x),
    2*exp(x), exp(x)*sin(x), exp(-x)*y, exp(-2*x), 1/exp(x), cos(x)**-2, tan(x)**Rational(1, 2), log(x)**-1,
    Abs(x)**-1, exp(x)**-1, exp(x**2), exp(x)**y, exp(sin(x)), Abs(x), Abs(x)**2, floor(x), ceiling(x)**2, frac(x),
    re(x), im(x)**2, conjugate(x), conjugate(x)**2, sign(x), arg(x), Max(x, y), Min(x, y, 2), Max(x, y)**2,
    # undefined functions
    f(x), f(x)**2, f(x, y), g(f(x)), f(x)*g(x), Function('alpha')(x), Function('f_1')(x), Function('foo')(x),
    Function('foo')(x)**2, f(x)**-1, f(x)**-2, f(x)**n, sin(x)**n,
    # relations and logic
    Eq(x, y), Ne(x, y), x < y, x <= y, x > y, x >= y, Eq(x + y, z**2), And(x < y, y < z), Or(x < y, y < z),
    Xor(x < y, y < z), Implies(x < y, y < z), Equivalent(x < y, y < z), And(Or(x < y, y < z), x < z),
    Or(And(x < y, y < z), x < z), true, false,
    # calculus
    Derivative(f(x), x), Derivative(f(x), x, x), Derivative(f(x, y), x, y), Derivative(f(x), x)*2, Derivative(f(x), x) + 1,
    Derivative(f(x) + g(x), x), Derivative(f(x), x)**2, Subs(f(x), x, 1), Limit(f(x), x, 0), Limit(f(x), x, oo),
    Limit(f(x), x, 0, '-'), Limit(f(x), x, 0, '+-'), Order(x**2), Order(x**2, (x, oo)), x + Order(x**2),
    Lambda(x, x**2), Lambda((x, y), x + y), Tuple(x, y), Tuple(x), Sum(x**k, (k, 0, n)), Sum(x + k, (k, 0, n)),
    Product(k, (k, 1, n)), Sum(k, (k, 1, n))*x, x*Sum(k, (k, 1, n)), Sum(k, (k, 1, n)) + 1,
    Sum(k*m, (k, 1, n), (m, 1, k)), Sum(k, (k, 1, n))**2, Integral(x**2, x), Integral(x + 1, x),
    Integral(f(x), x)*2, 2*Integral(f(x), x), Integral(f(x), x) + 1, Integral(-x, x), Integral(f(x), x)**2,
    # combinatorial functions
    factorial(n), factorial(n + 1), factorial2(n), subfactorial(n), binomial(n, k), binomial(n, k)**2,
    RisingFactorial(x, n), FallingFactorial(x, n), factorial(factorial(n)), factorial(2*n), bernoulli(n),
    bernoulli(n, x), bell(n), fibonacci(n), fibonacci(n)**2, lucas(n), tribonacci(n), euler(n), catalan(n),
    harmonic(n), genocchi(n), mobius(n), totient(n), totient(n)**2, divisor_sigma(n), primenu(n), primeomega(n),
    reduced_totient(n),
    # special functions
    gamma(x), gamma(x)**2, lowergamma(a, x), uppergamma(a, x), beta(x, y), loggamma(x), digamma(x), polygamma(n, x),
    zeta(x), zeta(x, a), zeta(x)**2, dirichlet_eta(x), lerchphi(z, a, b), polylog(a, z), polylog(a, z)**2,
    besselj(n, x), besselj(n, x)**2, bessely(n, x), besseli(n, x), besselk(n, x), hankel1(n, x), hankel1(n, x)**2,
    jn(n, x), yn(n, x), airyai(x), airybi(x)**2, airyaiprime(x), erf(x), erfc(x), erfi(x)**2, Ei(x), Si(x), Ci(x),
    Chi(x), Shi(x), li(x), expint(n, x), fresnels(x), fresnelc(x)**2, erfinv(x), elliptic_k(x), elliptic_f(x, m),
    elliptic_e(m), elliptic_e(x, m), elliptic_pi(n, m), elliptic_pi(n, x, m), elliptic_k(x)**2, legendre(n, x),
    legendre(n, x)**2, chebyshevt(n, x), chebyshevu(n, x), hermite(n, x), laguerre(n, x), assoc_laguerre(n, a, x),
    assoc_legendre(n, m, x), jacobi(n, a, b, x), gegenbauer(n, a, x), DiracDelta(x), DiracDelta(x, 2), Heaviside(x),
    Heaviside(x)**2, KroneckerDelta(n, k), KroneckerDelta(n, k + 1), LeviCivita(n, k, m),
    SingularityFunction(x, a, n), LambertW(x), LambertW(x, k), hyper((a, b), (z,), x),
    meijerg((a,), (b,), (m,), (n,), x), mathieuc(a, b, z), mathieus(a, b, z), mathieucprime(a, b, z),
    Ynm(n, m, x, y),
    # sets and the rest
    Interval(0, 1), Interval(0, 1, True, False), Interval(0, oo), FiniteSet(1, 2, 3), FiniteSet(x, y),
    Union(Interval(0, 1), Interval(2, 3)), Intersection(Interval(0, x), Interval(y, 3)),
    Complement(Interval(0, x), FiniteSet(y)), EmptySet, Reals, Integers, Naturals, Naturals0, Rationals, Complexes,
    Piecewise((x, x < 0), (y, True)), Piecewise((x, x < 0), (y, x > 1)), 2*Piecewise((x, x < 0), (y, True)),
    Mod(x, y), Mod(x, y) + 1, 2*Mod(x, y), Mod(x + 1, y), Mod(x, y)**2, Mod(2*x, y), AccumBounds(0, 1),
    Dict({x: 1, y: 2}), Contains(x, Interval(0, 1)), ConditionSet(x, x**2 > 1, Reals),
    ImageSet(Tuple(n), 2*n, Integers), PowerSet(FiniteSet(1, 2)), ProductSet(Interval(0, 1), Interval(0, 1)),
    ProductSet(Interval(0, 1), FiniteSet(1, 2)),
]

#: the expression, what this package prints, why it is not what SymPy prints
DIFFERENT: list[tuple[Basic, str, str]] = [
    (factorial(n)**2, r'\left(n!\right)^{2}', 'SymPy: n!^{2}'),
    (airyaiprime(x)**2, r'\left(Ai^\prime\left(x\right)\right)^{2}', r'SymPy: {Ai^\prime}^{2}\left(x\right)'),
    (atan2(y, x), r'\operatorname{atan2}{\left(y,x \right)}', r'#30844: \operatorname{atan}_{2}'),
    (erf2(x, y), r'\operatorname{erf2}{\left(x,y \right)}', r'#30844: \operatorname{erf}_{2}'),
    (divisor_sigma(n, k), r'\sigma_{k}\left(n\right)', r'#30844: \sigma_k'),
    (Limit(sin(x)/x, x, 0), r'\lim_{x \to 0^+} \frac{\sin{\left(x \right)}}{x}',
     'SymPy brackets every product and sum under a limit; a product needs none'),
    (Piecewise((x, x < 0), (y, True)) + 1,
     r'\left(\begin{cases} x & \text{for}\: x < 0 \\y & \text{otherwise} \end{cases}\right) + 1',
     'cases are open on the right: bracketed in a sum'),
    (Not(x < y), r'\neg \left(x < y\right)', r'#30844: \neg x < y'),
]
