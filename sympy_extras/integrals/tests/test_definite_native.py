"""Tests of the native definite integration driver on core objects: the
values against hand computations and mpmath's quadrature, the
unevaluated Integral node, the numerical evaluation of integrands, the
helpers which replaced SymPy's ``Poly``, ``singularities`` and
``lambdify``."""
from __future__ import annotations

import mpmath

from sympy_extras.core import Integer, Rational, I, Symbol, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.evalf import numeric_value, to_float
from sympy_extras.core.relational import eq
from sympy_extras.functions import Piecewise, abs_, asin, atan, cos, exp, gamma, log, sin
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.definite import (DefiniteIntegrator, _coefficients, _numeric_function, _quadrature,
                                             _singular_points, conditional_integral, definite_integral,
                                             verify_numerically)
from sympy_extras.integrals.integral import Integral, is_unevaluated
from sympy_extras.settings import configure

x, k, c, y = Symbol('x'), Symbol('k'), Symbol('c'), Symbol('y')
a = Symbol('a', positive=True)


def _number(e: Expr) -> complex:
    value = numeric_value(e, 80)
    assert value is not None, e
    return complex(value)


def test_values_against_hand_computations() -> None:
    # Gradshteyn-Ryzhik 3.241.2 (Euler's reflection), 3.893.1, the
    # Gaussian integral, Euler's gamma integral and Frullani-free
    # elementary ones computed by hand
    assert definite_integral(x**k / (x + 3), (x, 0, oo), (k > -1) & (k < 0)) == -3**k * pi / sin(pi * k)
    assert definite_integral(exp(-a * x) * cos(k * x), (x, 0, oo), k > 0) == a / (a**2 + k**2)
    assert definite_integral(exp(-x**2), (x, -oo, oo)) == sqrt(pi)
    assert definite_integral(exp(-x) * x**k, (x, 0, oo), k > -1) == gamma(k + 1)
    assert definite_integral(x**2, (x, 0, 1)) == Rational(1, 3)
    assert definite_integral(abs_(x - 1), (x, 0, 3)) == Rational(5, 2)
    assert definite_integral(1 / (x**2 + c**2), (x, 0, oo), c > 0) == pi / (2 * c)
    # swapped bounds, an empty range, a constant over an infinite range
    assert definite_integral(exp(-x), (x, oo, 0)) == -1
    assert definite_integral(exp(-x), (x, 2, 2)) == 0
    assert definite_integral(Integer(3), (x, 0, oo)) == oo


def test_values_against_quadrature() -> None:
    cases: list[tuple[Expr, int, Expr]] = [(log(1 + x) / (1 + x**2), 0, Integer(1)), (atan(x) / (x * (1 + x**2)), 0, oo),
             (sqrt(x) * sqrt(2 - x), 0, Integer(2)), (log(x)**2 / (1 + x**2), 0, oo)]
    for f, lower, upper in cases:
        value = definite_integral(f, (x, lower, upper))
        assert not is_unevaluated(value), f
        g = _numeric_function(f, x)
        with mpmath.workdps(30):
            hi = mpmath.inf if upper == oo else mpmath.mpf(to_float(upper))
            expected = complex(mpmath.quad(lambda u: g(mpmath.mpc(u)), [lower, 1, hi] if hi != 1 else [lower, hi]))
        assert abs(_number(value) - expected) < 1e-15, (f, value)


def test_divergence_and_conditions() -> None:
    assert definite_integral(1 / x, (x, 0, 1)) == oo
    assert definite_integral(sin(x) / x, (x, 0, oo)) == pi / 2
    found = conditional_integral(exp(-a * x), x, Integer(0), oo)
    assert found == ConditionalValue(1 / a)
    # the condition comes from the Mellin strips: -1 < k < 0
    value = definite_integral(x**k / (x + 3), (x, 0, oo))
    assert isinstance(value, Piecewise)
    assert is_unevaluated(value.args[-1].expr)


def test_unevaluated_integral_node() -> None:
    node = Integral(exp(-y * x**2), x, Integer(0), oo)
    assert str(node) == 'Integral(exp(-x**2*y), (x, 0, oo))'
    assert node.free_symbols == {y}
    assert node.subs(x, 2) == node
    assert node.subs(y, 2) == Integral(exp(-2 * x**2), x, Integer(0), oo)
    assert node.diff(x) == 0
    # Leibniz's rule in a bound
    F = Integral(exp(x), x, Integer(0), y)
    assert F.diff(y) == exp(y)
    assert is_unevaluated(1 + node) and not is_unevaluated(x)
    assert node.doit() == sqrt(pi) / (2 * sqrt(y)) or is_unevaluated(node.doit())


def test_numeric_function_of_piecewise() -> None:
    g = _numeric_function(Piecewise((x**2, x < 1), (2 - x, True)), x)
    assert abs(complex(g(mpmath.mpc(0.5))) - 0.25) < 1e-15
    assert abs(complex(g(mpmath.mpc(1.5))) - 0.5) < 1e-15
    # a real power of a negative base takes the principal branch
    h = _numeric_function(x**Rational(1, 3), x)
    assert abs(complex(h(mpmath.mpc(-8))) - (1 + 1.7320508075688772j)) < 1e-12


def test_quadrature() -> None:
    found = _quadrature(abs_(x - Rational(1, 2)), x, Integer(0), Integer(1), [Rational(1, 2)])
    assert found is not None and abs(found - 0.25) < 1e-12
    # a segment of the complex plane through a branch point
    z = Symbol('z')
    found = _quadrature(Integer(2)**log(z), z, -I, I)
    with mpmath.workdps(20):
        segment = mpmath.quad(lambda u: mpmath.power(2, mpmath.log(-1j + 2j * u)) * 2j, [0, 0.5, 1])
    assert found is not None and abs(found - complex(segment)) < 1e-12
    # a pole inside the range: not trusted
    assert _quadrature(1 / (x - Rational(1, 3))**2, x, Integer(0), Integer(1)) is None


def test_verify_numerically() -> None:
    assert verify_numerically(1 / a, exp(-a * x), x, Integer(0), oo) is True
    assert verify_numerically(2 / a, exp(-a * x), x, Integer(0), oo) is False
    # the branch at an isolated value is checked at its point
    value = Piecewise((Integer(2), eq(y, 1)), ((y**2 - 1) / log(y), True))
    assert verify_numerically(value, y**x, x, Integer(0), Integer(2), y > 0) is True
    wrong = Piecewise((Integer(3), eq(y, 1)), ((y**2 - 1) / log(y), True))
    assert verify_numerically(wrong, y**x, x, Integer(0), Integer(2), y > 0) is False


def test_coefficients_with_irrational_coefficients() -> None:
    # the typed Poly refuses sqrt(2) and pi as coefficients
    assert _coefficients(sqrt(2) * x**2 + a * x + pi, x) == [pi, a, sqrt(2)]
    assert _coefficients(sin(x) + 1, x) is None
    assert _coefficients(Integer(3), x) == [Integer(3)]


def test_singular_points() -> None:
    assert _singular_points(1 / x, x, c, y) == [Integer(0)]
    # no point of a range the facts of the bounds exclude
    assert _singular_points(1 / x, x, a, a + 1) == []
    # a zero against symbolic bounds, which the solver leaves as a
    # ConditionSet under the bounds, from the whole real line
    assert _singular_points(1 / (x - c), x, Integer(0), Integer(1)) == [c]
    # the complex cube roots of x**3 - c**3 are dropped
    assert _singular_points(1 / (x**3 - c**3), x, Integer(0), Integer(1)) == [c]
    assert _singular_points(1 / sin(x), x, Integer(1), Integer(7)) == [pi, 2 * pi]


def test_definite_integrator_of_parametric() -> None:
    integrator = DefiniteIntegrator(None)
    assert integrator.integrate(exp(-a * x), x, Integer(0), oo, 0) == ConditionalValue(1 / a)
    assert integrator.verify(1 / a, exp(-a * x), x, Integer(0), oo) is True


def test_without_numerical_checks() -> None:
    with configure(numerical_checks=False):
        assert definite_integral(exp(-x**2), (x, -oo, oo)) == sqrt(pi)


def test_beta_integrals_by_the_arcsine_substitution() -> None:
    # the regression of the port: sin(asin(sqrt(u))) stays unevaluated on
    # the core, and the Beta integrals of the quarter periods were lost
    # (Maxima's rtest_integrate 876, sin(x)**(1/3)*cos(x)**(1/3) over a
    # period, left unevaluated; the value is the old one, checked by
    # quadrature)
    f = sin(x)**Rational(1, 3) * cos(x)**Rational(1, 3)
    assert definite_integral(f, (x, 0, pi / 2)) == gamma(Rational(2, 3))**2 / (2 * gamma(Rational(4, 3)))
    value = definite_integral(f, (x, 0, 2 * pi))
    assert not is_unevaluated(value)
    assert abs(_number(value) - (1.5400426634543828858 + 2.6674321389266885873j)) < 1e-15


def test_quadrature_through_a_logarithmic_endpoint() -> None:
    # the regression of the port: log(cos(x) + 1) is log(0) at x = pi,
    # where the numerical rule of the node gives nothing; the quadrature
    # takes the node as one at an end (as lambdify's -inf did) instead of
    # giving up, and the value -pi*log(2) checks again
    found = _quadrature(log(cos(x) + 1), x, Integer(0), pi)
    assert found is not None and abs(found - _number(-pi * log(2))) < 1e-10
    assert verify_numerically(-pi * log(2), log(cos(x) + 1), x, Integer(0), pi) is True
    u = Symbol('u')
    found = _quadrature(u**2 * Integral(Integer(1), x, Integer(0), u), u, Integer(0), Integer(1))
    assert found is not None and abs(found - 0.25) < 1e-12


def test_samples_at_quadratic_roots_in_radicals() -> None:
    # the regression of the port: the sample a = IndexedRoot(a**2 - 3, a, 0)
    # left the hidden zero sqrt(3 - a**2) unproved, the check gave nothing
    # and the area of the parameter cell stayed unevaluated (the old
    # samples were SymPy's CRootOf, written -sqrt(3))
    b = Symbol('b')
    value = -b * sqrt(3 - b**2) / 2 - 3 * asin(sqrt(3) * b / 3) / 2 + 3 * pi / 4
    assert verify_numerically(value, sqrt(3 - y**2), y, b, sqrt(3), [eq(b**2 - 3, 0)]) is True
    found = definite_integral(2 * sqrt(3 - y**2), (y, b, sqrt(3)), [eq(b**2 - 3, 0)], special_values=False)
    assert not is_unevaluated(found)


def test_integral_node_values() -> None:
    # the value of a definite Integral node went to SymPy's evalf through
    # the bridge; now mpmath's tanh-sinh quadrature of the native
    # integrand, confirmed at a higher precision
    from sympy_extras.functions import besselj
    with mpmath.workprec(80):
        series = mpmath.nsum(lambda n: (-1)**(n + 1) / n**n, [1, mpmath.inf])
        cases = [(Integral(exp(-x**2), x, Integer(0), oo), mpmath.sqrt(mpmath.pi) / 2),
                 (Integral(log(x), x, Integer(0), Integer(1)), mpmath.mpf(-1)),
                 (Integral(x**x, x, Integer(0), Integer(1)), series),
                 (Integral(besselj(0, x), x, Integer(0), Integer(3)), mpmath.quad(lambda t: mpmath.besselj(0, t), [0, 3])),
                 (Integral(x**Rational(-3, 2), x, Integer(161), oo), 2 / mpmath.sqrt(161)),
                 # a single tanh-sinh rule over (-326, 500) misses the peak and estimates no error
                 (Integral(exp(-x**2), x, Integer(-326), Integer(500)), mpmath.sqrt(mpmath.pi))]
        for e, known in cases:
            assert e.is_number
            value = numeric_value(e, 80)
            assert value is not None and abs(value - known) < mpmath.mpf(10)**-20, e
    # not accurate: no value (SymPy's evalf prints garbage digits)
    assert numeric_value(Integral(sin(x) / x, x, Integer(0), oo), 53) is None
    assert Integral(exp(-x), x).is_number is False
