"""Tests of the real antiderivatives of x**n * Q**(m/2)."""
from __future__ import annotations

from sympy import symbols, sqrt, Rational, asin, asinh, log, simplify, diff, S, N, Integral

from sympy_extras._typing import as_expr
from sympy_extras.integrals.radicals import quadratic_radical_antiderivative

x = symbols('x')


def _checks(f: object) -> bool:
    F = quadratic_radical_antiderivative(as_expr(f), x)
    return F is not None and simplify(diff(F, x) - as_expr(f)) == 0


def test_the_table_is_real_and_correct() -> None:
    # SymPy writes the arcsine as I*log(-I*x + sqrt(1 - x**2)); the table
    # gives the real forms, checked by differentiation
    F = quadratic_radical_antiderivative(sqrt(1 - x**2), x)
    assert F == x * sqrt(1 - x**2) / 2 + asin(x) / 2
    assert quadratic_radical_antiderivative(x**2 / sqrt(x**2 + 1), x) == x * sqrt(x**2 + 1) / 2 - asinh(x) / 2
    assert quadratic_radical_antiderivative(1 / sqrt(2 * x - x**2), x) == -asin(1 - x)
    for f in [x**2 * sqrt(1 - x**2), x**3 / sqrt(x**2 + 1), (2 * x - x**2)**Rational(3, 2), sqrt(x**2 - 1),
              x**2 / (1 - x**2)**Rational(3, 2), (x + 1)**2 * sqrt(1 - x**2), 1 / sqrt(x**2 + 2 * x + 5),
              x**4 * sqrt(3 - 2 * x - x**2) + x, 1 / (x**2 + 1)**Rational(5, 2), x * (x**2 + 1)**Rational(5, 2),
              sqrt(2 * x + 3), x**2 / sqrt(2 * x + 3)]:
        assert _checks(f), f
    # the leading coefficient of the recurrence vanishes for n + m + 1 = 0
    assert _checks(x**2 / (1 - x**2)**Rational(3, 2)) and _checks(x**4 / (1 - x**2)**Rational(5, 2))
    # a + b x with a > 0 and two real roots: the logarithm without Abs, a
    # primitive up to a constant on each component
    F = quadratic_radical_antiderivative(sqrt(x**2 - 1), x)
    assert F is not None and F.has(log) and simplify(diff(F, x) - sqrt(x**2 - 1)) == 0


def test_products_of_radicals_of_linear_factors() -> None:
    # the cells of a region hand the driver sqrt(1 - y)*sqrt(y + 1): one
    # radicand of degree two where both factors are real (the bug: the
    # split form fell through to SymPy's complex Piecewise, and the three
    # intersecting cylinders stayed unevaluated on it)
    from sympy_extras.integrals.conditions import numerically_equal
    from sympy_extras._typing import as_boolean
    y = symbols('y')
    F = quadratic_radical_antiderivative(2 * sqrt(1 - y) * sqrt(y + 1), y)
    assert F == y * sqrt(1 - y**2) + asin(y)
    for f in [sqrt(1 - y)**3 * sqrt(y + 1), y**2 * sqrt(1 - y) * sqrt(y + 1), sqrt(2 - y) * sqrt(y) * (y + 1)]:
        F = quadratic_radical_antiderivative(f, y)
        assert F is not None and numerically_equal(diff(F, y), f, [as_boolean(y > 0), as_boolean(y < 1)]), f
    # a reciprocal radical: not one radicand for the table, the two square
    # roots combined into sqrt(1 - y**2) by the Euler route
    F = quadratic_radical_antiderivative(sqrt(1 - y) / sqrt(1 + y), y)
    assert F is not None and numerically_equal(diff(F, y), sqrt(1 - y) / sqrt(1 + y),
                                               [as_boolean(y > 0), as_boolean(y < 1)])
    # a quadratic factor in the product: not one radicand
    assert quadratic_radical_antiderivative(sqrt(1 - y) * sqrt(1 + y**2), y) is None


def test_outside_the_table() -> None:
    assert quadratic_radical_antiderivative(sqrt(x**3 + 1), x) is None          # a cubic
    assert quadratic_radical_antiderivative(sqrt(1 - x**2) * sqrt(x**2 + 1), x) is None  # two radicands
    assert quadratic_radical_antiderivative((x**2 + 1)**Rational(1, 3), x) is None  # a cube root
    assert quadratic_radical_antiderivative(1 / (x**2 + 1), x) is None          # no radical
    assert quadratic_radical_antiderivative(sqrt((x - 1)**2), x) is None        # a perfect square
    assert quadratic_radical_antiderivative(sqrt(-1 - x**2), x) is None         # never positive


def test_symbolic_algebraic_bounds() -> None:
    # the bug: the slice of a cylinder, Integral(2*sqrt(1 - y**2), (y,
    # -sqrt(1 - x**2), x)) with -sqrt(2)/2 < x < 0, was left unevaluated;
    # three things stood in the way, each fixed: the complex-log form of
    # the antiderivative, whose limit at the bound failed; the leak guard
    # of one_sided_limit, which rejected the symbol of the bound; and the
    # sampler of the numerical check, which had no point for an
    # assumption with the irrational bound -sqrt(2)/2
    from sympy_extras.integrals import definite_integral
    y = symbols('y')
    found = definite_integral(2 * sqrt(1 - y**2), (y, -sqrt(1 - x**2), x), [x > -sqrt(2) / 2, x < 0])
    assert not found.has(Integral)
    assert abs(float(N(found.subs(x, -S.Half))) - float(N(Integral(2 * sqrt(1 - y**2), (y, -sqrt(S(3)) / 2, -S.Half))))) < 1e-12
    # the symmetric slice: [y sqrt(1 - y**2)/2 + asin(y)/2] between -+sqrt(1 - x**2), with sqrt(x**2) = x
    found = definite_integral(sqrt(1 - y**2), (y, -sqrt(1 - x**2), sqrt(1 - x**2)), [x > 0, x < 1])
    assert not found.has(Integral) and simplify(found - (x * sqrt(1 - x**2) + asin(sqrt(1 - x**2)))) == 0


def test_negative_powers_of_x() -> None:
    # Maxima's rtestint 20-30: Q**(m/2)/x**k by the recurrence solved for
    # the lowest power, from 1/(x*sqrt(Q)) (GR 2.266: a logarithm for a
    # positive constant term, an arcsine for a negative one, checked in
    # Mathematica on both sides of zero) and from Q taken out of the radical
    from sympy_extras.integrals.indefinite import is_antiderivative
    assert quadratic_radical_antiderivative(1 / (x * sqrt(x**2 + 1)), x) == -log((2 * sqrt(x**2 + 1) + 2) / x)
    assert quadratic_radical_antiderivative(1 / (x * sqrt(x**2 - 1)), x) == -asin(1 / sqrt(x**2))
    for f in [1 / (x * sqrt(1 - x**2)), 1 / (x**2 * sqrt(1 - x**2)), (1 - x**2)**Rational(3, 2) / x,
              1 / (x * (x**2 + 1)**Rational(3, 2)), 1 / (x**3 * sqrt(x**2 + 1)) + 1 / x, sqrt(x**2 + 2 * x + 5) / x**2,
              1 / (x**2 * (1 - x**2)**Rational(3, 2))]:
        F = quadratic_radical_antiderivative(f, x)
        assert F is not None and is_antiderivative(F, f, x) is True, f
    # the signs from the assumptions, as the census records them
    a, b, c = symbols('a b c')
    Q = a + b * x + c * x**2
    facts = [a > 0, b > 0, c > 0, 4 * a * c - b**2 > 0]
    for f in [sqrt(Q) / x, sqrt(Q) / x**2, 1 / (x**2 * sqrt(Q)), 1 / (x**3 * sqrt(Q))]:
        F = quadratic_radical_antiderivative(f, x, facts)
        assert F is not None and is_antiderivative(F, f, x, facts) is True, f
    assert quadratic_radical_antiderivative(sqrt(Q) / x, x) is None       # the sign of a undecided
    # x divides Q: not in the table, found by an Euler substitution
    F = quadratic_radical_antiderivative(sqrt(b * x + c * x**2) / x, x, [b > 0, c > 0])
    assert F is not None and is_antiderivative(F, sqrt(b * x + c * x**2) / x, x, [b > 0, c > 0]) is True


def test_euler_substitutions_for_rational_functions_of_a_radical() -> None:
    # not of the form x**n*Q**(m/2): the Euler substitutions make the
    # integrand a rational function (Maxima's rtest_integrate 854, FriCAS's
    # in1186a and in143a, a linear radical)
    from sympy_extras.integrals.radicals import euler_substitution_antiderivative
    for f in (1 / ((x + 3) * sqrt(x**2 - 1)), sqrt(x**2 + x) / (x**2 + 1)**2, sqrt(x + 1) / (x**2 + 1),
              1 / ((x + 2) * sqrt(1 - x**2)), sqrt(x**2 + 1) / (x - 1)):
        F = euler_substitution_antiderivative(as_expr(f), x)
        assert F is not None, f
        difference = as_expr(F.diff(x) - f)
        for point in (Rational(3, 2), Rational(7, 3)) if f.has(sqrt(x**2 - 1)) else (Rational(1, 3), Rational(3, 4)):
            assert abs(N(difference.subs(x, point), 20)) < 1e-15, (f, point)
        # reached through the public function
        assert quadratic_radical_antiderivative(as_expr(f), x) is not None
    assert euler_substitution_antiderivative(1 / (x * sqrt(1 - x**2) + sqrt(1 - x**2)), x) == -sqrt(1 - x**2) / (x + 1)
    # a cubic radicand is not a quadratic one
    assert euler_substitution_antiderivative(1 / ((x + 1) * sqrt(x**3 + 1)), x) is None


def test_odd_powers_of_a_quadratic_under_the_root() -> None:
    # sqrt(Q**3) is Q*sqrt(Q) where Q > 0: the radical table's entries
    # for the odd powers of sqrt(Q) (Maxima's rtestint with symbolic
    # coefficients)
    from sympy_extras.integrals import is_antiderivative, verified_antiderivative
    from sympy_extras.integrals.radicals import _odd_power
    a, b, c = symbols('a b c')
    Q = as_expr(a + b * x + c * x**2)
    assert _odd_power(as_expr(Q**3), x) == (Q, 3)
    assert _odd_power(as_expr(Q**2), x) == (Q**2, 1)
    assert _odd_power(as_expr(x**2 + 1), x) == (x**2 + 1, 1)
    facts = [a > 0, b > 0, c > 0, 4 * a * c - b**2 > 0]
    f = as_expr(sqrt(Q**3) / x)
    found = verified_antiderivative(f, x, facts)
    assert found is not None and found[1] == 'radicals'
    assert is_antiderivative(found[0], f, x, facts) is True
