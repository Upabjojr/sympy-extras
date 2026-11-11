"""Padé approximants (the order of contact checked on SymPy's series,
closed forms from the literature), rational interpolation (the values at
the points) and series reversion (composition, known inverse series)."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest
import sympy

from sympy_extras.core import I, Integer, Rational, Symbol, pi
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, exp, log, sin, sqrt, tan
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.series.approximants import (pade, pade_from_coefficients, rational_interpolate, revert_coefficients,
                                              series_reversion, taylor_coefficients)

x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
sx, sy = sympy.Symbol('x'), sympy.Symbol('y')


def _degrees(r: sympy.Expr) -> tuple[int, int]:
    numerator, denominator = sympy.fraction(sympy.cancel(r))
    if numerator == 0:
        return 0, 0
    return int(sympy.degree(numerator, sx)), int(sympy.degree(denominator, sx))


def _contact(f: sympy.Expr, r: sympy.Expr, point: sympy.Expr, order: int) -> bool:
    """The expansion of f - r at the point starts at the given order or
    later."""
    h = sympy.Symbol('h')
    difference = (f - r).subs(sx, point + h)
    leading = sympy.series(difference, h, 0, order).removeO()
    # sin(1)**2 + cos(1)**2 = 1 is needed for the expansions away from the origin: the numerators are reduced
    # modulo that relation (SymPy's series also produces sin(2) and cos(2))
    s1, c1 = sympy.Symbol('s1'), sympy.Symbol('c1')
    for k in range(order):
        coefficient = leading.coeff(h, k).subs({sympy.sin(1): s1, sympy.cos(1): c1, sympy.sin(2): 2*s1*c1,
                                                sympy.cos(2): c1**2 - s1**2})
        numerator = sympy.fraction(sympy.together(coefficient))[0]
        if sympy.rem(sympy.expand(numerator), s1**2 + c1**2 - 1, s1) != 0:
            return False
    return True


def test_pade_known() -> None:
    # Baker, Graves-Morris, Pade Approximants, Section 1.2: the table of exp
    assert sympy.cancel(to_sympy(pade(exp(x), x, 1, 1)) - (2 + sx)/(2 - sx)) == 0
    assert sympy.cancel(to_sympy(pade(exp(x), x, 3, 3)) - (120 + 60*sx + 12*sx**2 + sx**3)/(120 - 60*sx + 12*sx**2 - sx**3)) == 0
    assert sympy.cancel(to_sympy(pade(exp(x), x, 0, 2)) - 2/(2 - 2*sx + sx**2)) == 0
    # log(1 + x) ~ 2x/(2 + x), tan(x) ~ 3x/(3 - x**2), 1/(1 - x) is its own approximant
    assert sympy.cancel(to_sympy(pade(log(1 + x), x, 1, 1)) - 2*sx/(2 + sx)) == 0
    assert sympy.cancel(to_sympy(pade(tan(x), x, 1, 2)) - 3*sx/(3 - sx**2)) == 0
    assert pade(1/(1 - x), x, 2, 3) == -1/(x - 1)
    # [m/0] is the Taylor polynomial
    assert sympy.expand(to_sympy(pade(cos(x), x, 4, 0)) - (1 - sx**2/2 + sx**4/24)) == 0


def test_pade_order_of_contact() -> None:
    functions: list[tuple[Expr, Expr]] = [(exp(x), Integer(0)), (sin(x), Integer(0)), (log(x), Integer(1)),
                                          (sqrt(1 + x), Integer(0)), (exp(x)/(1 + x), Integer(0)), (cos(x), Integer(1)),
                                          (1/(2 + x + x**2), Integer(0)), (tan(x), Integer(0))]
    for f, point in functions:
        for m, n in [(0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2)]:
            r = sympy_expr(to_sympy(pade(f, x, m, n, point)))
            shifted = sympy.cancel(r.subs(sx, sx + sympy_expr(to_sympy(point))))
            p, q = _degrees(shifted)
            assert p <= m and q <= n
            try:
                strict = pade(f, x, m, n, point, strict=True)
            except ValueError:
                continue
            assert strict == pade(f, x, m, n, point)
            assert _contact(sympy_expr(to_sympy(f)), r, sympy_expr(to_sympy(point)), m + n + 1)


def test_pade_with_parameters() -> None:
    r = sympy_expr(to_sympy(pade(exp(a*x), x, 2, 1)))
    assert _contact(sympy.exp(sympy.Symbol('a')*sx), r, sympy.Integer(0), 4)
    # coefficients with kernels which are not symbols: exp(1) at the point 1, pi
    r = sympy_expr(to_sympy(pade(exp(x), x, 1, 1, 1)))
    assert sympy.simplify(r - sympy.E*(1 + sx)/(3 - sx)) == 0
    assert sympy.simplify(to_sympy(pade(pi + x + x**2, x, 1, 1)) - (sympy.pi + (1 - sympy.pi)*sx)/(1 - sx)) == 0
    assert sympy.simplify(to_sympy(pade(1/(1 - I*x), x, 0, 1)) - 1/(1 - sympy.I*sx)) == 0


def test_pade_degenerate() -> None:
    # 1 + x**2 has no [1/1] approximant: the Pade form x/x reduces to 1, which only agrees to the order 2
    assert pade(1 + x**2, x, 1, 1) == 1
    with pytest.raises(ValueError):
        pade(1 + x**2, x, 1, 1, strict=True)
    assert pade(Integer(0), x, 2, 2) == 0 and pade(x**3, x, 1, 1) == 0
    with pytest.raises(ValueError):
        pade(exp(x), x, -1, 2)
    with pytest.raises(ValueError):
        pade(log(x), x, 1, 1)
    with pytest.raises(ValueError):
        pade(1/x, x, 1, 1)
    with pytest.raises(ValueError):
        pade_from_coefficients([1, 2], x, 1, 1)


def test_pade_from_coefficients_recovers_rational_functions() -> None:
    rnd = random.Random(51)
    for _ in range(25):
        m, n = rnd.randint(0, 3), rnd.randint(0, 3)
        numerator = sum(rnd.randint(-3, 3)*sx**k for k in range(m + 1))
        denominator = 1 + sum(rnd.randint(-3, 3)*sx**k for k in range(1, n + 1))
        target = sympy.cancel(numerator/denominator)
        series = sympy.series(target, sx, 0, m + n + 3).removeO()
        coefficients = [Rational(int(c.p), int(c.q)) for c in (series.coeff(sx, k) for k in range(m + n + 3))]
        # the series of a rational function of type [m/n] is reproduced by every [m'/n'] with m' >= m, n' >= n
        assert sympy.cancel(to_sympy(pade_from_coefficients(coefficients, x, m, n)) - target) == 0
        assert sympy.cancel(to_sympy(pade_from_coefficients(coefficients, x, m + 1, n + 1)) - target) == 0


def test_taylor_coefficients() -> None:
    assert taylor_coefficients(exp(x), x, 4) == [1, 1, Rational(1, 2), Rational(1, 6)]
    assert taylor_coefficients(x**2, x, 5) == [0, 0, 1, 0, 0] and taylor_coefficients(Integer(7), x, 2) == [7, 0]
    assert taylor_coefficients(1/x, x, 3, 2) == [Rational(1, 2), Rational(-1, 4), Rational(1, 8)]
    assert taylor_coefficients(exp(x), x, 0) == []
    with pytest.raises(ValueError):
        taylor_coefficients(sqrt(x), x, 3)


def test_rational_interpolation_random() -> None:
    rnd = random.Random(52)
    found = missing = 0
    for _ in range(60):
        count = rnd.randint(1, 6)
        nodes = rnd.sample(range(-5, 6), count)
        values = [Fraction(rnd.randint(-4, 4), rnd.randint(1, 3)) for _ in nodes]
        k = rnd.randint(0, count - 1)
        points: list[tuple[Expr, Expr]] = [(Integer(p), Rational(v.numerator, v.denominator)) for p, v in zip(nodes, values)]
        r = rational_interpolate(points, x, k)
        if r is None:
            # no interpolant: the linear problem t*y_i = r(x_i) has only solutions vanishing at some node; with the
            # generic degrees there are count + 1 unknowns for count equations, so one solution up to a factor
            missing += 1
            unknowns = sympy.symbols('c0:%d' % (count + 1))
            numerator = sum(c*sx**i for i, c in enumerate(unknowns[:k + 1]))
            denominator = sum(c*sx**i for i, c in enumerate(unknowns[k + 1:]))
            equations = [numerator.subs(sx, p) - sympy.Rational(v.numerator, v.denominator)*denominator.subs(sx, p)
                         for p, v in zip(nodes, values)]
            solution = sympy.solve(equations, unknowns, dict=True)[0]
            general = sympy.cancel(denominator.subs(solution))
            assert any(general.subs(sx, p) == 0 for p in nodes)
            continue
        found += 1
        s = sympy_expr(to_sympy(r))
        p_degree, q_degree = _degrees(s)
        assert (s == 0 or p_degree <= k) and q_degree <= count - 1 - k
        for p, v in zip(nodes, values):
            assert s.subs(sx, p) == sympy.Rational(v.numerator, v.denominator)
    assert found > 30 and missing > 2


def test_rational_interpolation_known() -> None:
    # the interpolating polynomial
    points: list[tuple[Expr, Expr]] = [(Integer(k), Integer(k**3 - 2*k)) for k in range(4)]
    assert rational_interpolate(points, x, 3) == x**3 - 2*x
    # a Moebius function sending two points to the same value is constant: (2, 2) is unattainable
    assert rational_interpolate([(0, 1), (1, 1), (2, 2)], x, 1) is None
    assert rational_interpolate([(0, 1), (1, 1), (2, 1)], x, 1) == 1
    # symbolic nodes and values
    r = rational_interpolate([(a, 1/a), (2*a, 1/(2*a)), (3*a, 1/(3*a))], x, 0)
    assert r == 1/x
    with pytest.raises(ValueError):
        rational_interpolate([(0, 1), (0, 2)], x, 1)
    with pytest.raises(ValueError):
        rational_interpolate([(0, 1), (1, 2)], x, 2)


def test_series_reversion() -> None:
    for f in (sin(x), x*exp(x), x + x**2, tan(x), 2*x/(1 - x), exp(x) - 1, log(1 + x) + x**2):
        n = 7
        g = sympy_expr(to_sympy(series_reversion(f, x, y, n)))
        composed = sympy.series(g.subs(sy, sympy_expr(to_sympy(f))), sx, 0, n).removeO()
        assert sympy.expand(composed - sx) == 0
    # the inverse of x*exp(x) is Lambert's W: the coefficients (-k)**(k-1)/k!
    coefficients = revert_coefficients(taylor_coefficients(x*exp(x), x, 8))
    assert coefficients == [0] + [Rational((-k)**(k - 1), sympy.factorial(k)) for k in range(1, 8)]
    # away from the origin: the inverse of exp at 1 is log, of x**2 at x = 3 is sqrt at 9
    g = sympy_expr(to_sympy(series_reversion(x**2, x, y, 5, 3)))
    assert sympy.expand(g - sympy.series(sympy.sqrt(sy), sy, 9, 5).removeO()) == 0
    # with a parameter
    g = sympy_expr(to_sympy(series_reversion(a*x + x**2, x, y, 4)))
    sa = sympy.Symbol('a')
    assert sympy.simplify(sympy.series(g.subs(sy, sa*sx + sx**2), sx, 0, 4).removeO() - sx) == 0
    with pytest.raises(ValueError):
        series_reversion(x**2, x, y, 4)
    assert series_reversion(sin(x), x, y, 1) == 0
