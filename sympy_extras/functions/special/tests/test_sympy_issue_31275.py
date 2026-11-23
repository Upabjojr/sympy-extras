"""Regression tests for the findings of sympy/sympy#31275 in the special
functions which the ports shared with SymPy (V88-V91, V95, V97, V98,
V104-V106), each checked against mpmath, and for the neighbouring errors
of the same code found while fixing them."""
from __future__ import annotations

from typing import Callable

import mpmath

from sympy_extras.core import I, Integer, Rational, Symbol, oo
from sympy_extras.core.expr import Expr
from sympy_extras.core.order import Order
from sympy_extras.functions import (BesselJ, BesselY, Ei, Li, LiNode, Piecewise, SingularityFunction, SphericalJ,
                                    SphericalY, besseli, besselj, besselk, bessely, conjugate, hn1, hyper, jn, li, log,
                                    sin, yn)
from sympy_extras.functions.special.hyper import Hyper

x, z = Symbol('x'), Symbol('z')
nu = Symbol('nu')
k = Symbol('k', integer=True)


def _without_order(e: Expr) -> Expr:
    return e.xreplace({o: Integer(0) for o in e.atoms(Order)})


def _number(e: Expr, digits: int = 50) -> mpmath.mpc:
    from sympy_extras.functions import im, re
    v = e.evalf(digits)
    return mpmath.mpc(str(re(v)), str(im(v)))


BESSEL: list[tuple[Callable[[Expr, Expr], Expr], Callable[[mpmath.mpf, mpmath.mpf], mpmath.mpc]]] = [
    (besselj, mpmath.besselj), (bessely, mpmath.bessely), (besseli, mpmath.besseli), (besselk, mpmath.besselk)]


def _series_is_right(f: Callable[[Expr, Expr], Expr], reference: Callable[[mpmath.mpf, mpmath.mpf], mpmath.mpc],
                     order: Rational, inner: Expr, n: int) -> bool:
    """The remainder of the series to ``O(x**n)`` is of that order: it does
    not grow, divided by ``x**n``, from 1/100 to 1/10000 (a missing term in
    ``x**m``, ``m < n``, grows by ``100**(n - m)``)."""
    truncated = _without_order(f(order, inner).series(x, n=n))
    ratios = []
    for denominator in (100, 10000):
        point = Rational(1, denominator)
        t = mpmath.mpf(1)/denominator
        exact = reference(mpmath.mpf(order.p)/order.q, _number(inner.subs(x, point)).real)
        error = abs(_number(truncated.subs(x, point)) - exact)
        ratios.append(mpmath.mpf(0) if error < mpmath.mpf(10)**-40*abs(exact) else error/t**n)
    return bool(ratios[1] < 10*ratios[0] + mpmath.mpf(10)**-20)


def test_bessel_series_of_composite_arguments() -> None:
    # V88: bessely(1, sin(x)) to O(x**4) missed x**3/(60*pi): the series of sin(x)/2, truncated at
    # O(x**4), was put into 1/r
    mpmath.mp.dps = 60
    try:
        series = _without_order(bessely(Integer(1), sin(x)).series(x, n=4))
        for denominator in (1000, 100000):
            t = mpmath.mpf(1)/denominator
            error = _number(series.subs(x, Rational(1, denominator))) - mpmath.bessely(1, mpmath.sin(t))
            assert abs(error)/t**3 < mpmath.mpf(1)/1000, denominator  # 1/(60*pi) = 0.0053 before
        orders: list[Rational] = [Integer(0), Integer(1), Integer(2), Rational(1, 2), Rational(-1, 2), Rational(-4, 3)]
        for order in orders:
            for f, reference in BESSEL:
                if f is bessely and order.q != 1:
                    continue  # no expansion for a noninteger order, as in SymPy
                for inner in (sin(x), x + x**2, sin(x)**2):
                    assert _series_is_right(f, reference, order, inner, 4), (f, order, inner)
    finally:
        mpmath.mp.dps = 15


def test_bessel_series_of_monomials() -> None:
    # the neighbours of V88 in the same code: terms lost for a negative order, for a small n and for a
    # monomial of degree above 1
    mpmath.mp.dps = 60
    try:
        cases: list[tuple[Callable[[Expr, Expr], Expr], Callable[[mpmath.mpf, mpmath.mpf], mpmath.mpc], Rational, Expr,
                          int]] = [
            (besselj, mpmath.besselj, Rational(-1, 2), x, 4), (besselj, mpmath.besselj, Rational(-4, 3), x, 6),
            (besseli, mpmath.besseli, Rational(-4, 3), x, 4), (bessely, mpmath.bessely, Integer(2), x, 2),
            (bessely, mpmath.bessely, Integer(2), x, 1), (bessely, mpmath.bessely, Integer(3), x, 2),
            (besselk, mpmath.besselk, Rational(5, 2), x**2, 4), (besselk, mpmath.besselk, Rational(-4, 3), x**2, 2),
            (besselj, mpmath.besselj, Integer(0), x**5, 4), (besselk, mpmath.besselk, Integer(2), 2*x, 4)]
        for f, reference, order, inner, n in cases:
            assert _series_is_right(f, reference, order, inner, n), (f, order, inner, n)
    finally:
        mpmath.mp.dps = 15
    assert besselj(Integer(0), x**5).series(x, n=4).removeO() == 1
    assert bessely(Integer(2), x).series(x, n=2).removeO().coeff(x, 0) != 0


def test_asymptotic_series_of_the_modified_bessel_functions() -> None:
    # V89, V90: DLMF 10.40.1 and 10.40.2 for nu = 0 are 1 +- 1/(8*x) + 9/(128*x**2) +- 75/(1024*x**3)
    assert str(besseli(Integer(0), x).series(x, oo, n=4)) == (
        'sqrt(2)*(sqrt(1/x) + (1/x)**(3/2)/8 + 9*(1/x)**(5/2)/128 + 75*(1/x)**(7/2)/1024 + O((1/x)**(9/2), (x, oo)))'
        '*exp(x)/(2*sqrt(pi))')
    assert str(besselk(Integer(0), x).series(x, oo, n=4)) == (
        'sqrt(2)*sqrt(pi)*(sqrt(1/x) - (1/x)**(3/2)/8 + 9*(1/x)**(5/2)/128 - 75*(1/x)**(7/2)/1024 '
        '+ O((1/x)**(9/2), (x, oo)))*exp(-x)/2')
    mpmath.mp.dps = 30
    try:
        orders: list[Rational] = [Integer(0), Integer(1), Rational(1, 3), Integer(2)]
        for order in orders:
            for f, reference in ((besseli, mpmath.besseli), (besselk, mpmath.besselk)):
                for point in (oo, -oo):
                    series = _without_order(f(order, x).series(x, point, n=4))
                    for size in (50, 200):
                        at = -size if point == -oo else size
                        exact = reference(mpmath.mpf(order.p)/order.q, at)
                        assert abs(_number(series.subs(x, Integer(at)), 30)/exact - 1) < 2*mpmath.mpf(size)**-4, \
                            (f, order, point, size)
    finally:
        mpmath.mp.dps = 15


def test_spherical_bessel_rewrites_need_an_integer_order() -> None:
    # V91: Y(-nu - 1/2) = (-1)**nu*J(nu + 1/2) for an integer nu only
    order = Rational(3, 10) + I/5
    assert jn(order, z).rewrite(BesselY) == jn(order, z) and jn(nu, z).rewrite(BesselY) == jn(nu, z)
    assert jn(nu, z).rewrite(SphericalY) == jn(nu, z) and yn(nu, z).rewrite(SphericalJ) == yn(nu, z)
    assert hn1(nu, z).rewrite(SphericalJ) == hn1(nu, z) and hn1(nu, z).rewrite(SphericalY) == hn1(nu, z)
    point = mpmath.mpf(17)/10
    spherical = mpmath.sqrt(mpmath.pi/(2*point))*mpmath.besselj(mpmath.mpc(0.8, 0.2), point)
    refused = (-1)**mpmath.mpc(0.3, 0.2)*mpmath.sqrt(mpmath.pi/(2*point))*mpmath.bessely(mpmath.mpc(-0.8, -0.2), point)
    assert abs(spherical - refused) > 0.1
    # ... which still holds, and is applied, for the integers; and by definition for besselj
    for target in (BesselY, SphericalY):
        rewritten = jn(k, z).rewrite(target)
        assert rewritten != jn(k, z)
        for integer in (0, 1, 2, -3):
            ours = _number(rewritten.subs(k, Integer(integer)).subs(z, Rational(17, 10)), 20)
            exact = mpmath.sqrt(mpmath.pi/(2*point))*mpmath.besselj(integer + mpmath.mpf(1)/2, point)
            assert abs(ours - exact) < 1e-12, (target, integer)
    assert jn(nu, z).rewrite(BesselJ) != jn(nu, z)


def test_li_rewrite_conjugate_and_series() -> None:
    p = Symbol('p', positive=True)
    r = Symbol('r', real=True)
    # V95: Ei(2*log(x)) is li(x**2) only where 2*log(x) = log(x**2)
    assert Ei(2*log(x)).rewrite(LiNode) == Ei(2*log(x)) and Ei(2*log(p)).rewrite(LiNode) == li(p**2)
    assert Ei(log(x)).rewrite(LiNode) == li(x)
    point = mpmath.mpc(-0.3, 0.9)
    assert abs(mpmath.ei(2*mpmath.log(point)) - mpmath.li(point**2)) > 1
    # V97: -r may be on the branch cut (-oo, 0), where li is not real
    assert conjugate(li(-r)) != li(-r) and conjugate(li(p)) == li(p)
    assert abs(mpmath.li(-1).imag) > 3
    # V98: the series in log(z) is an expansion where log(z) tends to 0 only
    assert li(z).series(z) == li(z) and Li(z).series(z) == li(z) - li(Integer(2))
    mpmath.mp.dps = 30
    try:
        for e, reference in ((li(1 + x), lambda t: mpmath.li(1 + t)), (li(2 + x), lambda t: mpmath.li(2 + t))):
            series = _without_order(e.series(x, n=4))
            errors = [abs(_number(series.subs(x, Rational(1, d)), 30) - reference(mpmath.mpf(1)/d))*d**4
                      for d in (100, 10000)]
            assert errors[1] < 10*errors[0], (e, errors)
    finally:
        mpmath.mp.dps = 15


def _series(ap: tuple[Expr, ...], bq: tuple[Expr, ...], argument: Expr) -> Hyper:
    node = hyper(ap, bq, argument)
    assert isinstance(node, Hyper)
    return node


def test_radius_of_convergence_of_a_series_which_ends() -> None:
    # V104: the numerator -1 ends the series before the denominator -4 or -2 divides by zero
    one, two, three = Integer(1), Integer(2), Integer(3)
    for ap, bq in (((-one, one, two), (Integer(-4),)), ((-one, one, three), (-two, two)), ((-one, one, three), (-two,))):
        assert _series(ap, bq, z).radius_of_convergence == oo, (ap, bq)
        assert _series(ap, bq, z).convergence_statement is True
    assert abs(mpmath.hyper([-1, 1, 2], [-4], 100) - 51) < 1e-9
    assert _series((-one,), (-two, -three), z).radius_of_convergence == oo
    # the division by zero first: not defined
    assert _series((one, two), (-one,), z).radius_of_convergence == 0
    assert Hyper([-three, one], [-two], z).radius_of_convergence == 0
    assert _series((one, two, three), (Integer(-4),), z).radius_of_convergence == 0
    # the general rule
    a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    assert _series((a, b), (c,), z).radius_of_convergence == 1 and _series((one,), (two, three), z).radius_of_convergence == oo
    assert _series((one, two, three), (Integer(4),), z).radius_of_convergence == 0
    # V105: symbolic integers: the radius depends on them, and is never the 1 of the general rule
    n = Symbol('n', integer=True, positive=True)
    m = Symbol('m', integer=True)
    radius = _series((one, 1 - n), (-m,), z).radius_of_convergence
    assert isinstance(radius, Piecewise)
    for values, expected in (({n: three, m: Integer(5)}, oo), ({n: three, m: two}, 1), ({n: Integer(6), m: two}, 0),
                             ({n: three, m: Integer(-2)}, oo), ({n: one, m: Integer(0)}, 1),
                             ({n: three, m: Integer(0)}, 0)):
        assert radius.subs(values) == expected, values
        assert _series((one, 1 - n.subs(values)), (-m.subs(values),), z).radius_of_convergence == expected, values
    assert _series((-n, one), (two,), z).radius_of_convergence == oo
    # equal parameters cancel, as in hyper: hyper((1, -2), (-2,), z) is hyper((1,), (), z)
    assert Hyper([-two, one], [-two], z).radius_of_convergence == 1


def test_singularity_function_series_at_its_point() -> None:
    # V106: on the right of the point <x>**1 is x, not 0
    length = Symbol('l', positive=True)
    beam = (SingularityFunction(x + length, 0, 1)/2 - SingularityFunction(x + length, length/2, 1)
            + SingularityFunction(x + length, length, 1)/2)
    assert beam.as_leading_term(x) == 0 and beam.as_leading_term(x, cdir=-1) == -x/2
    assert beam.nseries(x).removeO() == 0 and beam.series(x, dir='-').removeO() == -x/2
    for value, expected in ((Rational(1, 100), 0), (Rational(-1, 100), Rational(1, 200))):
        assert beam.subs(length, Integer(1)).subs(x, value) == expected
    assert SingularityFunction(x, 0, 2).as_leading_term(x) == x**2
    assert SingularityFunction(x, 0, 2).as_leading_term(x, cdir=-1) == 0
    assert SingularityFunction(x, 0, 0).as_leading_term(x) == 1 and SingularityFunction(-x, 0, 0).as_leading_term(x) == 0
    assert SingularityFunction(x, 0, 2).nseries(x) == x**2 and SingularityFunction(-x, 0, 1).nseries(x) == 0
