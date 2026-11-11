"""The native sample points, on ``Algebraic`` numbers and the typed
polynomials: hand computations, brute force and SymPy's numerical values
as the oracles."""
from __future__ import annotations

from fractions import Fraction
from math import floor, isqrt

import sympy

from sympy_extras.core import Symbol
from sympy_extras.polys.cad.samplepoints import (SamplePoint, Specialization, _join, _order, _real_roots_over,
    _sign_in_field, compare_real, floor_scaled, rational_above, rational_below, rational_between,
    rational_real_roots, real_roots_of, simplest_between)
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.polytools import Poly

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
sqrt2 = real_roots_of([1, 0, -2])[1]


def _value(a: Algebraic) -> float:
    return float(a)


def test_real_roots_and_comparisons() -> None:
    assert [round(float(r), 6) for r in real_roots_of([1, 0, -3, 0, 2])] == [-1.414214, -1.0, 1.0, 1.414214]
    m2, p2 = real_roots_of([1, 0, -2])
    cbrt2 = real_roots_of([1, 0, 0, -2])[0]
    assert compare_real(p2, Fraction(3, 2)) == -1 and compare_real(Fraction(3, 2), p2) == 1
    assert compare_real(p2, cbrt2) == 1 and compare_real(m2, p2) == -1 and compare_real(p2, p2) == 0
    # one number from two isolations (sqrt(2) as a root of 2*x**2 - 4, whose
    # minimal polynomial is the same): equal, although the intervals differ
    other = real_roots_of([2, 0, -4])[1]
    assert other is not p2 and compare_real(other, p2) == 0
    # close roots of different polynomials
    close = real_roots_of([1000000, 0, -2000001])[1]
    assert compare_real(p2, close) == -1 and compare_real(close, p2) == 1
    assert _order([Algebraic.from_rational(1), p2, Algebraic.from_rational(0), other, m2]) == [[4], [2], [0], [1, 3]]
    assert _order([]) == []
    found = rational_real_roots(Poly((2*x - 1)**2*(x**2 - 2)*(x**2 + 1), x))
    assert [(round(float(r), 6), m) for r, m in found] == [(-1.414214, 1), (0.5, 2), (1.414214, 1)]


def test_floor_and_rationals_between() -> None:
    for k in range(8):
        assert floor_scaled(sqrt2, k) == isqrt(2*4**k)
        assert floor_scaled(-sqrt2, k) == -isqrt(2*4**k) - 1
    assert floor_scaled(Fraction(7, 2), 1) == 7 and floor_scaled(Fraction(-7, 2), 0) == -4
    m2, p2 = real_roots_of([1, 0, -2])
    assert rational_between(m2, p2) == 0 and rational_between(1, p2) == Fraction(5, 4)
    assert rational_between(-1, 1) == 0 and rational_between(Fraction(1, 3), Fraction(1, 2)) == Fraction(2, 5)
    assert rational_below(p2) == 0 and rational_above(p2) == 2 and rational_below(m2) == -2
    assert rational_below(0) == -1 and rational_above(-3) == 0
    # the simplest rational between two bounds, by brute force
    for lo, hi in [(Fraction(1, 3), Fraction(1, 2)), (Fraction(5, 2), Fraction(3)), (Fraction(-5, 7), Fraction(-2, 3))]:
        q = simplest_between(lo, hi)
        assert lo < q < hi
        assert not any(lo < Fraction(n, d) < hi for d in range(1, q.denominator)
                       for n in range(floor(lo*d) - 1, floor(hi*d) + 2))
    # between the sorted roots of several polynomials
    roots = sorted([r for f in ([1, 0, -2], [1, 0, 0, -2], [1, 0, -3], [4, 0, -5], [1, -1]) for r in real_roots_of(f)],
                   key=_value)
    for a, b in zip(roots, roots[1:]):
        q = rational_between(a, b)
        assert compare_real(a, q) == -1 and compare_real(q, b) == -1 and q.denominator <= 16


def test_sign_in_field_and_join() -> None:
    p2 = real_roots_of([1, 0, -2])[1]
    assert _sign_in_field(p2, []) == 0
    # sqrt(2) - 1.4142 and sqrt(2) - 1.41421356238
    assert _sign_in_field(p2, [Fraction(1), Fraction(-14142, 10000)]) == 1
    assert _sign_in_field(p2, [Fraction(1), Fraction(-141421356238, 10**11)]) == -1
    p3 = real_roots_of([1, 0, -3])[1]
    gamma, K, theta_K, beta_K = _join(p2, p3)
    assert gamma.degree == 4 and abs(float(gamma) - (2**0.5 + 3**0.5)) < 1e-12
    assert _sign_in_field(gamma, K.sub(K.mul(theta_K, theta_K), K.from_int(2))) == 0
    assert _sign_in_field(gamma, K.sub(K.mul(beta_K, beta_K), K.from_int(3))) == 0
    assert _sign_in_field(gamma, theta_K) == 1 and _sign_in_field(gamma, K.sub(theta_K, beta_K)) == -1
    # conjugates and multiples give no bigger field
    gamma, K, theta_K, beta_K = _join(p2, real_roots_of([1, 0, -2])[0])
    assert gamma.degree == 2 and _sign_in_field(gamma, K.add(theta_K, beta_K)) == 0
    gamma, K, theta_K, beta_K = _join(p2, real_roots_of([1, 0, -8])[1])
    assert gamma.degree == 2 and _sign_in_field(gamma, K.sub(K.mul(K.from_int(2), theta_K), beta_K)) == 0


def test_sample_points() -> None:
    origin = SamplePoint()
    assert len(origin) == 0 and origin.degree == 1 and origin.sign(3, []) == 1 and origin.sign(0, []) == 0
    assert [float(r) for r in origin.real_roots(x**3 - x, [x]) or []] == [-1.0, 0.0, 1.0]
    assert origin.real_roots(Poly(0, x), [x]) is None and origin.real_roots(x**2 + 1, [x]) == []
    q = origin.extend(sqrt2)
    assert q.degree == 2 and q.theta is sqrt2 and q.values == (sqrt2,)
    assert q.sign(x**2 - 2, [x]) == 0 and q.sign(x - 1, [x]) == 1 and q.sign(x - 2, [x]) == -1
    assert q.real_roots(x**2 + y**2 - 1, [x, y]) == [] and q.real_roots(x**2 - 2, [x, y]) is None
    r = q.real_roots(y**2 - x, [x, y])
    assert r is not None and [r[0].minpoly, r[1].minpoly] == [[1, 0, 0, 0, -2]]*2
    # the same number again, a rational coordinate: no bigger field
    assert q.extend(sqrt2).degree == 2 and q.extend(Fraction(1, 2)).degree == 2
    assert q.extend(sqrt2).sign(x - y, [x, y]) == 0
    q2 = q.extend(r[1])
    assert q2.degree == 4 and q2.sign(y**2 - x, [x, y]) == 0 and q2.sign(y - x, [x, y]) == -1
    r3 = q2.real_roots(z**2 - x - y, [x, y, z])
    assert r3 is not None and len(r3) == 2
    q3 = q2.extend(r3[1])
    assert q3.degree == 8 and q3.sign(z**2 - x - y, [x, y, z]) == 0 and q3.sign(z - 2, [x, y, z]) == -1
    # several quadratic extensions
    pp = origin.extend(sqrt2).extend(real_roots_of([1, 0, -3])[1]).extend(real_roots_of([1, 0, -5])[1])
    assert pp.degree == 8 and pp.sign(x**2*y**2*z**2 - 30, [x, y, z]) == 0
    assert pp.sign(x*y*z - 6, [x, y, z]) == -1 and pp.sign(x*y*z - 5, [x, y, z]) == 1
    try:
        q3.sign(x, [x])
    except ValueError:
        pass
    else:
        raise AssertionError("one generator per coordinate")


def test_specialization() -> None:
    p = SamplePoint().extend(sqrt2)
    s = p.specialization((y**2 - x**2)*(y - 1)**2*(y**2 + 1), [x, y])
    assert isinstance(s, Specialization) and s.degree == 6 and s.leading_sign == 1
    assert [(round(float(r), 6), m) for r, m in s.roots] == [(-1.414214, 1), (1.0, 2), (1.414214, 1)]
    assert [s.sign_before(i) for i in range(4)] == [1, -1, -1, 1]
    s = p.specialization((x**2 - 2)*y, [x, y])
    assert s.degree == -1 and s.leading_sign == 0 and s.roots == []
    assert p.specialization(1 - x**3, [x, y]).leading_sign == -1
    assert p.specialization(y - x, [x, y]) is p.specialization(y - x, [x, y])
    s = SamplePoint().extend(Fraction(1, 2)).specialization(-(2*y - 1)**3*(y + x), [x, y])
    assert s.degree == 4 and s.leading_sign == -1
    assert [(r.rational_value(), m) for r, m in s.roots] == [(Fraction(-1, 2), 1), (Fraction(1, 2), 3)]


def test_real_roots_over_against_sympy() -> None:
    # the roots over QQ<theta>, theta the largest root of x**3 - 3*x + 1,
    # against SymPy's numerical roots of the polynomial with theta put in
    theta = real_roots_of([1, 0, -3, 1])[2]
    point = SamplePoint().extend(theta)
    K = point.field
    assert K is not None
    sx, sy = sympy.symbols('x y')
    t = sympy.CRootOf(sx**3 - 3*sx + 1, 2)
    for f, e in [((y - x)*(y + x)*(y**2 - x - 3), (sy - sx)*(sy + sx)*(sy**2 - sx - 3)),
                 (y**4 - x*y**2 - 1, sy**4 - sx*sy**2 - 1), ((3*y - x**2)**3, (3*sy - sx**2)**3)]:
        found = point.specialization(f, [x, y]).roots
        part = sympy.Poly(e, sy, sx).sqf_part().as_expr()
        numeric = sorted(complex(r).real for r in sympy.Poly(part.subs(sx, t.evalf(40)), sy).nroots(n=30)
                         if abs(complex(r).imag) < 1e-20)
        distinct = [v for i, v in enumerate(numeric) if i == 0 or abs(v - numeric[i - 1]) > 1e-12]
        assert [round(float(r), 9) for r, _ in found] == [round(v, 9) for v in distinct], f
    assert _real_roots_over(theta, K, [K.one, K.zero, K.one]) == []
