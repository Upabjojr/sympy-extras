from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy_extras.polys.cad.samplepoints_sympy import (SamplePoint, compare_real,
    simplest_between, rational_between, rational_below, rational_above, _floor_scaled)
from sympy.core.cache import clear_cache
from sympy.testing.pytest import raises
from sympy.abc import x, y, z, w


def _roots(f: Expr) -> list[Expr]:
    return Poly(f, x).real_roots(radicals=False)


def test_simplest_between() -> None:
    assert simplest_between(QQ(1, 3), QQ(1, 2)) == QQ(2, 5)
    assert simplest_between(QQ(5, 2), QQ(3)) == QQ(8, 3)
    assert simplest_between(QQ(-1), QQ(1)) == 0
    assert simplest_between(QQ(2), QQ(5)) == 3
    assert simplest_between(QQ(0), QQ(1)) == QQ(1, 2)
    assert simplest_between(QQ(1), QQ(2)) == QQ(3, 2)
    assert simplest_between(QQ(-3, 2), QQ(-1)) == QQ(-4, 3)
    assert simplest_between(QQ(7, 3), None) == 3
    assert simplest_between(QQ(-7, 3), None) == 0
    assert simplest_between(QQ(3), None) == 4
    assert simplest_between(QQ(99, 100), QQ(101, 100)) == 1
    assert simplest_between(QQ(1, 3), QQ(1, 2), hi_open=False) == QQ(1, 2)
    assert simplest_between(QQ(1, 3), QQ(1, 2), lo_open=False) == QQ(1, 3)
    assert simplest_between(QQ(0), QQ(1), lo_open=False) == 0
    assert simplest_between(QQ(-2), QQ(-1), hi_open=False) == -1
    assert simplest_between(QQ(-2), QQ(-1), lo_open=False) == -2
    assert simplest_between(QQ(3), None, lo_open=False) == 3
    assert simplest_between(QQ(1, 2), QQ(1, 2), False, False) == QQ(1, 2)
    assert simplest_between(QQ(7, 5), QQ(3, 2), hi_open=False) == QQ(3, 2)
    raises(ValueError, lambda: simplest_between(QQ(1), QQ(1), lo_open=False))
    assert simplest_between(QQ(100, 101), QQ(101, 102)) == QQ(2011, 2030) or \
        simplest_between(QQ(100, 101), QQ(101, 102)).denominator <= 2030
    raises(ValueError, lambda: simplest_between(QQ(1), QQ(1)))
    raises(ValueError, lambda: simplest_between(QQ(2), QQ(1)))

    # the result has the smallest denominator among all rationals in the
    # interval, checked by brute force
    for lo, hi in [(QQ(1, 3), QQ(1, 2)), (QQ(5, 2), QQ(3)), (QQ(-3, 2), QQ(-1)),
                   (QQ(7, 10), QQ(8, 10)), (QQ(-5, 7), QQ(-2, 3))]:
        q = simplest_between(lo, hi)
        assert lo < q < hi
        for d in range(1, q.denominator):
            for n in range(int(lo*d) - 1, int(hi*d) + 2):
                assert not (lo < QQ(n, d) < hi)


def test_compare_real() -> None:
    s2, ms2 = CRootOf(x**2 - 2, 1), CRootOf(x**2 - 2, 0)
    c2 = CRootOf(x**3 - 2, 0)
    assert compare_real(s2, Rational(3, 2)) == -1
    assert compare_real(Rational(3, 2), s2) == 1
    assert compare_real(s2, c2) == 1
    assert compare_real(c2, s2) == -1
    assert compare_real(s2, s2) == 0
    assert compare_real(ms2, s2) == -1
    assert compare_real(Rational(1), Rational(2)) == -1
    assert compare_real(Rational(2), Rational(2)) == 0
    assert compare_real(Rational(1), s2) == -1
    assert compare_real(Rational(2), s2) == 1
    # close roots of different polynomials
    a = CRootOf(x**2 - 2, 1)
    b = CRootOf(x**4 - 4, 1)  # this is sqrt(2) too but CRootOf canonicalizes
    assert a == b
    c = CRootOf(1000000*x**2 - 2000001, 1)
    assert compare_real(a, c) == -1
    assert compare_real(c, a) == 1
    # roots with a rational factor
    r0, r1 = Poly(2*x**2 - 9, x).real_roots(radicals=False)
    assert r1 == 3*CRootOf(2*x**2 - 1, 1)
    assert compare_real(r0, r1) == -1
    assert compare_real(r1, 2) == 1
    assert compare_real(r1, Rational(3, 2)) == 1
    assert compare_real(r1, CRootOf(x**2 - 5, 1)) == -1
    assert compare_real(r0, -2) == -1
    assert rational_between(r0, r1) == 0
    assert rational_between(2, r1) == Rational(33, 16)
    assert rational_below(r0) == -3
    assert rational_above(r1) == 3


def test_floor_scaled() -> None:
    s2 = CRootOf(x**2 - 2, 1)
    for k in range(6):
        assert _floor_scaled(s2, k) == floor(2**k*sqrt(2))
    ms2 = CRootOf(x**2 - 2, 0)
    for k in range(6):
        assert _floor_scaled(ms2, k) == floor(-2**k*sqrt(2))
    assert _floor_scaled(Rational(7, 2), 0) == 3
    assert _floor_scaled(Rational(7, 2), 1) == 7
    assert _floor_scaled(Rational(-7, 2), 0) == -4


def test_rational_between() -> None:
    s2, ms2 = CRootOf(x**2 - 2, 1), CRootOf(x**2 - 2, 0)
    assert rational_between(ms2, s2) == 0
    assert rational_between(1, s2) == Rational(5, 4)
    assert rational_between(Rational(1, 3), Rational(1, 2)) == Rational(2, 5)
    assert rational_between(-2, ms2) == Rational(-3, 2)
    assert rational_between(ms2, -1) == Rational(-5, 4)
    assert rational_between(-1, 1) == 0
    assert rational_between(Rational(9, 10), s2) == 1
    assert rational_between(ms2, 1) == 0
    assert rational_between(Rational(-1, 2), Rational(1, 3)) == 0
    assert rational_between(Rational(1, 3), Rational(1, 2)) == Rational(2, 5)
    assert rational_between(3, 4) == Rational(7, 2)
    raises(ValueError, lambda: rational_between(s2, 1))
    raises(ValueError, lambda: rational_between(s2, s2))

    # the result does not depend on the state of the isolating intervals
    s2.evalf(60)
    assert rational_between(1, s2) == Rational(5, 4)
    assert rational_between(ms2, s2) == 0

    roots = sorted(_roots((x**2 - 2)*(x**3 - 2)*(x**2 - 3)*(4*x**2 - 5)*(x - 1)),
                   key=lambda r: r.evalf())
    for a, b in zip(roots, roots[1:]):
        q = rational_between(a, b)
        assert q.is_Rational
        assert compare_real(a, q) == -1 and compare_real(q, b) == -1
        assert q.q <= 16


def test_rational_below_above() -> None:
    s2, ms2 = CRootOf(x**2 - 2, 1), CRootOf(x**2 - 2, 0)
    assert rational_below(s2) == 0
    assert rational_above(s2) == 2
    assert rational_below(ms2) == -2
    assert rational_above(ms2) == 0
    assert rational_below(2) == 0
    assert rational_above(2) == 3
    assert rational_below(-3) == -4
    assert rational_above(-3) == 0
    assert rational_below(Rational(-7, 2)) == -4
    assert rational_above(Rational(7, 2)) == 4
    assert rational_below(Rational(1, 2)) == 0
    assert rational_above(Rational(-1, 2)) == 0
    assert rational_below(0) == -1
    assert rational_above(0) == 1
    r = CRootOf(x**3 - 100, 0)
    assert rational_below(r) == 0
    assert rational_above(r) == 5
    r = CRootOf(x**3 + 100, 0)
    assert rational_below(r) == -5
    assert rational_above(r) == 0


def test_sample_point() -> None:
    origin = SamplePoint()
    assert len(origin) == 0 and origin.degree == 1
    assert origin.sign(3, []) == 1
    assert origin.sign(Rational(-1, 2), []) == -1
    assert origin.sign(0, []) == 0
    assert origin.real_roots(x**3 - x, [x]) == [-1, 0, 1]
    assert origin.real_roots(x**2 + 1, [x]) == []
    assert origin.real_roots(Poly(x**2 - 2, x), [x]) == [
        CRootOf(x**2 - 2, 0), CRootOf(x**2 - 2, 1)]
    assert origin.real_roots(Poly(0, x), [x]) is None
    assert origin.real_roots(7, [x]) == []
    raises(ValueError, lambda: origin.sign(x, [x]))
    raises(ValueError, lambda: origin.real_roots(x, [x, y]))

    p = origin.extend(Rational(1, 2))
    assert p.theta is None and len(p) == 1
    assert p.as_exprs() == (Rational(1, 2),)
    assert p.sign(2*x - 1, [x]) == 0
    assert p.sign(x - 1, [x]) == -1
    assert p.sign(Poly(x**2, x), [x]) == 1
    assert p.real_roots(y**2 - x, [x, y]) == [-sqrt(2)/2, sqrt(2)/2] or \
        p.real_roots(y**2 - x, [x, y]) == [CRootOf(2*x**2 - 1, 0), CRootOf(2*x**2 - 1, 1)]
    assert p.real_roots(x*y - y, [x, y]) == [0]
    assert p.real_roots((2*x - 1)*y, [x, y]) is None

    s2 = CRootOf(x**2 - 2, 1)
    q = origin.extend(s2)
    assert q.degree == 2 and q.theta == s2
    assert q.as_exprs() == (s2,)
    assert repr(q) == "SamplePoint(CRootOf(x**2 - 2, 1))"
    assert q.sign(x**2 - 2, [x]) == 0
    assert q.sign(x - 1, [x]) == 1
    assert q.sign(x - 2, [x]) == -1
    assert q.real_roots(x**2 + y**2 - 1, [x, y]) == []
    assert q.real_roots(x**2 - y**2 - 1, [x, y]) == [-1, 1]
    assert q.real_roots(x**2 - 2, [x, y]) is None
    assert q.real_roots(x**2 - 1, [x, y]) == []
    r = q.real_roots(y**2 - x, [x, y])
    assert r == [CRootOf(x**4 - 2, 0), CRootOf(x**4 - 2, 1)]

    # the same algebraic number again does not extend the field
    qq = q.extend(s2)
    assert qq.degree == 2 and qq.as_exprs() == (s2, s2)
    assert qq.sign(x - y, [x, y]) == 0
    # a rational coordinate neither
    p2 = q.extend(Rational(1, 2))
    assert p2.degree == 2
    assert repr(p2) == "SamplePoint(CRootOf(x**2 - 2, 1), 1/2)"
    assert p2.sign(x**2 + y**2 - 2, [x, y]) == 1
    assert p2.sign(2*y - 1, [x, y]) == 0
    assert p2.sign(x*y - 1, [x, y]) == -1

    # extension by a root over an algebraic point
    q2 = q.extend(r[1])
    assert q2.degree == 4
    e1, e2 = q2.as_exprs()
    assert abs(e1.evalf(20) - sqrt(2).evalf(20)) < 1e-15
    assert abs(e2.evalf(20) - 2**Rational(1, 4)) < 1e-15
    assert q2.sign(y**2 - x, [x, y]) == 0
    assert q2.sign(y - x, [x, y]) == -1
    assert q2.sign(y**4 - 2, [x, y]) == 0
    r3 = q2.real_roots(z**2 - x - y, [x, y, z])
    assert r3 is not None
    assert len(r3) == 2 and compare_real(r3[0], r3[1]) == -1
    q3 = q2.extend(r3[1])
    assert q3.degree == 8
    assert q3.sign(z**2 - x - y, [x, y, z]) == 0
    assert q3.sign(z - 1, [x, y, z]) == 1
    assert q3.sign(z - 2, [x, y, z]) == -1
    assert q3.real_roots(0*w, [x, y, z, w]) is None
    assert q3.real_roots(w**2 + 1, [x, y, z, w]) == []
    r4 = q3.real_roots(w**2 - z**2, [x, y, z, w])
    assert r4 is not None
    assert r4 == [-r3[1], r3[1]] or len(r4) == 2
    raises(ValueError, lambda: q3.sign(x, [x]))
    raises(ValueError, lambda: q3.real_roots(w, [x, y]))

    # distinct roots are sorted and duplicates removed
    roots = q2.real_roots((z**2 - x)*(z - y)*(z**2 - 3), [x, y, z])
    assert roots == [CRootOf(x**2 - 3, 0), CRootOf(x**4 - 2, 0),
                     CRootOf(x**4 - 2, 1), CRootOf(x**2 - 3, 1)]

    # coordinates given as a rational multiple of a root
    r0, r1 = Poly(2*x**2 - 9, x).real_roots(radicals=False)
    p = origin.extend(r1)
    assert p.degree == 2 and p.theta == r1
    assert p.sign(2*x**2 - 9, [x]) == 0
    assert p.sign(x - 2, [x]) == 1
    assert p.sign(x - 3, [x]) == -1
    q = p.extend(CRootOf(x**3 - 2, 0))
    assert q.degree == 6
    assert q.sign(y**3 - 2, [x, y]) == 0 and q.sign(2*x**2 - 9, [x, y]) == 0
    assert q.sign(x*y - 3, [x, y]) == -1
    q = origin.extend(CRootOf(x**3 - 2, 0)).extend(r0)
    assert q.degree == 6
    assert q.sign(2*y**2 - 9, [x, y]) == 0 and q.sign(y, [x, y]) == -1

    # several quadratic extensions
    pp = origin.extend(CRootOf(x**2 - 2, 1)).extend(CRootOf(x**2 - 3, 1))
    pp = pp.extend(CRootOf(x**2 - 5, 1))
    assert pp.degree == 8
    assert pp.sign(x**2*y**2*z**2 - 30, [x, y, z]) == 0
    assert pp.sign(x**2*y**2*z**2 - 29, [x, y, z]) == 1
    assert pp.sign(x*y*z - 6, [x, y, z]) == -1
    assert pp.sign(x*y*z - 5, [x, y, z]) == 1
    vals = [e.evalf(20) for e in pp.as_exprs()]
    assert all(abs(v - sqrt(n).evalf(20)) < 1e-15 for v, n in zip(vals, [2, 3, 5]))


def test_specialization() -> None:
    from sympy_extras.polys.cad.samplepoints_sympy import Specialization
    p = SamplePoint().extend(CRootOf(x**2 - 2, 1))
    s = p.specialization((y**2 - x**2)*(y - 1)**2*(y**2 + 1), [x, y])
    assert s.degree == 6 and s.leading_sign == 1
    assert [(compare_real(r, v), m) for (r, m), v in zip(s.roots, [CRootOf(x**2 - 2, 0), Rational(1), CRootOf(x**2 - 2, 1)])] == [(0, 1), (0, 2), (0, 1)]
    assert [s.sign_before(i) for i in range(4)] == [1, -1, -1, 1]
    raises(ValueError, lambda: s.sign_before(4))
    # a polynomial which vanishes at the point
    s = p.specialization((x**2 - 2)*y, [x, y])
    assert s.degree == -1 and s.leading_sign == 0 and s.roots == [] and s.sign_before(0) == 0
    # a constant, positive and negative
    assert p.specialization(x**2 - 1, [x, y]).degree == 0
    assert p.specialization(x**2 - 1, [x, y]).leading_sign == 1
    assert p.specialization(1 - x**3, [x, y]).leading_sign == -1
    # the result is kept
    assert p.specialization(y - x, [x, y]) is p.specialization(y - x, [x, y])
    assert isinstance(p.specialization(y - x, [x, y]), Specialization)
    # a rational point
    s = SamplePoint().extend(Rational(1, 2)).specialization(-(2*y - 1)**3*(y + x), [x, y])
    assert s.degree == 4 and s.leading_sign == -1 and s.roots == [(-Rational(1, 2), 1), (Rational(1, 2), 3)]
    assert [s.sign_before(i) for i in range(3)] == [-1, 1, -1]


def test_equal_roots_written_differently() -> None:
    from sympy_extras.polys.cad.samplepoints_sympy import _order
    # the same number as a root of x**2 - 2 and as twice a root of
    # 2*x**2 - 1 (the form real_roots gives depends on the polynomial it
    # was asked about): compare_real refined the two intervals forever,
    # since they never become disjoint; the indices among the roots of the
    # common minimal polynomial tell that they are equal
    a, b = CRootOf(x**2 - 2, 1), 2*CRootOf(2*x**2 - 1, 1)
    assert a != b and compare_real(a, b) == 0 and compare_real(b, a) == 0
    assert compare_real(CRootOf(x**2 - 2, 0), 2*CRootOf(2*x**2 - 1, 1)) == -1
    assert compare_real(-CRootOf(x**2 - 2, 0), 2*CRootOf(2*x**2 - 1, 1)) == 0
    assert _order([a, 1, b, CRootOf(x**2 - 2, 0), Rational(3, 2), Rational(3, 2)]) == [[3], [1], [0, 2], [4, 5]]
    assert _order([]) == [] and _order([0]) == [[0]]


def test_real_roots_over_an_algebraic_field() -> None:
    # the real roots of a polynomial over QQ<theta> by the Sturm counts of
    # the candidates, the roots of the norm (the native _real_roots_over,
    # through the specialization at a point of QQ<theta>), against SymPy's
    # roots over the algebraic field
    # SymPy's oracle first: a root CRootOf(y**3 - 3*y + 1, 2) found in y
    # is equal to theta, SymPy's cache then returns its powers for those of
    # theta, and Poly(..., y) takes them for occurrences of y (issue #25)
    clear_cache()
    theta = CRootOf(x**3 - 3*x + 1, 2)
    polys = [(y - x)*(y + x)*(y**2 - x - 3), (y**2 - 2)*(y - x)**2*(y**2 + 1), y**4 - x*y**2 - 1, (3*y - x**2)**3]
    oracle = [Poly(f.subs(x, theta), y, extension=True).real_roots(multiple=False) for f in polys]
    point = SamplePoint().extend(theta)
    for f, direct in zip(polys, oracle):
        found = point.specialization(f, [x, y]).roots
        assert all(compare_real(r, s) < 0 for (r, _), (s, _) in zip(found, found[1:]))
        assert all(abs(f.subs({x: theta, y: r}).evalf(50)) < 1e-40 for r, _ in found)
        assert [m for _, m in found] == [m for _, m in direct]
        assert all(abs(r.evalf(50) - s.evalf(50)) < 1e-40 for (r, _), (s, _) in zip(found, direct))


def test_the_roots_of_each_factor_are_isolated_alone() -> None:
    # the whole test suite hung twice in a stack of
    # test_the_description_holds_on_the_cells_where_the_formula_does (seven
    # seconds alone): SamplePoint.specialization took the roots of a
    # polynomial with several factors from Poly.real_roots, which makes the
    # isolating intervals of all the factors disjoint together
    # (ComplexRootOf._reals_sorted, RealInterval.refine_disjoint), and with
    # the intervals SymPy had kept from the earlier tests that loop did not
    # end. The roots of each irreducible factor are now isolated alone and
    # merged by _order, so SymPy never compares intervals of two factors.
    from unittest.mock import patch
    from sympy.polys.rootisolation import RealInterval
    from sympy.polys.rootoftools import ComplexRootOf
    from sympy_extras.polys.cad.samplepoints_sympy import rational_real_roots
    original = ComplexRootOf._reals_sorted
    factors_seen: list[int] = []

    def spy(cls: type[ComplexRootOf], reals: list[tuple[RealInterval, Poly, int]]
            ) -> list[tuple[RealInterval, Poly, int]]:
        factors_seen.append(len({f for _, f, _ in reals}))
        return original(reals)

    f = Poly((2*y - 1)**2 * (y**2 - 2) * (2*y**2 - 1) * (y**3 - 3*y + 1) * y, y)
    with patch.object(ComplexRootOf, '_reals_sorted', classmethod(spy)):
        found = rational_real_roots(f)
        point = SamplePoint().extend(Rational(1, 3))
        specialized = point.specialization(f.as_expr() + 0*x, [x, y])
    # since the port to the native layer SymPy only names the roots
    # (CRootOf of one irreducible factor at a time)
    assert all(n == 1 for n in factors_seen)
    assert [m for _, m in found] == [1, 1, 1, 1, 1, 2, 1, 1, 1]
    assert [m for _, m in specialized.roots] == [m for _, m in found]
    assert [m for _, m in found] == [1, 1, 1, 1, 1, 2, 1, 1, 1]
    values = [r for r, _ in found]
    assert all(compare_real(a, b) < 0 for a, b in zip(values, values[1:]))
    assert [m for _, m in specialized.roots] == [m for _, m in found]
    # the same roots as SymPy's own, when SymPy finishes
    assert values == [r for r, _ in f.real_roots(multiple=False, radicals=False)]
