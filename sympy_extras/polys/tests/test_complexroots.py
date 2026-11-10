"""Complex root isolation against known roots and numerical roots
(mpmath, the oracle)."""
from __future__ import annotations

import random
from fractions import Fraction

import mpmath
from sympy.testing.pytest import raises

from sympy_extras.polys.complexroots import (Rectangle, dup_count_complex_roots, dup_isolate_all_roots_sqf,
                                              dup_isolate_complex_roots, dup_isolate_complex_roots_sqf,
                                              dup_refine_complex_root)
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.sqfreetools import dup_sqf_p


def _poly(*coefficients: int) -> list[Fraction]:
    return [Fraction(c) for c in coefficients]


def _strictly_inside(r: Rectangle, re: Fraction, im: Fraction) -> bool:
    return r.re_min < re < r.re_max and r.im_min < im < r.im_max


def _numerical_roots(f: list[Fraction]) -> list[tuple[Fraction, Fraction]]:
    """The roots to 30 digits, as rational points."""
    with mpmath.workdps(40):
        roots = mpmath.polyroots([mpmath.mpf(c.numerator)/c.denominator for c in f], maxsteps=500, extraprec=300)
        scale = 10**30
        return [(Fraction(int(mpmath.nint(z.real*scale)), scale), Fraction(int(mpmath.nint(z.imag*scale)), scale))
                for z in roots]


def test_known_roots() -> None:
    # x**4 + 4 = (x**2 - 2 x + 2)(x**2 + 2 x + 2): the roots +-1 +- i
    rectangles = dup_isolate_complex_roots_sqf(_poly(1, 0, 0, 0, 4))
    assert len(rectangles) == 2
    assert _strictly_inside(rectangles[0], Fraction(-1), Fraction(1))
    assert _strictly_inside(rectangles[1], Fraction(1), Fraction(1))
    # x**2 + 1
    (rectangle,) = dup_isolate_complex_roots_sqf(_poly(1, 0, 1), Fraction(1, 1000))
    assert _strictly_inside(rectangle, Fraction(0), Fraction(1))
    assert rectangle.width < Fraction(1, 1000) and rectangle.height < Fraction(1, 1000)
    # real roots only, constants
    assert dup_isolate_complex_roots_sqf(_poly(1, 0, -2)) == []
    assert dup_isolate_complex_roots_sqf(_poly(3)) == []
    # (x - 1/2)**2 + 1/9: rational real and imaginary parts
    f = [Fraction(1), Fraction(-1), Fraction(1, 4) + Fraction(1, 9)]
    (rectangle,) = dup_isolate_complex_roots_sqf(f, Fraction(1, 10**20))
    assert _strictly_inside(rectangle, Fraction(1, 2), Fraction(1, 3))


def test_roots_of_unity() -> None:
    # x**n - 1: (n - 1)//2 roots above the axis, on the unit circle
    for n in range(1, 13):
        f = _poly(*([1] + [0]*(n - 1) + [-1]))
        rectangles = dup_isolate_complex_roots_sqf(f, Fraction(1, 10**6))
        assert len(rectangles) == (n - 1)//2
        for r in rectangles:
            x, y = r.center
            assert abs(x*x + y*y - 1) < Fraction(1, 10**5)
            assert r.im_min > 0


def test_against_numerical_roots() -> None:
    rng = random.Random(31)
    checked = 0
    for trial in range(120):
        n = rng.randint(1, 9)
        bound = rng.choice([2, 10, 1000])
        f = [Fraction(rng.randint(-bound, bound), rng.randint(1, 3)) for _ in range(n + 1)]
        if f[0] == 0:
            f[0] = Fraction(1)
        if not dup_sqf_p(f, QQ):
            continue
        checked += 1
        roots = _numerical_roots(f)
        upper = [z for z in roots if z[1] > Fraction(1, 10**20)]
        eps = Fraction(1, 100) if trial % 2 else None
        real, rectangles = dup_isolate_all_roots_sqf(f, eps)
        assert len(real) + 2*len(rectangles) == n
        assert len(rectangles) == len(upper)
        for r in rectangles:
            assert r.im_min > 0
            assert sum(1 for z in upper if _strictly_inside(r, z[0], z[1])) == 1
            if eps is not None:
                assert r.width < eps and r.height < eps
        # the rectangles are sorted and their interiors are disjoint
        assert rectangles == sorted(rectangles, key=lambda r: (r.re_min, r.im_min))
        for i, r in enumerate(rectangles):
            for s in rectangles[i + 1:]:
                assert (r.re_max <= s.re_min or s.re_max <= r.re_min or r.im_max <= s.im_min or s.im_max <= r.im_min)
        # the winding number of a random rectangle
        x0 = Fraction(rng.randint(-40, 40), 8)
        y0 = Fraction(rng.randint(-40, 40), 8)
        window = Rectangle(x0, x0 + Fraction(rng.randint(1, 40), 8), y0, y0 + Fraction(rng.randint(1, 40), 8))
        tolerance = Fraction(1, 10**20)
        near_boundary = any(min(abs(z[0] - window.re_min), abs(z[0] - window.re_max)) < tolerance
                            or min(abs(z[1] - window.im_min), abs(z[1] - window.im_max)) < tolerance for z in roots)
        if not near_boundary:
            assert dup_count_complex_roots(f, window) == sum(1 for z in roots if _strictly_inside(window, z[0], z[1]))
    assert checked > 100


def test_count_with_multiplicities_and_boundary() -> None:
    f = _poly(1, 0, 2, 0, 1)                      # (x**2 + 1)**2
    assert dup_count_complex_roots(f, Rectangle(Fraction(-1), Fraction(1), Fraction(0), Fraction(2))) == 2
    assert dup_count_complex_roots(f, Rectangle(Fraction(-1), Fraction(1), Fraction(-2), Fraction(2))) == 4
    assert dup_count_complex_roots(f, Rectangle(Fraction(1, 2), Fraction(1), Fraction(0), Fraction(2))) == 0
    # the root i on an edge, then at a corner
    raises(ValueError, lambda: dup_count_complex_roots(f, Rectangle(Fraction(0), Fraction(1), Fraction(0), Fraction(2))))
    raises(ValueError, lambda: dup_count_complex_roots(f, Rectangle(Fraction(0), Fraction(1), Fraction(1), Fraction(2))))
    # a real polynomial on a real edge without roots: Im f = 0 there
    g = _poly(1, 0, 1)
    assert dup_count_complex_roots(g, Rectangle(Fraction(-1), Fraction(1), Fraction(0), Fraction(2))) == 1
    # real roots on the real edge
    raises(ValueError, lambda: dup_count_complex_roots(_poly(1, 0, -1),
                                                      Rectangle(Fraction(-2), Fraction(2), Fraction(0), Fraction(2))))
    raises(ValueError, lambda: dup_isolate_complex_roots_sqf(f))
    raises(ValueError, lambda: Rectangle(Fraction(1), Fraction(1), Fraction(0), Fraction(1)))


def test_roots_close_to_the_real_axis_and_to_each_other() -> None:
    # (x - 1)**2 + 10**-12: the roots 1 +- 10**-6 i
    f = [Fraction(1), Fraction(-2), 1 + Fraction(1, 10**12)]
    (rectangle,) = dup_isolate_complex_roots_sqf(f)
    assert _strictly_inside(rectangle, Fraction(1), Fraction(1, 10**6))
    assert rectangle.im_min > 0
    # ((x - 1)**2 + 1)((x - 1 - 10**-5)**2 + 1): two roots 10**-5 apart
    d = Fraction(1, 10**5)
    g = [Fraction(1), Fraction(-2), Fraction(2)]
    h = [Fraction(1), -2*(1 + d), (1 + d)**2 + 1]
    product = [Fraction(0)]*5
    for i, a in enumerate(g):
        for j, b in enumerate(h):
            product[i + j] += a*b
    rectangles = dup_isolate_complex_roots_sqf(product)
    assert len(rectangles) == 2
    assert _strictly_inside(rectangles[0], Fraction(1), Fraction(1))
    assert _strictly_inside(rectangles[1], 1 + d, Fraction(1))


def test_refinement() -> None:
    f = _poly(1, 0, 0, 1, 1)                        # x**4 + x + 1
    for rectangle in dup_isolate_complex_roots_sqf(f):
        eps = Fraction(1, 10**40)
        fine = dup_refine_complex_root(f, rectangle, eps)
        assert fine.width < eps and fine.height < eps
        assert rectangle.re_min <= fine.re_min and fine.re_max <= rectangle.re_max
        assert rectangle.im_min <= fine.im_min and fine.im_max <= rectangle.im_max
        assert dup_count_complex_roots(f, fine) == 1
    # a rectangle with two roots, or none, is refused
    raises(ValueError, lambda: dup_refine_complex_root(_poly(1, 0, 0, 0, 4), Rectangle(
        Fraction(-2), Fraction(2), Fraction(1, 2), Fraction(2)), Fraction(1, 10)))
    raises(ValueError, lambda: dup_refine_complex_root(_poly(1, 0, 1), Rectangle(
        Fraction(1), Fraction(2), Fraction(1, 2), Fraction(2)), Fraction(1, 10)))


def test_rectangle() -> None:
    r = Rectangle(Fraction(-1), Fraction(1, 2), Fraction(1), Fraction(3))
    assert r.center == (Fraction(-1, 4), Fraction(2))
    assert r.conjugate() == Rectangle(Fraction(-1), Fraction(1, 2), Fraction(-3), Fraction(-1))
    assert r.conjugate().conjugate() == r and hash(r.conjugate().conjugate()) == hash(r)
    assert r.contains(Fraction(1, 2), Fraction(3)) and not r.contains(Fraction(1), Fraction(2))
    assert repr(r) == 'Rectangle(-1, 1/2, 1, 3)'


def test_multiplicities() -> None:
    # (x**2 + 1)**3 (x**2 + 1 + 10**-8)**2 (x**2 - 2 x + 5) (x - 3)
    close = [Fraction(1), Fraction(0), 1 + Fraction(1, 10**8)]
    parts = [(_poly(1, 0, 1), 3), (close, 2), (_poly(1, -2, 5), 1), (_poly(1, -3), 1)]
    f = [Fraction(1)]
    for g, k in parts:
        for _ in range(k):
            product = [Fraction(0)]*(len(f) + len(g) - 1)
            for i, a in enumerate(f):
                for j, b in enumerate(g):
                    product[i + j] += a*b
            f = product
    result = dup_isolate_complex_roots(f)
    assert sorted(k for _, k in result) == [1, 2, 3]
    by_multiplicity = {k: r for r, k in result}
    assert _strictly_inside(by_multiplicity[3], Fraction(0), Fraction(1))
    assert not _strictly_inside(by_multiplicity[2], Fraction(0), Fraction(1))
    assert _strictly_inside(by_multiplicity[1], Fraction(1), Fraction(2))
    # the rectangles of i and of i sqrt(1 + 10**-8) were separated
    r, s = by_multiplicity[3], by_multiplicity[2]
    assert r.im_max < s.im_min or s.im_max < r.im_min or r.re_max < s.re_min or s.re_max < r.re_min
    assert dup_isolate_complex_roots(_poly(5)) == []
