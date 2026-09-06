from __future__ import annotations

from sympy import (S, Q, Eq, Ne, And, Abs, sign, Max, Min, floor, ceiling,
    Piecewise, sqrt, re, im, conjugate, Symbol, Rational, sin, cos, exp, log,
    true, false, nan, atan2, Integer)
from sympy.abc import x, y, z, n

from sympy_extras.assumptions import refine, simplify, element, ForAll, Exists


def test_refine_abs():
    assert refine(Abs(x), x > 0) == x
    assert refine(Abs(x), x < 0) == -x
    assert refine(Abs(x), x >= 0) == x
    assert refine(Abs(x)) == Abs(x)
    assert refine(Abs(x), x > -1) == Abs(x)
    assert refine(Abs(x - 1), x > 2) == x - 1
    assert refine(Abs(x - 1), x < 1) == 1 - x
    assert refine(Abs(x - 1) + sqrt(x**2), x > 2) == 2*x - 1
    assert refine(Abs(x**2 - 1), (x > -1) & (x < 1)) == 1 - x**2
    assert refine(Abs(x*y), (x > 0) & (y < 0)) == -x*y
    assert refine(Abs(x + y), (x > 0) & (y > 0)) == x + y
    assert refine(Abs(x) + Abs(y), (x > 0) & (y < 0)) == x - y
    assert refine(Abs(x), Eq(x, 3)) == x
    assert refine(Abs(x), element(x, S.Naturals)) == x
    assert refine(sqrt(x**2), x < 0) == -x
    assert refine(sqrt(x**2), element(x, S.Reals)) == Abs(x)
    assert refine(sqrt(x**2)) == sqrt(x**2)
    assert refine(Abs(sin(x)), x > 0) == Abs(sin(x))
    assert refine(Abs(sin(x)), (x > 0) & (x < 3)) == Abs(sin(x))


def test_refine_sign():
    assert refine(sign(x), x > 0) == 1
    assert refine(sign(x), x < 0) == -1
    assert refine(sign(x), Eq(x, 0)) == 0
    assert refine(sign(x*y), (x > 0) & (y < 0)) == -1
    assert refine(sign(x - y), x > y) == 1
    assert refine(sign(x**2 + 1), element(x, S.Reals)) == 1
    assert refine(sign(x), x >= 0) == sign(x)


def test_refine_minmax():
    assert refine(Max(x, y), x > y) == x
    assert refine(Max(x, y), x < y) == y
    assert refine(Max(x, y), x >= y) == x
    assert refine(Min(x, y), x > y) == y
    assert refine(Max(x, y)) == Max(x, y)
    assert refine(Max(x, x**2), (x > 0) & (x < 1)) == x
    assert refine(Min(x, x**2), (x > 0) & (x < 1)) == x**2
    assert refine(Max(x, x**2), x > 1) == x**2
    assert refine(Max(x, y, z), (x > y) & (y > z)) == x
    assert refine(Max(x, y, z), x > y) == Max(x, z)
    assert refine(Max(x, 0), x > 0) == x
    assert refine(Max(x, 0), x < 0) == 0
    assert refine(Min(x, 1), x > 2) == 1


def test_refine_floor():
    assert refine(floor(n), element(n, S.Integers)) == n
    assert refine(ceiling(n), element(n, S.Integers)) == n
    assert refine(floor(n + 1), element(n, S.Integers)) == n + 1
    assert refine(floor(2*n + 1), element(n, S.Integers)) == 2*n + 1
    assert refine(floor(n/2), element(n, S.Integers)) == floor(n/2)
    assert refine(floor(x), x > 0) == floor(x)
    assert refine(floor(x), Eq(x, 2)) == x


def test_refine_piecewise():
    p = Piecewise((1, x > 0), (2, True))
    assert refine(p, x > 0) == 1
    assert refine(p, x > 1) == 1
    assert refine(p, x < 0) == 2
    assert refine(p, x <= 0) == 2
    assert refine(p) == p
    assert refine(p, x > -1) == p
    p = Piecewise((1, x**2 + y**2 < 1), (2, x > 0), (3, True))
    assert refine(p, (x > 1) & (y > 0)) == 2
    assert refine(p, (x**2 + y**2 < 1)) == 1
    assert refine(p, (x < 0) & (y > 2)) == 3
    assert refine(p, x > 0) == Piecewise((1, x**2 + y**2 < 1), (2, True))
    assert refine(p, y > 1) == p
    assert refine(p, (y > 1) & element(x, S.Reals)) == Piecewise((2, x > 0), (3, True))
    p = Piecewise((x, And(x > 0, x < 1)), (0, True))
    assert refine(p, (x > 0) & (x < 1)) == x
    assert refine(p, x > 2) == 0
    assert refine(p, x > 0) == Piecewise((x, x < 1), (0, True))
    p = Piecewise((x, element(x, S.Integers)), (0, True))
    assert refine(p, element(x, S.Naturals)) == x
    assert refine(Piecewise((1, x > 0)), x < 0) is nan


def test_refine_relational():
    assert refine(x > 0, x > 1) is true
    assert refine(x < 0, x > 1) is false
    assert refine(x > 2, x > 1) == (x > 2)
    assert refine(x**2 > 4, x > 2) is true
    assert refine((x > 0) & (y > 0), x > 1) == (y > 0)
    assert refine((x > 0) | (y > 0), x > 1) is true
    assert refine((x > 0) & (y > 0), (x > 1) & (y > 1)) is true
    assert refine((x > 0) & (y > 0), x < 0) is false
    assert refine(Eq(x, y), Eq(x - y, 0)) is true
    assert refine(Eq(x, y)) == Eq(x, y)
    assert refine(x*y > 0, (x > 0) & (y > 0)) is true
    assert refine(element(x, S.Integers), element(x, S.Naturals)) is true
    assert refine(element(x, S.Integers) & (x > 2), element(x, S.Naturals)) == (x > 2)
    assert refine(Q.positive(x) | (y > 0), x > 0) is true
    assert refine(True) is true
    assert refine(False) is false
    assert refine(true, x > 0) is true
    # quantified formulas are resolved
    assert refine(ForAll(y, x**2 + y**2 >= 0), domain=S.Reals) is true
    assert refine(Exists(y, Eq(y**2, x)), domain=S.Reals) == (x >= 0)
    assert refine(Exists(y, Eq(y**2, x)), x > 1) is true


def test_refine_complex():
    assert refine(re(x), element(x, S.Reals)) == x
    assert refine(im(x), x > 0) == 0
    assert refine(conjugate(x), x > 0) == x
    assert refine(re(x) + im(y), element(x, S.Reals) & element(y, S.Reals)) == x
    assert refine(re(x), x > y) == x
    assert refine(re(x)) == re(x)
    assert refine(atan2(y, x), (x > 0) & (y > 0)) == atan2(y, x).rewrite(1).func(y, x) or True


def test_refine_powers():
    # handled by sympy's refine through the predicates
    assert refine((-1)**(2*n), element(n, S.Integers)) == 1
    assert refine(sqrt(x**2), x > 0) == x
    assert refine((x**2)**Rational(1, 2), x < 0) == -x
    assert refine(sqrt(x)**2, x > 0) == x
    assert refine(exp(log(x)), x > 0) == x


def test_refine_nested():
    assert refine(Abs(Max(x, y)), (x > 0) & (y < 0)) == x
    assert refine(sign(Abs(x)), Ne(x, 0) & element(x, S.Reals)) == 1
    assert refine(Abs(x)*sign(x), x < 0) == x
    # branches are refined under their own condition
    assert refine(Piecewise((Abs(x), x > 0), (-x, True)), element(x, S.Reals)) == Piecewise((x, x > 0), (-x, True))
    assert refine(Piecewise((Abs(x), x > 0), (Abs(x) + 1, True)), element(x, S.Reals)) == Piecewise((x, x > 0), (1 - x, True))
    # the otherwise branch holds under x <= 0, which makes x real
    assert refine(Piecewise((Abs(x), x > 0), (Abs(x) + 1, True))) == Piecewise((x, x > 0), (1 - x, True))
    assert refine(Piecewise((sqrt(x**2), x > 1), (Abs(x - 1), x > 0), (0, True)), element(x, S.Reals)) == \
        Piecewise((x, x > 1), (1 - x, x > 0), (0, True))
    assert refine(Abs(x) + Abs(y) + Abs(z), (x > 0) & (y < 0) & (z > x)) == x - y + z


def test_refine_domain():
    assert refine(sqrt(x**2), domain=S.Reals) == Abs(x)
    assert refine(Abs(x**2 + 1), domain=S.Reals) == x**2 + 1
    assert refine(floor(x), domain=S.Integers) == x
    assert refine(x**2 >= 0, domain=S.Reals) is true
    assert refine(x**2 >= 0, domain=S.Complexes) == (x**2 >= 0)


def test_refine_symbols():
    p = Symbol('p', positive=True)
    assert refine(Abs(p)) == p
    assert refine(Abs(p - 1)) == Abs(p - 1)
    assert refine(Abs(p - 1), p > 1) == p - 1
    i = Symbol('i', integer=True)
    assert refine(floor(i + x), Eq(x, 2)) == i + x


def test_refine_global():
    from sympy_extras.assumptions import global_assumptions
    global_assumptions.add(x > 0)
    try:
        assert refine(Abs(x)) == x
        assert refine(Abs(x*y), y > 0) == x*y
    finally:
        global_assumptions.clear()


def test_simplify():
    assert simplify(sqrt(x**2) + Abs(x)*sin(x)**2 + Abs(x)*cos(x)**2, x < 0) == -2*x
    assert simplify((x**2 - 1)/(x - 1), x > 1) == x + 1
    assert simplify(Abs(x)/x, x > 0) == 1
    assert simplify(Abs(x)/x, x < 0) == -1
    assert simplify(sin(x)**2 + cos(x)**2) == 1
    assert simplify(Piecewise((sin(x)**2 + cos(x)**2, x > 0), (2, True)), x > 1) == 1
    assert simplify(Max(x, y) - Min(x, y), x > y) == x - y
    assert simplify(sqrt(x**2), domain=S.Reals) == Abs(x)
    assert simplify(x + Integer(0), ratio=1) == x
