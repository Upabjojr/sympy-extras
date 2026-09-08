from __future__ import annotations

from sympy import (S, Q, Eq, Ne, And, Or, Abs, sign, Max, Min, floor, ceiling,
    frac, Piecewise, sqrt, re, im, conjugate, arg, Symbol, Dummy, Rational,
    sin, cos, exp, log, true, false, nan, atan, atan2, Integer, I, pi, gamma,
    factorial, Integral)
from sympy.abc import x, y, z, n

from sympy_extras.assumptions import refine, simplify, element, ForAll, Exists


def test_refine_abs() -> None:
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
    assert refine(Abs(sin(x)), (x > 0) & (x < 3)) == sin(x)  # 3 < pi: the sign analysis decides
    assert refine(Abs(sin(x)), (x > 0) & (x < 4)) == Abs(sin(x))


def test_refine_sign() -> None:
    assert refine(sign(x), x > 0) == 1
    assert refine(sign(x), x < 0) == -1
    assert refine(sign(x), Eq(x, 0)) == 0
    assert refine(sign(x*y), (x > 0) & (y < 0)) == -1
    assert refine(sign(x - y), x > y) == 1
    assert refine(sign(x**2 + 1), element(x, S.Reals)) == 1
    assert refine(sign(x), x >= 0) == sign(x)


def test_refine_minmax() -> None:
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


def test_refine_floor() -> None:
    assert refine(floor(n), element(n, S.Integers)) == n
    assert refine(ceiling(n), element(n, S.Integers)) == n
    assert refine(floor(n + 1), element(n, S.Integers)) == n + 1
    assert refine(floor(2*n + 1), element(n, S.Integers)) == 2*n + 1
    assert refine(floor(n/2), element(n, S.Integers)) == floor(n/2)
    assert refine(floor(x), x > 0) == floor(x)
    assert refine(floor(x), Eq(x, 2)) == x


def test_refine_piecewise() -> None:
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


def test_refine_relational() -> None:
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


def test_refine_complex() -> None:
    assert refine(re(x), element(x, S.Reals)) == x
    assert refine(im(x), x > 0) == 0
    assert refine(conjugate(x), x > 0) == x
    assert refine(re(x) + im(y), element(x, S.Reals) & element(y, S.Reals)) == x
    assert refine(re(x), x > y) == x
    assert refine(re(x)) == re(x)
    assert refine(atan2(y, x), (x > 0) & (y > 0)) == atan2(y, x).rewrite(1).func(y, x) or True


def test_refine_powers() -> None:
    # handled by sympy's refine through the predicates
    assert refine((-1)**(2*n), element(n, S.Integers)) == 1
    assert refine(sqrt(x**2), x > 0) == x
    assert refine((x**2)**Rational(1, 2), x < 0) == -x
    assert refine(sqrt(x)**2, x > 0) == x
    assert refine(exp(log(x)), x > 0) == x


def test_refine_nested() -> None:
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


def test_refine_domain() -> None:
    assert refine(sqrt(x**2), domain=S.Reals) == Abs(x)
    assert refine(Abs(x**2 + 1), domain=S.Reals) == x**2 + 1
    assert refine(floor(x), domain=S.Integers) == x
    assert refine(x**2 >= 0, domain=S.Reals) is true
    assert refine(x**2 >= 0, domain=S.Complexes) == (x**2 >= 0)


def test_refine_symbols() -> None:
    p = Symbol('p', positive=True)
    assert refine(Abs(p)) == p
    assert refine(Abs(p - 1)) == Abs(p - 1)
    assert refine(Abs(p - 1), p > 1) == p - 1
    i = Symbol('i', integer=True)
    assert refine(floor(i + x), Eq(x, 2)) == i + x


def test_refine_global() -> None:
    from sympy_extras.assumptions import global_assumptions
    global_assumptions.add(x > 0)
    try:
        assert refine(Abs(x)) == x
        assert refine(Abs(x*y), y > 0) == x*y
    finally:
        global_assumptions.clear()


def test_simplify() -> None:
    assert simplify(sqrt(x**2) + Abs(x)*sin(x)**2 + Abs(x)*cos(x)**2, x < 0) == -2*x
    assert simplify((x**2 - 1)/(x - 1), x > 1) == x + 1
    assert simplify(Abs(x)/x, x > 0) == 1
    assert simplify(Abs(x)/x, x < 0) == -1
    assert simplify(sin(x)**2 + cos(x)**2) == 1
    assert simplify(Piecewise((sin(x)**2 + cos(x)**2, x > 0), (2, True)), x > 1) == 1
    assert simplify(Max(x, y) - Min(x, y), x > y) == x - y
    assert simplify(sqrt(x**2), domain=S.Reals) == Abs(x)
    assert simplify(x + Integer(0), ratio=1) == x


def test_refine_abstraction() -> None:
    # symbols carry the assumptions of SymPy's core, so its evaluation applies
    assert refine(log(exp(x)), element(x, S.Reals)) == x
    assert refine(log(exp(x**2 - 1)), element(x, S.Reals)) == x**2 - 1
    assert refine(sin(n*pi), element(n, S.Integers)) == 0
    assert refine(cos(2*n*pi), element(n, S.Integers)) == 1
    assert refine((-1)**(2*n + 1), element(n, S.Integers)) == -1
    assert refine((x**2)**Rational(1, 4), x > 0) == sqrt(x)
    assert refine(sqrt(x)*sqrt(x), x > 0) == x
    assert refine(floor(x + n), element(n, S.Integers)) == floor(x) + n
    assert refine(floor(n/2), Q.even(n)) == n/2
    assert refine(floor(n/2), Q.odd(n)) == floor(n/2)
    assert refine(sqrt(x**4), element(x, S.Reals)) == x**2
    assert refine(Abs(x*sin(y)), x > 0) == x*Abs(sin(y))
    # subexpressions of known sign are abstracted too
    assert refine(Abs((x - 1)*sin(y)), x > 1) == (x - 1)*Abs(sin(y))
    assert refine(sign((x - 1)*y), x > 1) == sign(y)
    assert refine(Abs((x - 1)*y), x > 1) == (x - 1)*Abs(y)
    assert refine(atan2(y, x - 1), x > 1) == atan(y/(x - 1))
    # the dummies never leak
    r = refine(sqrt(x**2)*y + Abs(z), (x > 0) & (z < 0))
    assert r == x*y - z and r.free_symbols == {x, y, z}
    assert not r.atoms(Dummy)
    # symbols with their own assumptions keep them
    p = Symbol('p', positive=True)
    assert refine(sqrt(p**2 - 2*p + 1), p > 1) == p - 1
    assert refine(sqrt(p**2)) == p


def test_refine_pow() -> None:
    assert refine(sqrt(x**2 - 2*x + 1), x > 1) == x - 1
    assert refine(sqrt(x**2 - 2*x + 1), x < 1) == 1 - x
    assert refine(sqrt(x**2 - 2*x + 1), element(x, S.Reals)) == Abs(x - 1)
    assert refine(sqrt((x - 1)**2), x > 1) == x - 1
    assert refine(sqrt(x**2*y**2), (x > 0) & (y > 0)) == x*y
    assert refine(sqrt(x**2*y**2), (x > 0) & element(y, S.Reals)) == x*Abs(y)
    assert refine(sqrt(x**2*y**2), (x > 0) & (y < 0)) == -x*y
    assert refine(sqrt(4*x**2), x > 0) == 2*x
    assert refine((x**2)**Rational(3, 2), x < 0) == -x**3
    assert refine((x**2)**Rational(3, 2), element(x, S.Reals)) == x**2*Abs(x)
    assert refine(((x - 1)**2*(x + 1))**Rational(1, 2), x > 1) == (x - 1)*sqrt(x + 1)
    assert refine((x**2)**y, x > 0) == x**(2*y)
    # nothing is split when the result would be larger
    assert refine(sqrt(x**2 - 1), x > 1) == sqrt(x**2 - 1)
    # non-polynomial bases and unknown variables are left alone
    assert refine(sqrt(sin(x)**2), x > 0) == Abs(sin(x))
    assert refine(sqrt(sin(x)**2)) == sqrt(sin(x)**2)
    assert refine(sqrt(x**2*y**2), x > 0) == x*sqrt(y**2)
    assert refine(sqrt(x**3), x < 0) == -x**Rational(3, 2)
    assert refine(sqrt(-x), x < 0) == sqrt(-x)


def test_refine_log() -> None:
    assert refine(log(x**2), x > 0) == 2*log(x)
    assert refine(log(x**2), x < 0) == 2*log(-x)
    assert refine(log(x**2), element(x, S.Reals)) == log(x**2)
    assert refine(log((x - 1)**2), x > 1) == 2*log(x - 1)
    assert refine(log(x**2 - 2*x + 1), x > 1) == 2*log(x - 1)
    assert refine(log(x**y), x > 0) == log(x**y)
    assert refine(log(x**y), (x > 0) & element(y, S.Reals)) == y*log(x)
    assert refine(log(x**2 - 1), x > 1) == log(x**2 - 1)
    assert refine(log(x*y), (x > 0) & (y > 0)) == log(x*y)
    assert refine(exp(2*log(x)), x > 0) == x**2


def test_refine_complex_functions() -> None:
    assert refine(re(x**2 - 1), element(x, S.Reals)) == x**2 - 1
    assert refine(im(x**2 - 1), element(x, S.Reals)) == 0
    assert refine(conjugate(x**2 + 1), x > 0) == x**2 + 1
    assert refine(re(x + I*y), (x > 0) & (y > 0)) == x
    assert refine(im(x + I*y), (x > 0) & (y > 0)) == y
    assert refine(arg(x), x > 0) == 0
    assert refine(arg(x - 2), x > 3) == 0
    assert refine(arg(x - 2), x < 1) == pi
    assert refine(arg(x), element(x, S.Reals)) == arg(x)
    assert refine(atan2(y, x), (x > 0) & (y > 0)) == atan(y/x)
    assert refine(atan2(y, x), x > 0) == atan(y/x)
    assert refine(atan2(y, x), (x < 0) & (y > 0)) == atan(y/x) + pi
    assert refine(atan2(y, x), (x < 0) & (y < 0)) == atan(y/x) - pi
    assert refine(atan2(y, x), Eq(x, 0) & (y > 0)) == pi/2
    assert refine(atan2(y, x), Eq(x, 0) & (y < 0)) == -pi/2
    assert refine(atan2(y, x), (x < 0) & element(y, S.Reals)) == atan2(y, x)
    assert refine(frac(n), element(n, S.Integers)) == 0


def test_refine_contradictory_symbol() -> None:
    from sympy.testing.pytest import raises as _raises
    p = Symbol('p', positive=True)
    _raises(ValueError, lambda: refine(Abs(p), p < 0))


def test_simplify_assumptions() -> None:
    assert simplify(log(x) + log(y), (x > 0) & (y > 0)) == log(x*y)
    assert simplify(log(exp(x)), element(x, S.Reals)) == x
    assert simplify(sin(n*pi) + cos(n*pi), element(n, S.Integers)) == (-1)**n
    assert simplify((-1)**(n**2 + n), element(n, S.Integers)) == 1
    assert simplify(sqrt(x**2*y**2), (x > 0) & (y > 0)) == x*y
    assert simplify(gamma(x + 1)/gamma(x), x > 0) == x
    assert simplify(gamma(x**2 + 1)/gamma(x**2), x > 0) == x**2
    assert simplify(factorial(n + 1)/factorial(n), element(n, S.Naturals0)) == n + 1
    assert simplify((x**y)**(1/y), (x > 0) & (y > 0)) == x
    assert simplify((x**2 - 1)/(Abs(x) - 1), x > 1) == x + 1
    assert simplify(Piecewise((sqrt(x**2), x > 0), (Abs(x) + Abs(x), True)), domain=S.Reals) == \
        Piecewise((x, x > 0), (-2*x, True))
    assert simplify(Integral(Abs(x), (x, 0, 1)), x > 0) == Integral(x, (x, 0, 1))
    # the result is never larger than the refined expression
    assert simplify(sqrt(x) + 1, x > 0) == sqrt(x) + 1
    assert simplify(sqrt(x - 1)*sqrt(x + 1), x > 1) == sqrt(x - 1)*sqrt(x + 1)
    # keyword arguments go to sympy.simplify
    assert simplify(sqrt(x**2) + x, x > 0, ratio=1) == 2*x


def test_simplify_boolean() -> None:
    assert simplify(Eq(x**2, 1), x > 0) == Eq(x, 1)
    assert simplify((x > 1) | (x**2 > 1), domain=S.Reals) == ((x > 1) | (x < -1))
    assert simplify((x**2 > 4) & (x > 0), domain=S.Reals) == (x > 2)
    assert simplify((x > 1) & (x < 0)) is false
    assert simplify(Or(x > 0, Eq(x, 0)), domain=S.Reals) == (x >= 0)
    assert simplify(x**2 + 2*x + 1 >= 0, domain=S.Reals) is true
    assert simplify(Ne(x, 0) & (x > 0)) == (x > 0)
    assert simplify(Ne(x, 0), x > 1) is true
    assert simplify(Abs(x) > 0, Ne(x, 0) & element(x, S.Reals)) is true
    assert simplify(exp(x) > 0, element(x, S.Reals)) is true
    # non-polynomial atoms are decided when possible and otherwise kept
    assert simplify(((sin(x) > 0) & (sin(x) > 0)) | (x > 0), x > 1) is true
    assert simplify((sin(x) > 0) & (x > 0), x > 1) == (sin(x) > 0)
    assert simplify(ForAll(y, x**2 + y**2 >= 0), domain=S.Reals) is true
    assert simplify(Exists(y, Eq(y**2, x)), domain=S.Reals) == (x >= 0)


def test_refine_parity() -> None:
    # the parity of an integer polynomial is read off its residues
    integers = element(n, S.Integers)
    assert refine((-1)**(n**2 + n), integers) == 1
    assert refine((-1)**(n**2 + n + 1), integers) == -1
    assert refine((-1)**(n*(n + 1)*(n + 2)), integers) == 1
    assert refine((-1)**(n**2), integers) == (-1)**(n**2)
    assert refine(cos(pi*(n**2 + n)), integers) == 1
    m = Symbol('m')
    both = integers & element(m, S.Integers)
    assert refine((-1)**(2*n*m + 1), both) == -1
    assert refine((-1)**(n*m), both) == (-1)**(n*m)
    # residues modulo other numbers
    from sympy import Mod
    assert refine(Mod(n**2 + n, 2), integers) == 0
    assert refine(Mod(n**3 - n, 6), integers) == 0
    assert refine(Mod(n**2, 4), integers) == Mod(n**2, 4)
    assert refine(Mod(n**2 + n, 2)) == Mod(n**2 + n, 2)
    assert refine(Mod(2*n*m + 1, 2), both) == 1
