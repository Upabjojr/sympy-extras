"""Tests of the port of SymPy's Fourier series on the typed core: SymPy's
own tests (``sympy/series/tests/test_fourier.py``; the expected series
are facts, hand computations), and the differences of the port."""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/tests/test_fourier.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the core's objects; the sums compared by their difference.
# ---------------------------------------------------------------------
from __future__ import annotations

from functools import lru_cache

from pytest import raises

from sympy_extras.core import Add, Expr, Integer, Rational, S, Symbol, oo, pi
from sympy_extras.core.function import expand
from sympy_extras.functions import cos, exp, log, piecewise, sin, sinc, tan
from sympy_extras.polys.polytools import cancel
from sympy_extras.series.fourier import FiniteFourierSeries, FourierSeries, fourier_series

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def _fourier(e: Expr) -> FourierSeries:
    assert isinstance(e, FourierSeries), e
    return e


def _finite(e: Expr) -> FiniteFourierSeries:
    assert isinstance(e, FiniteFourierSeries), e
    return e


def _same(a: Expr, b: Expr) -> bool:
    return cancel(expand(a - b)) == 0


@lru_cache()
def _get_examples() -> tuple[FourierSeries, FourierSeries, FourierSeries]:
    fo = _fourier(fourier_series(x, (x, -pi, pi)))
    fe = _fourier(fourier_series(x**2, (-pi, pi)))
    fp = _fourier(fourier_series(piecewise((0, x < 0), (pi, True)), (x, -pi, pi)))
    return fo, fe, fp


def test_FourierSeries() -> None:
    fo, fe, fp = _get_examples()

    assert fourier_series(1, (-pi, pi)) == 1
    assert _fourier(piecewise((0, x < 0), (pi, True)).fourier_series((x, -pi, pi))).truncate() == fp.truncate()
    assert fo.function == x
    assert fo.x == x
    assert fo.period == (-pi, pi)

    assert fo.term(3) == 2*sin(3*x) / 3
    assert fe.term(3) == -4*cos(3*x) / 9
    assert fp.term(3) == 2*sin(3*x) / 3

    assert fo.as_leading_term(x) == 2*sin(x)
    assert fe.as_leading_term(x) == pi**2 / 3
    assert fp.as_leading_term(x) == pi / 2

    assert fo.truncate() == 2*sin(x) - sin(2*x) + (2*sin(3*x) / 3)
    assert fe.truncate() == -4*cos(x) + cos(2*x) + pi**2 / 3
    assert fp.truncate() == 2*sin(x) + (2*sin(3*x) / 3) + pi / 2

    fot = fo.truncate(n=None)
    s = [S.Zero, 2*sin(x), -sin(2*x)]
    for i, t in enumerate(fot):
        if i == 3:
            break
        assert s[i] == t

    def _check_iter(f: FourierSeries, i: int) -> None:
        for ind, t in enumerate(f):
            assert t == f[ind]
            if ind == i:
                break

    _check_iter(fo, 3)
    _check_iter(fe, 3)
    _check_iter(fp, 3)

    assert fo.subs(x, x**2) == fo

    raises(ValueError, lambda: fourier_series(x, (0, 1, 2)))
    raises(ValueError, lambda: fourier_series(x, (x, 0, oo)))
    raises(ValueError, lambda: fourier_series(x*y, (0, oo)))


def test_FourierSeries_2() -> None:
    p = piecewise((0, x < 0), (x, True))
    f = _fourier(fourier_series(p, (x, -2, 2)))
    assert _same(f.term(3), 2*sin(3*pi*x / 2) / (3*pi) - 4*cos(3*pi*x / 2) / (9*pi**2))
    assert _same(f.truncate(), 2*sin(pi*x / 2) / pi - sin(pi*x) / pi - 4*cos(pi*x / 2) / pi**2 + S.Half)


def test_square_wave() -> None:
    """The Fourier series approximates a discontinuous function."""
    square_wave = piecewise((1, x < pi), (-1, True))
    s = _fourier(fourier_series(square_wave, (x, 0, 2*pi)))
    assert _same(s.truncate(3), 4 / pi * sin(x) + 4 / (3 * pi) * sin(3 * x) + 4 / (5 * pi) * sin(5 * x))
    assert _same(s.sigma_approximation(4), 4 / pi * sin(x) * sinc(pi / 4) + 4 / (3 * pi) * sin(3 * x) * sinc(3 * pi / 4))


def test_sawtooth_wave() -> None:
    s = _fourier(fourier_series(x, (x, 0, pi)))
    assert _same(s.truncate(4), pi/2 - sin(2*x) - sin(4*x)/2 - sin(6*x)/3)
    s = _fourier(fourier_series(x, (x, 0, 1)))
    assert _same(s.truncate(4), S.Half - sin(2*pi*x)/pi - sin(4*pi*x)/(2*pi) - sin(6*pi*x)/(3*pi))


def test_FourierSeries__operations() -> None:
    fo, fe, fp = _get_examples()

    fes = _fourier(_fourier(fe.scale(-1)).shift(pi**2))
    assert fes.truncate() == 4*cos(x) - cos(2*x) + 2*pi**2 / 3
    assert _fourier(fp.shift(-pi/2)).truncate() == 2*sin(x) + (2*sin(3*x) / 3) + (2*sin(5*x) / 5)
    fos = _fourier(fo.scale(3))
    assert fos.truncate() == 6*sin(x) - 3*sin(2*x) + 2*sin(3*x)
    fx = _fourier(_fourier(fe.scalex(2)).shiftx(1))
    assert fx.truncate() == -4*cos(2*x + 2) + cos(4*x + 4) + pi**2 / 3
    fl = _fourier(_fourier(_fourier(_fourier(_fourier(fe.scalex(3)).shift(-pi)).scalex(2)).shiftx(1)).scale(4))
    assert fl.truncate() == -16*cos(6*x + 6) + 4*cos(12*x + 12) - 4*pi + 4*pi**2 / 3

    raises(ValueError, lambda: fo.shift(x))
    raises(ValueError, lambda: fo.shiftx(sin(x)))
    raises(ValueError, lambda: fo.scale(x*y))
    raises(ValueError, lambda: fo.scalex(x**2))


def test_FourierSeries__neg() -> None:
    fo, fe, fp = _get_examples()
    assert _fourier(-fo).truncate() == -2*sin(x) + sin(2*x) - (2*sin(3*x) / 3)
    assert _fourier(-fe).truncate() == +4*cos(x) - cos(2*x) - pi**2 / 3


def test_FourierSeries__add__sub() -> None:
    fo, fe, fp = _get_examples()

    assert fo + fo == fo.scale(2)
    assert fo - fo == 0
    assert -fe - fe == fe.scale(-2)

    assert _fourier(fo + fe).truncate() == 2*sin(x) - sin(2*x) - 4*cos(x) + cos(2*x) + pi**2 / 3
    assert _fourier(fo - fe).truncate() == 2*sin(x) - sin(2*x) + 4*cos(x) - cos(2*x) - pi**2 / 3

    assert isinstance(fo + 1, Add)

    raises(ValueError, lambda: fo + fourier_series(x, (x, 0, 2)))


def test_FourierSeries_finite() -> None:
    assert _finite(fourier_series(sin(x))).truncate(1) == sin(x)
    assert _finite(fourier_series(sin(x)*log(y)*exp(z), (x, pi, -pi))).truncate() == sin(x)*log(y)*exp(z)
    assert _finite(fourier_series(sin(x)**6)).truncate(oo) == \
        -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 + Rational(5, 16)
    assert _finite(fourier_series(sin(x) ** 6)).truncate() == \
        -15 * cos(2 * x) / 32 + 3 * cos(4 * x) / 16 + Rational(5, 16)
    assert _finite(fourier_series(sin(4*x+3) + cos(3*x+4))).truncate(oo) == \
        -sin(4)*sin(3*x) + sin(4*x)*cos(3) + cos(4)*cos(3*x) + sin(3)*cos(4*x)
    assert _finite(fourier_series(sin(x)+cos(x)*tan(x))).truncate(oo) == 2*sin(x)
    assert _finite(fourier_series(cos(pi*x), (x, -1, 1))).truncate(oo) == cos(pi*x)
    assert _finite(fourier_series(cos(3*pi*x + 4) - sin(4*pi*x)*log(pi*y), (x, -1, 1))).truncate(oo) == \
        -log(pi*y)*sin(4*pi*x) - sin(4)*sin(3*pi*x) + cos(4)*cos(3*pi*x)


def test_finite_operations_keep_every_term() -> None:
    # SymPy's FiniteFourierSeries.scale, shiftx and scalex rebuild the
    # series from truncate(), its first three nonzero terms: the scaled
    # series of sin(x)**6 lost its term cos(6*x)
    s = _finite(fourier_series(sin(x)**6))
    assert _same(_finite(s.scale(2)).truncate(oo), 2*s.truncate(oo))
    assert _same(_finite(s.scalex(2)).truncate(oo), s.truncate(oo).subs(x, 2*x))
    assert _same(_finite(s.shiftx(1)).truncate(oo).expand(trig=True), s.truncate(oo).subs(x, x + 1).expand(trig=True))
    assert _same(_finite(-s).truncate(oo), -s.truncate(oo))


def test_finite_sum() -> None:
    # the sum of two finite series is the finite series of the sum
    s = _finite(fourier_series(sin(x)**2)) + _finite(fourier_series(cos(2*x)))
    assert _same(_finite(s).truncate(oo), S.Half + cos(2*x)/2)
    assert _finite(fourier_series(sin(x))) + _finite(fourier_series(-sin(x))) == 0


def test_coefficients_converge_to_the_function() -> None:
    # an independent check: the partial sums at a point inside the interval
    # approach the function (x**2 on (-pi, pi) at x = 1, error O(1/n))
    fe = _get_examples()[1]
    partial = add_terms(fe, 60)
    assert abs(float(partial.subs(x, Integer(1)).evalf(20)) - 1.0) < 1e-3


def add_terms(s: FourierSeries, n: int) -> Expr:
    return Add(*s[:n]).doit()
