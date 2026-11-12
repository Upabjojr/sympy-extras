"""Tests of the integrals by series expansion on the objects of the core
(:mod:`sympy_extras.integrals.series`): the values against mpmath's
quadrature and series (independent sources), the tabulated Fourier series
against their coefficients, and the agreement with the SymPy face."""
from __future__ import annotations

import mpmath
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import Catalan, Dummy, Integer, Rational, S, Symbol, oo, pi
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, exp, log, sin
from sympy_extras.functions.special.zeta_functions import PolyLog
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.series import (Expansion, _linear_partial_fractions, expansions, fourier_expansion,
                                           fourier_integral, series_integral, sum_series)

x = Symbol('x')
a = Symbol('a', positive=True)
n = Symbol('n', integer=True, positive=True)


def _number(e: Expr) -> complex:
    return complex(e.evalf(30))


def test_fourier_expansions_from_the_table() -> None:
    e = fourier_expansion(log(2 * sin(x / 2)), x, 0, 2 * pi)
    assert isinstance(e, Expansion)
    k = e.index
    assert (e.coefficient, e.basis, e.start, e.constant) == (-1 / k, cos(k * x), 1, 0)
    e = fourier_expansion(log(sin(x)), x, 0, pi)
    assert e is not None and e.constant == -log(2) and e.basis == cos(2 * e.index * x)
    # a positive constant in the logarithm goes into the mean
    e = fourier_expansion(log(3 * sin(x)), x, 0, pi)
    assert e is not None and e.constant == log(3) - log(2)
    assert fourier_expansion(log(sin(x)), x, 0, 2 * pi) is None
    assert fourier_expansion(log(-sin(x)), x, 0, pi) is None
    # the Poisson kernel: numeric P and Q give the root inside the unit circle
    e = fourier_expansion(1 / (5 + 3 * cos(x)), x, 0, 2 * pi)
    assert e is not None and e.coefficient == Rational(-1, 3)**e.index / 2 and e.constant == Rational(1, 4)
    assert fourier_expansion(1 / (3 + 5 * cos(x)), x, 0, pi) is None


def test_every_index_has_a_name_of_its_own() -> None:
    # the typed polynomials map their symbols by name: the indices of two
    # expansions must not be taken for one symbol
    first = fourier_expansion(log(sin(x)), x, 0, pi)
    second = fourier_expansion(log(sin(x)), x, 0, pi)
    assert first is not None and second is not None and first.index.name != second.index.name


def test_sums_against_mpmath() -> None:
    j = Dummy('j', integer=True, nonnegative=True)
    k = Dummy('k', integer=True, positive=True)
    assert sum_series((-1)**j / (2 * j + 1)**2, j, 0) == Catalan
    assert sum_series(1 / (j + 1)**2, j, 0) == pi**2 / 6
    assert sum_series(Integer(2)**j, j, 0) is None
    # the partial fractions diverge separately: the digamma values cancel
    found = sum_series(2 / (k * (4 * k**2 - 1)), k, 1)
    assert found is not None and abs(_number(found) - float(4 * mpmath.log(2) - 2)) < 1e-14
    found = sum_series(1 / (k * (4 * k**2 + 1)), k, 1)
    expected = mpmath.nsum(lambda i: 1 / (i * (4 * i**2 + 1)), [1, mpmath.inf])
    assert found is not None and abs(_number(found) - complex(expected)) < 1e-12
    # the Clausen function
    found = sum_series(sin(k) / k**2, k, 1)
    assert found is not None and found.has(PolyLog) and abs(_number(found) - float(mpmath.clsin(2, 1))) < 1e-12


def test_partial_fractions_over_the_roots() -> None:
    k = Dummy('k', integer=True, positive=True)
    # 1/(k*(4*k**2 + 1)) = 1/k - (1/2)/(k - I/2) - (1/2)/(k + I/2)
    terms = _linear_partial_fractions(1 / (k * (4 * k**2 + 1)), k)
    assert terms is not None and sorted(str(t) for t in terms) == sorted(
        ['(1, 0, 1)', '(-1/2, I/2, 1)', '(-1/2, -I/2, 1)'])
    # a double pole: 1/(k + 1)**2 + 1/(k + 1)
    terms = _linear_partial_fractions((k + 2) / (k + 1)**2, k)
    assert terms is not None and sorted(str(t) for t in terms) == ['(1, -1, 1)', '(1, -1, 2)']
    # a polynomial part has no decomposition into proper terms
    assert _linear_partial_fractions(k**2 / (k + 1), k) is None


def test_fourier_integrals_against_quadrature() -> None:
    assert fourier_integral(log(sin(x)), x, 0, pi) == ConditionalValue(-pi * log(2))
    found = fourier_integral(log(sin(x))**2, x, 0, pi)
    expected = mpmath.quad(lambda t: mpmath.log(mpmath.sin(t))**2, [0, mpmath.pi / 2, mpmath.pi])
    assert found is not None and abs(_number(found.value) - float(expected)) < 1e-12
    assert fourier_integral(log(2 * sin(x / 2)) * cos(n * x), x, 0, 2 * pi) == ConditionalValue(-pi / n)
    assert fourier_integral(log(sin(x)) * cos(3 * x), x, 0, pi) == ConditionalValue(0)
    # a possibly resonant harmonic in the rest is refused
    assert fourier_integral(log(sin(x)) * cos(x)**2, x, 0, pi) is None
    assert fourier_integral(log(sin(x)), x, pi, 0) is None


def test_series_integrals_against_quadrature() -> None:
    found = series_integral(log(1 - x) / x, x, 0, 1)
    assert found == ConditionalValue(-pi**2 / 6)
    found = series_integral(x / (exp(x) - 1), x, 0, oo)
    assert found == ConditionalValue(pi**2 / 6)
    # the terms of exp(-x)*log(1 + x) integrate to a divergent series
    assert series_integral(exp(-x) * log(1 + x), x, 0, oo) is None
    assert series_integral(log(1 - x) / x, x, S.Half, 1) is None


def test_expansions_of_the_factors() -> None:
    found = list(expansions(x / (exp(x) - 1), x, S.Zero, oo))
    assert found and found[0][0].factor == 1 / (exp(x) - 1) and found[0][0].coefficient == 1
    assert found[0][0].start == 1 and found[0][1] == x


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(series_integral)([1], x, 0, 1))


def test_agreement_with_the_face() -> None:
    import sympy
    from sympy_extras.integrals.series_sympy import fourier_integral as face_fourier_integral
    from sympy_extras.interop import to_sympy
    X = sympy.Symbol('x')
    face = face_fourier_integral(sympy.log(sympy.sin(X)) * sympy.log(sympy.cos(X)), X, 0, sympy.pi / 2)
    native = fourier_integral(log(sin(x)) * log(cos(x)), x, 0, pi / 2)
    assert face is not None and native is not None and face.value == to_sympy(native.value)
