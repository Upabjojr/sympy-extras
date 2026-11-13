"""The native formal power series and Fourier series converted to SymPy
(:func:`sympy_extras.interop.to_sympy`), and ``Expr.fps`` and
``Expr.fourier_series`` of expressions with wrapped SymPy nodes, which
SymPy computes through the bridge."""
from __future__ import annotations

import sympy
from sympy.series.formal import FormalPowerSeries as SymPyFPS, FormalPowerSeriesProduct as SymPyProduct
from sympy.series.fourier import FiniteFourierSeries as SymPyFinite, FourierSeries as SymPyFourier

from sympy_extras.core import Symbol, as_expr, pi
from sympy_extras.functions import exp, sin
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.series.formal import FormalPowerSeries, fps
from sympy_extras.series.fourier import fourier_series

x = Symbol('x')
X = sympy.Symbol('x')


def test_fps_to_sympy() -> None:
    s = fps(sin(x))
    assert isinstance(s, FormalPowerSeries)
    converted = to_sympy(s)
    assert isinstance(converted, SymPyFPS)
    assert converted.truncate(8) == sympy.sin(X).series(X, 0, 8)
    assert converted.function == sympy.sin(X)
    e = fps(exp(x))
    assert isinstance(e, FormalPowerSeries)
    product = to_sympy(e.product(s, x))
    assert isinstance(product, SymPyProduct)
    assert product.truncate(4) == (sympy.exp(X)*sympy.sin(X)).series(X, 0, 4)
    # the polynomial case: the coefficients by the Coeff function of both
    polynomial = to_sympy(fps(x**3 + 2*x + 1))
    assert isinstance(polynomial, SymPyFPS)
    assert polynomial.truncate() == X**3 + 2*X + 1 + sympy.O(X**6)


def test_fourier_series_to_sympy() -> None:
    converted = to_sympy(fourier_series(x**2, (x, -pi, pi)))
    assert isinstance(converted, SymPyFourier)
    assert converted.truncate() == -4*sympy.cos(X) + sympy.cos(2*X) + sympy.pi**2/3
    finite = to_sympy(fourier_series(sin(x)**2))
    assert isinstance(finite, SymPyFinite)
    assert finite.truncate() == sympy.S.Half - sympy.cos(2*X)/2


def test_foreign_expressions_go_through_the_bridge() -> None:
    Y = sympy.Symbol('y')
    constant = as_expr(from_sympy(sympy.Integral(sympy.exp(-Y**2), (Y, 0, 1))))
    assert constant._has_foreign_class()
    e = constant * as_expr(from_sympy(sympy.sin(X)))
    found = to_sympy(e.fps(x))
    assert isinstance(found, SymPyFPS)
    assert sympy.simplify(found.truncate(4).removeO()
                          - sympy.Integral(sympy.exp(-Y**2), (Y, 0, 1))*(X - X**3/6)) == 0
    series = to_sympy((constant * x).fourier_series((x, -pi, pi)))
    assert isinstance(series, SymPyFourier)
