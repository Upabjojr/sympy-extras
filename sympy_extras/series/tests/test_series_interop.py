"""The native formal power series and Fourier series converted to SymPy
(:func:`sympy_extras.interop.to_sympy`), and ``Expr.fps`` and
``Expr.fourier_series`` of expressions with wrapped SymPy nodes, which
SymPy computes through the bridge."""
from __future__ import annotations

import sympy
from sympy.series.formal import FormalPowerSeries as SymPyFPS, FormalPowerSeriesProduct as SymPyProduct
from sympy.series.fourier import FiniteFourierSeries as SymPyFinite, FourierSeries as SymPyFourier

from sympy_extras.core import Expr, Symbol, as_expr, pi
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


def _same(native: Expr, theirs: sympy.Basic) -> bool:
    converted = to_sympy(native)
    assert isinstance(converted, sympy.Expr) and isinstance(theirs, sympy.Expr)
    return bool(sympy.simplify(converted - theirs) == 0)


def test_recursive_seq_against_sympy() -> None:
    from sympy.series.sequences import RecursiveSeq as SymPyRecursiveSeq

    from sympy_extras.core import Function
    from sympy_extras.series.sequences import RecursiveSeq
    y, n, a, b = Function('y'), Symbol('n'), Symbol('a'), Symbol('b')
    Yf, N = sympy.Function('y'), sympy.Symbol('n')
    A, B = sympy.symbols('a b')
    batch = [
        (RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1]),
         SymPyRecursiveSeq(Yf(N - 1) + Yf(N - 2), Yf(N), N, [0, 1])),
        (RecursiveSeq(2*y(n - 1) - y(n - 2) + n, y(n), n, [3, -1]),
         SymPyRecursiveSeq(2*Yf(N - 1) - Yf(N - 2) + N, Yf(N), N, [3, -1])),
        (RecursiveSeq(y(n - 1) + a*y(n - 2) - b*y(n - 3), y(n), n, [a, b, a + b]),
         SymPyRecursiveSeq(Yf(N - 1) + A*Yf(N - 2) - B*Yf(N - 3), Yf(N), N, [A, B, A + B])),
        (RecursiveSeq(n*y(n - 1)**2 - y(n - 2), y(n), n, [1, 2]),
         SymPyRecursiveSeq(N*Yf(N - 1)**2 - Yf(N - 2), Yf(N), N, [1, 2])),
    ]
    for native, theirs in batch:
        converted = to_sympy(native)
        assert converted == theirs
        assert str(native) == str(theirs)
        assert native.degree == len(theirs.initial)
        assert to_sympy(native.recurrence) == theirs.recurrence
        assert all(_same(p, q) for p, q in zip(native[:9], theirs[:9]))
        assert _same(native.coeff(7), theirs.coeff(7))
    # order 3 with the default (dummy) initial values: the same terms in them
    native_seq = RecursiveSeq(y(n - 1) + y(n - 3), y(n), n)
    converted_seq = to_sympy(native_seq)
    assert isinstance(converted_seq, SymPyRecursiveSeq)
    assert all(sympy.expand(to_sympy(p) - q) == 0 for p, q in zip(native_seq[:10], converted_seq[:10]))
    # the linear recurrences of the recursive sequences
    for native, theirs in batch[:2]:
        assert [to_sympy(c) for c in native.find_linear_recurrence(12)] == theirs.find_linear_recurrence(12)
    # SymPy 1.14 finds none for symbolic coefficients: it compares the
    # unexpanded products with the terms structurally
    assert batch[2][1].find_linear_recurrence(12) == []
    assert batch[2][0].find_linear_recurrence(12) == [1, a, -b]


def test_find_linear_recurrence_against_sympy() -> None:
    from sympy_extras.core import oo, sqrt
    from sympy_extras.functions.combinatorial.numbers import fibonacci, lucas
    from sympy_extras.series.sequences import SeqPer, sequence
    n, y = Symbol('n'), Symbol('y')
    N, Y = sympy.Symbol('n'), sympy.Symbol('y')
    phi, Phi = (1 + sqrt(5))/2, (1 + sympy.sqrt(5))/2
    batch = [
        (sequence(n**2), sympy.sequence(N**2), 10),
        (sequence(n**3 - 2*n + 7), sympy.sequence(N**3 - 2*N + 7), 12),
        (sequence(23*n**4 + 91*n**2), sympy.sequence(23*N**4 + 91*N**2), 10),
        (sequence(2**n), sympy.sequence(2**N), 10),
        (sequence(3*(-5)**n/7), sympy.sequence(3*(-5)**N/7), 10),
        (sequence(3*5**n + 12), sympy.sequence(3*5**N + 12), 20),
        (sequence(x + y*(-2)**(-n), (n, 0, oo)), sympy.sequence(X + Y*(-2)**(-N), (N, 0, sympy.oo)), 30),
        (sequence(sqrt(5)*(phi**n - (-phi)**(-n))/5), sympy.sequence(sympy.sqrt(5)*(Phi**N - (-Phi)**(-N))/5), 10),
        (sequence((phi**n - (1 - phi)**n)/sqrt(5)), sympy.sequence((Phi**N - (1 - Phi)**N)/sympy.sqrt(5)), 12),
        (sequence(fibonacci(n)), sympy.sequence(sympy.fibonacci(N)), 14),
        (sequence(lucas(n)), sympy.sequence(sympy.lucas(N)), 15),
        (sequence(n*2**n), sympy.sequence(N*2**N), 12),
        (SeqPer((1, 2, 3), (n, 0, oo)), sympy.SeqPer((1, 2, 3), (N, 0, sympy.oo)), 12),
    ]
    for native, theirs, terms in batch:
        coeffs = native.find_linear_recurrence(terms)
        assert [to_sympy(c) for c in coeffs] == theirs.find_linear_recurrence(terms)
        assert [to_sympy(c) for c in native.find_linear_recurrence(terms, 2)] == \
            theirs.find_linear_recurrence(terms, 2)
        found, gf = native.find_linear_recurrence(terms, gfvar=x)
        their_found, their_gf = theirs.find_linear_recurrence(terms, gfvar=X)
        assert [to_sympy(c) for c in found] == their_found
        assert gf is not None and their_gf is not None
        assert sympy.cancel(to_sympy(gf) - their_gf) == 0
    assert sequence(n**2).find_linear_recurrence(10, 2, x) == ([], None)
