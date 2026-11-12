"""Tests of the native ComplexRootOf (no SymPy: the properties of the roots
and their order, checked by their values and by hand computations)."""
from fractions import Fraction

from sympy_extras._testing import untyped
from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.functions import sqrt
from sympy_extras.polys.cad.indexedroot import IndexedRoot
from sympy_extras.polys.complexrootof import (ComplexRootOf, algebraic_number, all_roots, complex_rectangles,
                                              polynomial_roots, real_roots, root_count, rootof)
from sympy_extras.polys.numberfields.minpoly import minimal_polynomial
from sympy_extras.polys.numberfields.qqbar import algebraic_roots
from sympy_extras.polys.roots import polynomial_root, radical_form
from sympy.testing.pytest import raises

x, y = Symbol('x'), Symbol('y')


def _value(e: Expr) -> complex:
    found = numeric_value(e, 120)
    assert found is not None
    return complex(found)


def _residual(f: Expr, r: Expr) -> float:
    return abs(_value(f.xreplace({x: r})))


def test_roots_are_roots() -> None:
    for f in [x**5 - x - 1, x**4 + 1, x**6 + x + 1, 3*x**7 - 5*x**3 + 2*x - 11, (x**2 + 1)*(x**3 - x - 1)]:
        found = all_roots(f, radicals=False)
        assert len(found) == Poly_degree(f)
        assert all(_residual(f, r) < 1e-25 for r in found)
        values = [_value(r) for r in found]
        # distinct roots of a square-free polynomial
        assert all(abs(values[i] - values[j]) > 1e-10 for i in range(len(values)) for j in range(i))


def Poly_degree(f: Expr) -> int:
    from sympy_extras.polys.polytools import Poly
    return Poly(f, x).degree()


def test_order_real_first_then_conjugate_pairs() -> None:
    found = all_roots(x**5 - x - 1, radicals=False)
    values = [_value(r) for r in found]
    assert values[0].imag == 0 and abs(values[0].real - 1.1673039782614187) < 1e-14
    # the conjugate below the real axis first
    for k in (1, 3):
        assert values[k].imag < 0 and abs(values[k].conjugate() - values[k + 1]) < 1e-14
    # the pairs of x**5 - x - 1 by real part (SymPy: -0.76..., then 0.18...)
    assert values[1].real < values[3].real


def test_roots_on_the_imaginary_axis() -> None:
    # x**4 + 6*x**2 + 4: the roots i*sqrt(3 -+ sqrt(5)) lie on the first
    # vertical cut; SymPy's rectangles are [0, 3] x [0, 3/2] and [0, 3] x [3/2, 3]
    f = [1, 0, 6, 0, 4]
    upper = [r for r in algebraic_roots(f) if not r.is_real and r.enclosure(Fraction(1))[3] > 0]
    boxes = [box for box, _ in complex_rectangles(f, upper)]
    assert boxes == [(Fraction(0), Fraction(0), Fraction(3), Fraction(3, 2)),
                     (Fraction(0), Fraction(3, 2), Fraction(3), Fraction(3))]
    values = [_value(ComplexRootOf(x**4 + 6*x**2 + 4, k)) for k in range(4)]
    # the real parts are exactly zero
    assert all(v.real == 0 for v in values)
    assert [round(v.imag, 6) for v in values] == [-0.874032, 0.874032, -2.288246, 2.288246]
    assert ComplexRootOf(x**4 + 6*x**2 + 4, 1).is_imaginary is True
    assert ComplexRootOf(x**5 - x - 1, 1).is_imaginary is False


def test_factors_in_sympys_order() -> None:
    # the complex roots factor by factor (SymPy: -I, I, -2*I, 2*I), not by value
    assert all_roots((x**2 + 4)*(x**2 + 1), radicals=False) == [
        ComplexRootOf(x**2 + 1, 0), ComplexRootOf(x**2 + 1, 1), ComplexRootOf(x**2 + 4, 0), ComplexRootOf(x**2 + 4, 1)]
    # the multiplicities repeat the roots
    assert all_roots((x - 1)**2*(x**2 + 2)**2) == [Integer(1), Integer(1), -sqrt(2)*I, -sqrt(2)*I, sqrt(2)*I, sqrt(2)*I]
    assert real_roots((x**3 - x - 1)*(x**2 - 2)) == [-sqrt(2), ComplexRootOf(x**3 - x - 1, 0), sqrt(2)]
    assert root_count((x**3 - x - 1)*(x**2 + 1)) == (1, 5)


def test_rootof_evaluates() -> None:
    assert rootof(2*x - 3, 0) == Rational(3, 2)
    assert rootof(x**2 - 2, 1) == sqrt(2)
    assert rootof(x**2 - 2, 1, radicals=False) == ComplexRootOf(x**2 - 2, 1)
    # the content and the sign are taken out, the root is that of its factor
    assert rootof(-2*x**3 + 2*x + 2, 0) == ComplexRootOf(x**3 - x - 1, 0)
    assert rootof((x - 3)*(x**3 - x - 1), 0) == ComplexRootOf(x**3 - x - 1, 0)
    assert rootof((x - 3)*(x**3 - x - 1), 1) == 3
    assert rootof(x**3 - x - 1, -1) == ComplexRootOf(x**3 - x - 1, 2)
    # an integer basis of the roots is taken out, as SymPy's preprocess_roots does
    assert rootof(x**2 + 16, 0, radicals=False) == 4*ComplexRootOf(x**2 + 1, 0)
    raises(IndexError, lambda: rootof(x**3 - x - 1, 3))
    raises(ValueError, lambda: rootof(Integer(3), 0, x))
    raises(NotImplementedError, lambda: rootof(x**2 - sqrt(2), 0, x))


def test_node_does_not_evaluate() -> None:
    r = ComplexRootOf(x**2 - 4, 1)
    assert isinstance(r, ComplexRootOf) and r.doit() == r
    assert r.is_canonical() is False and ComplexRootOf(x**3 - x - 1, 2).is_canonical() is True
    assert abs(_value(r) - 2) < 1e-30
    assert r._evaluated(r.args) == Integer(2)
    raises(ValueError, lambda: ComplexRootOf(x*y - 1, 0))
    raises(IndexError, lambda: ComplexRootOf(x**2 - 2, 2))
    raises(TypeError, lambda: untyped(ComplexRootOf)(3, 0))


def test_bound_variable() -> None:
    r = ComplexRootOf(x**3 - x - 1, 1)
    assert r == ComplexRootOf(y**3 - y - 1, 1) and hash(r) == hash(ComplexRootOf(y**3 - y - 1, 1))
    assert r.free_symbols == set() and r.is_number
    assert r.subs(x, 2) == r and r.xreplace({x: y}) == r
    assert r.subs(r, 3) == 3
    assert str(r) == 'CRootOf(x**3 - x - 1, 1)'
    assert r.func(*r.args) == r


def test_facts() -> None:
    real = ComplexRootOf(x**3 - x - 1, 0)
    assert real.is_real and real.is_positive and not real.is_rational and real.is_algebraic
    negative = ComplexRootOf(x**3 - x + 1, 0)
    assert negative.is_negative
    other = ComplexRootOf(x**3 - x - 1, 1)
    assert other.is_real is False and other.is_zero is False and other.is_finite and other.is_algebraic


def test_evalf() -> None:
    r = ComplexRootOf(x**5 - x - 1, 2)
    assert str(r.evalf(10)) == '-0.7648844336 + 0.352471546*I'
    # many digits, exactly refined
    value = numeric_value(r, 400)
    assert value is not None
    assert abs(complex(value) - complex(-0.7648844336005847, 0.35247154603172626)) < 1e-15
    assert abs(_value(ComplexRootOf(x**2 + 4, 1)) - 2j) < 1e-30


def test_minimal_polynomial_and_radicals() -> None:
    r = ComplexRootOf(x**5 - x - 1, 3)
    assert minimal_polynomial(r, y) == y**5 - y - 1
    assert minimal_polynomial(r + 1, y) == y**5 - 5*y**4 + 10*y**3 - 10*y**2 + 4*y - 1
    assert polynomial_root(r) is not None
    assert radical_form(r) is None
    # a root of a quadratic is written in radicals; that of a cyclotomic factor too
    assert radical_form(ComplexRootOf(x**2 + 1, 1)) == I
    assert radical_form(ComplexRootOf(x**4 + 1, 3)) == sqrt(2)/2 + sqrt(2)*I/2
    # the IndexedRoot of the CAD has its minimal polynomial too
    assert minimal_polynomial(IndexedRoot(x**3 - x - 1, x, 0), y) == y**3 - y - 1


def test_algebraic_number() -> None:
    r = ComplexRootOf(x**3 - 2, 1)
    a = algebraic_number(r)
    assert a is not None and a.minpoly == [1, 0, 0, -2] and not a.is_real
    assert algebraic_number(I) == algebraic_number(ComplexRootOf(x**2 + 1, 1))
    b = algebraic_number(ComplexRootOf(x**3 - 2, 2))
    assert b is not None and a.conjugate() == b
    assert algebraic_number(x) is None


def test_polynomial_roots() -> None:
    found = polynomial_roots([Fraction(1), Fraction(0), Fraction(-1), Fraction(-1)], x)
    assert found == [ComplexRootOf(x**3 - x - 1, k) for k in range(3)]
    assert polynomial_roots([Fraction(1, 2), Fraction(0), Fraction(-1)], x) == [-sqrt(2), sqrt(2)]
    assert polynomial_roots([Fraction(1), Fraction(0), Fraction(1)], x, real=True) == []
