"""The native ComplexRootOf against SymPy's CRootOf: the order of the
roots, the isolating rectangles and the conversion both ways."""
import random
from fractions import Fraction

import sympy
from sympy.core.cache import clear_cache
from sympy.polys.domains import ZZ as SymPyZZ
from sympy.polys.rootisolation import dup_isolate_complex_roots_sqf

from sympy_extras.core import Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.complexrootof import ComplexRootOf, complex_rectangles, polynomial_roots
from sympy_extras.polys.numberfields.qqbar import algebraic_roots

X = sympy.Symbol('x')
x = Symbol('x')

Box = tuple[Fraction, Fraction, Fraction, Fraction]


def _fraction(q: object) -> Fraction:
    return Fraction(str(q))


def _polynomials(seed: int, count: int) -> list[sympy.Poly]:
    """Random polynomials: dense ones, sparse ones (roots on the axes and
    on the lines of the cuts), and products of quadratics with dyadic
    roots (equal real parts, roots on the cuts), some reducible."""
    rng = random.Random(seed)
    found: list[sympy.Poly] = []
    while len(found) < count:
        kind = rng.random()
        if kind < 0.3:
            degree = rng.randint(2, 7)
            coefficients = [1] + [rng.choice([0, 0, 1, -1, 2, 4, 6, -3]) for _ in range(degree - 1)] \
                + [rng.choice([1, 2, 4, -1])]
            p: sympy.Expr = sympy.Poly(coefficients, X).as_expr()
        elif kind < 0.5:
            p = sympy.Integer(1)
            for _ in range(rng.randint(1, 3)):
                a = rng.choice([0, 1, -1, 2, -2, sympy.Rational(1, 2)])
                b = rng.choice([1, 2, 4, sympy.Rational(1, 2), 3])
                p = p*((X - a)**2 + b**2)
            if rng.random() < 0.5:
                p = p*(X**3 - X - 1)
        else:
            degree = rng.randint(2, 7)
            coefficients = [rng.randint(-9, 9) for _ in range(degree + 1)]
            coefficients[0] = coefficients[0] or 1
            p = sympy.Poly(coefficients, X).as_expr()
            if rng.random() < 0.3:
                p = p*sympy.Poly([rng.randint(-5, 5) for _ in range(3)] + [1], X).as_expr()
        poly = sympy.Poly(p, X)
        if poly.degree() >= 1:
            found.append(poly)
    return found


def _sympy_rectangles(f: list[int]) -> list[Box]:
    found = dup_isolate_complex_roots_sqf(f, SymPyZZ, blackbox=True)
    return [(_fraction(r.ax), _fraction(r.ay), _fraction(r.bx), _fraction(r.by)) for r in found if not r.conj]


def _native(e: object) -> Expr:
    converted = from_sympy(sympy.sympify(e))
    assert isinstance(converted, Expr)
    return converted


def test_rectangles_and_order_as_sympy() -> None:
    for poly in _polynomials(7, 40):
        clear_cache()
        square_free = sympy.Poly(sympy.sqf_part(poly.as_expr()), X).clear_denoms()[1].primitive()[1]
        f = [int(c) for c in square_free.all_coeffs()]
        upper = [r for r in algebraic_roots(f) if not r.is_real and r.enclosure(Fraction(1))[3] > 0]
        assert [box for box, _ in complex_rectangles(f, upper)] == _sympy_rectangles(f), f
        mine = polynomial_roots([_fraction(c) for c in poly.all_coeffs()], x, radicals=False)
        clear_cache()
        theirs = poly.all_roots(radicals=False)
        assert mine == [_native(r) for r in theirs], poly
        clear_cache()
        assert polynomial_roots([_fraction(c) for c in poly.all_coeffs()], x) == [_native(r) for r in poly.all_roots()]


def test_values_as_sympy() -> None:
    for f in [X**5 - X - 1, X**4 + 6*X**2 + 4]:
        for k in range(sympy.degree(f, X)):
            clear_cache()
            r = sympy.CRootOf(f, k)
            native = from_sympy(r)
            assert isinstance(native, ComplexRootOf)
            assert sympy.Abs(to_sympy(native.evalf(30)) - sympy.N(r, 30)) < sympy.Float('1e-28')


def test_conversion_both_ways() -> None:
    clear_cache()
    for e in [sympy.CRootOf(X**5 - X - 1, 2), sympy.CRootOf(X**5 - X - 1, 0), sympy.CRootOf(X**2 + 16, 0),
              1 + sympy.CRootOf(X**3 - X - 1, 1)**2]:
        native = from_sympy(e)
        assert to_sympy(native) == e
        assert str(native) == str(e)
    node = from_sympy(sympy.CRootOf(X**5 - X - 1, 3))
    assert isinstance(node, ComplexRootOf) and node.index == 3
    assert to_sympy(ComplexRootOf(x**5 - x - 1, 3)) == sympy.CRootOf(X**5 - X - 1, 3)
    # SymPy's own cache returns the root of an equal PurePoly in an earlier
    # generator (CRootOf(y**3 - y - 1, 0) printed with x after a CRootOf in x):
    # the native node keeps its own variable and is equal to it
    y = Symbol('y')
    assert ComplexRootOf(y**3 - y - 1, 0) == ComplexRootOf(x**3 - x - 1, 0)
    assert str(ComplexRootOf(y**3 - y - 1, 0)) == 'CRootOf(y**3 - y - 1, 0)'
