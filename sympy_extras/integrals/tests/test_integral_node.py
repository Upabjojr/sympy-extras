"""The constructor of the unevaluated ``Integral`` converts the Python
numbers, as the constructors of ``Sum``, ``Product`` and ``Limit`` do."""
from __future__ import annotations

from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import Float, Integer, Rational, Symbol, srepr, sstr
from sympy_extras.core.basic import Basic
from sympy_extras.core.limits import Limit
from sympy_extras.integrals.integral import Integral

x = Symbol('x')


def _nodes_only(e: Basic) -> bool:
    return isinstance(e, Basic) and all(_nodes_only(a) for a in e.args)


def test_integral_with_python_bounds() -> None:
    # Integral(x, x, 0, 1) stored the ints as they were given: str() failed
    # with "'int' object has no attribute '_sympystr'"
    e = Integral(x, x, 0, 1)
    assert e.bounds == (Integer(0), Integer(1)) and _nodes_only(e)
    assert str(e) == sstr(e) == 'Integral(x, (x, 0, 1))'
    assert srepr(e) == "Integral(Symbol('x'), Symbol('x'), Integer(0), Integer(1))"
    assert e == Integral(x, x, Integer(0), Integer(1)) and hash(e) == hash(Integral(x, x, Integer(0), Integer(1)))
    assert Integral(x, x, Fraction(1, 2), 0.5).bounds == (Rational(1, 2), Float(0.5))
    assert Integral(1, x).function == Integer(1) and _nodes_only(Integral(1, x))
    assert e.doit() == Rational(1, 2)


def test_integral_refuses_what_is_not_a_number() -> None:
    raises(TypeError, lambda: untyped(Integral)('x', x))
    raises(TypeError, lambda: untyped(Integral)(x, 'x'))
    raises(TypeError, lambda: Integral(x, x, 0))


def test_the_other_operators_convert_python_numbers_too() -> None:
    for e, text in ((Sum(x, (x, 0, 1)), 'Sum(x, (x, 0, 1))'), (Sum(1, (x, 0, 1)), 'Sum(1, (x, 0, 1))'),
                    (Product(x, (x, 1, 2)), 'Product(x, (x, 1, 2))'), (Product(2, (x, 1, 2)), 'Product(2, (x, 1, 2))'),
                    (Limit(x, x, 0), "Limit(x, x, 0, dir='+')"), (Limit(1, x, 0), "Limit(1, x, 0, dir='+')")):
        assert _nodes_only(e) and str(e) == text


def test_doit_evaluates_the_integrals_of_the_integrand() -> None:
    # Integral(Integral(x*y, x), y).doit() stayed: the integrators do not look into the inner node
    # (found with sympy/sympy#31017; SymPy's deep doit evaluates it)
    y = Symbol('y')
    nested = Integral(Integral(x*y, x), y).doit()
    assert not nested.has(Integral) and nested.diff(y).diff(x) == x*y
    assert Integral(Integral(x*y, x, 0, y), y, 0, 1).doit() == Rational(1, 8)
    assert Integral(1 + Integral(2*x, x), x, 0, 3).doit() == 12
    # an inner integral which is not found stays, and so does the outer one
    assert Integral(Integral(x**x, x), y).doit().has(Integral)
