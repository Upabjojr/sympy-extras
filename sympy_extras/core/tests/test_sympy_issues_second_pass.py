"""Regression tests of the second pass over the reports against SymPy
(sympy/sympy#31275, #31017, #30902, #30900, #30844) in the core: the
parser, ``Derivative``, the printing of ``Order`` and of unevaluated
function nodes, the series of ``Abs`` and the powers on their cut."""
from __future__ import annotations

import mpmath
import pytest
import sympy
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import (Derivative, Function, I, Integer, O, Rational, Symbol, SympifyError, evaluate,
                               oo, sqrt, srepr, sympify)
from sympy_extras.core.limits import Limit, limit
from sympy_extras.core.parsing import ParseUnsupported, parse
from sympy_extras.functions import Abs, Max, abs_, legendre, sin
from sympy_extras.functions.elementary.trigonometric import Sin
from sympy_extras.functions.special.polynomials import Legendre
from sympy_extras.integrals.integral import Integral
from sympy_extras.interop import to_sympy

x, y, a, c = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('c')
f = Function('f')


def test_parser_reads_the_unevaluated_nodes() -> None:
    # sympify('Integral(x**2,(x,0,1))') raised SympifyError "cannot parse" without the bridge:
    # Integral, Sum, Product, Limit and Derivative were outside the grammar
    assert parse('Integral(x**2,(x,0,1))') == Integral(x**2, x, 0, 1)
    assert parse('Integral(x**2, x)') == parse('Integral(x**2, (x,))') == parse('Integral(x**2)') == Integral(x**2, x)
    assert parse('Integral(x, [x, 0, 1])') == Integral(x, x, 0, 1)
    assert parse('Sum(x**2,(x,0,1))') == Sum(x**2, (x, 0, 1))
    assert parse('Sum(x*y, (x, 0, 1), (y, 1, oo))') == Sum(x*y, (x, 0, 1), (y, 1, oo))
    assert parse('Product(x,(x,1,3))') == Product(x, (x, 1, 3))
    assert parse('Limit(sin(x)/x, x, 0)') == Limit(sin(x)/x, x, 0)
    assert parse("Limit(1/x, x, 0, '-')") == parse('Limit(1/x, x, 0, dir="-")') == Limit(1/x, x, 0, '-')
    assert parse("Limit(1/x, x, 0, dir='+-')") == Limit(1/x, x, 0, '+-')
    assert parse('Limit(x, x, oo)') == Limit(x, x, oo)
    assert parse('Derivative(x**2, x)') == parse('Derivative(x**2)') == Derivative(x**2, x)
    assert parse('Derivative(x**3, x, 2)') == parse('Derivative(x**3, (x, 2))') == Derivative(x**3, (x, 2))
    assert parse('Derivative(x*y, x, 2, y)') == Derivative(x*y, (x, 2), y)
    assert parse('Derivative(x*y, x, y, 2)') == Derivative(x*y, x, (y, 2))
    # inside an expression, and the parentheses of a factor are not a tuple
    assert parse('2*Integral((x + 1)*y, (x, 0, 1)) + 1') == 2*Integral((x + 1)*y, x, 0, 1) + 1
    assert parse('Integral((x), (x, 0, (1)))') == Integral(x, x, 0, 1)
    assert sympify('Integral(x**2,(x,0,1))') == Integral(x**2, x, 0, 1)
    # the values are those of the nodes
    assert parse('Integral(x**2,(x,0,1))').doit() == Rational(1, 3)
    assert parse('Sum(x**2,(x,0,3))').doit() == 14 and parse('Derivative(x**3, x, 2)').doit() == 6*x


def test_parser_nodes_as_sympy_constructs_them() -> None:
    # SymPy's Integral(f, x, y) integrates over x first: the nested integrals here
    nested = parse('Integral(x*y, (x, 0, y), (y, 0, 1))')
    assert nested == Integral(Integral(x*y, x, 0, y), y, 0, 1)
    assert to_sympy(nested) == sympy.sympify('Integral(x*y, (x, 0, y), (y, 0, 1))')
    # SymPy's constructors: an order 0 dropped, derivatives and sums of the same kind merged
    assert parse('Derivative(x**2, x, 0)') == x**2
    assert parse('Derivative(Derivative(f(x), x), x)') == Derivative(f(x), (x, 2))
    assert parse('Sum(Sum(x*y,(x,0,1)),(y,1,2))') == Sum(x*y, (x, 0, 1), (y, 1, 2))
    for text in ['Integral(x**2,(x,0,1))', 'Sum(x*y,(x,0,1),(y,1,2))', 'Product(x,(x,1,3))', 'Limit(sin(x)/x, x, 0)',
                 "Limit(1/x, x, 0, '-')", 'Derivative(x**3, x, 2)', 'Derivative(x*y, x, y, 2)', 'Integral(x)',
                 'Derivative(f(x, y), y, x)', 'Limit(x, x, oo)', 'Integral(x*y, x, y)']:
        assert to_sympy(parse(text)) == sympy.sympify(text), text
    # without evaluation the classes, which build what they are given
    with evaluate(False):
        inner = Sum(x*y, (x, 0, 1))
    assert parse('Sum(Sum(x*y,(x,0,1)),(y,1,2))', evaluate=False) == Sum(inner, (y, 1, 2))
    assert parse('Derivative(x**2, x, 0)', evaluate=False) == Derivative(x**2, (x, 0))


def test_parser_refuses_the_forms_the_nodes_have_not() -> None:
    # a sum without bounds, a symbolic order, a variable which is not a symbol, a direction which
    # is none: outside the grammar, not a node of another meaning ((x, a) of an integral was in
    # this list: it is the substitution in the indefinite integral since, test_parsing)
    for text in ['Integral(x, (x, 1, 2, 3))', 'Sum(x, x)', 'Sum(x)', 'Derivative(f(x), (x, n))', 'Derivative(f(x), f(x))',
                 'Derivative(x, 2)', "Limit(1/x, x, 0, 'up')", 'Limit(1/x, x, x)', 'Limit(1/x, x)', 'Integral(x*y)',
                 "Integral(x, 'x')", 'Integral(x, y=1)', 'Limit()', 'Derivative(x**2, x, -1)', 'x = 1', "'x'",
                 '[x, 1]', 'f((x, 1))']:
        with pytest.raises(ParseUnsupported):
            parse(text)


def test_derivative_accepts_a_count_after_a_variable() -> None:
    # Derivative(x, x, 2) raised TypeError: 'int' object is not subscriptable
    assert Derivative(x**3, x, 2) == Derivative(x**3, (x, 2)) == Derivative(x**3, x, x)
    assert Derivative(x**3, x, Integer(2)).variable_count == ((x, 2),)
    assert Derivative(x*y, x, 2, y, 3).variable_count == ((x, 2), (y, 3))
    assert Derivative(x**3, x, 2).doit() == 6*x and Derivative(x, x, 2).doit() == 0
    assert str(Derivative(x, x, 2)) == 'Derivative(x, (x, 2))'
    # SymPy's errors: a count which follows no variable, a negative order
    raises(TypeError, lambda: untyped(Derivative)(x**3, 2))
    raises(TypeError, lambda: Derivative(x**3, (x, 2), 3))
    raises(ValueError, lambda: Derivative(x**3, x, -1))
    raises(ValueError, lambda: Derivative(x**3, (x, -1)))


def test_order_prints_a_variable_its_expression_has_not() -> None:
    # str(O(1, x)) was 'O(1)' (SymPy's too, sympy/sympy#30844), which is the order without variables
    assert str(O(1, x)) == 'O(1, x)' and str(O(1)) == 'O(1)' and O(1, x) != O(1)
    assert str(O(x**2)) == 'O(x**2)' and str(O(x*y)) == 'O(x*y, x, y)' and str(O(x, (x, oo))) == 'O(x, (x, oo))'
    assert str(sin(x).series(x, 0, 0)) == 'O(1, x)'
    for e in [O(1, x), O(x**2), O(x*y)]:
        assert sympify(str(e)) == e


def test_srepr_of_a_function_node_which_is_not_evaluated() -> None:
    # srepr(Sin(0)) was "sin(Integer(0))", the call which gives 0: the class builds the node
    assert srepr(Sin(Integer(0))) == 'Sin(Integer(0))' and srepr(Sin(Integer(0)) + 1) == 'Add(Sin(Integer(0)), Integer(1))'
    assert srepr(Legendre(Integer(2), x)) == "Legendre(Integer(2), Symbol('x'))"
    assert srepr(Abs(Integer(-1))) == 'Abs(Integer(-1))'
    # the nodes the functions return are printed with the functions, as SymPy's
    assert srepr(sin(x)) == "sin(Symbol('x'))" and srepr(legendre(a, x)) == "legendre(Symbol('a'), Symbol('x'))"
    assert srepr(Max(x, y)) == "Max(Symbol('x'), Symbol('y'))" and srepr(f(x)) == "Function('f')(Symbol('x'))"
    names: dict[str, object] = {'Sin': Sin, 'sin': sin, 'Integer': Integer, 'Symbol': Symbol, 'Add': sympify}
    for node in [Sin(Integer(0)), sin(x), Sin(x)]:
        assert eval(srepr(node), names) == node


def test_series_of_a_power_of_abs_from_the_left() -> None:
    # (1/Abs(x**5 + x**6)).series(x, 0, 3, dir='-') raised NotImplementedError (SymPy's too), and
    # Abs(x**5 + x**6).as_leading_term(x) "Abs has no _eval_as_leading_term routine"
    e = 1/abs_(x**5 + x**6)
    left = e.series(x, 0, 3, dir='-')
    assert left.removeO() == -1/x**5 + x**(-4) - 1/x**3 + x**(-2) - 1/x + 1 - x + x**2
    right = e.series(x, 0, 3)
    assert right.removeO() == x**(-5) - 1/x**4 + x**(-3) - 1/x**2 + 1/x - 1 + x - x**2
    for series, point in ((left, Rational(-1, 100)), (right, Rational(1, 100))):
        exact = 1/abs(float(point)**5 + float(point)**6)
        assert abs(complex(series.removeO().subs(x, point).evalf(30)) - exact) < 1e-5*exact
    assert abs_(x**5 + x**6).as_leading_term(x) == x**5 and abs_(x**5 + x**6).as_leading_term(x, cdir=-1) == -x**5
    assert e.as_leading_term(x, cdir=-1) == -1/x**5
    assert abs_(x - a).as_leading_term(x) == abs_(a) and abs_(x - 1).as_leading_term(x) == 1
    assert (1/abs_(sin(x)*(1 + x))).series(x, 0, 3, dir='-').removeO() == -1/x + 1 - 7*x/6 + 7*x**2/6
    assert (1/abs_(I*x + x**2)).series(x, 0, 3, dir='-').removeO() == -1/x + x/2
    assert limit(x**5*e, x, 0, '-') == -1 and limit(x**5*e, x, 0) == 1
    # an argument without a leading term (oscillating, bounds) keeps SymPy's rule of functions: the
    # absolute value of the bounds made the Dirichlet test of sum(sin(n)/sqrt(n)) fail
    t = Symbol('t', positive=True)
    assert abs_(sin(1/t)).as_leading_term(t) == abs_(sin(1/t))


def test_power_on_its_cut_from_a_side_the_parameters_decide() -> None:
    # (sympy/sympy#31275 V212) limit(sqrt((-a**2 + x**2)/(1 - x**2)), a, 1, '-') was I for a complex
    # symbol x: it is -I where im(1/(1 - x**2)) < 0 (Mathematica: -I, I, I, I at the points below)
    e = sqrt((-a**2 + x**2)/(1 - x**2))
    found = limit(e, a, 1, '-')
    for point, expected in ((1 - I, -I), (Rational(1, 2), I), (2 + I, I), (Integer(3), I)):
        assert found.subs(x, point) == expected, point
        near = complex(e.subs(x, point).subs(a, 1 - Rational(1, 10**12)).evalf(30))
        assert abs(near - complex(expected.evalf())) < 1e-9, point
    real = Symbol('x', real=True)
    assert limit(e.subs(x, real), a, 1, '-') == I
    # the same for any power of a negative number approached along a symbolic direction
    assert limit(sqrt(-1 + c*a), a, 0).subs(c, -I) == -I and limit(sqrt(-1 + c*a), a, 0).subs(c, I) == I
    assert limit(sqrt(-1 + I*a), a, 0, '-') == -I and limit(sqrt(-1 - I*a), a, 0, '-') == I
    p = Symbol('p', positive=True)
    assert limit(sqrt(-1 + p*a), a, 0) == I and limit(sqrt(-1 - I*p*a), a, 0) == -I
    cube = ((-1 + c*a)**Rational(1, 3)).series(a, 0, 2).removeO()
    for direction in (Rational(3, 2) - 2*I, 1 + I):
        value = complex(cube.subs(c, direction).subs(a, Rational(1, 1000)).evalf(30))
        exact = complex(mpmath.mpc(-1 + complex(direction.evalf())/1000)**(mpmath.mpf(1)/3))
        assert abs(value - exact) < 1e-5, direction


def test_sympify_error_for_what_no_parser_reads() -> None:
    # (without the bridge the message is "cannot parse"; with it SymPy's error is quoted)
    raises(SympifyError, lambda: sympify('Sum(x, x)'))
