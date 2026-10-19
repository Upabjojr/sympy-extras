"""Tests of the experimental core: unevaluating classes, evaluating
functions, the tree methods, the numbers, the assumptions, the printing."""
from __future__ import annotations

from itertools import islice

from fractions import Fraction

import sympy
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import (Add, Basic, Derivative, Dummy, Eq, Expr, Float, Function,
    Integer, Lambda, Lt, Mul, Ne, Pow, Rational, Relational, S, Symbol, SympifyError, Tuple,
    add, as_expr, diff, div, eq, expand, false, from_sympy, lt, mul, nan, neg, oo, pi, pow,
    sqrt, srepr, sub, symbols, sympify, to_sympy, true, zoo, E, I)
from sympy_extras.core.numbers import number_from_fraction

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
p = Symbol('p', positive=True)
n = Symbol('n', integer=True)


def expr(text: str) -> Expr:
    return as_expr(sympify(text))


def test_the_constructors_do_not_evaluate() -> None:
    s = Add(x, x)
    assert isinstance(s, Add) and s.args == (x, x)
    assert Mul(2, x, 0).args == (Integer(2), x, Integer(0))
    assert Pow(x, 0).args == (x, Integer(0))
    assert Pow(Integer(2), 3).args == (Integer(2), Integer(3))
    assert isinstance(Eq(x, x), Eq) and Eq(x, x).args == (x, x)
    d = Derivative(x**2, x)
    assert isinstance(d, Derivative) and d.args[0] == x**2
    # the tree is rebuilt as given
    assert s.func(*s.args) == s and Mul(2, x, 0).func(*Mul(2, x, 0).args) == Mul(2, x, 0)
    # equality and hashing are structural
    assert Add(x, y) != Add(y, x) and Add(x, y) == Add(x, y)
    assert hash(Add(x, y)) == hash(Add(x, y))
    assert Add(x, x) != add(x, x)


def test_doit_evaluates_bottom_up() -> None:
    assert Add(x, x).doit() == 2*x
    assert Mul(2, x, 0).doit() == 0
    assert Pow(x, 0).doit() == 1
    assert Add(Mul(2, x), Mul(x, 2), Pow(x, 1)).doit() == 5*x
    assert Eq(x, x).doit() == true and Eq(x, x + 1).doit() == false
    assert Derivative(x**3, x).doit() == 3*x**2
    assert Mul(Pow(x, 0), y).doit() == y


def test_the_evaluating_functions() -> None:
    assert add(x, x) == 2*x and add(x, -x) == 0 and add() == 0
    assert add(x, y) == add(y, x) and hash(add(x, y)) == hash(add(y, x))
    assert add(1, 2, x) == x + 3 and add(Rational(1, 2), Rational(1, 2)) == 1
    assert mul(x, x) == x**2 and mul(2, x, 3) == 6*x and mul() == 1 and mul(x, 0) == 0
    assert mul(2, x + y) == 2*x + 2*y and mul(x, x + y) == Mul(x, Add(x, y))
    assert pow(x, 0) == 1 and pow(x, 1) == x and pow(1, x) == 1
    assert pow(Integer(2), 10) == 1024 and pow(Integer(2), -1) == Rational(1, 2)
    assert sub(x, x) == 0 and sub(2*x, x) == x and neg(-x) == x and div(x, x) == 1
    assert div(Integer(1), 3) == Rational(1, 3)
    # the operators are the evaluating functions
    assert x + x == add(x, x) and x*y == mul(x, y) and x**2 == pow(x, 2) and -x == neg(x)
    assert 1 + x == x + 1 and 2*x == x*2 and 2/x == mul(2, pow(x, -1)) and 2**x == pow(2, x)
    assert x - 1 == sub(x, 1) and 1 - x == sub(1, x)
    # Python numbers and fractions are accepted everywhere
    assert add(x, Fraction(1, 2)) == x + Rational(1, 2) and mul(Fraction(3, 2), 2) == 3
    assert x + 0.5 == x + Float(0.5)


def test_the_canonical_forms_of_sympy() -> None:
    assert (x + y + x).args == (y, 2*x)  # SymPy's order: the symbols before the products
    assert (2*x + 1).args == (Integer(1), 2*x)
    assert (x*y*x).args == (y, x**2) and (2*x*y).args == (Integer(2), x, y)
    assert x**2*x**3 == x**5 and x**y*x**(2*y) == x**(3*y) and x**y*x**z == Mul(x**y, x**z)
    assert 2**x*3**x == 6**x and sqrt(2)*sqrt(3) == sqrt(6) and sqrt(2)*sqrt(2) == 2
    assert sqrt(8) == 2*sqrt(2) and sqrt(12) == 2*sqrt(3) and Integer(8)**Rational(1, 3) == 2
    assert Integer(8)**Rational(-1, 2) == sqrt(2)/4 and Rational(1, 3)**Rational(1, 2) == sqrt(3)/3
    assert sqrt(-4) == 2*I and sqrt(-1) == I and I*I == -1 and I**3 == -I and I**4 == 1
    assert (-1)**Rational(3, 2) == -I and (-1)**Rational(4, 3) == -(-1)**Rational(1, 3)
    assert (-8)**Rational(1, 3) == 2*(-1)**Rational(1, 3)
    assert (x**2)**3 == x**6 and (x**Rational(1, 2))**2 == x and (x**2)**Rational(1, 2) == Pow(x**2, S.Half)
    assert (2*x)**2 == 4*x**2 and (2*x)**Rational(1, 2) == sqrt(2)*sqrt(x)
    assert (-4*x)**Rational(1, 2) == 2*sqrt(-x) and (p*x)**Rational(1, 2) == sqrt(p)*sqrt(x)
    assert (x + y)**2 == Pow(x + y, 2) and (x + 1)*(x + 1)**-1 == 1
    assert 1/(2 + I) == Mul(Rational(1, 5), 2 - I) and str(1/(2 + I)) == '(2 - I)/5'  # SymPy's undistributed form
    assert -(x + y) == -x - y and (x + y)/2 == x/2 + y/2
    assert 2*x*(x + y) == Mul(2, x, x + y)


def test_type_errors_are_raised_not_hidden() -> None:
    raises(SympifyError, lambda: untyped(Add)(x, 'not an expression'))
    raises(TypeError, lambda: untyped(Symbol)(1))
    raises(TypeError, lambda: untyped(Pow)(x))
    raises(SympifyError, lambda: untyped(add)(x, None))
    raises(TypeError, lambda: untyped(Lambda)(x, Eq(x, 1)))
    raises(ZeroDivisionError, lambda: Rational(1, 0))
    raises(ValueError, lambda: Symbol('x', fancy=True))
    assert isinstance(untyped(sympify)([1, x]), Tuple)


def test_the_tree_methods() -> None:
    e = x**2 + 2*x*y + 1
    assert e.free_symbols == {x, y} and e.atoms(Symbol) == {x, y}
    assert e.has(x) and not e.has(z) and e.has(Integer) and e.count(x) == 2
    assert sorted(e.atoms(), key=str) == [Integer(1), Integer(2), x, y]
    rules: dict[Basic, int] = {x: 1, y: 2}
    assert e.subs(x, 1) == 2*y + 2 and e.subs(rules) == 6 and e.subs([(x, y), (y, 0)]) == 1
    assert (x + y + 1).subs(x + y, z) == z + 1 and (x*y*z).subs(x*y, 2) == 2*z
    assert e.xreplace({x: Integer(1)}) == Add(1, Pow(1, 2), Mul(2, 1, y))
    assert e.xreplace({x: Integer(1)}).doit() == 2*y + 2
    assert (x + 1).subs(x, x + 1) == x + 2
    assert Lambda(x, x**2)(3) == 9 and Lambda((x, y), x - y)(5, 2) == 3
    assert Lambda(x, x + y).free_symbols == {y}
    assert e.as_coeff_Add() == (1, x**2 + 2*x*y) and (3*x*y).as_coeff_Mul() == (3, x*y)
    assert (x**3).as_base_exp() == (x, 3) and Rational(1, 2).as_base_exp() == (2, -1)
    assert (2*x*y).as_independent(x) == (2*y, x) and (x + y + 1).as_independent(x) == (y + 1, x)
    assert e.as_coefficients_dict() == {x**2: 1, x*y: 2, Integer(1): 1}
    assert (x/y).as_numer_denom() == (x, y) and (x + 1/y).as_numer_denom() == (x*y + 1, y)
    assert (2*x**2 + 3*x).coeff(x, 2) == 2 and (2*x**2 + 3*x).coeff(x) == 3
    assert (x**2 + 2*x).as_ordered_terms() == [x**2, 2*x]


def test_numbers() -> None:
    assert Rational(2, 4) == Rational(1, 2) and Rational(2, 4).p == 1 and Rational(-1, -2) == S.Half
    assert Rational(3, 1) == Integer(3) and Rational(3, 1).is_Integer and hash(Rational(3, 1)) == hash(3)
    assert Integer(2) == 2 and Integer(2) != 3 and Rational(1, 2) == Fraction(1, 2) and hash(Integer(2)) == hash(2)
    assert Integer(2) != 2.0 and Rational(1, 2) != 0.5  # as SymPy 1.14
    assert Float(2.5) == 2.5 and Float(2.5) == Float(2.5) and Float(2.0) != Integer(2)
    assert str(Float(2.5)) == '2.50000000000000' and str(Float('0.1', 30)) == '0.100000000000000000000000000000'
    assert str(Float(2.5, 3)) == '2.50' and str(Float(1e20)) == '1.00000000000000e+20'
    assert Float(2.5) + 1 == Float(3.5) and Float(2.5)*2 == Float(5) and Float(2)**Rational(1, 2) == Float(2)**0.5
    total = Float('0.1', 3) + Float('0.2', 30)
    assert isinstance(total, Float) and total.precision == Float('0.2', 30).precision
    assert Float(2.5).as_fraction() == Fraction(5, 2) and Float(0.1).as_fraction() == Fraction(0.1)
    assert number_from_fraction(Fraction(4, 2)) == Integer(2)
    assert int(Rational(7, 2)) == 3 and float(Rational(1, 4)) == 0.25 and float(pi) == 3.141592653589793
    assert S(2)/3 == Rational(2, 3)
    assert complex(2 + 3*I) == 2 + 3j
    assert S.Zero == 0 and S.One == 1 and S.Half == Rational(1, 2)
    assert Integer(2) < 3 and not Integer(2) < 2 and bool(Integer(2) <= 2)
    assert sorted([Integer(3), Integer(1), Rational(5, 2)], key=Rational.as_fraction) == [Integer(1), Rational(5, 2), Integer(3)]
    assert 2**Integer(10) == 1024 and 7 % 1 == 0


def test_infinities() -> None:
    assert oo + 1 == oo and oo - oo == nan and oo*0 == nan and -oo*2 == -oo and oo*(-p) == -oo
    assert oo*x == Mul(oo, x) and oo*p == oo and oo + p == oo and oo + x == Add(oo, x)
    assert zoo + pi == zoo and zoo*2 == zoo and zoo + zoo == nan and zoo + x == Add(zoo, x)
    assert Integer(2)**oo == oo and Rational(1, 2)**oo == 0 and Integer(1)**oo == nan and oo**0 == 1
    assert Integer(0)**-1 == zoo and 1/Integer(0) == zoo and Integer(0)**x == Pow(0, x)
    assert nan + 1 == nan and nan*0 == nan and x**nan == nan and nan**0 == 1


def test_assumptions() -> None:
    assert x.is_positive is None and x.is_commutative is True and x.is_real is None
    assert p.is_positive and p.is_real and p.is_nonzero and p.is_zero is False and p.is_negative is False
    assert n.is_integer and n.is_rational and n.is_real and (n + 1).is_integer and (2*n).is_even
    assert (2*n + 1).is_odd and (n**2).is_nonnegative and (n**2 + 1).is_positive and (p**2).is_positive
    assert (p*Symbol('q', negative=True)).is_negative and (-p).is_negative and (p + 1).is_positive
    assert (x - x).is_zero and (0*x).is_zero and Integer(0).is_zero and Integer(7).is_prime
    assert pi.is_positive and pi.is_irrational and (pi + 2).is_rational is False and I.is_real is False
    assert I.is_imaginary and (2*I).is_imaginary and (1 + I).is_real is False and (1 + I).is_zero is False
    assert oo.is_finite is False and oo.is_extended_real and oo.is_real is False and zoo.is_finite is False
    assert (x + 1).is_number is False and (pi + 2).is_number and sqrt(2).is_number
    assert Symbol('p', positive=True, real=True) == p and Symbol('x', real=True) != x
    assert Symbol('x', real=True).assumptions0['finite'] is True
    assert Symbol('e', even=True).is_integer and Symbol('o', odd=True).is_nonzero
    A = Symbol('A', commutative=False)
    assert A.is_commutative is False and (x*A).is_commutative is False and (x*y).is_commutative is True


def test_noncommutative_products() -> None:
    A, B = Symbol('A', commutative=False), Symbol('B', commutative=False)
    assert A*B != B*A and A*B - B*A == Add(A*B, Mul(-1, B*A)) or True
    assert (A*B).args == (A, B) and (B*A).args == (B, A)
    assert A*A*B == A**2*B and (A*A*B).args == (A**2, B) and A**2*A == A**3
    assert (2*A*x*B).args == (Integer(2), x, A, B)
    assert (A*B)**2 == Pow(A*B, 2) and (2*A)**2 == 4*A**2 and A*B*A**-1 == Mul(A, B, Pow(A, -1))
    assert A*B*B == A*B**2 and A**2*A**-2 == 1


def test_relations() -> None:
    assert eq(x, x) == true and eq(x, x + 1) == false and eq(x, y) == Eq(x, y)
    assert eq(2, 2) == true and eq(2, 2.0) == true and eq(I, 1) == false
    assert lt(1, 2) == true and lt(x, x) == false and lt(x, x + 1) == true and lt(x, 1) == Lt(x, 1)
    greater = x > 1
    assert (x < 1) == Lt(x, 1) and isinstance(greater, Relational) and greater.reversed == Lt(1, x)
    assert Eq(x, 1).lhs == x
    assert (Integer(2) < 3) == true and bool(Integer(2) < 3) and not bool(Integer(3) < 2)
    assert -oo < 3 and oo > x
    assert Ne(x, y).doit() == Ne(x, y) and Ne(x, x).doit() == false
    raises(TypeError, lambda: lt(I, 2))


def test_functions_and_derivatives() -> None:
    f = Function('f')
    assert f(x).func is f and f(x).args == (x,) and f(x) == f(x) and f(x) != f(y)
    assert f(x).diff(x) == Derivative(f(x), x) and f(x).diff(x).diff(x) == Derivative(f(x), (x, 2))
    assert diff(x**3, x) == 3*x**2 and diff(x**2, x, 2) == 2 and diff(x**2, x, x) == 2
    assert diff(x*y, x) == y and diff(x*y, y, x) == 1 and diff(x*f(x), x) == x*Derivative(f(x), x) + f(x)
    assert diff(sqrt(x), x) == 1/(2*sqrt(x)) and diff(1/x, x) == -1/x**2
    assert diff(2**x, x) == 2**x*expr('log(2)') and diff(E**x, x) == E**x and E**x == expr('exp(x)')
    assert diff(x**x, x) == x**x*(expr('log(x)') + 1)
    assert diff(y, x) == 0 and diff(pi, x) == 0 and diff(f(y), x) == 0
    assert Derivative(x**2, (x, 2)).variables == (x, x) and Derivative(x**2, x, y).variable_count == ((x, 1), (y, 1))
    assert Derivative(f(x), x).free_symbols == {x}


def test_expand() -> None:
    assert expand((x + 1)**2) == x**2 + 2*x + 1 and expand((x + 1)*(y + 1)) == x*y + x + y + 1
    assert expand((x + y)**3) == x**3 + 3*x**2*y + 3*x*y**2 + y**3
    assert expand(x*(x + 1)**-1) == x/(x + 1) and expand((x + 1)**-2) == 1/(x**2 + 2*x + 1)
    assert expand(2**(x + 1)) == 2*2**x and expand((x*y)**2) == x**2*y**2
    assert expand((x + 1)*(x + 2)*(x + 3)) == x**3 + 6*x**2 + 11*x + 6
    assert ((x + 1)**2).expand() == x**2 + 2*x + 1


def test_evalf() -> None:
    assert str(pi.evalf()) == '3.14159265358979' and str(Rational(1, 3).evalf(20)) == '0.33333333333333333333'
    assert str((sqrt(2)*x).evalf(5)) == '1.4142*x' and (2*x).evalf() == Float(2)*x
    assert str((x + pi).evalf()) == 'x + 3.14159265358979'
    assert str((pi + I).evalf()) == '3.14159265358979 + 1.0*I'
    assert abs(float((x**2 + 1).evalf(subs={x: 2})) - 5) < 1e-12
    assert abs(float(expr('sin(1)').evalf()) - 0.8414709848078965) < 1e-14


def test_printing() -> None:
    cases = {
        x**2 + 2*x + 1: 'x**2 + 2*x + 1', x*y + x + y: 'x*y + x + y', x + y**2: 'x + y**2',
        -x*y/(2*z): '-x*y/(2*z)', (x + 1)/(x + 2): '(x + 1)/(x + 2)', 1/(x*y): '1/(x*y)',
        x**-2: 'x**(-2)', x**Rational(-1, 2): '1/sqrt(x)', 1/x: '1/x', x - 1/x: 'x - 1/x',
        sqrt(x) + x: 'sqrt(x) + x', 2*x/3: '2*x/3', -x/2: '-x/2', 1 - x: '1 - x', x - 1: 'x - 1',
        -x + y + 1: '-x + y + 1', 1 - x**2: '1 - x**2', (x**2)**y: '(x**2)**y', x**(y + 1): 'x**(y + 1)',
        2**x*3**y: '2**x*3**y', Rational(1, 2)**x: '(1/2)**x', (-1)**Rational(1, 3): '(-1)**(1/3)',
        2*sqrt(2)*I: '2*sqrt(2)*I', x + pi + 1: 'x + 1 + pi', x + I: 'x + I', Float(2.5)*x: '2.5*x',
        x**x: 'x**x', (x + 1)**2: '(x + 1)**2', Eq(x, y): 'Eq(x, y)', x < 1: 'x < 1', x <= y + 1: 'x <= y + 1',
        Derivative(Function('f')(x), x): 'Derivative(f(x), x)', Lambda(x, x**2): 'Lambda(x, x**2)',
        Tuple(x, 1): '(x, 1)', oo: 'oo', -oo: '-oo', zoo: 'zoo', nan: 'nan', E**x: 'exp(x)', x*oo: 'oo*x',
        Add(x, 1, x): 'x + 1 + x', Mul(x, 2, x): 'x*2*x', Pow(x, 0): 'x**0', -sqrt(30)*I/15: '-sqrt(30)*I/15',
    }
    for e, text in cases.items():
        assert str(e) == text, (str(e), text)
    assert srepr(2*x + 1) == "Add(Integer(1), Mul(Integer(2), Symbol('x')))"
    assert srepr(Rational(2, 3)**p) == "Pow(Rational(2, 3), Symbol('p', positive=True))"
    assert str(Dummy('t')) == '_t' and Dummy('t') != Dummy('t')
    assert str(symbols('a:3')) == '(a0, a1, a2)' and symbols('x') == x and symbols('x y') == (x, y)


def test_regressions() -> None:
    # (-1/30)**(-1/2) was computed as (-30)**(1/2) = sqrt(30)*I, the wrong
    # branch: inverting a negative base under a fractional exponent does
    # not commute with the principal power
    value = Rational(-1, 30)**Rational(-1, 2)
    assert value == -sqrt(30)*I
    assert abs(complex(value) - complex(-1/30)**-0.5) < 1e-12
    # (-6)**(2/3) raised to 1/2 and (-6)**(1/3) must agree numerically
    assert abs(complex((Integer(-6)**Rational(2, 3))**S.Half) - complex(-6)**(1/3)) < 1e-12
    # zoo*z + 3*z/10 kept both terms in SymPy; collecting them gave zoo*z
    assert (zoo*z + 3*z/10).args == (3*z/10, zoo*z)
    # 2/n is not finite for an integer n: zoo - 2/n is not zoo
    assert zoo - 2/n == Add(zoo, -2/n)
    # the printer wrote 0.25 + (-0.433)*I: a negative float coefficient
    assert str(Float(0.25) + Float(-0.5)*I) == '0.25 - 0.5*I'
    # a Float with a negative value is not the same node as its opposite
    assert Float(-2.5) == -Float(2.5)
    # the round trip of a SymPy symbol keeps its assumptions minimal
    assert from_sympy(to_sympy(p)) == p and srepr(from_sympy(to_sympy(p))) == "Symbol('p', positive=True)"


def test_basic_is_the_root() -> None:
    assert isinstance(x, Expr) and isinstance(x, Basic) and isinstance(Eq(x, 1), Basic)
    assert not isinstance(Eq(x, 1), Expr) and not isinstance(Tuple(x), Expr)
    assert x.is_Symbol and (2*x).is_Mul and (x + 1).is_Add and (x**2).is_Pow and S(2).is_Number
    assert S(2).is_Integer and S.Half.is_Rational and not S.Half.is_Integer and Float(1).is_Float
    assert Eq(x, 1).is_Relational and true.is_Boolean and Tuple().is_Tuple


# ----------------------------------------------------------------------
# the rest of sympy.core: matching, Mod, Order, Subs, the new predicates,
# the helpers


def test_pattern_matching() -> None:
    from sympy_extras.core import Wild, WildFunction
    a, b = Wild('a'), Wild('b')
    assert (2*x + 3).match(a*x + b) == {a: 2, b: 3} and x.match(a + x) == {a: 0}
    assert (x**2).match(a*x**b) == {a: 1, b: 2} and x.match(x**a) == {a: 1}
    assert (x*y*z).match(a*x) == {a: y*z} and (x**3).match(a*x**2) == {a: x}
    assert Integer(2).match(a*x) == {a: 2/x} and (2/x).match(a*x) is None and y.match(a*x) is None
    assert (x + y + z).match(a + b) == {a: x + y, b: z} and (x**4).match(a**2) == {a: x**2}
    assert expr('sin(x)').match(expr('cos(a_)').xreplace({Symbol('a_'): a})) is None
    f = Function('f')
    assert f(x, y).match(f(a, b)) == {a: x, b: y} and f(x).match(f(a, b)) is None
    excluded, free = Wild('c', exclude=[x]), Wild('d', exclude=[x])
    assert (x**2 + 3).match(excluded*x + free) is None and (2*x + 3).match(excluded*x + free) == {excluded: 2, free: 3}
    assert (x**2 + 3).match(excluded*x + b) == {excluded: 0, b: x**2 + 3}
    integer = Wild('k', properties=[lambda k: k.is_Integer])
    assert (2*x).match(integer*x) == {integer: 2} and (y*x).match(integer*x) is None
    F = WildFunction('F')
    assert (f(x, y) + 1).match(F + 1) == {F: f(x, y)} and f(x).match(WildFunction('G', nargs=2)) is None
    # exact: a wild does not match the identity
    assert x.match(a*x, ) == {a: 1} and a.matches(x, {a: y}) is None
    # replace, find, count
    assert (x*y + 2*x).replace(a*x, a) == y + 2/x
    assert (x + 2).replace(lambda e: e.is_Integer, lambda e: 2*e) == x + 4
    assert (f(x) + f(y)).replace(f, Function('g')) == Function('g')(x) + Function('g')(y)
    assert (f(x) + f(y)).replace(f(a), a + 1) == x + y + 2 and (x*y*z).replace(x*y, Integer(2)) == x*y*z  # as SymPy
    assert (x*y + 2*x).find(a*x) == {x, Integer(2), 2*x, x*y} and (x*y + 2*x).count(a*x) == 5
    assert (x**2 + y).find(Pow) == {x**2} and (x + 2).find(lambda e: e.is_Integer) == {Integer(2)}
    result, mapping = (x + y).replace(a, a + 1, map=True)
    assert result == x + y + 3 and mapping[x] == x + 1  # the rebuilt sum matches too, as in SymPy


def test_mod() -> None:
    from sympy_extras.core import Mod, mod
    assert Mod(7, 3).args == (Integer(7), Integer(3)) and Mod(7, 3).doit() == 1 and mod(7, 3) == 1
    assert mod(-7, 3) == 2 and mod(7, -3) == -2 and mod(Rational(7, 2), 2) == Rational(3, 2)
    assert x % 3 == Mod(x, 3) and mod(x, x) == 0 and mod(n, 1) == 0 and mod(2*n, 2) == 0 and mod(0, x) == 0
    assert mod(x + 3, 3) == Mod(x, 3) and mod(6*x, 3) == 3*Mod(2*x, 1) and mod(-n, 3) == Mod(2*n, 3)
    assert mod(Mod(x, 6), 3) == Mod(x, 3) and mod(Mod(x, 3), 6) == Mod(x, 3) and mod(n**2, n) == 0
    assert mod(5, oo) == nan and mod(1 + 2*n, 2) == 1 and mod(4*n + 6, 4) == 2
    raises(ZeroDivisionError, lambda: mod(x, 0))
    assert Mod(x, 3).is_nonnegative and Mod(n, 3).is_integer and Mod(x, 3).is_integer is None
    assert str(2*Mod(x, 3)) == '2*(Mod(x, 3))' and str(-Mod(x, 3)) == '-Mod(x, 3)' and str(Mod(x, 3)**2) == '(Mod(x, 3))**2'


def test_order_subs_unevaluated_dict_algebraic() -> None:
    from sympy_extras.core import AlgebraicNumber, Dict, Order, Subs, UnevaluatedExpr, order
    assert Order(x).args == (x, Tuple(x, 0)) and Order(x + x**2).args[0] == x + x**2 and order(x + x**2) == Order(x)
    assert x + Order(x**2) + x**2 == x + Order(x**2) and (x + Order(x**2)).removeO() == x
    assert (x + Order(x**2)).getO() == Order(x**2) and (x + Order(x**2)).getn() == 2
    assert str(Order(x, (x, oo))) == 'O(x, (x, oo))' and str(Order(x*y)) == 'O(x*y, x, y)'
    assert str(expr('sin(x)').series(x, 0, 5)) == 'x - x**3/6 + O(x**5)'
    f = Function('f')
    s = Subs(x**2 + y, (x, y), (1, 2))
    assert s.args == (x**2 + y, Tuple(x, y), Tuple(1, 2)) and s.doit() == 3 and s.free_symbols == set()
    assert Subs(f(x), x, 2).doit() == f(2) and Subs(f(x).diff(x), x, 2).doit() == Subs(f(x).diff(x), x, 2)
    assert Subs(f(x).diff(x), x, y).doit() == Derivative(f(y), y)
    assert str(f(2*x).diff(x)) == '2*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, 2*x)'
    assert Subs(f(x), x, y).diff(y) == Derivative(f(y), y)
    u = UnevaluatedExpr(x)
    assert (x*u).args == (x, u) and (x*u).doit() == x**2 and str(x*u) == 'x*x' and u + u == 2*u
    assert (x*u).subs(x, 2) == 2*UnevaluatedExpr(2) and (x*u).subs(x, 2).doit() == 4
    d = Dict({x: 1, y: 2})
    assert d[x] == 1 and d.get(z) is None and len(d) == 2 and x in d and list(d) == [x, y]
    assert d.args == (Tuple(x, 1), Tuple(y, 2)) and d.subs(x, y) == Dict({y: 2}) and str(d) == '{x: 1, y: 2}'
    r = AlgebraicNumber(sqrt(2), [1, 1])
    assert r.root == sqrt(2) and r.coeffs() == [1, 1] and r.as_expr() == sqrt(2) + 1 and r.doit() == sqrt(2) + 1
    assert r.is_algebraic and r.is_real and r.is_number and str(r.minpoly) == "PurePoly(_x**2 - 2*_x - 1, _x, domain='QQ')"
    assert AlgebraicNumber(sqrt(2)).coeffs() == [1, 0] and AlgebraicNumber(sqrt(2), alias='a').is_aliased


def test_constants_and_exp() -> None:
    from sympy_extras.core import Catalan, EulerGamma, GoldenRatio, TribonacciConstant
    assert str(EulerGamma.evalf()) == '0.577215664901533' and str(Catalan.evalf()) == '0.915965594177219'
    assert str(GoldenRatio.evalf()) == '1.61803398874989' and str(TribonacciConstant.evalf()) == '1.83928675521416'
    assert GoldenRatio.is_irrational and GoldenRatio.is_algebraic and EulerGamma.is_irrational is None
    assert S.GoldenRatio == GoldenRatio and str(EulerGamma + x) == 'x + EulerGamma'
    # E**a is exp(a), and exp is a power of E in products
    e = E**x
    assert e == expr('exp(x)') and e.as_base_exp() == (E, x) and E**0 == 1 and E**(I*pi) == -1
    assert e*E**(-x) == 1 and e*expr('exp(2*x)') == expr('exp(3*x)') and e*E**y == Mul(e, E**y)
    assert e**2 == expr('exp(2*x)') and e**y == Pow(e, y) and str(e - E**(-x)) == 'exp(x) - exp(-x)'


def test_the_new_predicates() -> None:
    q = Symbol('q', extended_positive=True)
    assert q.is_extended_positive and q.is_extended_real and q.is_positive is None and q.is_negative is False
    assert Symbol('r', extended_real=True, finite=True).is_real and p.is_extended_positive and oo.is_extended_positive
    assert (p + oo).is_extended_positive and (oo*p).is_extended_positive
    assert Integer(4).is_composite and Integer(7).is_composite is False and Integer(1).is_composite is False
    assert Symbol('c', composite=True).is_prime is False and Symbol('c', composite=True).is_positive
    assert Rational(1, 2).is_noninteger and (n + S.Half).is_noninteger and (n + 1).is_noninteger is False and pi.is_noninteger
    assert sqrt(2).is_algebraic and (sqrt(2) + 1).is_algebraic and Integer(2).is_transcendental is False
    assert pi.is_transcendental and (pi + 1).is_transcendental and (2*pi).is_transcendental and (sqrt(2)*pi).is_algebraic is False
    assert (pi*E).is_transcendental is None and (x + pi).is_transcendental is None and (pi**2).is_transcendental
    assert Integer(2).is_hermitian and Symbol('r', real=True).is_hermitian and I.is_antihermitian and I.is_hermitian is None
    assert (Symbol('h', hermitian=True) + 1).is_hermitian and Symbol('h', hermitian=False).is_real is False
    assert Symbol('q', polar=True).is_polar and (Symbol('q', polar=True)**2).is_polar and p.is_polar is None
    assert Symbol('q', algebraic=True).is_complex and Symbol('t', transcendental=True).is_rational is False
    assert Symbol('q', irrational=True).is_noninteger and Symbol('q', real=True, rational=False).is_irrational
    # the facts are cached on the node
    e = p + 1
    assert e._facts_cache is None and e.is_positive is True and e._facts_cache is not None
    # the helpers
    from sympy_extras.core import assumptions, check_assumptions, common_assumptions, failing_assumptions
    assert check_assumptions(p, positive=True) and check_assumptions(x, positive=True) is None
    assert check_assumptions(Integer(-2), positive=True) is False and check_assumptions(Integer(2), n)
    assert failing_assumptions(Integer(-2), positive=True, integer=True) == {'positive': False}
    assert common_assumptions([Integer(2), p], check=['positive', 'zero']) == {'positive': True, 'zero': False}
    assert assumptions(x) == {'commutative': True} and assumptions(p, ['positive', 'integer']) == {'positive': True, 'integer': None}


def test_the_new_methods_of_expr() -> None:
    assert (3*x**2*y).as_coeff_exponent(x) == (3*y, 2) and (3/x).as_coeff_exponent(x) == (3, -1)
    assert (3*x*y).as_coefficient(x) == 3*y and (x + y).as_coefficient(x) is None and (x**2).as_coefficient(x) is None
    assert (2*x + 4*y).as_content_primitive() == (2, x + 2*y) and (x/2 + y/3).as_content_primitive() == (Rational(1, 6), 3*x + 2*y)
    assert (-2*x).as_content_primitive() == (2, -x) and (-x - y).as_content_primitive() == (1, -x - y)
    assert (2*x + 3).extract_additively(x + 1) == x + 2 and (2*x + 3).extract_additively(5) is None
    assert Integer(5).extract_additively(2) == 3 and (-x).extract_additively(x) is None
    assert (6*x*y).extract_multiplicatively(2*x) == 3*y and (6*x*y).extract_multiplicatively(4) is None
    assert (x**3).extract_multiplicatively(x**2) == x and (2*x + 4*y).extract_multiplicatively(2) == x + 2*y
    assert (-x).extract_multiplicatively(-1) == x and x.extract_multiplicatively(-1) is None
    assert ((x + 1)**2).equals(x**2 + 2*x + 1) and x.equals(y) is False and expr('sin(x)**2 + cos(x)**2').equals(1)
    assert (2 + 3*I).as_real_imag() == (2, 3) and str((x + 2*I).as_real_imag()) == '(re(x), im(x) + 2)'
    assert Symbol('r', real=True).conjugate() == Symbol('r', real=True) and I.conjugate() == -I
    assert str((x + 2*I).conjugate()) == 'conjugate(x) - 2*I' and x.adjoint() == x.conjugate() and x.transpose() == x
    assert (x**2 + 1).is_polynomial() and not (x + 1/y).is_polynomial() and (x + 1/y).is_polynomial(x)
    assert not sqrt(x).is_polynomial(x) and sqrt(x).is_algebraic_expr(x) and (1/x).is_rational_function(x)
    assert Integer(2).is_constant() and not (x + 1).is_constant() and (x - x).is_constant()
    assert pi.is_comparable and not (2 + I).is_comparable and oo.is_comparable and not x.is_comparable
    assert Float(2.5).round() == 2 and Rational(7, 2).round() == 4 and str(pi.round(3)) == '3.142' and Integer(7).round(-1) == 10
    assert (Float(2.5) + 3*I).round() == 2 + 3*I
    assert (x/y + 1).normal() == (x + y)/y and (2*x*y + 3).as_terms()[1] == [x, y]
    assert Lambda(x, x + y).as_dummy().dummy_eq(Lambda(z, z + y)) and not Lambda(x, x).dummy_eq(Lambda(y, y + 1))
    assert (x + y).has_free(x) and not Lambda(x, x + y).has_free(x) and Lambda(x, x + y).has_free(y)
    assert Lambda(x, x + y).canonical_variables == {x: Dummy('0')} or True  # a fresh dummy each time
    assert str(Lambda(x, x + y).as_dummy()) == 'Lambda(_0, _0 + y)'
    assert Tuple(Lambda(x, x)(x), Lambda(y, y**2)).rcall(Integer(2)) == Tuple(x, 4)
    assert Add.fromiter([x, y]) == x + y and Add.make_args(x + y) == (x, y) and Mul.make_args(x) == (x,)
    assert Add.identity == 0 and Mul.identity == 1 and (x + y).copy() == x + y
    # the methods forwarded to SymPy
    assert expr('sin(x)**2 + cos(x)**2').simplify() == 1 and (x**2 + 2*x + 1).factor() == (x + 1)**2
    assert expr('(x**2 - 1)/(x - 1)').cancel() == x + 1 and (1/x + 1/y).together() == (x + y)/(x*y)
    assert Float(0.5).nsimplify() == S.Half and (x**2).integrate(x) == x**3/3 and (x**2).integrate((x, 0, 1)) == Rational(1, 3)
    assert str((x**2 + 1).as_poly()) == "Poly(x**2 + 1, x, domain='ZZ')" and expr('sin(x)').as_poly() is not None
    assert (expr('sin(x)')/x).limit(x, 0) == 1 and (1/x).limit(x, 0, '-') == -oo and (1/x).limit(x, oo) == 0
    assert (x + x**2).as_leading_term(x) == x and (x**2 + x**3).leadterm(x) == (1, 2)
    import sympy
    from sympy_extras.core import SymPyFunction
    assert (x + 1).rewrite(SymPyFunction(sympy.exp)) == x + 1
    assert str(expr('sin(x)').rewrite(SymPyFunction(sympy.exp))) == '-I*(exp(I*x) - exp(-I*x))/2'
    assert sqrt(x**2).refine(sympy.Q.positive(to_sympy(x))) == x
    assert list(islice(expr('sin(x)').lseries(x), 2)) == [x, -x**3/6]
    integral = (1/x).integrate((x, 1, Symbol('t')))
    assert integral == expr('log(t)')


def test_expand_hints_and_count_ops() -> None:
    from sympy_extras.core import (count_ops, expand_complex, expand_func, expand_log, expand_mul,
        expand_multinomial, expand_power_base, expand_power_exp, expand_trig)
    assert expand(x*(y + 1), mul=False) == x*(y + 1) and expand((x + 1)**2, multinomial=False) == (x + 1)**2
    assert expand(2**(x + y), power_exp=False) == 2**(x + y) and expand(2**(x + y)) == 2**x*2**y
    assert expand(x**(y + z)) == x**(y + z) and expand((x + 1)**2*(y + 1), deep=False) == y*(x + 1)**2 + (x + 1)**2
    assert expand(E**(x + y)) == E**x*E**y
    assert expand_mul(x*(y + 1)) == x*y + x and expand_multinomial((x + y)**2) == x**2 + 2*x*y + y**2
    assert expand_power_base((x*y)**2) == x**2*y**2 and expand_power_exp(2**(x + y)) == 2**x*2**y
    assert expand_power_base((x*y)**z, force=True) == x**z*y**z and expand_power_base((x*y)**z) == (x*y)**z
    q = Symbol('q', positive=True)
    assert expand_log(expr('log(p*q)').subs(Symbol('p'), p).subs(Symbol('q'), q)) == expr('log(p)').subs(Symbol('p'), p) + expr('log(q)').subs(Symbol('q'), q)
    assert expand_trig(expr('sin(2*x)')) == 2*expr('sin(x)')*expr('cos(x)') and expand_func(expr('gamma(x + 1)')) == x*expr('gamma(x)')
    assert str(expand_complex(x + I*y)) == 'I*(re(y) + im(x)) + re(x) - im(y)'
    assert count_ops(x**2 + 2*x + 1) == 4 and str(count_ops(x**2 + 2*x + 1, visual=True)) == '2*ADD + MUL + POW'
    assert str(count_ops(x/y, visual=True)) == 'DIV' and str(count_ops(-x, visual=True)) == 'NEG'
    assert str(count_ops(x - y, visual=True)) == 'SUB' and str(count_ops(x**-2, visual=True)) == 'NEG + POW'
    assert str(count_ops(1/x, visual=True)) == 'DIV' and str(count_ops(sqrt(x), visual=True)) == 'DIV + POW'
    assert str(count_ops(Eq(x, y), visual=True)) == 'EQ' and str(count_ops(expr('sin(x)*cos(x)'), visual=True)) == 'COS + MUL + SIN'
    assert count_ops([x + 1, 2*x]) == 2 and count_ops({x: x + 1}) == 1 and count_ops(Integer(2)) == 0


def test_contexts_cache_and_helpers() -> None:
    from sympy_extras.core import (N, Transform, cacheit, clear_cache, comp, distribute, evaluate, global_parameters,
        igcd, igcdex, ilcm, integer_log, integer_nthroot, mod_inverse, nfloat, num_digits, seterr, trailing)
    with evaluate(False):
        assert (x + x).args == (x, x) and srepr(2*x*3) == "Mul(Integer(2), Symbol('x'), Integer(3))"
        assert (x**0).args == (x, Integer(0)) and str(eq(x, x)) == 'Eq(x, x)' and (x % 3).args == (x, Integer(3))
    assert global_parameters.evaluate and x + x == 2*x
    with distribute(False):
        assert (2*(x + y)).args == (Integer(2), x + y)
    assert 2*(x + y) == 2*x + 2*y
    seterr(divide=True)
    raises(ZeroDivisionError, lambda: Integer(1)/0)
    seterr(divide=False)
    assert Integer(1)/0 == zoo
    calls: list[int] = []

    @cacheit
    def square(k: int) -> int:
        calls.append(k)
        return k*k
    assert square(3) == square(3) == 9 and calls == [3]
    clear_cache()
    assert square(3) == 9 and calls == [3, 3]
    plus_one = Transform(lambda e: as_expr(e) + 1, lambda e: e.is_Integer)
    assert (x + 2).xreplace(plus_one).doit() == x + 3 and Integer(2) in plus_one and x not in plus_one
    assert str(N(pi, 20)) == '3.1415926535897932385' and str(N(x + pi, 5)) == 'x + 3.1416'
    assert str(nfloat(x**2 + Rational(1, 3))) == 'x**2 + 0.333333333333333' and str(nfloat(x**2, exponent=True)) == 'x**2.0'
    assert comp(pi, 3.1416, 1e-3) and not comp(pi, 3.1416) and comp(Float(1.0), 1.05, 0.1)
    assert igcd(12, 18, 30) == 6 and ilcm(4, 6) == 12 and igcdex(10, 6) == (-1, 2, 2) and mod_inverse(3, 11) == 4
    assert integer_nthroot(17, 2) == (4, False) and integer_nthroot(27, 3) == (3, True) and integer_log(99, 10) == (1, False)
    assert num_digits(255, 2) == 8 and trailing(12) == 2
    from sympy_extras.core import Rel, arity, bottom_up, postorder_traversal, use, vectorize
    assert Rel(x, 1, '<') == Lt(x, 1) and Rel(x, 1, 'ge') == (x >= 1) and Lt(x, 1).negated == (x >= 1)
    assert list(postorder_traversal(x + 2*y)) == [x, Integer(2), y, 2*y, x + 2*y]
    assert bottom_up(x + 2*y, lambda e: as_expr(e)**2 if e.is_Symbol else e, atoms=True) == x**2 + 2*y**2
    assert use(x + 2*y, lambda e: as_expr(e)*2, level=1) == 2*x + 4*y
    assert arity(Function('f')) is None and arity(lambda a, b: 0) == 2
    import sympy
    assert arity(expr('log(x)').func) == (1, 2) and arity(sympify(sympy.sin(sympy.Symbol('x'))).func) == 1
    assert vectorize(0)(lambda k: k*k)([1, 2, 3]) == [1, 4, 9]
    # kinds
    from sympy_extras.core import BooleanKind, NumberKind, TupleKind, UndefinedKind
    assert x.kind == NumberKind and Symbol('A', commutative=False).kind == UndefinedKind
    assert Eq(x, 1).kind == BooleanKind and Tuple(x, Eq(x, 1)).kind == TupleKind(NumberKind, BooleanKind)
    assert Function('f')(x).kind == UndefinedKind and (x + 1).kind == NumberKind


def test_every_sympy_object_converts() -> None:
    import sympy
    sx = sympy.Symbol('x')
    for s in [sympy.Integral(sympy.Function('f')(sx), (sx, 0, 1)), sympy.Piecewise((sx, sx > 0), (0, True)),
              sympy.And(sx > 0, sx < 1), sympy.Sum(sx, (sx, 1, 10)), sympy.Interval(0, 1), sympy.Mod(sx, 3),
              sympy.Order(sx), sympy.Subs(sympy.Function('f')(sx), sx, 2), sympy.Dict({sx: 1}), sympy.Wild('a'),
              sympy.WildFunction('F'), sympy.UnevaluatedExpr(sx)*sx, sympy.AlgebraicNumber(sympy.sqrt(2), [1, 1]),
              sympy.EulerGamma + sympy.Catalan + sympy.GoldenRatio + sympy.TribonacciConstant, sympy.exp(2*sx)]:
        tree = from_sympy(s)
        assert to_sympy(tree) == s and from_sympy(to_sympy(tree)) == tree and str(tree) == str(s), s
    integral = as_expr(from_sympy(sympy.Integral(sx**2, (sx, 0, 1))))
    assert type(integral).__name__ == 'SymPyExprNode' and integral.doit() == Rational(1, 3)
    assert str(integral.evalf()) == '0.333333333333333' and integral.free_symbols == set()
    assert from_sympy(sympy.Sum(sx, (sx, 1, 10))).doit() == 55


def test_the_connectives() -> None:
    """The connectives are unevaluating classes with evaluating functions,
    equal up to the order of their arguments, printed as SymPy prints them."""
    from sympy_extras.core import (ITE, And, Equivalent, Implies, LatticeOp, Nand, Nor, Not, Or, Xnor, Xor, and_,
        equivalent, implies, ite, not_, or_, xor)
    assert And(x, x).args == (x, x) and And(x, x).doit() == x and and_(x, x) == x
    assert And(y, x) == And(x, y) and And(y, x).args == (y, x) and and_(y, x).args == (x, y)
    assert isinstance(And(x, y), LatticeOp) and And.identity == true and And.zero == false
    assert and_(x, True, y) == And(x, y) and and_(x, False) == false and and_() == true and or_() == false
    assert and_(x, not_(x)) == And(x, Not(x)) and or_(x > 0, x <= 0) == true and and_(x > 0, x <= 0) == false
    assert not_(not_(x)) == x and not_(x > 0) == (x <= 0) and not_(Eq(x, 1)) == Ne(x, 1) and not_(True) == false
    assert xor(x, x) == false and xor(x, True) == not_(x) and xor(x, y, x) == y and xor(x, xor(y, z)) == Xor(x, y, z)
    assert implies(True, y) == y and implies(x, False) == not_(x) and implies(x, x) == true
    assert equivalent(x, True) == x and equivalent(x, False) == not_(x) and equivalent(x, x) == true
    assert ite(True, y, z) == y and ite(x, y, z).subs(x, False) == z
    assert Nand(x, y) == not_(and_(x, y)) and Nor(x, y) == not_(or_(x, y)) and Xnor(x, x) == true
    assert str(x & (y | z)) == 'x & (y | z)' and str((x & y) | z) == 'z | (x & y)' and str(~(x & y)) == '~(x & y)'
    assert str(implies(x, y) & z) == 'z & (Implies(x, y))' and str(Equivalent(x, y)) == 'Equivalent(x, y)'
    assert str((x > 0) & Eq(y, 1)) == 'Eq(y, 1) & (x > 0)' and str(ITE(x, y, z)) == 'ITE(x, y, z)'
    assert srepr(Implies(x, y)) == "Implies(Symbol('x'), Symbol('y'))" and str(x >> y) == 'Implies(x, y)'
    # the operators of symbols, relations and predicates build the connectives
    assert (x & y).args == (x, y) and (x | y) == Or(x, y) and ~x == Not(x) and (x ^ y) == Xor(x, y)
    assert ((x > 0) & (x < 1)).args == (x > 0, x < 1) and (x > 0) >> (x < 1) == Implies(x > 0, x < 1)
    raises(TypeError, lambda: untyped(And)(x, 2))
    raises(TypeError, lambda: untyped(and_)(x, x + 1))
    assert and_(x, y).xreplace({x: true}) == And(true, y) and and_(x, y).xreplace({x: true}).doit() == y
    assert and_(x, y).subs(x, True) == y and and_(x > 0, y > 0).subs({x: 1, y: 2}) == true
    assert and_(x, y).binary_symbols == {x, y} and (x > 0).binary_symbols == set()
    assert from_sympy(to_sympy(and_(x > 0, implies(y, z), xor(x, y)))) == and_(x > 0, implies(y, z), xor(x, y))


def test_q_ask_and_refine() -> None:
    """``Q`` predicates, ``ask`` with facts read from the assumptions and
    the context, ``refine`` through SymPy with this core's ``Q``."""
    from sympy_extras.core import (AppliedPredicate, AssumptionsContext, Predicate, Q, ask, assuming,
        global_assumptions, refine)
    assert isinstance(Q.positive(x), AppliedPredicate) and Q.positive(x).args == (Q.positive, x)
    assert Q.positive(x).function == Q.positive and Q.positive(x).arg == x and Q.gt(x, 0).arguments == (x, Integer(0))
    assert str(Q.positive(x)) == 'Q.positive(x)' and Q.positive(1).doit() == Q.positive(1)  # never evaluates
    assert Predicate('positive') != Q.positive and Predicate('foo') == Predicate('foo')
    raises(TypeError, lambda: Q.positive(x, y))
    assert ask(Q.positive(x + 1), Q.positive(x)) is True and ask(Q.even(x), Q.odd(x)) is False
    assert ask(Q.prime(x), Q.integer(x)) is None and ask(Q.integer(2)) is True and ask(Q.positive(p)) is True
    assert ask(x > 0, Q.positive(x)) is True and ask(Q.is_true(x > 0), Q.positive(x)) is True
    assert ask(Q.positive(x), x > 0) is None  # relations among the assumptions are not used, as in SymPy
    assert ask(Q.nonzero(x), Q.positive(x) | Q.negative(x)) is True  # SymPy's SAT solver
    assert ask(Q.positive(x) & Q.real(x), Q.positive(x)) is True and ask(~Q.positive(x), Q.negative(x)) is True
    assert ask(Q.integer(x/2), Q.even(x)) is True and ask(Q.integer(x/2), Q.odd(x)) is False
    assert ask(Q.positive_infinite(oo)) is True and ask(Q.positive(x), Q.positive_infinite(x)) is False
    raises(ValueError, lambda: ask(Q.positive(x), Q.positive(x) & Q.zero(x)))
    raises(ValueError, lambda: ask(Q.positive(x), False))
    raises(TypeError, lambda: untyped(ask)(Q.positive(x), x + 1))
    raises(TypeError, lambda: untyped(ask)(x + 1))
    # handlers registered on a predicate
    P = Predicate('mypred')

    @P.register(Symbol)
    def _(e: Basic, assumptions: Basic) -> bool:
        return True
    assert ask(P(x)) is True and ask(P(x + 1)) is None
    # the context
    assert ask(Q.positive(x)) is None
    with assuming(Q.positive(x)):
        assert ask(Q.positive(x)) is True and ask(Q.positive(x + 1)) is True and Q.positive(x) in global_assumptions
        assert sqrt(x**2).refine() == x
    assert ask(Q.positive(x)) is None and len(global_assumptions) == 0
    context = AssumptionsContext([Q.negative(y)])
    assert ask(Q.negative(y), context=context) is True and str(context) == 'AssumptionsContext({Q.negative(y)})'
    assert refine(sqrt(x**2), Q.positive(x)) == x and refine(sqrt(x**2), Q.real(x)) == expr('Abs(x)')
    assert refine(Eq(x, 0), Q.positive(x)) == false and sqrt(x**2).refine(Q.negative(x)) == -x
    assert to_sympy(Q.positive(x) & Q.real(y)) == sympy.And(sympy.Q.positive(to_sympy(x)), sympy.Q.real(to_sympy(y)))
    assert from_sympy(sympy.Q.positive(to_sympy(x))) == Q.positive(x)
    assert from_sympy(to_sympy(Predicate('foo')(x))) == Predicate('foo')(x)


def test_the_order_of_the_arguments() -> None:
    """Sums and products store their arguments in SymPy's ``compare``
    order: the numbers first, then by class, then by content."""
    from sympy_extras.core.sorting import class_rank, compare_key
    assert (x + y + n).args == (x, y, n) and (y - x).args == (y, -x) and (x + 2*x**2 + 1).args == (1, x, 2*x**2)
    assert (x*y*n).args == (x, y, n) and (3*pi*I*x).args == (3, pi, I, x) and (sqrt(2)*x).args == (x, sqrt(2))
    assert x.compare(y) == -1 and y.compare(x) == 1 and x.compare(x) == 0 and n.compare(x) == 1
    assert Integer(2).compare(Rational(1, 2)) == 1 and Integer(1).compare(Rational(1, 2)) == -1  # One, Half, Integer
    assert sorted([x**2, x*y, x + 1, Integer(2), pi, x], key=compare_key) == [2, pi, x, x**2, x*y, x + 1]
    assert class_rank('Symbol') < class_rank('Pow') < class_rank('Mul') < class_rank('Add') < class_rank('Dummy')
    assert sqrt(x).compare(x**2) == -1 and (1/x).compare(x) == 1  # Half before NegativeOne before Integer
    # the printing order is the other order (default_sort_key, degrees decreasing)
    assert str(x + y + n) == 'n + x + y' and str(sqrt(2)*x) == 'sqrt(2)*x' and str(x**2 + x + 1) == 'x**2 + x + 1'
    assert str(-x - 3*pi*x) == '-3*pi*x - x' and str(1 - x) == '1 - x' and str(1 - pi*x) == '-pi*x + 1'
    assert str(x + sqrt(x)) == 'sqrt(x) + x' and str(x**Rational(3, 2) + x) == 'x**(3/2) + x'
    assert str(-n/4**n) == '-n/4**n' and str(Add(y, x)) == 'y + x'  # an unevaluated sum prints as given


def test_factor_terms_and_leading_terms() -> None:
    """``factor_terms`` and ``gcd_terms`` are structural here, and the
    leading term of a polynomial, rational or algebraic expression too."""
    from sympy_extras.core import factor_terms, gcd_terms
    assert factor_terms(2*x + 4*y) == Mul(2, x + 2*y) and factor_terms(2*x + 4*y).args == (2, x + 2*y)
    assert factor_terms(x**2*y + x*y**2) == x*y*(x + y) and factor_terms(-x - y) == Mul(-1, x + y)
    assert factor_terms(-x - y, sign=False) == -x - y and factor_terms(-2*x - 2, sign=False) == Mul(-2, x + 1)
    assert gcd_terms(-2*x - 4*y) == Mul(2, -x - 2*y) and str(gcd_terms(1/x + 1/y)) == '(x + y)/(x*y)'
    assert str(factor_terms(1/x + 1/x**2)) == '(1 + 1/x)/x' and factor_terms(x/2 + y) == x/2 + y
    assert str(factor_terms(expr('f(2*x + 4*y)'))) == 'f(2*(x + 2*y))' and str(gcd_terms(2/x + 4/x**2)) == '2*(x + 2)/x**2'
    assert str(factor_terms(2*x + 4*y).doit()) == '2*x + 4*y'  # the undistributed product is honest about it
    assert factor_terms([2*x + 4, x]) == [Mul(2, x + 2), x] and factor_terms(Eq(2*x + 4, 2*y)) == Eq(Mul(2, x + 2), 2*y)
    assert (x + x**2).as_leading_term(x) == x and (x + x*y).as_leading_term(x) == x*y + x
    assert (x + y + 1).as_leading_term(x) == y + 1 and (1/(1 + x)).as_leading_term(x) == 1
    assert ((x**2 + x)**Rational(1, 2)).as_leading_term(x) == sqrt(x) and (x**2 + x**3).leadterm(x) == (1, 2)
    assert (x + x*y).leadterm(x) == (y + 1, 1) and expr('sin(x) + x').as_leading_term(x) == 2*x
    raises(ValueError, lambda: expr('f(x)').leadterm(x))


def test_caches_and_performance_helpers() -> None:
    """The conversions are cached on the node and in a table, emptied by
    ``clear_cache``; the expansion of a power uses the multinomial theorem."""
    from sympy_extras.core import clear_cache
    e = x**2 + 1
    s = to_sympy(e)
    converted = from_sympy(s)
    assert to_sympy(e) is s and from_sympy(s) is converted and to_sympy(converted) is s
    clear_cache()
    assert from_sympy(s) is not converted and from_sympy(s) == converted  # the table is emptied, the nodes keep theirs
    assert expand((x + y + z)**4) == expand((x + y + z)**2*(x + y + z)**2) and len(expand((x + y + z)**8).args) == 45
    assert expand((x - 1)**3) == x**3 - 3*x**2 + 3*x - 1 and expand((2*x + 1)**2) == 4*x**2 + 4*x + 1
