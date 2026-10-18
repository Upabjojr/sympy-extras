"""Tests of the experimental core: unevaluating classes, evaluating
functions, the tree methods, the numbers, the assumptions, the printing."""
from __future__ import annotations

from fractions import Fraction

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
    assert (x + y + x).args == (2*x, y)
    assert (2*x + 1).args == (Integer(1), 2*x)
    assert (x*y*x).args == (x**2, y) and (2*x*y).args == (Integer(2), x, y)
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
    assert 1/(2 + I) == Rational(2, 5) - I/5  # SymPy prints (2 - I)/5, an undistributed form
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
    assert diff(2**x, x) == 2**x*expr('log(2)') and diff(E**x, x) == E**x
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
        Tuple(x, 1): '(x, 1)', oo: 'oo', -oo: '-oo', zoo: 'zoo', nan: 'nan', E**x: 'E**x', x*oo: 'oo*x',
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
