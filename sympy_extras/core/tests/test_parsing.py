"""The native parser of sympify against SymPy's parse_expr (evaluating)."""
from __future__ import annotations

import math
import random

import pytest
import sympy
from sympy.parsing.sympy_parser import parse_expr

from sympy_extras.core import (Add, And, Basic, Derivative, Eq, Equivalent, Expr, Float, Function, Ge, Gt, I, Implies,
    Integer, Le, Lt, Mod, Mul, Ne, Not, Order, sympify,
    Or, Pow, Rational, Relational, Symbol, Xor, as_expr, pow, srepr, sstr)
from sympy_extras.core.numbers import ImaginaryUnit
from sympy_extras.core.parsing import ParseUnsupported, parse
from sympy_extras.interop import to_sympy

TEXTS = ['(x**2 - 1)/(x - 1)', 'Abs(x)', 'cos(3)', 'cos(a_)', 'exp(-1)', 'exp(-x)', 'exp(2*x)', 'f(2*x + 4*y)', 'f(x)',
         'gamma(x + 1)', 'log(1/(4096*x**4))', 'log(2)', 'log(3*z/8 - 3*pi/8)', 'log(p*q)', 'sin(x)**2 + cos(x)**2',
         'sin(x)*cos(x)', 'x + I*y', 'x**2 + 1', '2/3', '1/2 + 1/3', '-x**2', '2**-1', '2**3**2', '-2**2', 'x - y - z',
         'x/y/z', '2.5*x', '1e-3', '.5', '3.', 'log(8, 2)', 'sqrt(8)', 'besselj(0, x)', 'Max(x, 2, 3)', 'oo - oo',
         'E**(I*pi)', 'zoo', 'x*(y + 1)**2', '+x', '--x', 'x**-2', 'sin(cos(tan(x)))', 'f(x, y)', 'GoldenRatio**2',
         'EulerGamma + Catalan', 'x - 2*x', 'nan', '(x)', 'x*2*y', 'atan2(y, x)', 'binomial(n, k)', 'factorial(5)',
         'Rational(1, 2)', 'Integer(7)']


def test_parse_matches_sympy() -> None:
    for text in TEXTS:
        assert sympy.srepr(to_sympy(parse(text), evaluate=False)) == sympy.srepr(parse_expr(text)), text


def test_outside_the_grammar() -> None:
    for text in ['not an expression', 'x if y else z', 'x +', '(x', 'x == y', "f'(x)", 'lambda x: x']:
        with pytest.raises(ParseUnsupported):
            parse(text)


# -- printing then parsing: the value of an unevaluated tree is kept
# (sympy/sympy#30844: the string printers were reviewed for missing
# parentheses; ours printed 1/(1/x) as 1/1/x and x/(1/y) as x/1/y)

_VALUES = {'x': complex(1.3), 'y': complex(0.7), 'z': complex(2.1)}


def _value(e: Expr) -> complex:
    """The value of an unevaluated tree of sums, products and powers."""
    if isinstance(e, Symbol):
        return _VALUES[e.name]
    if isinstance(e, Rational):
        return complex(e.p / e.q)
    if isinstance(e, Float):
        return complex(float(e.as_fraction()))
    if isinstance(e, ImaginaryUnit):
        return 1j
    if isinstance(e, Add):
        return sum((_value(a) for a in e.args), complex(0))
    if isinstance(e, Mul):
        product = complex(1)
        for a in e.args:
            product *= _value(a)
        return product
    if isinstance(e, Pow):
        return complex(_value(e.base) ** _value(e.exp))
    if isinstance(e, Mod):
        p, q = _value(e.args[0]).real, _value(e.args[1]).real
        return complex(p - q * math.floor(p / q))
    raise TypeError(type(e))


def _random_leaf(r: random.Random) -> Expr:
    k = r.random()
    if k < 0.4:
        return Symbol(r.choice(['x', 'y', 'z']))
    if k < 0.7:
        return Integer(r.choice([-3, -2, -1, 1, 2, 3]))
    if k < 0.85:
        return Rational(r.choice([-3, -1, 1, 2, 5]), r.choice([2, 3, 7]))
    if k < 0.93:
        return Float(r.choice(['-2.5', '0.5', '1.25', '-0.125']))
    return I


def _random_tree(r: random.Random, depth: int) -> Expr:
    if depth == 0 or r.random() < 0.15:
        return _random_leaf(r)
    k = r.random()
    if k < 0.33:
        return Add(*[_random_tree(r, depth - 1) for _ in range(r.choice([2, 2, 3]))])
    if k < 0.7:
        return Mul(*[_random_tree(r, depth - 1) for _ in range(r.choice([2, 2, 3]))])
    if r.random() < 0.5:
        return Pow(_random_tree(r, depth - 1), _random_tree(r, depth - 1))
    return Pow(_random_tree(r, depth - 1), r.choice([Integer(-2), Integer(-1), Integer(2), Rational(-1, 2),
                                                     Rational(1, 2), Rational(-3, 2)]))


def test_printed_unevaluated_trees_keep_their_value() -> None:
    r = random.Random(30844)
    checked = 0
    for _ in range(4000):
        e = _random_tree(r, r.choice([1, 2, 2, 3]))
        try:
            expected = _value(e)
        except (ZeroDivisionError, OverflowError):
            continue
        text = sstr(e)
        try:
            found = _value(parse(text, evaluate=False))
        except (ZeroDivisionError, OverflowError):
            continue
        assert abs(found - expected) <= 1e-9 * max(1.0, abs(expected)), (srepr(e), text)
        checked += 1
    assert checked > 3000


def test_quotient_powers_are_parenthesized() -> None:
    x, y = Symbol('x'), Symbol('y')
    minus_one, half = Integer(-1), Rational(-1, 2)
    assert sstr(Pow(Pow(x, minus_one), minus_one)) == '1/(1/x)'
    assert sstr(Mul(x, Pow(Pow(y, minus_one), minus_one))) == 'x/(1/y)'
    assert sstr(Mul(Integer(-1), x, Pow(Rational(2, 3), minus_one))) == '-x/(2/3)'
    assert sstr(Pow(Pow(x, half), minus_one)) == '1/(1/sqrt(x))'
    assert sstr(Pow(Pow(x, minus_one), Integer(2))) == '(1/x)**2'
    assert sstr(Pow(Pow(x, minus_one), y)) == '(1/x)**y'
    assert sstr(Pow(x, Pow(y, minus_one))) == 'x**(1/y)'
    # the noncommutative powers are not printed as quotients
    a = Symbol('A', commutative=False)
    assert sstr(Pow(Pow(a, minus_one), minus_one)) == '(A**(-1))**(-1)'
    for text in ['1/(1/x)', 'x/(1/y)', '(1/x)**2', '1/(1/sqrt(x))', 'x/(y/z)', '(-x)**2', '-x**2',
                 'x - (y + z)', 'x**(y**z)', '(x**y)**z', '(-2)**x', '(2/3)**x', '2*(-x)', 'x*(-2)', '(-1/2)**x']:
        assert sstr(parse(text, evaluate=False)) == text


def test_evaluated_expressions_round_trip() -> None:
    for text in TEXTS + ['Mod(x, 3)', 'x**(1/y)', '1/sqrt(x + 1)', '(x + 1)**(-2)/y', '-(x + y)/(2*z)',
                         '2**(-x)', '(-x)**(1/3)', 'x**y**z', '(x**y)**z', 'sqrt(2)*x/3', '1/(2*sqrt(x))',
                         '-2.5*x**2', '1.0e-20*x', 'Mod(x, 3)**2 + 2*Mod(y, 5)', 'exp(-x)/x', 'I*x/2']:
        if text in ('nan', 'oo - oo'):
            continue  # nan is not equal to itself
        e = parse(text)
        assert parse(sstr(e)) == e, text


def test_mod_is_read_as_the_modulo() -> None:
    # it was read as an undefined function named Mod
    x = Symbol('x')
    assert parse('Mod(7, 3)') == Integer(1)
    assert parse('Mod(x, 3)') == Mod(x, Integer(3)) and isinstance(parse('Mod(x, 3)'), Mod)
    assert parse('Mod(7, 3)', evaluate=False) == Mod(Integer(7), Integer(3))


def test_unevaluated_roots() -> None:
    # sqrt, cbrt and root evaluated their argument without evaluation
    half = Rational(1, 2)
    assert parse('sqrt(4)', evaluate=False) == Pow(Integer(4), half)
    assert parse('sqrt(x*4)', evaluate=False) == Pow(Mul(Symbol('x'), Integer(4)), half)
    assert parse('cbrt(8)', evaluate=False) == Pow(Integer(8), Rational(1, 3))
    assert parse('root(8, 3)', evaluate=False) == Pow(Integer(8), Rational(1, 3))
    assert parse('sqrt(4)') == Integer(2)
    # and the power of an unevaluated 0*3 lost the zero: sqrt(0*3) was sqrt(3)
    assert pow(Mul(Integer(0), Integer(3)), half) != Pow(Integer(3), half)
    assert parse('sqrt(0*3)', evaluate=False) == Pow(Mul(Integer(0), Integer(3)), half)


def _truth(e: Basic, booleans: dict[str, bool], numbers: dict[str, int]) -> bool:
    """The truth value of an unevaluated tree of connectives and relations."""
    if isinstance(e, Symbol):
        return booleans[e.name]
    if isinstance(e, And):
        return all(_truth(a, booleans, numbers) for a in e.args)
    if isinstance(e, Or):
        return any(_truth(a, booleans, numbers) for a in e.args)
    if isinstance(e, Not):
        return not _truth(e.args[0], booleans, numbers)
    if isinstance(e, Xor):
        return sum(_truth(a, booleans, numbers) for a in e.args) % 2 == 1
    if isinstance(e, Implies):
        return not _truth(e.args[0], booleans, numbers) or _truth(e.args[1], booleans, numbers)
    if isinstance(e, Equivalent):
        return len({_truth(a, booleans, numbers) for a in e.args}) == 1
    if isinstance(e, Relational):
        values = {Symbol(name): Integer(value) for name, value in numbers.items()}
        left, right = as_expr(e.lhs.xreplace(values).doit()), as_expr(e.rhs.xreplace(values).doit())
        assert isinstance(left, Rational) and isinstance(right, Rational)
        p, q = left.as_fraction(), right.as_fraction()
        return {'<': p < q, '>': p > q, '<=': p <= q, '>=': p >= q, '==': p == q, '!=': p != q}[e.rel_op]
    raise TypeError(type(e))


def test_printed_booleans_and_relations_keep_their_truth_value() -> None:
    # the printed form is read by Python's grammar (SymPy's parse_expr, with
    # the values in place of the names) and must have the truth value of
    # the tree: the parentheses of ~, &, ^, | and of the relations
    a, b, c, x, y = (Symbol(n) for n in 'abcxy')
    trees: list[Basic] = [
        Not(Lt(x, y)), And(Lt(x, y), Gt(x, y*y)), Or(And(a, b), c), And(Or(a, b), c), Not(And(a, b)), Not(Or(a, b)),
        Xor(a, And(b, c)), Xor(a, Or(b, c)), And(a, Xor(b, c)), Or(a, Xor(b, c)), Implies(a, Implies(b, c)),
        Implies(Implies(a, b), c), And(Implies(a, b), c), Not(Eq(x, y)), Not(Not(a)), Lt(x + 1, 2*y),
        And(Eq(x, y), Ne(y, x*x)), Xor(Lt(x, y), c), Or(Lt(x, y), c), Equivalent(a, And(b, c)), Not(Xor(a, b)),
        Not(Implies(a, b)), And(Not(a), b), Xor(Not(a), b), Lt(x, Mul(Integer(-1), y)),
        Or(Not(Ge(x, y)), Le(x, 2)), Not(Equivalent(a, b)), Xor(Xor(a, b), c), And(And(a, b), c),
        Or(a, And(b, Not(c))), Xor(Or(a, b), And(b, c)), Not(Xor(Not(a), Or(b, c))), Or(Xor(a, b), Xor(b, c)),
        And(Lt(Add(x, Mul(Integer(-1), y)), Integer(0)), Not(Ge(Pow(x, Integer(2)), y)))]
    for tree in trees:
        text = sstr(tree)
        for flags in [(p, q, r) for p in (False, True) for q in (False, True) for r in (False, True)]:
            for numbers in [{'x': 1, 'y': 2}, {'x': 2, 'y': 1}, {'x': -1, 'y': -1}]:
                booleans = dict(zip('abc', flags))
                names: dict[str, sympy.Basic] = {n: sympy.true if v else sympy.false for n, v in booleans.items()}
                names.update({n: sympy.Integer(v) for n, v in numbers.items()})
                assert bool(parse_expr(text, local_dict=names)) == _truth(tree, booleans, numbers), (text, flags, numbers)


def test_names_of_sympy_are_not_undefined_functions() -> None:
    # Eq(x, y), Derivative(...), O(x), S(1), diff(...) were read as undefined functions of
    # those names and true, Reals as symbols: they are outside the grammar (for the bridge)
    for text in ['Eq(x, y)', 'Derivative(x**2, x)', 'O(x)', 'S(1)/2', 'Float(1.5)', 'diff(x**2, x)', 'And(x, y)',
                 'AccumBounds(-1, 1)', 'Integral(x, x)', 'Mod(x, y, z)', 'true', 'x + Reals', 'N(pi)']:
        with pytest.raises(ParseUnsupported):
            parse(text)
        with pytest.raises(ParseUnsupported):
            parse(text, evaluate=False)
    # the other names are symbols and undefined functions, as before
    x = Symbol('x')
    assert parse('F(x) + gamma + S') == Function('F')(x) + Symbol('gamma') + Symbol('S')
    # and with the bridge the printed forms read back
    y = Symbol('y')
    printed: list[Basic] = [Eq(x, y), Lt(x + 1, 2*y), And(Lt(x, y), Ne(x, 1)), Or(x, Xor(y, Not(x))),
                            Derivative(Function("f")(x), x, x), Order(x**2, x), Implies(x, y), Mod(x, 3) + 1]
    for e in printed:
        assert sympify(sstr(e)) == e, sstr(e)
