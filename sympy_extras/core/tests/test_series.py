"""Oracle tests of the native Order and series machinery against SymPy."""
import random

import sympy

from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.core.order import Order, order
from sympy_extras.core.symbol import Symbol

s = sympy
x, y = s.symbols('x y')
X, Y = Symbol('x'), Symbol('y')
SPOINTS = {'0': s.Integer(0), '1': s.Integer(1), 'oo': s.oo}


def _ours(e: Basic) -> str:
    return str(to_sympy(e, evaluate=False))


SERIES_CASES = ['cos(x)', 'exp(x)', '1/(1-x)', '1/x', 'sin(x)/x', 'sqrt(1+x)', 'x**x', 'exp(x)/x', '1/(x**2+x)',
                '(1+x)**y', 'log(1+x)', 'tan(x)', 'x/(exp(x)-1)', '1/sin(x)', '(x+y)**3', 'sqrt(x+x**2)',
                'log(x)*x', '(exp(x)-1)/x', 'exp(1/x)*x', 'x**(1/3)+x', 'log(cos(x))', 'sqrt(1-x)/x**2',
                'log(sin(x))', 'exp(exp(x))', 'log(x**2+x)', 'exp(x*log(x)+x)', 'log(-x+x**2)', 'exp(-1/x)',
                'log(exp(x)-1)', 'x**y*log(x)', 'log(1-x)/x', 'exp(sqrt(x))', 'log(sqrt(x)+x)',
                'sec(x)', 'csc(x)', 'cot(x)', 'tan(x+1)', 'cosh(x)', 'tanh(x)', 'coth(x)', 'sech(x)', 'csch(x)',
                'tan(pi/2+x)', 'cot(pi+x)', 'sin(pi+x)', 'cos(pi/2+x)', 'sec(pi/2+x)', 'csc(pi+x)', 'tanh(sin(x))',
                'cos(log(x))*x', 'sin(sqrt(x))']


def _sympy_expr(text: str) -> sympy.Expr:
    e = s.sympify(text)
    assert isinstance(e, s.Expr)
    return e


def test_series_matches_sympy() -> None:
    for text in SERIES_CASES:
        e = _sympy_expr(text)
        for point, n in (('0', 6), ('1', 3)):
            try:
                want = str(e.series(x, SPOINTS[point], n))
            except Exception as error:
                want = type(error).__name__
            try:
                got = _ours(expr_from_sympy(e).series(X, expr_from_sympy(SPOINTS[point]), n))
            except Exception as error:
                got = type(error).__name__
            assert got == want, (text, point)


def test_as_leading_term_matches_sympy() -> None:
    for text in SERIES_CASES:
        e = _sympy_expr(text)
        try:
            want = str(e.as_leading_term(x))
        except Exception as error:
            want = type(error).__name__
        try:
            got = _ours(expr_from_sympy(e).as_leading_term(X))
        except Exception as error:
            got = type(error).__name__
        assert got == want, text


def test_order_matches_sympy() -> None:
    rng = random.Random(1)
    atoms: list[sympy.Expr] = [x, y, x**2, 1/x, s.log(x), s.exp(x), s.sin(x), s.sqrt(x), s.Integer(2),
                               s.Rational(1, 3), x*y]

    def rnd(d: int) -> sympy.Expr:
        if d == 0:
            return rng.choice(atoms)
        op = rng.choice('+*/^')
        a, b = rnd(d - 1), rnd(d - 1)
        if op == '+':
            return a + b
        if op == '*':
            return a * b
        if op == '/':
            return a / b if b != 0 else a
        return a ** rng.choice([s.Integer(2), s.Integer(-1), s.Rational(1, 2), s.Integer(3)])

    specs: list[list[tuple[str, str]]] = [[('x', '0')], [('x', 'oo')], [('x', '1')], [('x', '0'), ('y', '0')]]
    symbols = {'x': (x, X), 'y': (y, Y)}
    for _ in range(80):
        e = rnd(rng.randint(1, 3))
        spec = rng.choice(specs)
        try:
            want = str(s.Order(e, *[(symbols[v][0], SPOINTS[p]) for v, p in spec]))
        except Exception as error:
            want = type(error).__name__
        try:
            got = _ours(order(expr_from_sympy(e), *[Tuple(symbols[v][1], expr_from_sympy(SPOINTS[p]))
                                                    for v, p in spec]))
        except Exception as error:
            got = type(error).__name__
        assert got == want, (e, spec)


def test_order_arithmetic() -> None:
    o = order(X**2)
    assert _ours(X + o + X**2) == 'x + O(x**2)'
    assert _ours(1 + order(1, X)) == str(1 + s.Order(1, x))
    assert _ours(X * order(X)) == 'O(x**2)'
    assert _ours(-o) == 'O(x**2)'
    assert _ours(order(X).subs(X, X - 1)) == 'O(x - 1, (x, 1))'
    ox, ox2 = order(X), order(X**2)
    assert isinstance(ox, Order) and isinstance(ox2, Order)
    assert ox.contains(X**2) is True and ox2.contains(X) is False
