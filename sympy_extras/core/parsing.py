"""The parser of ``sympify`` for strings: arithmetic expressions in Python's
syntax, as SymPy's ``parse_expr`` reads them with its standard
transformations.

Numbers are integers or decimals (``Integer``, ``Float``; ``1/2`` is the
quotient of two integers, ``Rational(1, 2)``); names are the constants
(``pi``, ``E``, ``I``, ``oo``, ...), the functions (``sin(x)``,
``log(x, 2)``, ``besselj(n, x)``: SymPy's names) and otherwise symbols, or
undefined functions when called (``f(x)``); the operators are ``+``,
``-``, ``*``, ``/``, ``**`` (right associative, binding tighter than a
unary sign: ``-x**2`` is ``-(x**2)``), with parentheses and calls.
Anything else is :class:`ParseUnsupported`: ``sympify`` then asks the
registered bridge, if any.

Examples
========

>>> from sympy_extras.core.parsing import parse
>>> parse('x**2 + 1'), parse('sin(x)**2 + cos(x)**2'), parse('(x**2 - 1)/(x - 1)')
(x**2 + 1, sin(x)**2 + cos(x)**2, (x**2 - 1)/(x - 1))
>>> parse('2/3 + 0.5'), parse('-x**2'), parse('2**-1'), parse('f(2*x + 4*y)'), parse('log(8, 2)')
(1.16666666666667, -x**2, 1/2, f(2*x + 4*y), 3)
>>> parse('x + x', evaluate=False)
x + x
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Union

from .basic import Basic
from .expr import Expr
# imported by sympify inside its function (the core is loaded by then)
from .function import DefinedFunction

__all__ = ['parse', 'ParseUnsupported']


class ParseUnsupported(ValueError):
    """The text is outside the grammar of the native parser."""


_TOKEN = re.compile(r'\s*(?:(?P<number>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+)'
                    r'|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>\*\*|[-+*/(),]))')

#: a token: its kind ('number', 'name', 'op', 'end') and its text
Token = tuple[str, str]


def _tokens(text: str) -> list[Token]:
    out: list[Token] = []
    position = 0
    stripped = text.rstrip()
    while position < len(stripped):
        m = _TOKEN.match(stripped, position)
        if m is None or m.end() == position:
            raise ParseUnsupported('cannot parse %r at %d' % (text, position))
        kind = m.lastgroup
        assert kind is not None
        out.append((kind, m.group(kind)))
        position = m.end()
    out.append(('end', ''))
    return out


class _Parser:
    """Recursive descent over the tokens, building the core's expressions."""

    def __init__(self, text: str, evaluate: bool) -> None:
        self.tokens = _tokens(text)
        self.position = 0
        self.evaluate = evaluate
        self.text = text

    def peek(self) -> Token:
        return self.tokens[self.position]

    def take(self) -> Token:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def expect(self, op: str) -> None:
        kind, value = self.take()
        if kind != 'op' or value != op:
            raise ParseUnsupported('expected %r in %r' % (op, self.text))

    def parse(self) -> Expr:
        e = self.sum()
        if self.peek()[0] != 'end':
            raise ParseUnsupported('unexpected %r in %r' % (self.peek()[1], self.text))
        return e

    def sum(self) -> Expr:
        e = self.product()
        while self.peek() in (('op', '+'), ('op', '-')):
            op = self.take()[1]
            right = self.product()
            e = self.add(e, right if op == '+' else self.negate(right))
        return e

    def product(self) -> Expr:
        e = self.unary()
        while self.peek() in (('op', '*'), ('op', '/')):
            op = self.take()[1]
            right = self.unary()
            e = self.mul(e, right if op == '*' else self.power_of(right, -1))
        return e

    def unary(self) -> Expr:
        if self.peek() == ('op', '-'):
            self.take()
            return self.negate(self.unary())
        if self.peek() == ('op', '+'):
            self.take()
            return self.unary()
        return self.power()

    def power(self) -> Expr:
        base = self.call()
        if self.peek() == ('op', '**'):
            self.take()
            return self.pow(base, self.unary())
        return base

    def call(self) -> Expr:
        kind, value = self.take()
        if kind == 'number':
            return self.number(value)
        if kind == 'op' and value == '(':
            e = self.sum()
            self.expect(')')
            return e
        if kind != 'name':
            raise ParseUnsupported('unexpected %r in %r' % (value, self.text))
        if self.peek() == ('op', '('):
            self.take()
            args: list[Expr] = []
            if self.peek() != ('op', ')'):
                args.append(self.sum())
                while self.peek() == ('op', ','):
                    self.take()
                    args.append(self.sum())
            self.expect(')')
            return self.function(value, args)
        return self.name(value)

    # -- the leaves and the nodes

    def number(self, text: str) -> Expr:
        from .numbers import Float, Integer
        if text.isdigit():
            return Integer(int(text))
        return Float(text)

    def name(self, text: str) -> Expr:
        from .numbers import Catalan, E, EulerGamma, GoldenRatio, I, TribonacciConstant, nan, oo, pi, zoo
        from .symbol import Symbol
        constants: dict[str, Expr] = {'pi': pi, 'E': E, 'I': I, 'oo': oo, 'zoo': zoo, 'nan': nan,
                                      'EulerGamma': EulerGamma, 'Catalan': Catalan, 'GoldenRatio': GoldenRatio,
                                      'TribonacciConstant': TribonacciConstant}
        found = constants.get(text)
        return found if found is not None else Symbol(text)

    def function(self, text: str, args: list[Expr]) -> Expr:
        from .function import Function
        from .numbers import Integer, Rational
        if text in ('Rational', 'Integer'):
            if not all(isinstance(a, Integer) for a in args) or not 1 <= len(args) <= (2 if text == 'Rational' else 1):
                raise ParseUnsupported('%s of %s in %r' % (text, args, self.text))
            p, q = int(args[0]), (int(args[1]) if len(args) == 2 else 1)
            return Rational(p, q) if q != 1 else Integer(p)
        f = _function_of(text, self.evaluate, len(args))
        if f is None:
            return as_expr_checked(Function(text)(*args), self.text)
        return as_expr_checked(f(*args), self.text)

    def add(self, a: Expr, b: Expr) -> Expr:
        from .add import Add, add
        return add(a, b) if self.evaluate else Add(a, b)

    def mul(self, a: Expr, b: Expr) -> Expr:
        from .mul import Mul, mul
        return mul(a, b) if self.evaluate else Mul(a, b)

    def negate(self, a: Expr) -> Expr:
        from .numbers import Integer, Number
        if not self.evaluate and isinstance(a, Number):
            return -a
        return self.mul(Integer(-1), a)

    def pow(self, a: Expr, b: Expr) -> Expr:
        from .power import Pow, pow
        return pow(a, b) if self.evaluate else Pow(a, b)

    def power_of(self, a: Expr, n: int) -> Expr:
        from .numbers import Integer
        return self.pow(a, Integer(n))


def as_expr_checked(b: Basic, text: str) -> Expr:
    if not isinstance(b, Expr):
        raise ParseUnsupported('%r is not an expression' % (text,))
    return b


#: a function of the core by SymPy's name
NamedFunction = Callable[..., Basic]


def _function_of(name: str, evaluate: bool, count: int) -> Optional[NamedFunction]:
    """The evaluating function of that name (``sympy_extras.functions``,
    then the classes' ``apply``), or the class itself without evaluation;
    ``None`` for an unknown name (an undefined function)."""
    import sympy_extras.functions as functions
    classes = _classes_by_name()
    cls = classes.get(name)
    if not evaluate and cls is not None and (cls.variadic or count in cls.nargs):
        return cls
    candidate: Union[object, None] = vars(functions).get(name)
    if callable(candidate) and not (isinstance(candidate, type) and issubclass(candidate, DefinedFunction)):
        return candidate
    if cls is not None:
        return cls.apply
    return None


_CLASSES: dict[str, type[DefinedFunction]] = {}


def _classes_by_name() -> dict[str, type[DefinedFunction]]:
    if not _CLASSES:
        import sympy_extras.functions  # noqa: F401 (the classes register by being defined)

        def walk(c: type) -> None:
            for s in c.__subclasses__():
                if issubclass(s, DefinedFunction) and s.function_name != 'f':
                    _CLASSES.setdefault(s.function_name, s)
                walk(s)
        walk(DefinedFunction)
    return _CLASSES


def parse(text: str, evaluate: bool = True) -> Expr:
    """The expression of ``text`` (``ParseUnsupported`` outside the grammar)."""
    return _Parser(text, evaluate).parse()
