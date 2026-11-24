"""The parser of ``sympify`` for strings: arithmetic expressions in Python's
syntax, as SymPy's ``parse_expr`` reads them with its standard
transformations.

Numbers are integers or decimals (``Integer``, ``Float``; ``1/2`` is the
quotient of two integers, ``Rational(1, 2)``); names are the constants
(``pi``, ``E``, ``I``, ``oo``, ...), the functions (``sin(x)``,
``log(x, 2)``, ``besselj(n, x)``: SymPy's names) and otherwise symbols, or
undefined functions when called (``f(x)``), except the other names which
SymPy calls (``Eq(x, y)``, ``O(x)``, ``S(1)``, ``diff(x**2, x)``) and the names of the Booleans and of the sets (``true``,
``Reals``), which are outside the grammar; the operators are ``+``,
``-``, ``*``, ``/``, ``**`` (right associative, binding tighter than a
unary sign: ``-x**2`` is ``-(x**2)``), with parentheses and calls. The
unevaluated nodes ``Integral``, ``Sum``, ``Product``, ``Limit`` and
``Derivative`` are read with SymPy's argument forms (``(x, 0, 1)``,
``x, 2``, ``(x, 2)``, ``dir='-'``) as far as the native nodes have them: an
integral in several variables is the nested integrals and ``(x, a)`` of an
integral, the antiderivative at ``a``, is ``Subs(Integral(f, x), x, a)``; a
symbolic order of differentiation, a sum or a product without bounds
(SymPy refuses them too) and a variable which is not a symbol are outside
the grammar.
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
>>> parse('Integral(x**2, (x, 0, 1))'), parse('Sum(1/k**2, (k, 1, oo))'), parse('Derivative(f(x), x, 2)')
(Integral(x**2, (x, 0, 1)), Sum(k**(-2), (k, 1, oo)), Derivative(f(x), (x, 2)))
>>> parse("Limit(1/x, x, 0, dir='-')"), parse('Integral(x*y, x, (y, 0, 1))')
(Limit(1/x, x, 0, dir='-'), Integral(Integral(x*y, x), (y, 0, 1)))
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Callable, Optional, Union

from ._foreign_names import SYMPY_CALLABLE_NAMES, SYMPY_CONSTANT_NAMES
from .basic import Basic
from .expr import Expr
# imported by sympify inside its function (the core is loaded by then)
from .function import DefinedFunction
from .symbol import Symbol

__all__ = ['parse', 'ParseUnsupported']


class ParseUnsupported(ValueError):
    """The text is outside the grammar of the native parser."""


_TOKEN = re.compile(r'\s*(?:(?P<number>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+)'
                    r'|(?P<name>[A-Za-z_][A-Za-z_0-9]*)'
                    r"""|(?P<string>'[^'\\]*'|"[^"\\]*")"""
                    r'|(?P<op>\*\*|[-+*/(),=\[\]]))')

#: a token: its kind ('number', 'name', 'string', 'op', 'end') and its text
Token = tuple[str, str]

#: the unevaluated nodes read with SymPy's argument forms
_NODE_NAMES = frozenset({'Integral', 'Sum', 'Product', 'Limit', 'Derivative'})


@dataclass(frozen=True)
class _Argument:
    """An argument of ``Integral``, ``Sum``, ``Product``, ``Limit`` or
    ``Derivative``: an expression (one item), a tuple or a list of
    expressions (``grouped``), or a string (``Limit``'s direction), with
    the name of its keyword if it has one."""

    items: tuple[Expr, ...]
    grouped: bool
    string: Optional[str]
    keyword: Optional[str]


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
        if self.peek() == ('op', '(') and value in _NODE_NAMES:
            self.take()
            arguments: list[_Argument] = []
            if self.peek() != ('op', ')'):
                arguments.append(self.argument())
                while self.peek() == ('op', ','):
                    self.take()
                    arguments.append(self.argument())
            self.expect(')')
            try:
                return self.node(value, arguments)
            except ParseUnsupported:
                raise
            except (TypeError, ValueError, NotImplementedError) as error:
                # what the constructors refuse (a limit at a point depending
                # on the variable, a negative order, ...)
                raise ParseUnsupported('%s in %r' % (error, self.text)) from None
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

    # -- the arguments of Integral, Sum, Product, Limit and Derivative

    def argument(self) -> _Argument:
        keyword: Optional[str] = None
        if self.peek()[0] == 'name' and self.tokens[self.position + 1] == ('op', '='):
            keyword = self.take()[1]
            self.take()
        kind, value = self.peek()
        if kind == 'string':
            self.take()
            return _Argument((), False, value[1:-1], keyword)
        if kind == 'op' and value in ('(', '['):
            group = self.group()
            if group is not None:
                return _Argument(group, True, None, keyword)
        return _Argument((self.sum(),), False, None, keyword)

    def group(self) -> Optional[tuple[Expr, ...]]:
        """The items of a tuple or of a list which is a whole argument;
        ``None``, with nothing consumed, for a parenthesised expression."""
        start = self.position
        close = ')' if self.take()[1] == '(' else ']'
        items: list[Expr] = []
        commas = 0
        while self.peek() != ('op', close):
            items.append(self.sum())
            if self.peek() != ('op', ','):
                break
            self.take()
            commas += 1
        self.expect(close)
        whole = self.peek() in (('op', ','), ('op', ')'))
        if whole and (commas > 0 or close == ']'):
            return tuple(items)
        self.position = start
        return None

    def node(self, name: str, arguments: list[_Argument]) -> Expr:
        if not arguments or arguments[0].grouped or arguments[0].string is not None \
                or arguments[0].keyword is not None:
            raise ParseUnsupported('%s(...) in %r' % (name, self.text))
        expr = arguments[0].items[0]
        rest = arguments[1:]
        if name == 'Limit':
            return self.limit_node(expr, rest)
        if any(a.string is not None or a.keyword is not None for a in rest):
            raise ParseUnsupported('%s(...) in %r' % (name, self.text))
        if name == 'Integral':
            return self.integral_node(expr, rest)
        if name == 'Derivative':
            return self.derivative_node(expr, rest)
        return self.sum_node(name, expr, rest)

    def the_symbol(self, e: Expr) -> Symbol:
        if not isinstance(e, Symbol):
            raise ParseUnsupported('%s is not a symbol in %r' % (e, self.text))
        return e

    def the_free_symbol(self, e: Expr) -> Symbol:
        # SymPy's: without variables, the only free symbol of the expression
        free = e.free_symbols
        if len(free) != 1:
            raise ParseUnsupported('the variable of %s is not given in %r' % (e, self.text))
        return next(iter(free))

    def limit_node(self, expr: Expr, rest: list[_Argument]) -> Expr:
        from .limits import Limit
        direction = '+'
        if len(rest) == 3 and rest[2].string is not None and rest[2].keyword in (None, 'dir'):
            direction = rest[2].string
            rest = rest[:2]
        if len(rest) != 2 or any(a.grouped or a.string is not None or a.keyword is not None for a in rest):
            raise ParseUnsupported('Limit(...) in %r' % (self.text,))
        return Limit(expr, self.the_symbol(rest[0].items[0]), rest[1].items[0], direction)

    def integral_node(self, expr: Expr, rest: list[_Argument]) -> Expr:
        integral = importlib.import_module('sympy_extras.integrals.integral')
        build: Callable[..., Basic] = integral.Integral
        if not rest:
            return as_expr_checked(build(expr, self.the_free_symbol(expr)), self.text)
        result = expr
        for a in rest:
            # SymPy's Integral(f, x, y) integrates over x first: the nested
            # integrals (the native node has one variable)
            if len(a.items) == 1:
                result = as_expr_checked(build(result, self.the_symbol(a.items[0])), self.text)
            elif len(a.items) == 3:
                result = as_expr_checked(build(result, self.the_symbol(a.items[0]), a.items[1], a.items[2]),
                                         self.text)
            elif len(a.items) == 2:
                # (x, a), the antiderivative at a: the substitution in the
                # indefinite integral (the native node has not this form)
                from .function import Subs
                variable = self.the_symbol(a.items[0])
                result = Subs(as_expr_checked(build(result, variable), self.text), variable, a.items[1])
            else:
                raise ParseUnsupported('the limit %s of an integral in %r' % (a.items, self.text))
        return result

    def sum_node(self, name: str, expr: Expr, rest: list[_Argument]) -> Expr:
        limits = importlib.import_module('sympy_extras.concrete.expr_with_limits')
        build: Callable[..., Basic]
        if name == 'Sum':
            build = limits.sum_ if self.evaluate else limits.Sum
        else:
            build = limits.product_ if self.evaluate else limits.Product
        triples: list[tuple[Symbol, Expr, Expr]] = []
        for a in rest:
            if not a.grouped or len(a.items) != 3:
                raise ParseUnsupported('%s requires values for lower and upper bounds in %r' % (name, self.text))
            triples.append((self.the_symbol(a.items[0]), a.items[1], a.items[2]))
        if not triples:
            raise ParseUnsupported('%s requires limits in %r' % (name, self.text))
        return as_expr_checked(build(expr, *triples), self.text)

    def derivative_node(self, expr: Expr, rest: list[_Argument]) -> Expr:
        from .function import Derivative
        from .numbers import Integer
        counts: list[tuple[Symbol, Expr]] = []
        after_symbol = False
        for a in rest:
            if a.grouped:
                if len(a.items) != 2:
                    raise ParseUnsupported('the variable %s of a derivative in %r' % (a.items, self.text))
                # (x, 2), or a symbolic order (x, n)
                counts.append((self.the_symbol(a.items[0]), a.items[1]))
                after_symbol = False
            elif isinstance(a.items[0], Integer):
                if not after_symbol:
                    raise ParseUnsupported('the order %s must follow a variable in %r' % (a.items[0], self.text))
                counts[-1] = (counts[-1][0], a.items[0])
                after_symbol = False
            else:
                counts.append((self.the_symbol(a.items[0]), Integer(1)))
                after_symbol = True
        if not rest:
            counts.append((self.the_free_symbol(expr), Integer(1)))
        if self.evaluate:
            # SymPy's constructor: the orders 0 dropped, the derivative of a
            # derivative merged
            counts = [c for c in counts if c[1] != 0]
            if isinstance(expr, Derivative):
                counts = list(expr.variable_orders) + counts
                expr = expr.expr
            if not counts:
                return expr
        try:
            return Derivative(expr, *counts)
        except (TypeError, ValueError) as error:
            # an order which is not a nonnegative integer
            raise ParseUnsupported('%s in %r' % (error, self.text))

    # -- the leaves and the nodes

    def number(self, text: str) -> Expr:
        from .numbers import Float, Integer
        if text.isdigit():
            return Integer(int(text))
        return Float(text)

    def name(self, text: str) -> Expr:
        from .numbers import Catalan, E, EulerGamma, GoldenRatio, I, TribonacciConstant, nan, oo, pi, zoo
        constants: dict[str, Expr] = {'pi': pi, 'E': E, 'I': I, 'oo': oo, 'zoo': zoo, 'nan': nan,
                                      'EulerGamma': EulerGamma, 'Catalan': Catalan, 'GoldenRatio': GoldenRatio,
                                      'TribonacciConstant': TribonacciConstant}
        found = constants.get(text)
        if found is None and text in SYMPY_CONSTANT_NAMES:
            raise ParseUnsupported('%s in %r' % (text, self.text))  # true, Reals: not symbols of that name
        return found if found is not None else Symbol(text)

    def function(self, text: str, args: list[Expr]) -> Expr:
        from .function import Function
        from .numbers import Integer, Rational
        if text in ('Rational', 'Integer'):
            if not all(isinstance(a, Integer) for a in args) or not 1 <= len(args) <= (2 if text == 'Rational' else 1):
                raise ParseUnsupported('%s of %s in %r' % (text, args, self.text))
            p, q = int(args[0]), (int(args[1]) if len(args) == 2 else 1)
            return Rational(p, q) if q != 1 else Integer(p)
        if text == 'Mod' and len(args) == 2:
            # Mod is an Application of the core, not a DefinedFunction: without
            # this it would be read as an undefined function named Mod
            from .mod import Mod, mod
            return mod(args[0], args[1]) if self.evaluate else Mod(args[0], args[1])
        if not self.evaluate:
            # the roots are functions which build powers: unevaluated ones here
            from .power import Pow
            if text == 'sqrt' and len(args) == 1:
                return Pow(args[0], Rational(1, 2))
            if text == 'cbrt' and len(args) == 1:
                return Pow(args[0], Rational(1, 3))
            if text == 'root' and len(args) == 2 and isinstance(args[1], Integer) and int(args[1]) != 0:
                return Pow(args[0], Rational(1, int(args[1])))
        f = _function_of(text, self.evaluate, len(args))
        if f is None:
            if text in SYMPY_CALLABLE_NAMES:
                # Eq(x, y), Derivative(f(x), x), O(x), S(1), diff(x**2, x): a class or
                # a function of SymPy which this parser has not, not an undefined
                # function of that name
                raise ParseUnsupported('%s(...) in %r' % (text, self.text))
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
        importlib.import_module('sympy_extras.functions')  # the classes register by being defined

        def walk(c: type) -> None:
            subclasses: list[type] = c.__subclasses__()
            for s in subclasses:
                if issubclass(s, DefinedFunction) and s.function_name != 'f':
                    _CLASSES.setdefault(s.function_name, s)
                walk(s)
        walk(DefinedFunction)
    return _CLASSES


def parse(text: str, evaluate: bool = True) -> Expr:
    """The expression of ``text`` (``ParseUnsupported`` outside the grammar)."""
    return _Parser(text, evaluate).parse()
