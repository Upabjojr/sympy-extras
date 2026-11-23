r"""A reader of the LaTeX the printer writes, for the tests only: it
evaluates a formula at given values of the symbols, with the conventions
of mathematical notation (a juxtaposition is a product, ``- a b`` is
``-(a b)``, ``a / b c`` would be ``(a / b) c``, a relation binds tighter
than a connective), independently of the printer.

The grammar covers what the random trees of ``test_latex_semantic.py``
print: numbers, symbols, sums, products, quotients, powers, roots, the
functions sin, cos, exp, log and the absolute value, ``mod``, relations
and connectives.
"""
from __future__ import annotations

import math
import re
from typing import Mapping, Union

#: a number or a truth value
Value = Union[float, bool]

_TOKEN = re.compile(r'\s*(\\[a-zA-Z]+|\\[^a-zA-Z]|\d+(?:\.\d+)?|.)', re.S)
_FUNCTIONS = {r'\sin': math.sin, r'\cos': math.cos, r'\log': math.log}
_RELATIONS = ('=', '<', '>', r'\leq', r'\geq', r'\neq')
_CLOSERS = ('}', r'\right', ')', ']', '&', '')


class ReadError(Exception):
    """The formula is outside the grammar of the reader."""


def _number(v: Value) -> float:
    if isinstance(v, bool):
        raise ReadError('a number is expected, got a truth value')
    return v


def _truth(v: Value) -> bool:
    if not isinstance(v, bool):
        raise ReadError('a truth value is expected, got a number')
    return v


class Reader:
    """A recursive descent over the tokens of a formula."""

    def __init__(self, tex: str, values: Mapping[str, float]) -> None:
        self.tokens: list[str] = [t for t in _TOKEN.findall(tex) if t.strip()]
        self.position: int = 0
        self.values: Mapping[str, float] = values

    def peek(self) -> str:
        return self.tokens[self.position] if self.position < len(self.tokens) else ''

    def take(self) -> str:
        token = self.peek()
        self.position += 1
        return token

    def expect(self, token: str) -> None:
        got = self.take()
        if got != token:
            raise ReadError('%r expected, got %r' % (token, got))

    # connectives, from the loosest

    def formula(self) -> Value:
        left = self.disjunction()
        while self.peek() in (r'\Rightarrow', r'\veebar', r'\Leftrightarrow', r'\not'):
            sign = self.take()
            if sign == r'\not':
                sign = self.take()
                right = self.disjunction()
                left = not self._binary(sign, _truth(left), _truth(right))
            else:
                left = self._binary(sign, _truth(left), _truth(self.disjunction()))
        return left

    @staticmethod
    def _binary(sign: str, a: bool, b: bool) -> bool:
        if sign == r'\Rightarrow':
            return (not a) or b
        if sign == r'\veebar':
            return a != b
        if sign == r'\Leftrightarrow':
            return a == b
        raise ReadError('unknown connective %r' % sign)

    def disjunction(self) -> Value:
        left = self.conjunction()
        while self.peek() == r'\vee':
            self.take()
            right = self.conjunction()
            left = _truth(left) or _truth(right)
        return left

    def conjunction(self) -> Value:
        left = self.negation()
        while self.peek() == r'\wedge':
            self.take()
            right = self.negation()
            left = _truth(left) and _truth(right)
        return left

    def negation(self) -> Value:
        if self.peek() == r'\neg':
            self.take()
            return not _truth(self.negation())
        return self.relation()

    def relation(self) -> Value:
        left = self.modulo()
        if self.peek() in _RELATIONS:
            sign = self.take()
            a, b = _number(left), _number(self.modulo())
            if self.peek() in _RELATIONS:
                raise ReadError('a chain of relations')
            return {'=': math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), '<': a < b, '>': a > b, r'\leq': a <= b,
                    r'\geq': a >= b, r'\neq': not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)}[sign]
        return left

    def modulo(self) -> Value:
        left = self.sum()
        if self.peek() == r'\bmod':
            self.take()
            a, b = _number(left), _number(self.sum())
            if self.peek() == r'\bmod':
                raise ReadError('a chain of mod')
            return a - b * math.floor(a / b)
        return left

    # arithmetic

    def sum(self) -> Value:
        if self.peek() == '-':
            self.take()
            total: Value = -_number(self.product())
        else:
            total = self.product()
        while self.peek() in ('+', '-'):
            sign = self.take()
            term = _number(self.product())
            total = _number(total) + term if sign == '+' else _number(total) - term
        return total

    def product(self) -> Value:
        value = self.power()
        while True:
            token = self.peek()
            if token == '/':
                self.take()
                value = _number(value) / _number(self.power())
            elif token in (r'\cdot', r'\times'):
                self.take()
                value = _number(value) * _number(self.power())
            elif token in _CLOSERS or token in ('+', '-', '|', ',', r'\bmod', r'\vee', r'\wedge', r'\Rightarrow',
                                                r'\veebar', r'\Leftrightarrow', r'\not', r'\neg') \
                    or token in _RELATIONS:
                return value
            else:
                value = _number(value) * _number(self.power())

    def power(self) -> Value:
        token = self.peek()
        if token in _FUNCTIONS or token == r'\operatorname':
            return self.function()
        base = self.atom()
        if self.peek() == '^':
            self.take()
            return _number(base) ** _number(self.group())
        return base

    def function(self) -> float:
        token = self.take()
        if token == r'\operatorname':
            raise ReadError('an unknown function')
        exponent = 1.0
        if self.peek() == '^':
            self.take()
            exponent = _number(self.group())
        self.expect('{')
        self.expect(r'\left')
        self.expect('(')
        argument = _number(self.formula())
        self.expect(r'\right')
        self.expect(')')
        self.expect('}')
        result = _FUNCTIONS[token](argument) ** exponent
        if self.peek() == '^':
            # \log{\left(x \right)}^{2}: the square of the logarithm
            self.take()
            result = result ** _number(self.group())
        return result

    def group(self) -> Value:
        self.expect('{')
        value = self.formula()
        self.expect('}')
        return value

    def atom(self) -> Value:
        token = self.take()
        if re.fullmatch(r'\d+(?:\.\d+)?', token):
            return float(token)
        if token in self.values:
            return self.values[token]
        if token == 'e':
            return math.e
        if token == r'\frac':
            return _number(self.group()) / _number(self.group())
        if token == r'\sqrt':
            index = 2.0
            if self.peek() == '[':
                self.take()
                index = float(self.take())
                self.expect(']')
            return _number(self.group()) ** (1 / index)
        if token == '{':
            value = self.formula()
            self.expect('}')
            return value
        if token == r'\left':
            opening = self.take()
            if opening == '(':
                value = self.formula()
                self.expect(r'\right')
                self.expect(')')
                return value
            if opening == '|':
                inner = _number(self.group())
                self.expect(r'\right')
                self.expect('|')
                return abs(inner)
            raise ReadError('unknown delimiter %r' % opening)
        if token == r'\text':
            self.expect('{')
            word = ''
            while self.peek() != '}':
                word += self.take()
            self.expect('}')
            if word in ('True', 'False'):
                return word == 'True'
        raise ReadError('unexpected %r' % token)


def read_latex(tex: str, values: Mapping[str, float]) -> Value:
    r"""The value of the formula ``tex`` at the values of the symbols.

    Examples
    ========

    >>> from sympy_extras.printing.tests.latex_reader import read_latex
    >>> read_latex(r'- x^{2} + \frac{y}{2}', {'x': 3.0, 'y': 4.0}), read_latex(r'\left(- x\right)^{2}', {'x': 3.0})
    (-7.0, 9.0)
    >>> read_latex(r'\neg \left(x < y\right) \vee x = 3', {'x': 3.0, 'y': 4.0})
    True
    """
    reader = Reader(tex, values)
    value = reader.formula()
    if reader.peek() != '':
        raise ReadError('unexpected %r at the end' % reader.peek())
    return value
