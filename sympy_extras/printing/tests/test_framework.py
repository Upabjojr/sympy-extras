"""The framework of the printers on a toy printer of its own: the
brackets come from the result of the child and its context, whatever the
node, and the rules are found by class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.printing.framework import (Binding, Context, Dispatcher, Position, Result, TreePrinter,
    needs_brackets)


@dataclass(frozen=True)
class Leaf:
    name: str


@dataclass(frozen=True)
class Negative(Leaf):
    pass


@dataclass(frozen=True)
class Power:
    base: Union[Leaf, Power]
    exponent: Union[Leaf, Power]


Node = Union[Leaf, Power]


class Toy(TreePrinter[Node, str]):
    """Leaves, and powers of them, bracketed with square brackets."""

    def rule(self, node: Node, context: Context[str]) -> Result[str]:
        found = TOY.lookup(type(node))
        if found is None:
            raise TypeError('no rule for %r' % (node,))
        return found(self, node, context)

    def bracket(self, result: Result[str], context: Context[str]) -> str:
        return '[%s]' % result.output


TOY: Dispatcher[Toy, Node, str] = Dispatcher()


@TOY.of(Leaf)
def _leaf(printer: Toy, node: Leaf, context: Context[str]) -> Result[str]:
    if context.exponent_offered is not None and node.name == 'sin':
        return Result('sin^%s' % context.exponent_offered, Binding.POWER, absorbed=True)
    return Result(node.name, Binding.ATOM)


@TOY.of(Negative)
def _negative(printer: Toy, node: Negative, context: Context[str]) -> Result[str]:
    return Result('-' + node.name, Binding.NEGATION, unsigned=node.name)


@TOY.of(Power)
def _power(printer: Toy, node: Power, context: Context[str]) -> Result[str]:
    exponent = printer.print(node.exponent, context.exponent()).output
    base = printer.print(node.base, context.base(exponent))
    if base.absorbed:
        return Result(base.output, base.binding)
    return Result('%s^%s' % (base.output, exponent), Binding.POWER)


def test_brackets_follow_the_printed_form() -> None:
    toy = Toy()
    top: Context[str] = Context()
    assert toy.print(Power(Leaf('x'), Leaf('y')), top).output == 'x^y'
    assert toy.print(Power(Negative('x'), Leaf('y')), top).output == '[-x]^y'
    assert toy.print(Power(Power(Leaf('x'), Leaf('y')), Leaf('z')), top).output == '[x^y]^z'
    assert toy.print(Power(Leaf('x'), Power(Leaf('y'), Negative('z'))), top).output == 'x^y^-z'
    # a base which renders the exponent itself is not bracketed, the power around it is
    assert toy.print(Power(Leaf('sin'), Leaf('2')), top).output == 'sin^2'
    assert toy.print(Power(Power(Leaf('sin'), Leaf('2')), Leaf('3')), top).output == '[sin^2]^3'


def test_dispatch_is_by_class_along_the_bases() -> None:
    assert TOY.lookup(Negative) is not None and TOY.lookup(Leaf) is not None and TOY.lookup(int) is None

    @dataclass(frozen=True)
    class Special(Negative):
        pass

    toy = Toy()
    top: Context[str] = Context()
    assert toy.print(Special('x'), top).output == '-x'  # the rule of the nearest base
    raises(TypeError, lambda: untyped(toy.print)(3, top))
    rule = TOY.lookup(Negative)
    assert rule is not None
    raises(TypeError, lambda: rule(toy, Leaf('x'), top))  # a rule checks the class of its node


def test_needs_brackets_table() -> None:
    top: Context[str] = Context()
    total = Result('a + b', Binding.SUM)
    minus = Result('-a', Binding.NEGATION, unsigned='a')
    quotient = Result('a / b', Binding.QUOTIENT)
    product = Result('a b', Binding.PRODUCT)
    power = Result('a^b', Binding.POWER)
    relation = Result('a = b', Binding.RELATION)
    open_ended = Result('sum a', Binding.PRODUCT, open_ended=True)
    scripted = Result('a^i', Binding.ATOM, scripted=True)
    applied = Result('f(a) b', Binding.PRODUCT, applies=True)
    rows = [
        # (result, context, brackets)
        (total, top, False), (total, top.delimited(), False), (total, top.term(), True),
        (total, top.factor(False), True), (total, top.base(None), True), (total, top.side(), False),
        (total, top.body(), True), (total, top.numerator(False), False), (total, top.numerator(True), True),
        (minus, top.term(), False), (minus, top.factor(False), True), (minus, top.base(None), True),
        (minus, top.exponent(), False), (minus, top.point(), True), (minus, top.left(Binding.NEGATION), True),
        (quotient, top.term(), False), (quotient, top.factor(False), True), (quotient, top.denominator(True), True),
        (product, top.denominator(True), True), (product, top.denominator(False), False),
        (product, top.factor(True), False), (product, top.base(None), True), (product, top.bare_argument(), False),
        (power, top.denominator(True), False), (power, top.base(None), True), (power, top.operand(), True),
        (relation, top.side(), True), (relation, top.connective(), False), (relation, top.negated(), True),
        (relation, top.term(), True), (open_ended, top.factor(True), True), (open_ended, top.factor(False), False),
        (open_ended, top.term(), False), (scripted, top.base(None), True), (scripted, top.factor(True), False),
        (scripted, top.point(), True), (applied, top.bare_argument(), True), (applied, top.factor(True), False),
    ]
    for result, context, expected in rows:
        assert needs_brackets(result, context) is expected, (result, context)


def test_contexts_inherit_their_state_and_never_their_offers() -> None:
    top: Context[str] = Context(inline=True)
    base = top.exponent().base('2')
    assert base.in_exponent and base.inline and base.exponent_offered == '2' and base.position is Position.BASE
    child = base.factor(False)
    assert child.exponent_offered is None and child.in_exponent and child.inline
    assert top.negated().negation_offered and not top.negated().term().negation_offered
    assert base.without_offers().exponent_offered is None and base.same() is base
    # an absorbed result is never bracketed: it stands for the parent
    assert not needs_brackets(Result('sin^2', Binding.POWER, absorbed=True), base)
