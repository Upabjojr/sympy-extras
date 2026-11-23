"""The pretty-printing rules of the core: numbers, symbols, sums,
products, powers, relations, logic, containers.

Each rule returns the picture with the binding of what it printed, and
gets the brackets of its children from the printer (see
:mod:`~sympy_extras.printing.pretty.engine`). The layouts are SymPy's
(``sympy/printing/pretty/pretty.py``); the bracketing is not: the
defects of sympy/sympy#30844 which it removes are named at the rules.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_Add, _print_Mul, _print_Pow, _print_Rational,
# _print_Relational, the boolean connectives, the containers), sympy/printing/pretty/stringpict.py
# (prettyForm.__add__, __mul__, __truediv__, __pow__) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; rules return typed results and take typed
# contexts; the brackets come from the binding of the printed child.
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from ...core.accumbounds import AccumBounds
from ...core.add import Add
from ...core.assume import AppliedPredicate, Predicate
from ...core.basic import Basic
from ...core.containers import Dict, Tuple
from ...core.expr import Expr, UnevaluatedExpr
from ...core.logic import And, BooleanAtom, BooleanFunction, Equivalent, Implies, Not, Or, Xor
from ...core.mod import Mod
from ...core.mul import Mul
from ...core.numbers import (AlgebraicNumber, CatalanType, ComplexInfinity, EulerGammaType, Exp1, Float,
    GoldenRatioType, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity, Number, NumberSymbol, Pi, Rational)
from ...core.power import Pow
from ...core.printing import printed_factors, printed_terms, sstr
from ...core.relational import Relational
from ...core.sorting import default_sort_key
from ...core.symbol import Symbol, Wild
from .engine import PrettyPrinter, RuleEntry, rule_for
from ..framework import Binding, Position
from .layout import Picture, PrettyContext, PrettyResult

__all__ = ['CORE_RULES', 'negative', 'integer_result', 'infix']


def negative(magnitude: Picture, binding_stacked: bool = False) -> PrettyResult:
    """The negation of what is printed as ``magnitude``."""
    sign = '- ' if magnitude.height() > 1 else '-'
    return PrettyResult(magnitude.ltext(sign), Binding.NEGATION, unsigned=magnitude, stacked=binding_stacked)


def integer_result(n: int) -> PrettyResult:
    """An integer; a negative one is not an atom."""
    if n < 0:
        return negative(Picture.text(str(-n)))
    return PrettyResult(Picture.text(str(n)), Binding.ATOM)


def infix(p: PrettyPrinter, args: Sequence[Basic], operator: str, binding: Binding, operand: PrettyContext
          ) -> PrettyResult:
    """The operands, each one printed in the context ``operand``, joined
    by ``operator``."""
    parts = [p.operand(a, operand) for a in args]
    return PrettyResult(p.join(parts, ' %s ' % operator), binding)


# ----------------------------------------------------------------------
# numbers and symbols

def _rational(p: PrettyPrinter, e: Rational, context: PrettyContext) -> PrettyResult:
    if e.q == 1:
        return integer_result(e.p)
    if context.display or (abs(e.p) >= 10 and e.q >= 10):
        magnitude = p.fraction(Picture.text(str(abs(e.p))), Picture.text(str(e.q)))
        if e.p < 0:
            # (SymPy's layout: the minus sign in the numerator)
            picture = p.fraction(Picture.text('-%d ' % -e.p), Picture.text(str(e.q)))
            return PrettyResult(picture, Binding.NEGATION, unsigned=magnitude, stacked=True)
        return PrettyResult(magnitude, Binding.FRACTION, stacked=True)
    # a quotient on one row is not an atom: (2/3)**x, x*(1/2) (sympy/sympy#30844, root cause 5)
    magnitude = Picture.text('%d/%d' % (abs(e.p), e.q))
    if e.p < 0:
        return negative(magnitude)
    return PrettyResult(magnitude, Binding.QUOTIENT)


def _float(p: PrettyPrinter, e: Float, context: PrettyContext) -> PrettyResult:
    # all the digits for a float alone, as SymPy prints it (full_prec='auto')
    full = p.settings.full_prec
    if full is None:
        full = context.position is Position.TOP
    text = sstr(e) if full else e.short_str()
    # 1.0e-5 is a product as a base: (1.0e-5)**2
    binding = Binding.PRODUCT if 'e' in text else Binding.ATOM
    if text.startswith('-'):
        # a negative float is not an atom: (-2.5)**n (sympy/sympy#30844, root cause 5)
        return negative(Picture.text(text[1:]))
    return PrettyResult(Picture.text(text), binding)


def _infinity(p: PrettyPrinter, e: Infinity, context: PrettyContext) -> PrettyResult:
    return p.atom(p.symbology.atom('Infinity', 'oo'))


def _negative_infinity(p: PrettyPrinter, e: NegativeInfinity, context: PrettyContext) -> PrettyResult:
    return negative(Picture.text(p.symbology.atom('Infinity', 'oo')))


def _complex_infinity(p: PrettyPrinter, e: ComplexInfinity, context: PrettyContext) -> PrettyResult:
    return p.atom('zoo')


def _nan(p: PrettyPrinter, e: NaN, context: PrettyContext) -> PrettyResult:
    return p.atom('nan')


def _pi(p: PrettyPrinter, e: Pi, context: PrettyContext) -> PrettyResult:
    return p.atom(p.symbology.atom('Pi', 'pi'))


def _exp1(p: PrettyPrinter, e: Exp1, context: PrettyContext) -> PrettyResult:
    return p.atom(p.symbology.atom('Exp1', 'e'))


def _golden_ratio(p: PrettyPrinter, e: GoldenRatioType, context: PrettyContext) -> PrettyResult:
    return p.atom(p.symbology.greek('phi', 'GoldenRatio'))


def _euler_gamma(p: PrettyPrinter, e: EulerGammaType, context: PrettyContext) -> PrettyResult:
    return p.atom(p.symbology.greek('gamma', 'EulerGamma'))


def _catalan(p: PrettyPrinter, e: CatalanType, context: PrettyContext) -> PrettyResult:
    return p.atom('G')


def _number_symbol(p: PrettyPrinter, e: NumberSymbol, context: PrettyContext) -> PrettyResult:
    return p.atom(e.name)


def _imaginary_unit(p: PrettyPrinter, e: ImaginaryUnit, context: PrettyContext) -> PrettyResult:
    return p.atom(p.symbology.atom('ImaginaryUnit', 'I'))


def _symbol(p: PrettyPrinter, e: Symbol, context: PrettyContext) -> PrettyResult:
    return p.atom(p.symbology.symbol(e.name))


def _wild(p: PrettyPrinter, e: Wild, context: PrettyContext) -> PrettyResult:
    # (SymPy prints a Wild as the symbol of the same name)
    return p.atom(p.symbology.symbol(e.name) + '_')


def _boolean_atom(p: PrettyPrinter, e: BooleanAtom, context: PrettyContext) -> PrettyResult:
    return p.atom(sstr(e))


def _unevaluated(p: PrettyPrinter, e: UnevaluatedExpr, context: PrettyContext) -> PrettyResult:
    # nothing is printed for the wrapper: the content stands in its place
    return p.rule(e.expr, context)


def _algebraic_number(p: PrettyPrinter, e: AlgebraicNumber, context: PrettyContext) -> PrettyResult:
    return p.rule(e.doit(), context)


# ----------------------------------------------------------------------
# sums

# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Add
def _add(p: PrettyPrinter, e: Add, context: PrettyContext) -> PrettyResult:
    return sum_of(p, printed_terms(e.args, p.settings.order), context)


def sum_of(p: PrettyPrinter, terms: Sequence[Expr], context: PrettyContext) -> PrettyResult:
    """The terms in a row, joined by ``+``; a term which is the negation
    of something is subtracted. A fraction among tall terms is stacked."""
    c = context.term()
    results: list[Optional[PrettyResult]] = []
    for term in terms:
        fraction = isinstance(term, Rational) and term.q > 1
        results.append(None if fraction else p.rule(term, c))
    large = any(r is not None and r.output.height() > 1 for r in results)
    parts: list[Picture] = []
    open_right = False
    for i, term in enumerate(terms):
        found = results[i]
        result = found if found is not None else p.rule(term, c.displayed() if large else c)
        magnitude = result.unsigned
        if magnitude is not None:
            # a - b, not a + -b
            sign = ' - ' if i > 0 else '- ' if magnitude.height() > 1 else '-'
            parts.append(magnitude.ltext(sign))
            open_right = result.open_ended
            continue
        closed = p.enclose(result, c)
        if i > 0:
            parts.append(Picture.text(' + '))
        parts.append(closed.output)
        open_right = closed.open_ended
    return PrettyResult(Picture.beside(*parts), Binding.SUM, open_ended=open_right)


# ----------------------------------------------------------------------
# products

def _product(p: PrettyPrinter, factors: Sequence[Expr], context: PrettyContext, alone: Optional[PrettyContext]
             ) -> tuple[Picture, bool]:
    """The factors joined by the multiplication sign, and whether the
    product is open to the right. A single factor of an unsigned
    fraction part stands in the context ``alone`` (the numerator, the
    denominator) and needs no brackets."""
    if not factors:
        return Picture.text('1'), False
    if len(factors) == 1 and alone is not None:
        result = p.print(factors[0], alone)
        return result.output, result.open_ended
    parts: list[Picture] = []
    last = len(factors) - 1
    open_right = False
    for i, f in enumerate(factors):
        # a factor which is open to the right captures the factors after it:
        # (sum of 1/k)*x, not the sum of x/k (sympy/sympy#30844)
        result = p.print(f, context.factor(followed=i < last))
        parts.append(result.output)
        open_right = result.open_ended
    return p.join(parts, p.symbology.xsym('*')), open_right


def _is_negative_number(e: Expr) -> bool:
    return (isinstance(e, Rational) and e.p < 0) or (isinstance(e, Float) and e.as_fraction() < 0)


def _absolute(e: Expr) -> Expr:
    """``-e`` for a negative rational or float."""
    if isinstance(e, Rational):
        return Rational(-e.p, e.q)
    if isinstance(e, Float):
        from ...core.numbers import number_from_fraction
        return Float(number_from_fraction(-e.as_fraction()), precision=e.precision)
    return e


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Mul
def _mul(p: PrettyPrinter, e: Mul, context: PrettyContext) -> PrettyResult:
    args = printed_factors(e.args, p.settings.order)
    first = args[0]
    signed = _is_negative_number(first)
    # an unevaluated product (a leading 1, a number after the first place)
    # is printed factor by factor: 2*3*x, x*(-1)*2
    plain = (isinstance(first, Rational) and first.p == 1 and first.q == 1) \
        or any(isinstance(a, Number) for a in args[1:])
    numerator: list[Expr] = []
    denominator: list[Expr] = []
    if plain:
        numerator = [_absolute(first) if signed else first] + args[1:]
    else:
        for item in args:
            if isinstance(item, Pow) and item.is_commutative is not False and isinstance(item.exp, Rational) \
                    and item.exp.p < 0:
                minus = Rational(-item.exp.p, item.exp.q)
                denominator.append(item.base if minus.p == 1 and minus.q == 1 else Pow(item.base, minus))
            elif isinstance(item, Rational):
                if abs(item.p) != 1:
                    numerator.append(Integer(abs(item.p)))
                if item.q != 1:
                    denominator.append(Integer(item.q))
            elif item is first and signed:
                numerator.append(_absolute(item))
            else:
                numerator.append(item)
    if not denominator:
        product, open_right = _product(p, numerator, context, None)
        if signed:
            return PrettyResult(product.ltext('-'), Binding.NEGATION, unsigned=product, open_ended=open_right)
        return PrettyResult(product, Binding.PRODUCT, open_ended=open_right)
    top, _ = _product(p, numerator, context, None if signed else context.numerator(False))
    bottom, _ = _product(p, denominator, context, context.denominator(False))
    magnitude = p.fraction(top, bottom)
    if signed:
        # (SymPy's layout: the minus sign in the numerator, followed by a space)
        picture = p.fraction(top.ltext('-').rtext(' '), bottom)
        return PrettyResult(picture, Binding.NEGATION, unsigned=magnitude, stacked=True)
    return PrettyResult(magnitude, Binding.FRACTION, stacked=True)


# ----------------------------------------------------------------------
# powers

def reciprocal(p: PrettyPrinter, denominator: Expr, context: PrettyContext) -> PrettyResult:
    """``1/denominator`` with a fraction bar."""
    bottom = p.operand(denominator, context.denominator(False))
    return PrettyResult(p.fraction(Picture.text('1'), bottom), Binding.FRACTION, stacked=True)


def raised(p: PrettyPrinter, base: Basic, exponent: Picture, context: PrettyContext) -> PrettyResult:
    """``base`` with the picture ``exponent`` raised. An exponent of one
    row is offered to the base: a function application puts it between
    the function and its arguments (``sin²(x)``) and then binds as a
    power: nothing is attached to it (sympy/sympy#30844:
    ``(sin(x)**2)**3`` printed a stray ``3``)."""
    one_row = exponent.height() == 1
    # (SymPy brackets a function application under a tall exponent)
    c = context.at(Position.BASE, Binding.POWER if one_row else Binding.APPLICATION, strict=True,
                   exponent_offered=exponent if one_row else None)
    result = p.rule(base, c)
    if result.absorbed:
        return PrettyResult(result.output, Binding.POWER)
    return PrettyResult(p.power(p.enclose(result, c).output, exponent), Binding.POWER)


def exponent_picture(p: PrettyPrinter, exponent: Basic, context: PrettyContext) -> Picture:
    """The picture of an exponent: raised, it needs no brackets, but a
    power in it gets them as in SymPy."""
    return p.operand(exponent, context.at(Position.EXPONENT, Binding.LOWEST, excluded=Binding.POWER))


def _root(p: PrettyPrinter, base: Expr, index: Expr, context: PrettyContext) -> Optional[PrettyResult]:
    square = isinstance(index, Integer) and index.p == 2
    index_picture = None if square else p.operand(index, context.subscript())
    if index_picture is not None and index_picture.height() != 1:
        return None
    radicand = p.operand(base, context.delimited())
    small = radicand.width() == 1 or (isinstance(base, Integer) and base.p >= 0)
    return p.root(radicand, index_picture, small)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Pow
def _pow(p: PrettyPrinter, e: Pow, context: PrettyContext) -> PrettyResult:
    base, exp = e.base, e.exp
    if base.is_commutative is not False:
        if isinstance(exp, Rational) and exp.p == -1 and exp.q == 1:
            return reciprocal(p, base, context)
        if p.settings.root_notation:
            index: Optional[Expr] = None
            if isinstance(exp, Rational) and exp.p == 1 and exp.q > 1:
                index = Integer(exp.q)
            elif isinstance(exp, Pow) and isinstance(exp.base, Symbol) and isinstance(exp.exp, Rational) \
                    and exp.exp.p == -1 and exp.exp.q == 1:
                index = exp.base
            if index is not None:
                rooted = _root(p, base, index, context)
                if rooted is not None:
                    return rooted
        if isinstance(exp, Rational) and exp.p < 0:
            return reciprocal(p, Pow(base, Rational(-exp.p, exp.q)), context)
    return raised(p, base, exponent_picture(p, exp, context), context)


# ----------------------------------------------------------------------
# modulo, relations, logic

def _mod(p: PrettyPrinter, e: Mod, context: PrettyContext) -> PrettyResult:
    # both operands are bracketed when they are not products, and the
    # result binds less than a sum: x mod (y + z), x mod (y mod z),
    # -(x mod y), (x mod y) + 1 (sympy/sympy#30844, sympy/sympy#23912)
    left = p.operand(e.args[0], context.at(Position.LEFT, Binding.PRODUCT, followed=True))
    right = p.operand(e.args[1], context.at(Position.RIGHT, Binding.PRODUCT, strict=True))
    return PrettyResult(Picture.beside(left, Picture.text(' mod '), right), Binding.MODULO)


def _relational(p: PrettyPrinter, e: Relational, context: PrettyContext) -> PrettyResult:
    # a side which is itself a relation or a formula is bracketed:
    # (x = y) = (y = z), (p ∧ q) = r (sympy/sympy#30844, root cause 3)
    left = p.operand(e.lhs, context.side())
    right = p.operand(e.rhs, context.side())
    operator = Picture.text(' %s ' % p.symbology.xsym(e.rel_op))
    return PrettyResult(Picture.beside(left, operator, right), Binding.RELATION)


def _sorted(args: Sequence[Basic]) -> list[Basic]:
    return sorted(args, key=default_sort_key)


def _connective(p: PrettyPrinter, args: Sequence[Basic], context: PrettyContext, name: str, atom: str,
                binding: Binding) -> PrettyResult:
    if not p.unicode:
        return p.function(name, args, context)
    # what binds less than a relation is bracketed: the other connectives
    return infix(p, args, p.symbology.atom(atom, name), binding, context.connective())


def _negatable(p: PrettyPrinter, args: Sequence[Basic], context: PrettyContext, name: str, atom: str,
               negated_atom: str) -> PrettyResult:
    """A connective which has a negated sign: it takes the negation its
    parent offers (``a ⇏ b``) and then stands for the negation."""
    if p.unicode and context.negation_offered:
        return replace(_connective(p, args, context, name, negated_atom, Binding.IMPLICATION), absorbed=True)
    return _connective(p, args, context, name, atom, Binding.IMPLICATION)


def _and(p: PrettyPrinter, e: And, context: PrettyContext) -> PrettyResult:
    return _connective(p, _sorted(e.args), context, 'And', 'And', Binding.AND)


def _or(p: PrettyPrinter, e: Or, context: PrettyContext) -> PrettyResult:
    return _connective(p, _sorted(e.args), context, 'Or', 'Or', Binding.OR)


def _xor(p: PrettyPrinter, e: Xor, context: PrettyContext) -> PrettyResult:
    return _connective(p, _sorted(e.args), context, 'Xor', 'Xor', Binding.XOR)


def _implies(p: PrettyPrinter, e: Implies, context: PrettyContext) -> PrettyResult:
    return _negatable(p, e.args, context, 'Implies', 'Arrow', 'NotArrow')


def _equivalent(p: PrettyPrinter, e: Equivalent, context: PrettyContext) -> PrettyResult:
    return _negatable(p, _sorted(e.args), context, 'Equivalent', 'Equiv', 'NotEquiv')


def _not(p: PrettyPrinter, e: Not, context: PrettyContext) -> PrettyResult:
    if not p.unicode:
        return p.function('Not', e.args, context)
    # the operand is bracketed unless it is an atom, an application or a
    # negation: ¬(x < y), ¬(x ∈ S) (sympy/sympy#30844, root cause 4); an
    # implication or an equivalence takes the negation into its sign
    c = context.at(Position.NEGATED, Binding.AFFIX, negation_offered=True)
    result = p.rule(e.args[0], c)
    if result.absorbed:
        return replace(result, absorbed=False)
    return PrettyResult(p.enclose(result, c).output.ltext(p.symbology.atom('Not', '~')), Binding.AFFIX)


def _boolean_function(p: PrettyPrinter, e: BooleanFunction, context: PrettyContext) -> PrettyResult:
    return p.function(type(e).__name__, e.args, context)


def _predicate(p: PrettyPrinter, e: Predicate, context: PrettyContext) -> PrettyResult:
    return p.opaque(sstr(e))


def _applied_predicate(p: PrettyPrinter, e: AppliedPredicate, context: PrettyContext) -> PrettyResult:
    return p.applied(Picture.text(sstr(e.function)), p.arguments(e.arguments, context), context)


# ----------------------------------------------------------------------
# containers

def _tuple(p: PrettyPrinter, e: Tuple, context: PrettyContext) -> PrettyResult:
    if len(e.args) == 1:
        inside = p.operand(e.args[0], context.delimited()).rtext(',')
        return PrettyResult(p.parens(inside, '(', ')', True), Binding.ATOM)
    return p.sequence(e.args, context, '(', ')')


def _dict(p: PrettyPrinter, e: Dict, context: PrettyContext) -> PrettyResult:
    items: list[Picture] = []
    for key, value in sorted(e.items(), key=lambda kv: default_sort_key(kv[0])):
        items.append(Picture.beside(p.operand(key, context.delimited()), Picture.text(': '),
                                    p.operand(value, context.delimited())))
    return PrettyResult(p.parens(p.join(items, ', '), '{', '}', True), Binding.ATOM)


def _accumbounds(p: PrettyPrinter, e: AccumBounds, context: PrettyContext) -> PrettyResult:
    return p.sequence([e.min, e.max], context, '<', '>')


CORE_RULES: tuple[RuleEntry, ...] = (
    rule_for(Rational, _rational),
    rule_for(Float, _float),
    rule_for(Infinity, _infinity),
    rule_for(NegativeInfinity, _negative_infinity),
    rule_for(ComplexInfinity, _complex_infinity),
    rule_for(NaN, _nan),
    rule_for(NumberSymbol, _number_symbol),
    rule_for(Pi, _pi),
    rule_for(Exp1, _exp1),
    rule_for(GoldenRatioType, _golden_ratio),
    rule_for(EulerGammaType, _euler_gamma),
    rule_for(CatalanType, _catalan),
    rule_for(ImaginaryUnit, _imaginary_unit),
    rule_for(Symbol, _symbol),
    rule_for(Wild, _wild),
    rule_for(BooleanAtom, _boolean_atom),
    rule_for(UnevaluatedExpr, _unevaluated),
    rule_for(AlgebraicNumber, _algebraic_number),
    rule_for(Add, _add),
    rule_for(Mul, _mul),
    rule_for(Pow, _pow),
    rule_for(Mod, _mod),
    rule_for(Relational, _relational),
    rule_for(And, _and),
    rule_for(Or, _or),
    rule_for(Xor, _xor),
    rule_for(Implies, _implies),
    rule_for(Equivalent, _equivalent),
    rule_for(Not, _not),
    rule_for(BooleanFunction, _boolean_function),
    rule_for(Predicate, _predicate),
    rule_for(AppliedPredicate, _applied_predicate),
    rule_for(Tuple, _tuple),
    rule_for(Dict, _dict),
    rule_for(AccumBounds, _accumbounds),
)
