"""The pretty-printing rules of the functions: applications, the
elementary functions with a notation of their own, the special
functions, piecewise functions.

The layouts are SymPy's (``sympy/printing/pretty/pretty.py``). The name
of a defined function is printed as it is: SymPy passes it through the
rules of symbol names, so that ``reduced_totient(n)`` gets ``totient``
as a subscript and ``divisor_sigma`` a Greek letter (sympy/sympy#30844).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_Function and the rules of the named
# functions, _print_hyper, _print_meijerg, _print_Piecewise, _print_binomial, _print_number_function,
# _print_KroneckerDelta, the elliptic integrals) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; immutable pictures; typed results and
# contexts; brackets from the binding of the printed operand.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ...core.expr import Expr
from ...core.function import Application, AppliedUndef, DefinedFunction, WildFunction
from ...core.logic import BooleanTrue
from ...core.numbers import E, Rational
from ...functions.combinatorial.factorials import Binomial, Factorial, Factorial2, Subfactorial
from ...functions.combinatorial.numbers import (Bell, Bernoulli, Catalan, Euler, Fibonacci, Lucas, PrimeNu,
    PrimeOmega, Tribonacci)
from ...functions.elementary.complexes import Abs, Conjugate
from ...functions.elementary.exponential import Exp, LambertW
from ...functions.elementary.integers import Ceiling, Floor
from ...functions.elementary.piecewise import ExprCondPair, Piecewise
from ...functions.special.bessel import AiryAi, AiryAiPrime, AiryBi, AiryBiPrime
from ...functions.special.beta_functions import Beta, BetaInc, BetaIncRegularized
from ...functions.special.delta_functions import DiracDelta, Heaviside
from ...functions.special.elliptic_integrals import EllipticE, EllipticF, EllipticK, EllipticPi
from ...functions.special.error_functions import ChiNode, Expint, FresnelC, FresnelS
from ...functions.special.gamma_functions import Gamma, LowerGamma, UpperGamma
from ...functions.special.hyper import Hyper, MeijerG
from ...functions.special.mathieu_functions import MathieuC, MathieuCPrime, MathieuS, MathieuSPrime
from ...functions.special.singularity_functions import SingularityFunctionNode
from ...functions.special.tensor_functions import KroneckerDeltaNode
from ...functions.special.zeta_functions import DirichletEta, LerchPhi, PolyLog, Stieltjes
from .engine import PrettyPrinter, RuleEntry, rule_for
from .layout import Binding, Context, Picture, Position, PrettyResult
from .rules_core import exponent_picture, raised
from .symbology import greek_unicode, sub

__all__ = ['FUNCTION_RULES']

ApplicationRule = Callable[[PrettyPrinter, Application, Context], PrettyResult]


def _application(p: PrettyPrinter, e: Application, context: Context) -> PrettyResult:
    # an applied undefined function (or a foreign one): its name is a symbol name
    return p.applied(Picture.text(p.symbology.symbol(e.name)), p.arguments(e.args, context))


def _defined(p: PrettyPrinter, e: DefinedFunction, context: Context) -> PrettyResult:
    name = e.name
    if p.unicode and name in greek_unicode:
        name = greek_unicode[name]  # zeta
    return p.function(name, e.args, context)


def _wild_function(p: PrettyPrinter, e: WildFunction, context: Context) -> PrettyResult:
    return p.atom(e.name + '_')


def _named(unicode_name: str, ascii_name: str) -> ApplicationRule:
    """The rule of a function printed under another name."""
    def rule(p: PrettyPrinter, e: Application, context: Context) -> PrettyResult:
        return p.function(unicode_name if p.unicode else ascii_name, e.args, context)
    return rule


def _greek(letter: str, ascii_name: str) -> ApplicationRule:
    return _named(greek_unicode[letter], ascii_name)


# ----------------------------------------------------------------------
# elementary functions

def _exp(p: PrettyPrinter, e: Exp, context: Context) -> PrettyResult:
    return raised(p, E, exponent_picture(p, e.args[0], context), context)


def _factorial(p: PrettyPrinter, e: Factorial, context: Context) -> PrettyResult:
    operand = p.operand(e.args[0], context.child(Position.AFFIX_OPERAND, Binding.FUNC))
    return PrettyResult(operand.rtext('!'), Binding.AFFIX)


def _factorial2(p: PrettyPrinter, e: Factorial2, context: Context) -> PrettyResult:
    operand = p.operand(e.args[0], context.child(Position.AFFIX_OPERAND, Binding.FUNC))
    return PrettyResult(operand.rtext('!!'), Binding.AFFIX)


def _subfactorial(p: PrettyPrinter, e: Subfactorial, context: Context) -> PrettyResult:
    operand = p.operand(e.args[0], context.child(Position.AFFIX_OPERAND, Binding.FUNC))
    return PrettyResult(operand.ltext('!'), Binding.AFFIX)


def _binomial(p: PrettyPrinter, e: Binomial, context: Context) -> PrettyResult:
    n = p.operand(e.args[0], context.argument())
    k = p.operand(e.args[1], context.argument())
    bar = Picture.blank(max(n.width(), k.width()))
    picture = p.parens(k.above(n, bar))
    return PrettyResult(picture.with_baseline((picture.baseline + 1)//2), Binding.ATOM)


def _conjugate(p: PrettyPrinter, e: Conjugate, context: Context) -> PrettyResult:
    # the bar covers the operand: it needs no brackets and the result is closed
    operand = p.operand(e.args[0], context.argument())
    return PrettyResult(operand.above(Picture.text(p.symbology.hobj('_', operand.width()))), Binding.ATOM)


def _abs(p: PrettyPrinter, e: Abs, context: Context) -> PrettyResult:
    return PrettyResult(p.parens(p.operand(e.args[0], context.argument()), '|', '|'), Binding.ATOM)


def _floor(p: PrettyPrinter, e: Floor, context: Context) -> PrettyResult:
    if not p.unicode:
        return p.function('floor', e.args, context)
    return PrettyResult(p.parens(p.operand(e.args[0], context.argument()), 'lfloor', 'rfloor'), Binding.ATOM)


def _ceiling(p: PrettyPrinter, e: Ceiling, context: Context) -> PrettyResult:
    if not p.unicode:
        return p.function('ceiling', e.args, context)
    return PrettyResult(p.parens(p.operand(e.args[0], context.argument()), 'lceil', 'rceil'), Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Piecewise
def _piecewise(p: PrettyPrinter, e: Piecewise, context: Context) -> PrettyResult:
    rows: list[tuple[Picture, Picture]] = []
    for pair in e.args:
        value = p.operand(pair.expr, context.argument())
        if isinstance(pair.cond, BooleanTrue):
            condition = Picture.text('otherwise')
        else:
            condition = p.operand(pair.cond, context.argument()).ltext('for ')
        rows.append((value, condition))
    widths = [max(row[j].width() for row in rows) for j in range(2)]
    table: Optional[Picture] = None
    for row in rows:
        cells: list[Picture] = []
        for j in range(2):
            delta = widths[j] - row[j].width()
            left = delta//2
            cells.append(Picture.beside(Picture.blank(left), row[j], Picture.blank(delta - left)))
        line = Picture.beside(cells[0], Picture.blank(2), cells[1])
        table = line if table is None else table.below(Picture.text(' '), line)
    if table is None:
        return p.function('Piecewise', (), context)
    picture = p.parens(table, '{', '')
    # the brace is open on the right: the result is bracketed everywhere
    return PrettyResult(picture.with_baseline(picture.height()//2), Binding.OPEN)


def _expr_cond_pair(p: PrettyPrinter, e: ExprCondPair, context: Context) -> PrettyResult:
    return p.sequence([e.expr, e.cond], context, '(', ')')


# ----------------------------------------------------------------------
# special functions

def _vector(p: PrettyPrinter, items: Sequence[Expr], context: Context) -> Picture:
    if not items:
        return Picture.text(' ')
    return p.arguments(items, context)


def _lettered(p: PrettyPrinter, letter: str, table: Picture, left: Picture, right: Picture) -> PrettyResult:
    """The stylised letter with its indices in front of the table."""
    above = table.height()//2 - 1
    below = table.height() - above - 1
    drawing = p.symbology.annotated(letter)
    sign = Picture.text('\n'*(above - drawing.top) + drawing.image + '\n'*(below - drawing.bottom),
                        above + drawing.size)
    return PrettyResult(_indices(sign, left, right, above, drawing.add, drawing.size, letter)
                        .right(Picture.text(' '), table), Binding.FUNC)


def _indices(sign: Picture, left: Picture, right: Picture, above: int, add: int, size: int, letter: str) -> Picture:
    if letter == 'F':
        return sign.left(left).right(right).with_baseline(above + (size + 1)//2)
    height = sign.baseline - above - 2
    block = left.below(*[Picture.text('')]*height, right) if height > 0 else left.below(right)
    return sign.with_baseline(above).right(block).with_baseline(above + add)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_hyper
def _hyper(p: PrettyPrinter, e: Hyper, context: Context) -> PrettyResult:
    argument = p.operand(e.argument, context.argument())
    argument = argument.with_baseline(argument.height()//2).ltext(' ')
    table = _vector(p, e.ap, context).below(Picture.text(' '), _vector(p, e.bq, context))
    table = table.with_baseline(table.height()//2).rtext(' ')
    table = p.parens(p.separated(table, argument, '', ''))
    return _lettered(p, 'F', table, Picture.text(str(len(e.ap))), Picture.text(str(len(e.bq))))


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_meijerg
def _meijerg(p: PrettyPrinter, e: MeijerG, context: Context) -> PrettyResult:
    argument = p.operand(e.argument, context.argument())
    argument = argument.with_baseline(argument.height()//2).ltext(' ')
    cells = [[_vector(p, e.an, context), _vector(p, e.aother, context)],
             [_vector(p, e.bm, context), _vector(p, e.bother, context)]]
    for i in range(2):
        width = max(cells[0][i].width(), cells[1][i].width())
        for j in range(2):
            cell = cells[j][i]
            left = (width - cell.width())//2
            cells[j][i] = Picture.beside(Picture.blank(left), cell, Picture.blank(width - left - cell.width()))
    top = Picture.beside(cells[0][0], Picture.blank(2), cells[0][1])
    bottom = Picture.beside(cells[1][0], Picture.blank(2), cells[1][1])
    table = top.below(Picture.text(' '), bottom)
    table = table.with_baseline(table.height()//2).rtext(' ')
    table = p.parens(p.separated(table, argument, '', ''))
    pp, pm = _same_width(Picture.text(str(len(e.ap))), Picture.text(str(len(e.bm))))
    pq, pn = _same_width(Picture.text(str(len(e.bq))), Picture.text(str(len(e.an))))
    upper = Picture.beside(pm, Picture.text(', '), pn)
    lower = Picture.beside(pp, Picture.text(', '), pq)
    return _lettered(p, 'G', table, upper, lower)


def _same_width(a: Picture, b: Picture) -> tuple[Picture, Picture]:
    difference = a.width() - b.width()
    if difference > 0:
        return a, b.left(Picture.blank(difference))
    if difference < 0:
        return a.left(Picture.blank(-difference)), b
    return a, b


def _polylog(p: PrettyPrinter, e: PolyLog, context: Context) -> PrettyResult:
    return _subscripted_name(p, e, context, 'Li', 'polylog')


def _expint(p: PrettyPrinter, e: Expint, context: Context) -> PrettyResult:
    return _subscripted_name(p, e, context, 'E', 'expint')


def _subscripted_name(p: PrettyPrinter, e: Application, context: Context, letter: str, name: str) -> PrettyResult:
    """``Li₂(z)`` when the first argument has a Unicode subscript form."""
    index = str(p.operand(e.args[0], context.argument()))
    if p.symbology.subscriptable(index):
        return p.function(letter + ''.join(sub[c] for c in index), e.args[1:], context)
    return p.function(name, e.args, context)


def _heaviside(p: PrettyPrinter, e: Heaviside, context: Context) -> PrettyResult:
    name = p.symbology.greek('theta', 'Heaviside')
    half = e.args[1]
    if isinstance(half, Rational) and half.p == 1 and half.q == 2:
        return p.function(name, e.args[:1], context)
    return p.function(name, e.args, context)


def _singularity(p: PrettyPrinter, e: SingularityFunctionNode, context: Context) -> PrettyResult:
    # <x - a> raised to n; the angle brackets close the shift
    shift = p.parens(p.operand(e.args[0] - e.args[1], context.argument()), '<', '>')
    return PrettyResult(p.power(shift, exponent_picture(p, e.args[2], context)), Binding.POW)


def _dirac_delta(p: PrettyPrinter, e: DiracDelta, context: Context) -> PrettyResult:
    if not p.unicode:
        return p.function('DiracDelta', e.args, context)
    delta = Picture.text(greek_unicode['delta'])
    if len(e.args) == 2:
        # the order of the derivative, between parentheses, as an exponent
        order = p.parens(p.operand(e.args[1], context.argument()))
        return p.applied(p.power(delta, order).rtext(' '), p.operand(e.args[0], context.argument()))
    return p.applied(delta, p.operand(e.args[0], context.argument()))


def _elliptic_k(p: PrettyPrinter, e: EllipticK, context: Context) -> PrettyResult:
    return p.function('K', e.args, context)


def _bar_separated(p: PrettyPrinter, left: Expr, right: Expr, context: Context) -> Picture:
    return p.separated(p.operand(left, context.argument()), p.operand(right, context.argument()), '', '')


def _elliptic_e(p: PrettyPrinter, e: EllipticE, context: Context) -> PrettyResult:
    if len(e.args) == 1:
        return p.function('E', e.args, context)
    return p.applied(Picture.text('E'), _bar_separated(p, e.args[0], e.args[1], context))


def _elliptic_f(p: PrettyPrinter, e: EllipticF, context: Context) -> PrettyResult:
    return p.applied(Picture.text('F'), _bar_separated(p, e.args[0], e.args[1], context))


def _elliptic_pi(p: PrettyPrinter, e: EllipticPi, context: Context) -> PrettyResult:
    name = Picture.text(p.symbology.greek('Pi', 'Pi'))
    if len(e.args) == 2:
        return p.applied(name, _bar_separated(p, e.args[0], e.args[1], context))
    inside = _bar_separated(p, e.args[1], e.args[2], context).ltext('; ')
    return p.applied(name, inside.left(p.operand(e.args[0], context.argument())))


def _number_function(unicode_name: str, ascii_name: str) -> ApplicationRule:
    """``B_n``, and ``B_n(x)`` with a second argument: the first argument
    is a subscript."""
    def rule(p: PrettyPrinter, e: Application, context: Context) -> PrettyResult:
        name = Picture.text(unicode_name if p.unicode else ascii_name)
        index = p.operand(e.args[0], context.child(Position.SUBSCRIPT, Binding.OPEN))
        head = name.right(Picture.blank(index.width()).below(index))
        if len(e.args) == 1:
            return PrettyResult(head, Binding.FUNC)
        return p.applied(head, p.arguments(e.args[1:], context))
    return rule


def _kronecker_delta(p: PrettyPrinter, e: KroneckerDeltaNode, context: Context) -> PrettyResult:
    c = context.child(Position.SUBSCRIPT, Binding.OPEN)
    indices = Picture.beside(p.operand(e.args[0], c), Picture.text(','), p.operand(e.args[1], c))
    return PrettyResult(p.subscripted(Picture.text(p.symbology.greek('delta', 'd')), indices), Binding.FUNC)


FUNCTION_RULES: tuple[RuleEntry, ...] = (
    rule_for(Application, _application),
    rule_for(AppliedUndef, _application),
    rule_for(DefinedFunction, _defined),
    rule_for(WildFunction, _wild_function),
    rule_for(Exp, _exp),
    rule_for(Factorial, _factorial),
    rule_for(Factorial2, _factorial2),
    rule_for(Subfactorial, _subfactorial),
    rule_for(Binomial, _binomial),
    rule_for(Conjugate, _conjugate),
    rule_for(Abs, _abs),
    rule_for(Floor, _floor),
    rule_for(Ceiling, _ceiling),
    rule_for(Piecewise, _piecewise),
    rule_for(ExprCondPair, _expr_cond_pair),
    rule_for(Hyper, _hyper),
    rule_for(MeijerG, _meijerg),
    rule_for(MathieuC, _named('C', 'C')),
    rule_for(MathieuS, _named('S', 'S')),
    rule_for(MathieuCPrime, _named("C'", "C'")),
    rule_for(MathieuSPrime, _named("S'", "S'")),
    rule_for(PolyLog, _polylog),
    rule_for(Expint, _expint),
    rule_for(LerchPhi, _greek('Phi', 'lerchphi')),
    rule_for(DirichletEta, _greek('eta', 'dirichlet_eta')),
    rule_for(Heaviside, _heaviside),
    rule_for(FresnelS, _named('S', 'S')),
    rule_for(FresnelC, _named('C', 'C')),
    rule_for(AiryAi, _named('Ai', 'Ai')),
    rule_for(AiryBi, _named('Bi', 'Bi')),
    rule_for(AiryAiPrime, _named("Ai'", "Ai'")),
    rule_for(AiryBiPrime, _named("Bi'", "Bi'")),
    rule_for(LambertW, _named('W', 'W')),
    rule_for(SingularityFunctionNode, _singularity),
    rule_for(Beta, _greek('Beta', 'B')),
    rule_for(BetaInc, _named("B'", "B'")),
    rule_for(BetaIncRegularized, _named('I', 'I')),
    rule_for(Gamma, _greek('Gamma', 'Gamma')),
    rule_for(UpperGamma, _greek('Gamma', 'Gamma')),
    rule_for(LowerGamma, _greek('gamma', 'lowergamma')),
    rule_for(DiracDelta, _dirac_delta),
    rule_for(ChiNode, _named('Chi', 'Chi')),
    rule_for(EllipticK, _elliptic_k),
    rule_for(EllipticE, _elliptic_e),
    rule_for(EllipticF, _elliptic_f),
    rule_for(EllipticPi, _elliptic_pi),
    rule_for(Euler, _number_function('E', 'E')),
    rule_for(Catalan, _number_function('C', 'C')),
    rule_for(Bernoulli, _number_function('B', 'B')),
    rule_for(Bell, _number_function('B', 'B')),
    rule_for(Lucas, _number_function('L', 'L')),
    rule_for(Fibonacci, _number_function('F', 'F')),
    rule_for(Tribonacci, _number_function('T', 'T')),
    rule_for(Stieltjes, _number_function(greek_unicode['gamma'], 'stieltjes')),
    rule_for(KroneckerDeltaNode, _kronecker_delta),
    rule_for(PrimeNu, _greek('nu', 'nu')),
    rule_for(PrimeOmega, _greek('Omega', 'Omega')),
)
