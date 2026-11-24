"""The pretty-printing rules of calculus: derivatives, integrals, sums,
products, limits, substitutions, order terms, anonymous functions.

The layouts are SymPy's (``sympy/printing/pretty/pretty.py``). A sum
sign, a product sign and a limit are *open to the right*: their result
says so, and a product brackets such a factor when another one follows
(SymPy lets ``Sum(1/k, (k, 1, n))*x`` print as the sum of ``x/k``,
sympy/sympy#30844). The point of a one-sided limit is bracketed when
the direction mark would attach to a part of it (root cause 6).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py (_print_Derivative, _print_Integral, _print_Sum,
# _print_Product, _print_Limit, _print_Subs, _print_Order, _print_Lambda),
# sympy/printing/conventions.py (requires_partial) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; immutable pictures; typed results and
# contexts; brackets from the binding of the printed operand.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from ...core.add import add
from ...concrete.expr_with_limits import ExprWithLimits, Limits, Product, Sum
from ...core.expr import Expr
from ...core.function import Derivative, Lambda, Subs
from ...core.limits import Limit
from ...core.numbers import Infinity, Integer, NegativeInfinity
from ...core.order import Order
from ...integrals.integral import Integral
from .engine import PrettyPrinter, RuleEntry, rule_for
from ..framework import Binding, Position
from .layout import Picture, PrettyContext, PrettyResult

__all__ = ['CALCULUS_RULES', 'requires_partial']


# PORTED FROM SYMPY: sympy/printing/conventions.py:requires_partial
def requires_partial(expr: Expr) -> bool:
    """Whether a derivative of ``expr`` is written with the partial
    sign: the expression has several free symbols which are not integers.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.printing.pretty.rules_calculus import requires_partial
    >>> x, y = symbols('x y')
    >>> requires_partial(x*y), requires_partial(x**2)
    (True, False)
    """
    if isinstance(expr, Derivative):
        return requires_partial(expr.expr)
    return sum(1 for s in expr.free_symbols if s.is_integer is not True) > 1


def _body(p: PrettyPrinter, e: Expr, context: PrettyContext, minimum: Binding) -> PrettyResult:
    c = context.at(Position.BODY, minimum)
    return p.enclose(p.rule(e, c), c)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Derivative
def _derivative(p: PrettyPrinter, e: Derivative, context: PrettyContext) -> PrettyResult:
    d = p.symbology.atom('Partial', 'd') if requires_partial(e.expr) else 'd'
    below: list[Picture] = []
    # the orders are integers or symbolic: a symbolic order is written as an exponent, as an order above 1 is
    total: Expr = Integer(0)
    for symbol, count in reversed(e.variable_orders):
        ds = p.operand(symbol, context.operand()).ltext(d)
        total = add(total, count)
        if count != 1:
            ds = p.power(ds, p.operand(count, context.delimited()))
        below.append(ds)
    denominator = p.join(below, ' ')
    top = Picture.text(d)
    if total != 1:
        top = p.power(top, p.operand(total, context.delimited()))
    width = max(top.width(), denominator.width())
    operator = top.below(Picture.text(p.symbology.hobj('-', width)), denominator)
    operator = operator.with_baseline(operator.baseline + 1)
    # the operand between the parentheses of the notation
    operand = p.parens(p.operand(e.expr, context.delimited()))
    return PrettyResult(Picture.beside(operator, operand), Binding.PRODUCT)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Integral
def _integral(p: PrettyPrinter, e: Integral, context: PrettyContext) -> PrettyResult:
    # an integral of an integral is a multiple integral: the signs in a
    # row, the innermost variable first after the integrand
    chain: list[Integral] = [e]
    while isinstance(chain[-1].function, Integral):
        chain.append(chain[-1].function)
    chain.reverse()
    # a sum is bracketed; the differential closes the integrand
    arg = _body(p, chain[0].function, context, Binding.NEGATION).output
    for integral in chain:
        variable = p.operand(integral.variable, context.operand())
        if variable.width() > 1:
            variable = p.parens(variable)
        arg = Picture.beside(arg, Picture.text(' d'), variable)
    ascii_mode = not p.unicode
    signs: Optional[Picture] = None
    for integral in chain:
        h = arg.height()
        height = h + 2 + (2 if ascii_mode else 0)
        sign = Picture.text(p.symbology.vobj('int', height), arg.baseline + (height - h)//2)
        bounds = integral.bounds
        if bounds is not None:
            lower = p.operand(bounds[0], context.delimited())
            upper = p.operand(bounds[1], context.delimited())
            if ascii_mode:
                # room, so that a bound is seen to belong to its sign
                upper = upper.left(Picture.blank(max(1, 3 - upper.width())))
                lower = lower.right(Picture.blank(max(1, 4 - lower.width())))
            sign = sign.above(upper).below(lower)
        if not ascii_mode:
            sign = sign.rtext(' ')
        signs = sign if signs is None else signs.left(sign)
    if signs is None:
        return PrettyResult(arg, Binding.PRODUCT)
    return PrettyResult(arg.left(signs), Binding.PRODUCT)


def _limits_pictures(p: PrettyPrinter, limit: Limits, context: PrettyContext) -> tuple[Picture, Picture]:
    """``variable = lower`` and the upper bound."""
    c = context.delimited()
    lower = Picture.beside(p.operand(limit[0], c), Picture.text(' = '), p.operand(limit[1], c))
    return lower, p.operand(limit[2], c)


def _flattened(e: ExprWithLimits) -> tuple[list[Limits], Expr]:
    """The limits, innermost first, and the operand of a sum of sums (a
    product of products): they are printed as one multiple sum."""
    limits = list(e.limits)
    function = e.function
    while type(function) is type(e) and isinstance(function, ExprWithLimits):
        limits = list(function.limits) + limits
        function = function.function
    return limits, function


def _sum_sign(required: int, ascii_mode: bool) -> tuple[int, int, list[str], int]:
    """The sign of a sum for an operand of the given height: the offset
    of its middle, its height, its rows and the correction of an odd
    height."""
    h = max(required, 2)
    d = h//2
    w = d + 1
    more = required % 2
    lines: list[str] = []
    if ascii_mode:
        lines.append('_'*w + ' ')
        lines.append('\\%s`' % (' '*(w - 1)))
        for i in range(1, d):
            lines.append('%s\\%s' % (' '*i, ' '*(w - i)))
        if more:
            lines.append('%s)%s' % (' '*d, ' '*(w - d)))
        for i in reversed(range(1, d)):
            lines.append('%s/%s' % (' '*i, ' '*(w - i)))
        lines.append('/' + '_'*(w - 1) + ',')
        return d, h + more, lines, more
    w += more
    d += more
    falling = '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT}'
    rising = '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT}'
    lines.append('_'*w)
    for i in range(0, d):
        lines.append('%s%s%s' % (' '*i, falling, ' '*(w - i - 1)))
    for i in reversed(range(0, d)):
        lines.append('%s%s%s' % (' '*i, rising, ' '*(w - i - 1)))
    lines.append('\N{OVERLINE}'*w)
    return d, h + 2*more, lines, more


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Sum
def _sum(p: PrettyPrinter, e: Sum, context: PrettyContext) -> PrettyResult:
    ascii_mode = not p.unicode
    # the operand of the sign: a sum is bracketed, so that what follows
    # the sign up to the next + or - is the operand
    limits, function = _flattened(e)
    picture = _body(p, function, context, Binding.NEGATION).output
    required = picture.height() + 2
    first = True
    max_upper = 0
    sign_height = 0
    adjustment = 0
    for limit in limits:
        lower, upper = _limits_pictures(p, limit, context)
        max_upper = max(max_upper, upper.height())
        d, h, lines, adjustment = _sum_sign(required, ascii_mode)
        sign = Picture(lines, 1)
        if first:
            sign_height = sign.height()
        sign = sign.above(upper).below(lower)
        if first:
            # the operand is centred on the sign
            picture = picture.with_baseline(picture.height()//2 - d)
            first = False
        pad = Picture([' ']*h, 1)
        picture = Picture.beside(sign, pad, picture)
    # an ASCII sign of odd height has its middle on a row
    ascii_adjustment = 1 if ascii_mode and not adjustment else 0
    picture = picture.with_baseline(max_upper + sign_height//2 + ascii_adjustment)
    return PrettyResult(picture, Binding.PRODUCT, open_ended=True)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Product
def _product(p: PrettyPrinter, e: Product, context: PrettyContext) -> PrettyResult:
    # (SymPy does not bracket a sum under a product sign: Product(k + 1, ...)
    # and Product(k, ...) + 1 print the same, sympy/sympy#30844)
    limits, function = _flattened(e)
    picture = _body(p, function, context, Binding.NEGATION).output
    symbology = p.symbology
    horizontal = symbology.xobj('-', 1)[0] if p.unicode else '_'
    corner = symbology.xobj('UpTack', 1)[0] if p.unicode else '_'
    vertical = symbology.xobj('|', 1)[0]
    func_height = picture.height()
    first = True
    max_upper = 0
    sign_height = 0
    for limit in limits:
        lower, upper = _limits_pictures(p, limit, context)
        width = (func_height + 2)*5//3 - 2
        lines = [horizontal + corner + horizontal*(width - 2) + corner + horizontal]
        for _ in range(func_height + 1):
            lines.append(' ' + vertical + ' '*(width - 2) + vertical + ' ')
        sign = Picture(lines, 1)
        max_upper = max(max_upper, upper.height())
        if first:
            sign_height = sign.height()
        sign = sign.above(upper).below(lower)
        if first:
            picture = picture.with_baseline(0)
            first = False
        padding = Picture([' ']*(sign.height() - 1), 1)
        picture = Picture.beside(sign, padding, picture)
    picture = picture.with_baseline(max_upper + sign_height//2)
    return PrettyResult(picture, Binding.PRODUCT, open_ended=True)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Limit
def _limit(p: PrettyPrinter, e: Limit, context: PrettyContext) -> PrettyResult:
    # the operand is bracketed unless it is a power, a function or an atom
    c_body = context.at(Position.BODY, Binding.ROOT)
    found = p.rule(e.expr, c_body)
    closed = p.enclose(found, c_body)
    body = closed.output
    arrow = p.symbology.xobj('-', 1)[0] + p.symbology.atom('Arrow', '>') if p.unicode else '->'
    point = e.point
    marked = e.direction != '+-' and not isinstance(point, (Infinity, NegativeInfinity))
    # the direction mark applies to the whole point: x -> (a + b)+, not
    # a + b+ (sympy/sympy#30844, root cause 6)
    c = context.at(Position.POINT, Binding.APPLICATION if marked else Binding.LOWEST)
    below = Picture.beside(p.operand(e.variable, context.delimited()), Picture.text(arrow), p.operand(point, c))
    if marked:
        if p.unicode:
            mark = p.symbology.atom('SuperscriptPlus' if e.direction == '+' else 'SuperscriptMinus', e.direction)
        else:
            mark = e.direction
        below = below.rtext(mark)
    sign = Picture.text('lim').below(below)
    if closed is found and body.height() - body.baseline > 1:
        # the rows of the operand under its baseline stand next to the
        # variable and the point: a column keeps them apart (SymPy
        # prints x -> oo and the denominator x of 1/x as "x─→∞x")
        sign = sign.rtext(' ')
    return PrettyResult(Picture.beside(sign, body), Binding.PRODUCT, open_ended=True)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Subs
def _subs(p: PrettyPrinter, e: Subs, context: PrettyContext) -> PrettyResult:
    picture = p.parens(p.operand(e.expr, context.delimited()))
    height = picture.height() if picture.height() > 1 else 2
    bar = Picture.text(p.symbology.vobj('|', height), picture.baseline)
    picture = picture.right(bar)
    baseline = picture.baseline
    assignments: list[Picture] = []
    for variable, value in zip(e.variables, e.point):
        assignments.append(Picture.beside(p.operand(variable, context.delimited()), Picture.text('='),
                                          p.operand(value, context.delimited())))
    picture = picture.with_baseline(picture.height() - 1).right(p.join(assignments, ', '))
    return PrettyResult(picture.with_baseline(baseline), Binding.PRODUCT)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Order
def _order(p: PrettyPrinter, e: Order, context: PrettyContext) -> PrettyResult:
    picture = p.operand(e.expr, context.delimited())
    variables, point = e.variables, e.point
    if any(not (v.is_Number and v == 0) for v in point) or len(variables) > 1:
        picture = picture.rtext('; ')
        if len(variables) > 1:
            picture = picture.right(p.sequence(variables, context, '(', ')').output)
        elif variables:
            picture = picture.right(p.operand(variables[0], context.delimited()))
        picture = picture.rtext(' %s ' % p.symbology.atom('Arrow', '->'))
        if len(point) > 1:
            picture = picture.right(p.sequence(point, context, '(', ')').output)
        else:
            picture = picture.right(p.operand(point[0], context.delimited()))
    return PrettyResult(p.parens(picture).ltext('O'), Binding.APPLICATION)


# PORTED FROM SYMPY: sympy/printing/pretty/pretty.py:PrettyPrinter._print_Lambda
def _lambda(p: PrettyPrinter, e: Lambda, context: PrettyContext) -> PrettyResult:
    arrow = ' %s ' % p.symbology.atom('ArrowFromBar', '->')
    signature = e.signature
    if len(signature) == 1:
        variables = p.operand(signature[0], context.delimited())
    else:
        variables = p.sequence(signature, context, '(', ')').output
    # nothing closes the body: the result is open
    body = p.operand(e.expr, context.at(Position.BODY, Binding.LOWEST))
    return PrettyResult(Picture.beside(variables, Picture.text(arrow), body), Binding.LOWEST)


CALCULUS_RULES: tuple[RuleEntry, ...] = (
    rule_for(Derivative, _derivative),
    rule_for(Integral, _integral),
    rule_for(Sum, _sum),
    rule_for(Product, _product),
    rule_for(Limit, _limit),
    rule_for(Subs, _subs),
    rule_for(Order, _order),
    rule_for(Lambda, _lambda),
)
