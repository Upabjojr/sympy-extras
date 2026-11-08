"""The ``expand(trig=True)`` rules of the trigonometric and hyperbolic
functions (SymPy's ``_eval_expand_trig``), as a typed table for
:func:`sympy_extras.core.function.expand`.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py, sympy/functions/elementary/hyperbolic.py
# (the _eval_expand_trig methods; ReciprocalTrigonometricFunction and
# ReciprocalHyperbolicFunction._calculate_reciprocal) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by class in place of methods; dummies in
# place of numbered_symbols('Y').
# ---------------------------------------------------------------------
from __future__ import annotations

from itertools import combinations
from typing import Callable

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import ExpandHints
from sympy_extras.core.mul import Mul, mul, neg
from sympy_extras.core.numbers import I, Integer
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Dummy

from . import hyperbolic as h, trigonometric as t

__all__ = ['rules']

#: a rule: the node and the hints of the call to its expansion
Rule = Callable[[Expr, 'ExpandHints'], Expr]

ONE = Integer(1)


def _arg(node: Expr) -> Expr:
    a = node.args[0]
    if not isinstance(a, Expr):
        raise TypeError('an expression is expected')
    return a


def _inv(e: Expr) -> Expr:
    return pow(e, Integer(-1))


def _two_terms(a: Add) -> tuple[Expr, Expr]:
    """SymPy's ``Add.as_two_terms``: the first term and the rest."""
    terms = [x for x in a.args if isinstance(x, Expr)]
    return terms[0], add(*terms[1:])


def _symmetric_poly(n: int, gens: list[Expr]) -> Expr:
    """SymPy's ``symmetric_poly``: the elementary symmetric polynomial of
    degree ``n``."""
    return add(*[mul(*c) for c in combinations(gens, n)])


def _integer(c: Expr) -> int:
    return c.p if isinstance(c, Integer) else 0


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sin._eval_expand_trig
def _sin(node: Expr, hints: ExpandHints) -> Expr:
    from sympy_extras.core.function import expand_mul
    from sympy_extras.functions.special.polynomials import chebyshevt, chebyshevu
    a = _arg(node)
    if isinstance(a, Add):
        x, y = _two_terms(a)
        sx, sy, cx, cy = _sin(t.Sin(x), hints), _sin(t.Sin(y), hints), _cos(t.Cos(x), hints), _cos(t.Cos(y), hints)
        return add(mul(sx, cy), mul(sy, cx))
    if isinstance(a, Mul):
        n, x = a.as_coeff_Mul(rational=True)
        if isinstance(n, Integer):
            if n.p % 2:
                return mul(pow(Integer(-1), Integer((n.p - 1) // 2)), chebyshevt(n, t.sin(x)))
            return expand_mul(mul(pow(Integer(-1), Integer(n.p // 2 - 1)), t.cos(x),
                                  chebyshevu(Integer(n.p - 1), t.sin(x))), deep=False)
    return t.sin(a)


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cos._eval_expand_trig
def _cos(node: Expr, hints: ExpandHints) -> Expr:
    from sympy_extras.functions.special.polynomials import chebyshevt
    a = _arg(node)
    if isinstance(a, Add):
        x, y = _two_terms(a)
        sx, sy, cx, cy = _sin(t.Sin(x), hints), _sin(t.Sin(y), hints), _cos(t.Cos(x), hints), _cos(t.Cos(y), hints)
        return add(mul(cx, cy), neg(mul(sx, sy)))
    if isinstance(a, Mul):
        coeff, terms = a.as_coeff_Mul(rational=True)
        if isinstance(coeff, Integer):
            return chebyshevt(coeff, t.cos(terms))
    return t.cos(a)


def _by_dummies(numerator: Expr, denominator: Expr, ys: list[Dummy], values: list[Expr]) -> Expr:
    rule: dict[Basic, Basic] = dict(zip(ys, values))
    q = mul(numerator, _inv(denominator))
    r = q._xreplace_evaluated(rule)
    if not isinstance(r, Expr):
        raise TypeError('an expression is expected')
    return r


def _real_dummy_power(base_of: Callable[[Expr], Expr], n: int, value: Expr, numerator_re: bool) -> Expr:
    from sympy_extras.core.function import expand
    from sympy_extras.functions import im, re
    z = Dummy('dummy', real=True)
    P = expand(pow(base_of(z), Integer(n)))
    q = mul(re(P), _inv(im(P))) if numerator_re else mul(im(P), _inv(re(P)))
    rule: dict[Basic, Basic] = {z: value}
    r = q._xreplace_evaluated(rule)
    if not isinstance(r, Expr):
        raise TypeError('an expression is expected')
    return r


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:tan._eval_expand_trig
def _tan(node: Expr, hints: ExpandHints) -> Expr:
    a = _arg(node)
    if isinstance(a, Add):
        n = len(a.args)
        TX = [_tan(t.Tan(_expr(x)), hints) for x in a.args]
        Y = [Dummy('Y') for _ in range(n)]
        p: list[Expr] = [Integer(0), Integer(0)]
        for i in range(n + 1):
            p[1 - i % 2] = add(p[1 - i % 2], mul(_symmetric_poly(i, list(Y)), Integer((-1)**((i % 4)//2))))
        return _by_dummies(p[0], p[1], Y, TX)
    if isinstance(a, Mul):
        coeff, terms = a.as_coeff_Mul(rational=True)
        if isinstance(coeff, Integer) and coeff.p > 1:
            return _real_dummy_power(lambda z: add(ONE, mul(I, z)), coeff.p, t.tan(terms), False)
    return t.tan(a)


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cot._eval_expand_trig
def _cot(node: Expr, hints: ExpandHints) -> Expr:
    a = _arg(node)
    if isinstance(a, Add):
        n = len(a.args)
        CX = [_cot(t.Cot(_expr(x)), hints) for x in a.args]
        Y = [Dummy('Y') for _ in range(n)]
        p: list[Expr] = [Integer(0), Integer(0)]
        for i in range(n, -1, -1):
            k = (n - i) % 2
            p[k] = add(p[k], mul(_symmetric_poly(i, list(Y)), Integer((-1)**(((n - i) % 4)//2))))
        return _by_dummies(p[0], p[1], Y, CX)
    if isinstance(a, Mul):
        coeff, terms = a.as_coeff_Mul(rational=True)
        if isinstance(coeff, Integer) and coeff.p > 1:
            return _real_dummy_power(lambda z: add(z, I), coeff.p, t.cot(terms), True)
    return t.cot(a)


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected')
    return b


def _hyperbolic_sum(node: Expr, hints: ExpandHints, cosine: bool) -> Expr:
    from sympy_extras.core.function import expand
    a = hints.expand(_arg(node), True)
    x = y = None
    if isinstance(a, Add):
        x, y = _two_terms(a)
    else:
        coeff, terms = a.as_coeff_Mul(rational=True)
        if not (isinstance(coeff, Integer) and coeff.p == 1) and isinstance(coeff, Integer) and not (
                isinstance(terms, Integer) and terms.p == 1):
            x = terms
            y = mul(Integer(coeff.p - 1), x)
    if x is not None and y is not None:
        if cosine:
            e = add(mul(h.cosh(x), h.cosh(y)), mul(h.sinh(x), h.sinh(y)))
        else:
            e = add(mul(h.sinh(x), h.cosh(y)), mul(h.sinh(y), h.cosh(x)))
        return expand(e, trig=True)
    return h.cosh(a) if cosine else h.sinh(a)


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:sinh._eval_expand_trig
def _sinh(node: Expr, hints: ExpandHints) -> Expr:
    return _hyperbolic_sum(node, hints, False)


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:cosh._eval_expand_trig
def _cosh(node: Expr, hints: ExpandHints) -> Expr:
    return _hyperbolic_sum(node, hints, True)


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:tanh._eval_expand_trig
def _tanh(node: Expr, hints: ExpandHints) -> Expr:
    from sympy_extras.functions import binomial
    a = _arg(node)
    if isinstance(a, Add):
        n = len(a.args)
        TX = [_tanh(h.Tanh(_expr(x)), hints) for x in a.args]
        p: list[Expr] = [Integer(0), Integer(0)]
        for i in range(n + 1):
            p[i % 2] = add(p[i % 2], _symmetric_poly(i, TX))
        return mul(p[1], _inv(p[0]))
    if isinstance(a, Mul):
        coeff, terms = a.as_coeff_Mul()
        if isinstance(coeff, Integer) and coeff.p > 1:
            T = h.tanh(terms)
            c = coeff.p
            num = [mul(binomial(c, k), pow(T, Integer(k))) for k in range(1, c + 1, 2)]
            den = [mul(binomial(c, k), pow(T, Integer(k))) for k in range(0, c + 1, 2)]
            return mul(add(*num), _inv(add(*den)))
    return h.tanh(a)


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:coth._eval_expand_trig
def _coth(node: Expr, hints: ExpandHints) -> Expr:
    from sympy_extras.functions import binomial
    a = _arg(node)
    if isinstance(a, Add):
        CX = [_coth(h.Coth(_expr(x)), hints) for x in a.args]
        n = len(a.args)
        p: list[list[Expr]] = [[], []]
        for i in range(n, -1, -1):
            p[(n - i) % 2].append(_symmetric_poly(i, CX))
        return mul(add(*p[0]), _inv(add(*p[1])))
    if isinstance(a, Mul):
        coeff, x = a.as_coeff_Mul(rational=True)
        if isinstance(coeff, Integer) and coeff.p > 1:
            c = h.Coth(x)
            q: list[list[Expr]] = [[], []]
            for i in range(coeff.p, -1, -1):
                q[(coeff.p - i) % 2].append(mul(binomial(coeff.p, i), pow(c, Integer(i))))
            return mul(add(*q[0]), _inv(add(*q[1])))
    return h.coth(a)


def _reciprocal(of: Callable[[Expr], Expr], rule: Rule) -> Rule:
    """SymPy's ``_calculate_reciprocal``: ``1/rule(of(arg))``."""
    def expanded(node: Expr, hints: ExpandHints) -> Expr:
        o = of(_arg(node))
        return _inv(rule(o, hints) if type(o) in _OWN else o)
    return expanded


_OWN: set[type] = set()


def rules() -> dict[type, Rule]:
    """The rules keyed by the class of the node."""
    table: dict[type, Rule] = {
        t.Sin: _sin, t.Cos: _cos, t.Tan: _tan, t.Cot: _cot, h.Sinh: _sinh, h.Cosh: _cosh, h.Tanh: _tanh,
        h.Coth: _coth}
    _OWN.update(table)
    table[t.Sec] = _reciprocal(t.cos, _cos)
    table[t.Csc] = _reciprocal(t.sin, _sin)
    table[h.Sech] = _reciprocal(h.cosh, _cosh)
    table[h.Csch] = _reciprocal(h.sinh, _sinh)
    return table
