"""``fraction``, ``numer``, ``denom``, ``collect`` and ``rcollect``.

Ported from SymPy's ``sympy.simplify.radsimp``: ``fraction`` splits a
product into its numerator and denominator without simplifying (``exact``
keeps powers with symbolic negative exponents as they are); ``collect``
groups the terms of a sum by the powers of the given symbols (or of any
pattern, powers of sums and derivatives included), ``collect_dict`` gives
the coefficients as SymPy's ``collect(..., evaluate=False)`` does;
``rcollect`` collects recursively.

>>> from sympy_extras.core import Symbol, Rational
>>> from sympy_extras.functions import exp
>>> x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
>>> fraction(2*x/(3*y)), fraction(x*exp(-y)), fraction(x**(-a), exact=True)
((2*x, 3*y), (x, exp(y)), (x**(-a), 1))
>>> collect(a*x**2 + b*x**2 + a*x - b*x + 3, x), collect_dict(a*x + b*x + y, x)
(x**2*(a + b) + x*(a - b) + 3, {x: a + b, 1: y})
>>> collect(a*x*y + b*x*y + x, [x, y])
x*(a*y + b*y + 1)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/radsimp.py (fraction, numer, denom, collect, rcollect) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; collect(evaluate=False) is collect_dict; the
# noncommutative symbols raise as in SymPy.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional, Sequence, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, factors_of, to_expr
from sympy_extras.core.function import Derivative, expand_power_base
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Exp1, Integer, Number, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol, Wild

__all__ = ['fraction', 'numer', 'denom', 'collect', 'collect_dict', 'rcollect']

Symbols = Union[ExprLike, Sequence[ExprLike]]
Transform = Callable[[Expr], Expr]


def _is_exp(e: Expr) -> bool:
    from sympy_extras.functions.elementary.exponential import Exp
    return isinstance(e, Exp)


def _unevaluated_mul(args: Sequence[Expr]) -> Expr:
    if not args:
        return Integer(1)
    if len(args) == 1:
        return args[0]
    return Mul(*args)


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:fraction
def fraction(expr: ExprLike, exact: bool = False) -> tuple[Expr, Expr]:
    """``(numerator, denominator)`` of a product, without simplification."""
    e = to_expr(expr)
    numer: list[Expr] = []
    denom: list[Expr] = []
    for term in factors_of(e):
        if term.is_commutative is not False and (isinstance(term, Pow) or _is_exp(term)):
            b, ex = term.as_base_exp()
            if ex.is_negative is True:
                if isinstance(ex, Integer) and ex.p == -1:
                    denom.append(b)
                elif exact:
                    if ex.is_constant():
                        denom.append(pow(b, mul(Integer(-1), ex)))
                    else:
                        numer.append(term)
                else:
                    denom.append(pow(b, mul(Integer(-1), ex)))
            elif ex.is_positive is True:
                numer.append(term)
            elif not exact and isinstance(ex, Mul):
                n, d = term.as_numer_denom()
                if not (isinstance(n, Integer) and n.p == 1):
                    numer.append(n)
                denom.append(d)
            else:
                numer.append(term)
        elif isinstance(term, Rational) and term.q != 1:
            if term.p != 1:
                numer.append(Integer(term.p))
            denom.append(Integer(term.q))
        else:
            numer.append(term)
    if exact:
        return _unevaluated_mul(numer), _unevaluated_mul(denom)
    return mul(*numer), mul(*denom)


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:numer
def numer(expr: ExprLike, exact: bool = False) -> Expr:
    return fraction(expr, exact)[0]


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:denom
def denom(expr: ExprLike, exact: bool = False) -> Expr:
    return fraction(expr, exact)[1]


Deriv = Optional[tuple[Symbol, Expr]]
ParsedTerm = tuple[Expr, Expr, Optional[Expr], Deriv]


def _parse_derivative(d: Derivative) -> tuple[Expr, tuple[Symbol, Expr]]:
    e, sym, order = d.expr, d.variables[0], 1
    for s in d.variables[1:]:
        if s == sym:
            order += 1
        else:
            raise NotImplementedError('Improve MV Derivative support in collect')
    while isinstance(e, Derivative):
        variables = e.variables
        s0 = variables[0]
        if any(s != s0 for s in variables):
            raise NotImplementedError('Improve MV Derivative support in collect')
        if s0 == sym:
            e, order = e.expr, order + len(variables)
        else:
            break
    return e, (sym, Integer(order))


def _parse_term(e: Expr) -> ParsedTerm:
    from sympy_extras.functions import exp
    rat_expo: Expr = Integer(1)
    sym_expo: Optional[Expr] = None
    sexpr: Expr = e
    deriv: Deriv = None
    if isinstance(e, Pow):
        if isinstance(e.base, Derivative):
            sexpr, deriv = _parse_derivative(e.base)
        else:
            sexpr = e.base
        if isinstance(e.base, Exp1):
            arg = e.exp
            if isinstance(arg, Rational):
                from sympy_extras.core.numbers import E
                sexpr, rat_expo = E, arg
            elif isinstance(arg, Mul):
                coeff, tail = arg.as_coeff_Mul(rational=True)
                sexpr, rat_expo = exp(tail), coeff
        elif isinstance(e.exp, Number):
            rat_expo = e.exp
        else:
            coeff, tail = e.exp.as_coeff_Mul()
            if isinstance(coeff, Number):
                rat_expo, sym_expo = coeff, tail
            else:
                sym_expo = e.exp
    elif _is_exp(e):
        arg = _as_expr(e.args[0])
        if isinstance(arg, Rational):
            from sympy_extras.core.numbers import E
            sexpr, rat_expo = E, arg
        elif isinstance(arg, Mul):
            coeff, tail = arg.as_coeff_Mul(rational=True)
            sexpr, rat_expo = exp(tail), coeff
    elif isinstance(e, Derivative):
        sexpr, deriv = _parse_derivative(e)
    return sexpr, rat_expo, sym_expo, deriv


def _make_expression(terms: Sequence[ParsedTerm]) -> Expr:
    product: list[Expr] = []
    for term, rat, sym, deriv in terms:
        if deriv is not None:
            var, order = deriv
            for _ in range(int(_as_int(order))):
                term = Derivative(term, var)
        if sym is None:
            product.append(term if isinstance(rat, Integer) and rat.p == 1 else pow(term, rat))
        else:
            product.append(pow(term, mul(rat, sym)))
    return mul(*product)


def _as_int(e: Expr) -> int:
    if not isinstance(e, Integer):
        raise TypeError('an integer is expected')
    return e.p


def _matches(e: Expr, pattern: Expr) -> bool:
    return e.match(pattern) is not None


def _parse_expression(terms: list[Optional[ParsedTerm]], pattern: Expr, exact: bool
                      ) -> Optional[tuple[list[ParsedTerm], list[ParsedTerm], Optional[Expr], bool]]:
    pattern_args = factors_of(pattern)
    if len([t for t in terms if t is not None]) < len(pattern_args):
        return None
    parsed_pattern = [_parse_term(elem) for elem in pattern_args]
    pending = list(terms)
    elems: list[ParsedTerm] = []
    common_expo: Optional[Expr] = None
    has_deriv = False
    for elem, e_rat, e_sym, e_ord in parsed_pattern:
        if isinstance(elem, Number) and isinstance(e_rat, Integer) and e_rat.p == 1 and e_sym is None:
            continue
        for j in range(len(pending)):
            current = pending[j]
            if current is None:
                continue
            term, t_rat, t_sym, t_ord = current
            if t_ord is not None:
                has_deriv = True
            if _matches(term, elem) and (t_sym == e_sym or (t_sym is not None and e_sym is not None
                                                             and _matches(t_sym, e_sym))):
                if exact is False:
                    expo = mul(t_rat, pow(e_rat, Integer(-1)))
                    if common_expo is None:
                        common_expo = expo
                    elif common_expo != expo:
                        common_expo = Integer(1)
                else:
                    if e_rat != t_rat or e_ord != t_ord:
                        continue
                elems.append(current)
                pending[j] = None
                break
        else:
            return None
    return [t for t in pending if t is not None], elems, common_expo, has_deriv


def _symbols(syms: Symbols) -> list[Expr]:
    if isinstance(syms, (list, tuple)):
        return [to_expr(s) for s in syms]
    if isinstance(syms, Expr):
        return [syms]
    if isinstance(syms, (int, float, Fraction)):
        return [to_expr(syms)]
    return [to_expr(s) for s in syms]


def _commutative_parts(e: Expr) -> tuple[list[Expr], list[Expr]]:
    parts = factors_of(e)
    return ([f for f in parts if f.is_commutative is not False], [f for f in parts if f.is_commutative is False])


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:collect
def _collect(e: Expr, syms: list[Expr], func: Optional[Transform], evaluate: bool, exact: Optional[bool],
             distribute_order_term: bool) -> Union[Expr, dict[Expr, Expr]]:
    def simple(x: Expr) -> bool:
        return isinstance(x, Symbol) or isinstance(mul(Integer(-1), x), Symbol) or bool(x.atoms(Wild))

    nonsyms = [s for s in syms if not simple(s)]
    if nonsyms:
        reps: dict[Basic, Basic] = {s: Dummy() for s in nonsyms}
        replaced = [_as_expr(reps.get(s, s)) for s in syms]
        rv = _collect(_as_expr(e.subs(reps)), replaced, func, evaluate, exact, distribute_order_term)
        urep: dict[Basic, Basic] = {v: k for k, v in reps.items()}
        if not isinstance(rv, dict):
            return _as_expr(rv._xreplace_evaluated(urep))
        return {_as_expr(_as_expr(urep.get(k, k))._xreplace_evaluated(urep)): _as_expr(v._xreplace_evaluated(urep))
                for k, v in rv.items()}
    if exact is None:
        found: set[Expr] = set()
        for i in (e.args if isinstance(e, Add) else (e,)):
            if not isinstance(i, Expr) or not i.has_free(*syms) or i in syms:
                continue
            if not isinstance(i, Mul) and i not in syms:
                found.add(i)
            else:
                dependent = [f for f in factors_of(i) if f.has_free(*syms)]
                g = _unevaluated_mul(dependent)
                if g not in syms:
                    found.add(g)
        is_simple = all(isinstance(i, Pow) and i.base in syms for i in found)
        syms = syms + [_as_expr(b) for b in ordered(list(found))]
        if not is_simple:
            return _collect(e, syms, func, evaluate, False, distribute_order_term)
    if evaluate:
        if isinstance(e, Add):
            o = e.getO()
            collected_args = [_collected(_as_expr(a), syms, func, exact, distribute_order_term)
                              for a in e.args if o is None or a != o]
            e = add(add(*collected_args), o if o is not None else Integer(0))
        elif isinstance(e, Mul):
            return mul(*[_collected(_as_expr(t), syms, func, exact, distribute_order_term) for t in e.args])
        elif isinstance(e, Pow):
            return pow(_collected(e.base, syms, func, exact, distribute_order_term), e.exp)
    syms = [expand_power_base(i, deep=False) for i in syms]
    order_term: Optional[Expr] = None
    if distribute_order_term:
        order_term = e.getO()
        if order_term is not None:
            if order_term.has(*syms):
                order_term = None
            else:
                e = e.removeO()
    summa = [expand_power_base(i, deep=False) for i in (e.args if isinstance(e, Add) else (e,))
             if isinstance(i, Expr)]
    collected: dict[Expr, list[Expr]] = {}
    disliked: Expr = Integer(0)
    for product in summa:
        c, nc = _commutative_parts(product)
        args = [_as_expr(a) for a in ordered(c)] + nc
        terms: list[Optional[ParsedTerm]] = [_parse_term(i) for i in args]
        small_first = True
        for symbol in syms:
            if isinstance(symbol, Derivative) and small_first:
                terms = list(reversed(terms))
                small_first = not small_first
            result = _parse_expression(terms, symbol, bool(exact))
            if result is not None:
                if symbol.is_commutative is False:
                    raise AttributeError('Can not collect noncommutative symbol')
                rest_terms, elems, _, has_deriv = result
                if not has_deriv:
                    margs: list[Expr] = []
                    for elem in elems:
                        expo = elem[1] if elem[2] is None else mul(elem[1], elem[2])
                        margs.append(pow(elem[0], expo))
                    index = mul(*margs)
                else:
                    index = _make_expression(elems)
                remaining = expand_power_base(_make_expression(rest_terms), deep=False)
                index = expand_power_base(index, deep=False)
                collected.setdefault(index, []).append(remaining)
                break
        else:
            disliked = add(disliked, product)
    result_dict = {k: add(*v) for k, v in collected.items()}
    if not (isinstance(disliked, Integer) and disliked.p == 0):
        result_dict[Integer(1)] = disliked
    if order_term is not None:
        result_dict = {k: add(v, order_term) for k, v in result_dict.items()}
    if func is not None:
        result_dict = {k: func(v) for k, v in result_dict.items()}
    if evaluate:
        return add(*[mul(k, v) for k, v in result_dict.items()])
    return result_dict


def _collected(e: Expr, syms: list[Expr], func: Optional[Transform], exact: Optional[bool],
               distribute_order_term: bool) -> Expr:
    result = _collect(e, syms, func, True, exact, distribute_order_term)
    if isinstance(result, dict):
        raise TypeError('an expression is expected')
    return result


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def collect(expr: ExprLike, syms: Symbols, func: Optional[Transform] = None, exact: Optional[bool] = False,
            distribute_order_term: bool = True) -> Expr:
    """The terms of a sum grouped by the powers of ``syms``."""
    result = _collect(to_expr(expr), _symbols(syms), func, True, exact, distribute_order_term)
    if isinstance(result, dict):
        raise TypeError('an expression is expected')
    return result


def collect_dict(expr: ExprLike, syms: Symbols, func: Optional[Transform] = None, exact: Optional[bool] = False,
                 distribute_order_term: bool = True) -> dict[Expr, Expr]:
    """SymPy's ``collect(..., evaluate=False)``: the coefficients of the
    powers of ``syms`` (the key ``1`` for the rest)."""
    result = _collect(to_expr(expr), _symbols(syms), func, False, exact, distribute_order_term)
    if not isinstance(result, dict):
        raise TypeError('a dictionary is expected')
    return result


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:rcollect
def rcollect(expr: ExprLike, *vars: ExprLike) -> Expr:
    """``collect`` applied to every sum of the tree, bottom up."""
    e = to_expr(expr)
    variables = [to_expr(v) for v in vars]
    if not e.args or not e.has(*variables):
        return e
    rebuilt = _as_expr(e._evaluated(tuple(rcollect(a, *variables) if isinstance(a, Expr) else a for a in e.args)))
    if isinstance(rebuilt, Add):
        return collect(rebuilt, variables)
    return rebuilt
