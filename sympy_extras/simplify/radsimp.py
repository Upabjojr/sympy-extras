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
# PORTED FROM SYMPY: sympy/simplify/radsimp.py (fraction, numer, denom, collect, rcollect, radsimp,
# rad_rationalize, split_surds, _split_gcd), sympy/core/mul.py (_unevaluated_Mul),
# sympy/core/exprtools.py (Factors of an expression) (SymPy 1.14)
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
from sympy_extras.core.containers import Tuple
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol, Wild

__all__ = ['fraction', 'numer', 'denom', 'collect', 'collect_dict', 'rcollect', 'radsimp', 'rad_rationalize',
           'split_surds']

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


def _truth(b: Basic) -> bool:
    """SymPy's ``bool`` of a relational: ``TypeError`` when undecided."""
    from sympy_extras.core.logic import false, true
    if b is true:
        return True
    if b is false:
        return False
    raise TypeError('cannot determine truth value of %s' % (b,))


def _lt(a: Expr, b: Expr) -> bool:
    from sympy_extras.core.relational import lt
    return _truth(lt(a, b))


# PORTED FROM SYMPY: sympy/core/mul.py:_unevaluated_Mul
def unevaluated_mul(*args: Expr) -> Expr:
    """A well-formed unevaluated product: numbers multiplied into one
    coefficient first, products flattened, the factors sorted."""
    from sympy_extras.core.sorting import compare_key
    pending = list(args)
    commutative: list[Expr] = []
    noncommutative: list[Expr] = []
    coefficient: Expr = Integer(1)
    i = 0
    while i < len(pending):
        a = pending[i]
        i += 1
        if isinstance(a, Mul):
            for f in a.args:
                f = _as_expr(f)
                if f.is_commutative is False:
                    noncommutative.append(f)
                else:
                    pending.append(f)
        elif isinstance(a, Number):
            coefficient = mul(coefficient, a)
        elif a.is_commutative is not False:
            commutative.append(a)
        else:
            noncommutative.append(a)
    commutative.sort(key=compare_key)
    if not (isinstance(coefficient, Integer) and coefficient.p == 1):
        commutative.insert(0, coefficient)
    return _unevaluated_mul(commutative + noncommutative)


# PORTED FROM SYMPY: sympy/core/exprtools.py:Factors.__init__ (of an expression)
def _factors_of(e: Expr) -> dict[Expr, Expr]:
    """The bases and exponents of a product (``I`` counted, a rational
    split into numerator and denominator)."""
    from sympy_extras.core.numbers import I
    commutative = [f for f in factors_of(e) if f.is_commutative is not False]
    noncommutative = [f for f in factors_of(e) if f.is_commutative is False]
    units = sum(1 for f in commutative if f == I)
    rest = [f for f in commutative if f != I]
    powers = dict(_unevaluated_mul(rest).as_powers_dict()) if rest else {}
    for f in list(powers):
        if isinstance(f, Rational) and f.q != 1:
            p, q = Integer(f.p), Integer(f.q)
            powers[p] = add(powers.get(p, Integer(0)), powers[f])
            powers[q] = add(powers.get(q, Integer(0)), mul(Integer(-1), powers[f]))
            powers.pop(f)
    if units:
        powers[I] = add(powers.get(I, Integer(0)), Integer(units))
    if noncommutative:
        powers[Mul(*noncommutative)] = Integer(1)
    return powers


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:_split_gcd
def _split_gcd(*a: Expr) -> tuple[Expr, list[Expr], list[Expr]]:
    from sympy_extras.polys.polytools import gcd
    g = a[0]
    b1 = [g]
    b2: list[Expr] = []
    for x in a[1:]:
        g1 = gcd(g, x)
        if isinstance(g1, Integer) and g1.p == 1:
            b2.append(x)
        else:
            g = g1
            b1.append(x)
    return g, b1, b2


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:split_surds
def split_surds(expr: Expr) -> tuple[Expr, Expr, Expr]:
    """``(g, a, b)`` with ``expr = a*sqrt(g) + b``, ``a`` collecting the
    surds whose radicands share the gcd ``g``."""
    from sympy_extras.functions import sqrt
    args = sorted((_as_expr(a) for a in expr.args), key=lambda t: t.sort_key())
    coeff_muls = [x.as_coeff_Mul() for x in args]
    surds = sorted((pow(x[1], Integer(2)) for x in coeff_muls if isinstance(x[1], Pow)), key=lambda t: t.sort_key())
    g, b1, b2 = _split_gcd(*surds)
    g2 = g
    if not b2 and len(b1) >= 2:
        b1n = [mul(x, pow(g, Integer(-1))) for x in b1]
        b1n = [x for x in b1n if not (isinstance(x, Integer) and x.p == 1)]
        g1, b1n, b2 = _split_gcd(*b1n)
        g2 = mul(g, g1)
    a1v: list[Expr] = []
    a2v: list[Expr] = []
    for c, sv in coeff_muls:
        if isinstance(sv, Pow) and sv.exp == Rational(1, 2):
            s1 = sv.base
            if s1 in b1:
                a1v.append(mul(c, sqrt(mul(s1, pow(g2, Integer(-1))))))
            else:
                a2v.append(mul(c, sv))
        else:
            a2v.append(mul(c, sv))
    return g2, add(*a1v), add(*a2v)


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:rad_rationalize
def rad_rationalize(num: Expr, den: Expr) -> tuple[Expr, Expr]:
    """``num/den`` with the square roots of the denominator removed."""
    from sympy_extras.functions import sqrt
    from .sqrtdenest import _mexpand
    if not isinstance(den, Add):
        return num, den
    g, a, b = split_surds(den)
    a = mul(a, sqrt(g))
    num = _mexpand(mul(add(a, mul(Integer(-1), b)), num))
    den = _mexpand(add(pow(a, Integer(2)), mul(Integer(-1), pow(b, Integer(2)))))
    return rad_rationalize(num, den)


def _ispow2(d: Expr, symbolic: bool, log2: bool = False) -> bool:
    from sympy_extras.functions import log
    if not isinstance(d, Pow):
        return False
    e = d.exp
    if (isinstance(e, Rational) and e.q == 2) or (symbolic and denom(e) == Integer(2)):
        return True
    if log2:
        q = 1
        if isinstance(e, Rational):
            q = e.q
        elif symbolic:
            dd = denom(e)
            if isinstance(dd, Integer):
                q = dd.p
        if q != 1 and isinstance(log(Integer(q), Integer(2)), Integer):
            return True
    return False


def _radsimp_numerator(rterms: list[tuple[Expr, Expr]]) -> Expr:
    """The conjugate factor of SymPy's ``_num`` for two, three or four
    radical terms ``(A, a)`` meaning ``sqrt(A)*a``."""
    from sympy_extras.functions import sqrt

    def s(i: int) -> Expr:
        return sqrt(rterms[i][0])

    def c(i: int) -> Expr:
        return rterms[i][1]

    def A(i: int) -> Expr:
        return rterms[i][0]

    if len(rterms) == 1:
        return sqrt(rterms[0][0])
    if len(rterms) == 2:
        return add(mul(s(0), c(0)), mul(Integer(-1), s(1), c(1)))
    if len(rterms) == 3:
        first = add(mul(s(0), c(0)), mul(s(1), c(1)), mul(Integer(-1), s(2), c(2)))
        second = add(mul(Integer(2), s(0), s(1), c(0), c(1)), mul(Integer(-1), A(0), pow(c(0), Integer(2))),
                     mul(Integer(-1), A(1), pow(c(1), Integer(2))), mul(A(2), pow(c(2), Integer(2))))
        return mul(first, second)
    if len(rterms) == 4:
        a, b, cc, d = c(0), c(1), c(2), c(3)
        AA, B, C, D = A(0), A(1), A(2), A(3)
        sA, sB, sC, sD = s(0), s(1), s(2), s(3)
        f1 = add(mul(sA, a), mul(sB, b), mul(Integer(-1), sC, cc), mul(Integer(-1), sD, d))
        f2 = add(mul(Integer(2), sA, sB, a, b), mul(Integer(-1), AA, pow(a, Integer(2))),
                 mul(Integer(-1), B, pow(b, Integer(2))), mul(Integer(-2), sC, sD, cc, d), mul(C, pow(cc, Integer(2))),
                 mul(D, pow(d, Integer(2))))
        f3 = add(mul(Integer(-8), sA, sB, sC, sD, a, b, cc, d), mul(pow(AA, Integer(2)), pow(a, Integer(4))),
                 mul(Integer(-2), AA, B, pow(a, Integer(2)), pow(b, Integer(2))),
                 mul(Integer(-2), AA, C, pow(a, Integer(2)), pow(cc, Integer(2))),
                 mul(Integer(-2), AA, D, pow(a, Integer(2)), pow(d, Integer(2))), mul(pow(B, Integer(2)), pow(b, Integer(4))),
                 mul(Integer(-2), B, C, pow(b, Integer(2)), pow(cc, Integer(2))),
                 mul(Integer(-2), B, D, pow(b, Integer(2)), pow(d, Integer(2))), mul(pow(C, Integer(2)), pow(cc, Integer(4))),
                 mul(Integer(-2), C, D, pow(cc, Integer(2)), pow(d, Integer(2))), mul(pow(D, Integer(2)), pow(d, Integer(4))))
        return mul(f1, f2, f3)
    raise NotImplementedError('radsimp of more than four radical terms')


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:radsimp
def radsimp(expr: ExprLike, symbolic: bool = True, max_terms: int = 4) -> Expr:
    """The denominator rationalized: square roots (and roots of index a
    power of two) removed from the denominator when it has at most
    ``max_terms`` radical terms; ``symbolic`` allows symbolic radicands.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sqrt
    >>> radsimp(1/(2 + sqrt(2))), radsimp(1/(sqrt(2) + sqrt(3))), radsimp(1/(1 + sqrt(Symbol('x'))))
    ((2 - sqrt(2))/2, -sqrt(2) + sqrt(3), (sqrt(x) - 1)/(x - 1))
    """
    from sympy_extras.core.exprtools import gcd_terms
    from sympy_extras.core.function import expand_mul
    from sympy_extras.core.numbers import ComplexInfinity, NaN
    from .powsimp import powdenest, powsimp
    from .signsimp import signsimp
    from .sqrtdenest import _mexpand, sqrtdenest
    e = to_expr(expr)

    def inverse(x: Expr) -> Expr:
        return pow(x, Integer(-1))

    def handle(x: Expr) -> Expr:
        from sympy_extras.functions import sqrt
        if not x.args:
            return x
        n, d = fraction(x)
        if not d.args and not n.args:
            return x
        if n.args:
            n = _as_expr(n._evaluated(tuple(handle(_as_expr(a)) for a in n.args)))
            return unevaluated_mul(n, handle(inverse(d)))
        if not (isinstance(n, Integer) and n.p == 1):
            return unevaluated_mul(n, handle(inverse(d)))
        if isinstance(d, Mul):
            return unevaluated_mul(*[handle(inverse(_as_expr(f))) for f in d.args])
        if not symbolic and d.free_symbols:
            return x
        if _ispow2(d, symbolic) and isinstance(d, Pow):
            d2 = pow(sqrtdenest(sqrt(d.base)), numer(d.exp))
            if d2 != d:
                return handle(inverse(d2))
        elif isinstance(d, Pow) and (d.exp.is_integer is True or d.base.is_positive is True):
            return pow(handle(inverse(d.base)), d.exp)
        if not (isinstance(d, Add) or _ispow2(d, symbolic)):
            return inverse(_as_expr(d._evaluated(tuple(handle(_as_expr(a)) for a in d.args))))
        keep = True
        d = _mexpand(d)
        if not d.args:
            return inverse(d)
        if d.is_number:
            from .simplify import nsimplify
            nd = nsimplify(d)
            if isinstance(nd, Number) and nd.equals(d) is True:
                return inverse(nd)
        while True:
            collected: dict[tuple[Expr, ...], list[Expr]] = {}
            for m in (d.args if isinstance(d, Add) else (d,)):
                m = _as_expr(m)
                p2: list[Expr] = []
                other: list[Expr] = []
                for i in factors_of(m):
                    if _ispow2(i, symbolic, log2=True) and isinstance(i, Pow):
                        p2.append(i.base if i.exp == Rational(1, 2) else pow(i.base, mul(Integer(2), i.exp)))
                    elif i == I_unit():
                        p2.append(Integer(-1))
                    else:
                        other.append(i)
                key = tuple(_as_expr(v) for v in ordered(p2))
                collected.setdefault(key, []).append(mul(*other))
            rterms = [(mul(*k), add(*collected[k])) for k in _ordered_tuples(list(collected))]
            nrad = len(rterms) - (1 if isinstance(rterms[0][0], Integer) and rterms[0][0].p == 1 else 0)
            if nrad < 1:
                break
            if nrad > max_terms:
                keep = False
                break
            if len(rterms) > 4:
                if all(isinstance(x, Integer) and isinstance(pow(y, Integer(2)), Rational) for x, y in rterms):
                    nd2, d = rad_rationalize(Integer(1), Add(*[mul(sqrt(x), y) for x, y in rterms]))
                    n = mul(n, nd2)
                else:
                    keep = False
                break
            num = powsimp(_radsimp_numerator(rterms))
            n = mul(n, num)
            d = mul(d, num)
            d = powdenest(_mexpand(d), force=symbolic)
            # SymPy's d.has(S.Zero, nan, zoo)
            if any(isinstance(t, (NaN, ComplexInfinity)) or (isinstance(t, Integer) and t.p == 0) for t in _atoms_of(d)):
                return x
            if not d.args:
                break
        if not keep:
            return x
        return unevaluated_mul(n, inverse(d))

    if not isinstance(e, Expr):
        return e
    coeff, rest = e.as_coeff_Add()
    rest = rest.normal()
    old = fraction(rest)
    n, d = fraction(handle(rest))
    if old != (n, d):
        if d.args:
            was = (n, d)
            n = signsimp(n, evaluate=False)
            d = signsimp(d, evaluate=False)
            u = _factors_of(unevaluated_mul(n, inverse(d)))
            u_expr = unevaluated_mul(*[pow(k, v) for k, v in u.items()])
            n, d = fraction(u_expr)
            if old == (n, d):
                n, d = was
        n = expand_mul(n)
        if isinstance(d, Number) or isinstance(d, Add):
            n2, d2 = fraction(gcd_terms(unevaluated_mul(n, inverse(d))))
            if isinstance(d2, Number) or _count(d2) <= _count(d):
                n, d = signsimp(n2), signsimp(d2)
                if isinstance(n, Mul) and isinstance(n.args[0], Number):
                    n = mul(*[_as_expr(a) for a in n.args])
    return add(coeff, unevaluated_mul(n, inverse(d)))


def I_unit() -> Expr:
    from sympy_extras.core.numbers import I
    return I


def _ordered_tuples(keys: list[tuple[Expr, ...]]) -> list[tuple[Expr, ...]]:
    """SymPy's ``ordered`` of the tuples of radicands (a Tuple each)."""
    wrapped = {Tuple(*k): k for k in keys}
    return [wrapped[_as_tuple(t)] for t in ordered(list(wrapped))]


def _as_tuple(b: Basic) -> Tuple:
    if not isinstance(b, Tuple):
        raise TypeError('a Tuple is expected')
    return b


def _atoms_of(e: Expr) -> list[Basic]:
    from sympy_extras.core.basic import preorder_traversal
    return list(preorder_traversal(e))


def _count(e: Expr) -> int:
    from sympy_extras.core.function import count_ops
    value = count_ops(e)
    return value.p if isinstance(value, Integer) else 0
