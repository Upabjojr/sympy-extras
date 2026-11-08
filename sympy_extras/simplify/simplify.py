"""``simplify``, ``logcombine``, ``posify`` and ``nsimplify`` of SymPy's
``simplify`` module.

``simplify`` tries the simplifications in SymPy's order (``cancel``,
``together``, ``factor_terms``, ``trigsimp``, ``logcombine``,
``combsimp``, ``powsimp``, ``exptrigsimp``, ``radsimp`` of the
denominator) and keeps the shortest result by ``count_ops`` unless it is
longer than ``ratio`` times the input. The steps for nodes not ported
yet (Integral, Sum, Product, Bessel functions, ``hyper``, Kronecker
deltas, Piecewise conditions, noncommutative products) go through SymPy,
only for expressions containing them.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import cos, sin, log, gamma
>>> x = Symbol('x')
>>> simplify(sin(x)**2 + cos(x)**2), simplify((x**3 + x**2 - x - 1)/(x**2 + 2*x + 1))
(1, x - 1)
>>> simplify(gamma(x + 1)/gamma(x)), simplify(log(2) + log(3))
(x, log(6))

Ported from SymPy's ``sympy.simplify.simplify``: ``logcombine`` joins
logarithms, ``a*log(x) + log(y)`` into ``log(x**a*y)``, when the
coefficient ``a`` is real and the arguments positive (or with ``force``);
``posify`` replaces the symbols of unknown sign by positive dummies and
returns the substitution back.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import log
>>> x, y = Symbol('x', positive=True), Symbol('y', positive=True)
>>> a = Symbol('a', real=True)
>>> logcombine(a*log(x) + log(y) - log(2)), logcombine(log(x) - log(y))
(log(x**a*y/2), log(x/y))
>>> z = Symbol('z')
>>> e, back = posify(1/z + z)
>>> e.free_symbols == {list(back)[0]}, e.subs(back) == 1/z + z
(True, True)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/simplify.py (simplify, inversecombine, kroneckersimp, logcombine, posify, nsimplify,
# _real_to_rational), sympy/simplify/radsimp.py (collect_abs) and the
# _eval_simplify of log, sign and the combinatorial functions (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; posify of a single expression (map it over a list); the
# formulas of mpmath.identify read by a small typed parser (no parse_expr).
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Callable, Optional

import mpmath
import sympy

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Number, Rational
from sympy_extras.core.power import pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.core.traversal import bottom_up

__all__ = ['simplify', 'logcombine', 'posify', 'nsimplify', 'collect_abs', 'inversecombine', 'kroneckersimp']


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def _is_log(e: Basic) -> bool:
    from sympy_extras.functions.elementary.exponential import Log
    return isinstance(e, Log)


def _arg(e: Expr) -> Expr:
    return _expr(e.args[0])


LogTerm = tuple[list[Expr], Expr]


# PORTED FROM SYMPY: sympy/simplify/simplify.py:logcombine
def logcombine(expr: ExprLike, force: bool = False) -> Expr:
    """Logarithms combined (see the module), bottom up."""
    from sympy_extras.functions.elementary.exponential import Log, log

    def gooda(a: Expr) -> bool:
        return not (isinstance(a, Integer) and a.p == -1) and (
            a.is_extended_real is True or (force and a.is_extended_real is not False))

    def goodlog(l: Expr) -> bool:
        a = _arg(l)
        return a.is_positive is True or (force and a.is_nonpositive is not False)

    def f(node: Basic) -> Basic:
        if not isinstance(node, (Add, Mul)):
            return node
        other: list[Expr] = []
        logs: list[tuple[list[Expr], list[Expr], list[Expr]]] = []
        log1: dict[tuple[Expr, ...], list[LogTerm]] = {}
        for a in (node.args if isinstance(node, Add) else (node,)):
            a = _expr(a)
            if _is_log(a) and goodlog(a):
                log1.setdefault((), []).append(([], a))
            elif not isinstance(a, Mul):
                other.append(a)
            else:
                ot: list[Expr] = []
                co: list[Expr] = []
                lo: list[Expr] = []
                for ai0 in a.args:
                    ai = _expr(ai0)
                    if isinstance(ai, Rational) and ai.p < 0:
                        ot.append(Integer(-1))
                        co.append(mul(Integer(-1), ai))
                    elif _is_log(ai) and goodlog(ai):
                        lo.append(ai)
                    elif gooda(ai):
                        co.append(ai)
                    else:
                        ot.append(ai)
                if len(lo) > 1:
                    logs.append((ot, co, lo))
                elif lo:
                    log1.setdefault(tuple(ot), []).append((co, lo[0]))
                else:
                    other.append(a)
        if len(other) == 1 and _is_log(other[0]):
            log1.setdefault((), []).append(([], other.pop()))
        if not logs and not log1:
            return node
        for o, e, l0 in logs:
            l = [_expr(v) for v in ordered(l0)]
            combined = log(pow(_arg(l.pop(0)), mul(*e)))
            while l:
                li = l.pop(0)
                combined = log(pow(_arg(li), combined))
            c = mul(*o)
            if _is_log(combined):
                log1.setdefault((c,), []).append(([], combined))
            else:
                other.append(mul(c, combined))
        joined: dict[Expr, Expr] = {}
        for key in list(log1):
            terms = log1.pop(key)
            inner = logcombine(mul(*[pow(_arg(lg), mul(*cs)) for cs, lg in terms]), force=force)
            joined[mul(*key)] = Log(inner)
        for k0 in ordered(list(joined)):
            k = _expr(k0)
            if k not in joined:
                continue
            minus_k = mul(Integer(-1), k)
            if minus_k in joined:
                num, den = k, minus_k
                if _ops(num) > _ops(den):
                    num, den = den, num
                numerator, denominator = joined.pop(num), joined.pop(den)
                other.append(mul(num, Log(mul(_arg(numerator), pow(_arg(denominator), Integer(-1))))))
            else:
                other.append(mul(k, joined.pop(k)))
        return add(*other)

    return _expr(bottom_up(to_expr(expr), f))


def _ops(e: Expr) -> int:
    from sympy_extras.core.function import count_ops
    value = count_ops(e)
    return value.p if isinstance(value, Integer) else 0


# PORTED FROM SYMPY: sympy/simplify/simplify.py:posify
def posify(eq: ExprLike) -> tuple[Expr, dict[Basic, Basic]]:
    """``eq`` with the symbols of unknown sign replaced by positive
    dummies of the same name, and the mapping back."""
    e = to_expr(eq)
    reps: dict[Basic, Basic] = {}
    for s in e.free_symbols:
        if isinstance(s, Symbol) and s.is_positive is None:
            assumptions = dict(s.assumptions0)
            assumptions['positive'] = True
            reps[s] = Dummy(s.name, **assumptions)
    replaced = _expr(e.subs(reps))
    return replaced, {r: s for s, r in reps.items()}


# ---------------------------------------------------------------------
# nsimplify

class _FormulaParser:
    """A recursive-descent reader of the formulas ``mpmath.identify``
    returns (numbers, ``+ - * / **``, parentheses, ``sqrt``, ``exp``,
    ``log`` and the names of the given constants)."""

    def __init__(self, text: str, names: dict[str, Expr]) -> None:
        self.text = text.replace(' ', '')
        self.pos = 0
        self.names = names

    def parse(self) -> Expr:
        e = self.expression()
        if self.pos != len(self.text):
            raise ValueError('unexpected %r in %r' % (self.text[self.pos:], self.text))
        return e

    def peek(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)

    def take(self, token: str) -> bool:
        if self.peek(token):
            self.pos += len(token)
            return True
        return False

    def expression(self) -> Expr:
        result = self.term()
        while True:
            if self.take('+'):
                result = add(result, self.term())
            elif self.peek('-'):
                self.pos += 1
                result = add(result, mul(Integer(-1), self.term()))
            else:
                return result

    def term(self) -> Expr:
        result = self.factor()
        while True:
            if self.peek('**'):
                return result
            if self.take('*'):
                result = mul(result, self.factor())
            elif self.take('/'):
                result = mul(result, pow(self.factor(), Integer(-1)))
            else:
                return result

    def factor(self) -> Expr:
        if self.take('-'):
            return mul(Integer(-1), self.factor())
        if self.take('+'):
            return self.factor()
        base = self.atom()
        if self.take('**'):
            return pow(base, self.factor())
        return base

    def atom(self) -> Expr:
        from sympy_extras.functions import exp, log, sqrt
        if self.take('('):
            e = self.expression()
            if not self.take(')'):
                raise ValueError('missing ) in %r' % self.text)
            return e
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        if self.pos > start:
            from fractions import Fraction
            from sympy_extras.core.numbers import number_from_fraction
            return number_from_fraction(Fraction(self.text[start:self.pos]))
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        name = self.text[start:self.pos]
        if not name:
            raise ValueError('unexpected %r in %r' % (self.text[self.pos:], self.text))
        functions = {'sqrt': sqrt, 'exp': exp, 'log': log}
        if name in functions and self.take('('):
            argument = self.expression()
            if not self.take(')'):
                raise ValueError('missing ) in %r' % self.text)
            return functions[name](argument)
        if name in self.names:
            return self.names[name]
        raise ValueError('unknown name %r in %r' % (name, self.text))


def _to_mpf(x: Expr, bprec: int) -> 'mpmath.mpf':
    import mpmath
    from sympy_extras.core.numbers import Float
    with mpmath.workprec(bprec):
        if isinstance(x, (Float, Rational)):
            f = x.as_fraction()
            return mpmath.mpf(f.numerator) / f.denominator
        value = x.evalf(int(bprec / 3.33) + 1)
        if isinstance(value, (Float, Rational)):
            f = value.as_fraction()
            return mpmath.mpf(f.numerator) / f.denominator
    raise ValueError('a real number is expected, got %s' % (x,))


def _float_atoms(e: Expr) -> list[Expr]:
    from sympy_extras.core.numbers import Float
    return [_expr(f) for f in e.atoms(Float)]


# PORTED FROM SYMPY: sympy/simplify/simplify.py:nsimplify
def nsimplify(expr: ExprLike, constants: tuple[ExprLike, ...] = (), tolerance: Optional[float] = None,
              full: bool = False, rational: Optional[bool] = None, rational_conversion: str = 'base10') -> Expr:
    """A simple exact expression whose value is ``expr`` (a number): a
    rational, or a combination of radicals and the given ``constants``
    found by ``mpmath.identify``; the floats of a symbolic expression made
    rational.

    >>> from sympy_extras.core import Float, Integer
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.core.numbers import pi
    >>> nsimplify(Float(0.5)), nsimplify(4/(1 + sqrt(5))), nsimplify(Float(1.414213562373095)), nsimplify(Float(0.333333333333333))
    (1/2, -1 + sqrt(5), sqrt(2), 1/3)
    >>> nsimplify(Float(3.14159265358979)*Float(2), [pi])
    2*pi
    """
    import mpmath
    from sympy_extras.core.numbers import Float, I, Infinity, NegativeInfinity, oo
    e = to_expr(expr)
    if isinstance(e, Integer):
        return e
    if isinstance(e, Float) and e.as_fraction() == e.as_fraction() and str(e) in ('inf', '-inf'):
        return oo if str(e) == 'inf' else mul(Integer(-1), oo)
    if isinstance(e, (Infinity, NegativeInfinity)):
        return e
    if rational or e.free_symbols:
        return _real_to_rational(e, tolerance, rational_conversion)
    tol: float
    if tolerance is None:
        dps = [15] + [mpmath.libmp.libmpf.prec_to_dps(f.precision) for f in _float_atoms(e) if isinstance(f, Float)]
        tol = 10.0 ** -min(dps)
    else:
        tol = tolerance
    prec = 30
    bprec = int(prec * 3.33)
    names: dict[str, Expr] = {'pi': _pi(), 'E': _e()}
    constants_dict: dict[str, 'mpmath.mpf'] = {}
    for constant in constants:
        c = to_expr(constant)
        names[str(c)] = c
        constants_dict[str(c)] = _to_mpf(c, bprec)
    value = e.evalf(prec)
    re, im = value.as_real_imag()
    re, im = _chop(re, prec), _chop(im, prec)
    if not (isinstance(re, Number) and isinstance(im, Number)):
        return e

    def nsimplify_real(x: Expr) -> Expr:
        orig = mpmath.mp.dps
        xv = _to_mpf(x, bprec)
        try:
            if not (tolerance or full):
                mpmath.mp.dps = 15
                rat = mpmath.pslq([xv, 1])
                if rat is not None:
                    return Rational(-int(rat[1]), int(rat[0])) if int(rat[0]) != 1 and int(rat[0]) != -1 \
                        else Integer(-int(rat[1]) * int(rat[0]))
            mpmath.mp.dps = prec
            newexpr = mpmath.identify(xv, constants=constants_dict, tol=tol, full=full)
            if not newexpr:
                raise ValueError('not identified')
            text = newexpr[0] if isinstance(newexpr, list) else newexpr
            result = _FormulaParser(str(text), names).parse()
            if not _is_zero_number(x) and _is_zero_number(result):
                raise ValueError('a nonzero number became 0')
            if result.is_finite is False and xv not in (mpmath.inf, mpmath.ninf):
                raise ValueError('infinite')
            return result
        finally:
            mpmath.mp.dps = orig

    try:
        if not _is_zero_number(re):
            re = nsimplify_real(re)
        if not _is_zero_number(im):
            im = nsimplify_real(im)
    except ValueError:
        if rational is None:
            return _real_to_rational(e, rational_conversion=rational_conversion)
        return e
    rv = add(re, mul(im, I))
    if rv != e or rational is False:
        return rv
    return _real_to_rational(e, rational_conversion=rational_conversion)


def _pi() -> Expr:
    from sympy_extras.core.numbers import pi
    return pi


def _e() -> Expr:
    from sympy_extras.core.numbers import E
    return E


def _is_zero_number(e: Expr) -> bool:
    return isinstance(e, Number) and e == 0


def _chop(x: Expr, prec: int) -> Expr:
    """SymPy's ``evalf(chop=True)`` on a part: a float below ``10**-prec``
    relative to 1 is 0."""
    from sympy_extras.core.numbers import Float
    if isinstance(x, Float) and abs(x.as_fraction()) < Rational(1, 10**prec).as_fraction():
        return Integer(0)
    return x


# PORTED FROM SYMPY: sympy/simplify/simplify.py:_real_to_rational
def _real_to_rational(expr: Expr, tolerance: Optional[float] = None, rational_conversion: str = 'base10') -> Expr:
    """The floats of ``expr`` replaced by rationals (by ``nsimplify`` of
    each, in base 10 when that fails, or exactly)."""
    import math
    from fractions import Fraction
    from sympy_extras.core.numbers import Float, number_from_fraction, zoo
    reps: dict[Basic, Basic] = {}
    reduce_num: Optional[int] = None
    if tolerance is not None and tolerance < 1:
        reduce_num = math.ceil(1 / tolerance)
    for fl in _float_atoms(expr):
        if not isinstance(fl, Float):
            continue
        value = fl.as_fraction()
        r: Expr
        if reduce_num is not None:
            r = number_from_fraction(value.limit_denominator(reduce_num))
        elif tolerance is not None and tolerance >= 1 and value.denominator != 1:
            r = number_from_fraction(Fraction(tolerance * round(float(value) / tolerance)).limit_denominator(int(tolerance)))
        elif rational_conversion == 'exact':
            r = number_from_fraction(value)
        elif rational_conversion != 'base10':
            raise ValueError("rational_conversion must be 'base10' or 'exact'")
        else:
            r = nsimplify(fl, rational=False)
            if value and _is_zero_number(r):
                r = number_from_fraction(value)
            elif not isinstance(r, Rational):
                if str(fl) in ('inf', '-inf'):
                    r = zoo
                elif value != 0:
                    magnitude = abs(value)
                    d = Fraction(10) ** int(math.log10(magnitude))
                    digits = Fraction(str(Float(magnitude / d)))
                    r = number_from_fraction(digits * d * (1 if value > 0 else -1))
                else:
                    r = Integer(0)
        reps[fl] = r
    return _expr(expr._xreplace_evaluated(reps))


# ---------------------------------------------------------------------
# simplify

Measure = Callable[[Expr], int]


# PORTED FROM SYMPY: sympy/simplify/radsimp.py:collect_abs
def collect_abs(expr: ExprLike) -> Expr:
    """The arguments of several ``Abs`` of a product joined under one:
    ``Abs(x + 1)/Abs(x**2 - 1)`` is ``Abs((x + 1)/(x**2 - 1))``."""
    from sympy_extras.core.expr import factors_of
    from sympy_extras.core.power import Pow
    from sympy_extras.functions.elementary.complexes import Abs
    from sympy_extras.functions import Abs as abs_

    def _abs(b: Basic) -> Basic:
        e = _expr(b)
        factors = list(factors_of(e)) if isinstance(e, Mul) else [e]
        c = [f for f in factors if f.is_commutative is not False]
        nc = [f for f in factors if f.is_commutative is False]
        a: list[Expr] = []
        o: list[Expr] = []
        for i in c:
            if isinstance(i, Abs):
                a.append(_expr(i.args[0]))
            elif isinstance(i, Pow) and isinstance(i.base, Abs) and i.exp.is_real is True:
                a.append(pow(_expr(i.base.args[0]), i.exp))
            else:
                o.append(i)
        if len(a) < 2 and not any(isinstance(i, Pow) and i.exp.is_negative is True for i in a):
            return e
        absarg = mul(*a)
        A = abs_(absarg)
        if not A.has(Abs):
            return mul(A, *o, *nc)
        if not isinstance(A, Abs):
            A = Abs(absarg)
        return mul(A, *o, *nc)

    e = _expr(to_expr(expr).replace(lambda x: isinstance(x, Mul), _abs))
    return _expr(e.replace(lambda x: isinstance(x, Pow), _abs))


def _inverse_table() -> dict[type, type]:
    """The function classes with their inverse of the first argument
    (SymPy's ``inverse(argindex=1)``)."""
    from sympy_extras.functions.elementary import hyperbolic as h, trigonometric as t
    from sympy_extras.functions.elementary.exponential import Exp, Log
    from sympy_extras.functions.special import error_functions as er
    pairs: list[tuple[type, type]] = [
        (Exp, Log), (Log, Exp), (t.Tan, t.Atan), (t.Cot, t.Acot), (t.Asin, t.Sin), (t.Acos, t.Cos), (t.Atan, t.Tan),
        (t.Acot, t.Cot), (t.Asec, t.Sec), (t.Acsc, t.Csc), (h.Sinh, h.Asinh), (h.Tanh, h.Atanh), (h.Coth, h.Acoth),
        (h.Asinh, h.Sinh), (h.Acosh, h.Cosh), (h.Atanh, h.Tanh), (h.Acoth, h.Coth), (h.Asech, h.Sech),
        (h.Acsch, h.Csch), (er.Erf, er.Erfinv), (er.Erfc, er.Erfcinv), (er.Erfinv, er.Erf), (er.Erfcinv, er.Erfc)]
    return dict(pairs)


# PORTED FROM SYMPY: sympy/simplify/simplify.py:inversecombine
def inversecombine(expr: ExprLike) -> Expr:
    """Functions of their inverses cancelled: ``log(exp(x))`` and
    ``exp(log(x))`` are ``x``, ``atan(tan(x))`` is ``x``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp, log, atan, tan
    >>> x = Symbol('x')
    >>> inversecombine(log(exp(x)) + exp(log(x)) + atan(tan(x)))
    3*x
    """
    from sympy_extras.core.numbers import E
    from sympy_extras.core.power import Pow
    from sympy_extras.functions.elementary.exponential import Exp, Log
    inverse = _inverse_table()

    def exponent(e: Basic) -> Optional[Expr]:
        if isinstance(e, Exp):
            return _expr(e.args[0])
        if isinstance(e, Pow) and e.base == E:
            return e.exp
        return None

    def f(b: Basic) -> Basic:
        rv = b
        if isinstance(rv, Log):
            ex = exponent(rv.args[0])
            if ex is not None:
                rv = ex
        elif type(rv) in inverse:
            if len(rv.args) == 1 and len(rv.args[0].args) == 1 and isinstance(rv.args[0], inverse[type(rv)]):
                rv = rv.args[0].args[0]
        if isinstance(rv, Pow) and rv.base == E and isinstance(rv.exp, Log):
            rv = rv.exp.args[0]
        elif isinstance(rv, Exp) and isinstance(rv.args[0], Log):
            rv = rv.args[0].args[0]
        return rv

    return _expr(bottom_up(to_expr(expr), f))


# PORTED FROM SYMPY: sympy/simplify/simplify.py:kroneckersimp
def kroneckersimp(expr: ExprLike) -> Expr:
    """Products of Kronecker deltas which cannot both be 1 set to 0.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import KroneckerDelta
    >>> i = Symbol('i')
    >>> kroneckersimp(1 + KroneckerDelta(0, i)*KroneckerDelta(1, i))
    1
    """
    from itertools import combinations
    from sympy_extras.core.logic import false, true
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.elementary.piecewise import Piecewise
    from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode

    def args_cancel(args1: tuple[Basic, ...], args2: tuple[Basic, ...]) -> bool:
        for i1 in range(2):
            for i2 in range(2):
                a1, a2 = _expr(args1[i1]), _expr(args2[i2])
                a3, a4 = _expr(args1[(i1 + 1) % 2]), _expr(args2[(i2 + 1) % 2])
                if eq(a1, a2) == true and eq(a3, a4) == false:
                    return True
        return False

    def cancel_kronecker_mul(b: Basic) -> Basic:
        deltas = [a for a in b.args if isinstance(a, KroneckerDeltaNode)]
        for delta1, delta2 in combinations(deltas, 2):
            if args_cancel(delta1.args, delta2.args):
                return mul(Integer(0), _expr(b))
        return b

    e = to_expr(expr)
    if not e.has(KroneckerDeltaNode):
        return e
    if e.has(Piecewise):
        e = _expr(e.rewrite(KroneckerDeltaNode))
    previous: Optional[Expr] = None
    while e != previous:
        previous = e
        e = _expr(e.replace(lambda x: isinstance(x, Mul), cancel_kronecker_mul))
    return e


def _has_variety(choices: list[Expr]) -> bool:
    return any(c != choices[0] for c in choices[1:])


def _through_sympy(e: Expr, step: Callable[['sympy.Basic'], 'sympy.Basic']) -> Expr:
    from sympy_extras.core.interop import expr_through_sympy
    return expr_through_sympy(e, step)


def _sympy_only() -> tuple[type, ...]:
    """The classes whose simplification is SymPy's (not ported)."""
    from sympy_extras.core.interop import SymPyExprNode
    return (SymPyExprNode,)


def _combinatorial() -> tuple[type, ...]:
    from sympy_extras.functions.combinatorial.factorials import (Binomial, Factorial, Factorial2, FallingFactorial,
                                                                  RisingFactorial, Subfactorial)
    return (Factorial, Factorial2, Subfactorial, RisingFactorial, FallingFactorial, Binomial)


def _sympy_simplify(e: Expr, ratio: float, rational: Optional[bool], inverse: bool, doit: bool) -> Expr:
    return _through_sympy(e, lambda s: sympy.simplify(s, ratio=ratio, rational=rational, inverse=inverse,
                                                       doit=doit))


def _eval_simplify(e: Expr, ratio: float, measure: Measure, rational: Optional[bool], inverse: bool,
                   doit: bool) -> Optional[Expr]:
    """The ``_eval_simplify`` of the node, ``None`` when it has none."""
    from sympy_extras.functions import log, sign
    from sympy_extras.functions.elementary.complexes import Sign
    from sympy_extras.functions.elementary.exponential import Log
    from sympy_extras.core.exprtools import factor_terms
    from sympy_extras.core.function import expand_log
    if isinstance(e, _sympy_only()):
        return _sympy_simplify(e, ratio, rational, inverse, doit)
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_simplify
    from sympy_extras.functions.special.bessel import BesselBase
    from sympy_extras.functions.special.hyper import Hyper
    if isinstance(e, Hyper):
        # PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper._eval_simplify
        from .hyperexpand import hyperexpand
        return hyperexpand(e)
    if isinstance(e, BesselBase):
        # PORTED FROM SYMPY: sympy/functions/special/bessel.py:BesselBase._eval_simplify
        return besselsimp(e)
    if isinstance(e, Piecewise):
        # PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise._eval_simplify
        return piecewise_simplify(e, doit=doit)
    if isinstance(e, Log):
        # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log._eval_simplify
        new = log(simplify(_expr(e.args[0]), ratio, measure, rational, inverse, doit))
        if inverse:
            new = inversecombine(new)
        new = expand_log(new, deep=True)
        return min([new, e], key=measure)
    if isinstance(e, Sign):
        # PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:sign._eval_simplify
        return sign(factor_terms(_expr(e.args[0])))
    if isinstance(e, _combinatorial()):
        # PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:CombinatorialFunction._eval_simplify
        from .gammasimp import combsimp
        new = combsimp(e)
        return new if measure(new) <= ratio * measure(e) else e
    return None


# PORTED FROM SYMPY: sympy/simplify/simplify.py:simplify
def simplify(expr: ExprLike, ratio: float = 1.7, measure: Measure = _ops, rational: Optional[bool] = False,
             inverse: bool = False, doit: bool = True) -> Expr:
    """The expression simplified: the shortest of SymPy's heuristics by
    ``measure``, the input when every result is more than ``ratio`` times
    longer."""
    from sympy_extras.core.exprtools import factor_terms
    from sympy_extras.core.function import Application, expand_log, expand_mul, expand_power_exp, nfloat
    from sympy_extras.core.numbers import Float
    from sympy_extras.core.power import Pow
    from sympy_extras.functions.elementary.complexes import Abs, Sign
    from sympy_extras.functions.elementary.exponential import Exp, Log
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_fold
    from sympy_extras.functions.special.gamma_functions import Gamma
    from sympy_extras.functions.special.hyper import Hyper, MeijerG
    from sympy_extras.functions.special.bessel import BesselBase
    from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode
    from sympy_extras.functions.elementary.complexes import PolarLift
    from sympy_extras.functions.elementary.exponential import ExpPolar
    from sympy_extras.core.interop import SymPyExprNode
    from sympy_extras.polys.polytools import cancel
    from sympy_extras.polys.rationaltools import together
    from .fu import HYPER, TRIG
    from .gammasimp import combsimp
    from .powsimp import powsimp
    from .radsimp import fraction, radsimp
    from .signsimp import signsimp
    from .sqrtdenest import _mexpand
    from .trigsimp import exptrigsimp, trigsimp
    from sympy.simplify.simplify import product_simplify, sum_simplify

    def shorter(*choices: Expr) -> Expr:
        options = list(choices)
        if not _has_variety(options):
            return options[0]
        return min(options, key=measure)

    def done(e: Expr) -> Expr:
        rv = e.doit() if doit else e
        return shorter(rv, collect_abs(rv))

    def again(e: Expr) -> Expr:
        return simplify(e, ratio, measure, rational, inverse, doit)

    e = to_expr(expr)
    if e.is_zero is True:
        return e if isinstance(e, Number) else Integer(0)
    hook = _eval_simplify(e, ratio, measure, rational, inverse, doit)
    if hook is not None:
        return hook
    original = e = collect_abs(signsimp(e))
    if not e.args:
        return e
    if inverse and e.has(Application):
        e = inversecombine(e)
        if not e.args:
            return e

    handled = (Add, Mul, Pow, Exp)

    def query(x: Basic) -> bool:
        return isinstance(x, Expr) and bool(x.args) and not isinstance(x, handled)

    def simplified(a: Basic) -> Basic:
        # SymPy simplifies a non-expression node argument by argument: the
        # expression and the condition of a piece
        from sympy_extras.functions.elementary.piecewise import ExprCondPair, _simplify_condition
        if isinstance(a, Expr):
            return again(a)
        if isinstance(a, ExprCondPair):
            return ExprCondPair(again(a.expr), _simplify_condition(a.cond))
        return a.simplify()

    def value(x: Basic) -> Basic:
        return x._evaluated(tuple(simplified(a) for a in x.args))

    e = _expr(e.replace(query, value))
    if not isinstance(e, handled):
        return done(e)
    if e.is_commutative is False:
        e = nc_simplify(e)
    floats = False
    if rational is not False and e.has(Float):
        floats = True
        e = nsimplify(e, rational=True)
    e = _expr(bottom_up(e, lambda w: w.normal() if isinstance(w, Expr) else w))
    c, p = powsimp(e).as_content_primitive()
    e = mul(c, p)
    _e = cancel(e)
    expr1 = shorter(_e, cancel(_mexpand(_e)))
    expr2 = shorter(together(e, deep=True), together(expr1, deep=True))
    e = expr2 if ratio == math.inf else shorter(expr2, expr1, e)
    e = factor_terms(e, sign=False)
    if e.has(Sign):
        e = _expr(e.rewrite(Abs))
    if e.has(Piecewise):
        e = done(piecewise_fold(e))
        if e.has(Piecewise):
            e = piecewise_fold(e)
            if e.has(KroneckerDeltaNode):
                e = kroneckersimp(e)
            if e.has(Piecewise):
                # SymPy's piecewise_simplify: the folded expression is a Piecewise
                from sympy_extras.functions.elementary.piecewise import piecewise_simplify
                if isinstance(e, Piecewise):
                    e = piecewise_simplify(e, doit=False)
                if e.has(Piecewise):
                    return shorter(e, factor_terms(e))
    if e.has(Hyper, MeijerG, ExpPolar, PolarLift):
        from .hyperexpand import hyperexpand
        e = hyperexpand(e)
    if e.has(KroneckerDeltaNode):
        e = kroneckersimp(e)
    if e.has(BesselBase):
        e = besselsimp(e)
    if e.has(*TRIG, *HYPER):
        e = trigsimp(e)
    if e.has(Log):
        e = shorter(expand_log(e, deep=True), logcombine(e))
    if e.has(*_combinatorial(), Gamma):
        e = combsimp(e)
    if e.has(SymPyExprNode):
        def concrete(s: sympy.Basic) -> sympy.Basic:
            if s.has(sympy.Sum):
                s = sum_simplify(s, ratio=ratio, measure=sympy.count_ops, rational=rational, inverse=inverse,
                                 doit=doit)
            if s.has(sympy.Integral):
                s = s.xreplace({i: sympy.factor_terms(i) for i in s.atoms(sympy.Integral)})
            if s.has(sympy.Product):
                s = product_simplify(s, ratio=ratio, measure=sympy.count_ops, rational=rational, inverse=inverse,
                                     doit=doit)
            return s
        e = _through_sympy(e, concrete)

    short = shorter(powsimp(e, combine='exp', deep=True), powsimp(e), e)
    short = shorter(short, cancel(short))
    short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
    if short.has(*TRIG, *HYPER, Exp):
        short = exptrigsimp(short)

    def hollow(x: Basic) -> bool:
        return (isinstance(x, Mul) and len(x.args) == 2 and isinstance(x.args[0], Number)
                and isinstance(x.args[1], Add) and x.is_commutative is not False)
    e = _expr(short.replace(hollow, lambda x: mul(*[_expr(a) for a in x.args])))

    numer, denom = e.as_numer_denom()
    if isinstance(denom, Add):
        n, d = fraction(radsimp(pow(denom, Integer(-1)), symbolic=False, max_terms=1))
        if not (isinstance(n, Integer) and n.p == 1):
            from sympy_extras.core.function import expand
            e = mul(expand(mul(numer, n)), pow(d, Integer(-1)))
    if e.could_extract_minus_sign():
        n, d = fraction(e)
        if not (isinstance(d, Integer) and d.p == 0):
            e = signsimp(mul(mul(Integer(-1), n), pow(mul(Integer(-1), d), Integer(-1))))
    if measure(e) > ratio * measure(original):
        e = original
    if floats and rational is None:
        e = _expr(nfloat(e, exponent=False))
    return done(e)


# PORTED FROM SYMPY: sympy/simplify/simplify.py:nc_simplify
def nc_simplify(expr: ExprLike, deep: bool = True) -> Expr:
    """A noncommutative product with its repeated subwords grouped into
    powers, the grouping which leaves the fewest arguments first
    (``a*b*a*b*c*a*b*c`` is ``a*b*(a*b*c)**2``); a sum term by term; the
    nested subexpressions too with ``deep``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.simplify.simplify import nc_simplify
    >>> a, b, c = symbols('a b c', commutative=False)
    >>> nc_simplify(a*b*a*b*c*a*b*c), nc_simplify(a**2*b*a**4*b*a**4)
    (a*b*(a*b*c)**2, a**2*(b*a**4)**2)
    >>> nc_simplify(a*b*a*b + 2*a*c*a**2*c*a**2*c*a), nc_simplify(b**-1*a**-1*(a*b)**2)
    ((a*b)**2 + 2*(a*c*a)**3, a*b)
    >>> nc_simplify(a**-1*b**-1*c*a), nc_simplify((a*b*a*b)**2*a*c*a*c, deep=False)
    ((b*a)**(-1)*c*a, (a*b*a*b)**2*(a*c)**2)
    """
    from sympy_extras.core.exprtools import factor_nc
    from sympy_extras.core.power import Pow
    e = to_expr(expr)

    def _overlaps(args: tuple[Expr, ...]) -> list[list[list[int]]]:
        # m[i][j]: the lengths of the overlaps of args[:i+1] with args[i+1+j:]
        # (the suffixes of the prefix which are prefixes of the suffix)
        m = [[([1, 0] if a == args[0] else [0]) for a in args[1:]]]
        for i in range(1, len(args)):
            overlaps: list[list[int]] = []
            for j in range(len(args) - i - 1):
                overlap: list[int] = []
                for v in m[i-1][j+1]:
                    if j + i + 1 + v < len(args) and args[i] == args[j+i+1+v]:
                        overlap.append(v + 1)
                overlap += [0]
                overlaps.append(overlap)
            m.append(overlaps)
        return m

    def _reduce_inverses(_args: tuple[Expr, ...]) -> tuple[int, tuple[Expr, ...]]:
        # consecutive negative powers made the inverse of a product of
        # positive powers (a**-1*b**-1*c is (b*a)**-1*c)
        inv_tot = 0
        inverses: list[Expr] = []
        out: list[Expr] = []
        for arg in _args:
            if isinstance(arg, Pow) and arg.exp.is_extended_negative is True:
                inverses = [pow(arg, Integer(-1))] + inverses
                inv_tot += 1
            else:
                if len(inverses) == 1:
                    out.append(pow(inverses[0], Integer(-1)))
                elif len(inverses) > 1:
                    out.append(pow(mul(*inverses), Integer(-1)))
                    inv_tot -= len(inverses) - 1
                inverses = []
                out.append(arg)
        if inverses:
            out.append(pow(mul(*inverses), Integer(-1)))
            inv_tot -= len(inverses) - 1
        return inv_tot, tuple(out)

    def get_score(s: Expr) -> int:
        # the number of arguments, nested ones included, the exponents not
        if isinstance(s, Pow):
            return get_score(s.base)
        if isinstance(s, (Add, Mul)):
            return sum(get_score(a) for a in s.args if isinstance(a, Expr))
        return 1

    def compare(s: Expr, alt_s: Expr) -> Expr:
        if s != alt_s and get_score(alt_s) < get_score(s):
            return alt_s
        return s

    def base_args(t: Expr) -> tuple[Expr, ...]:
        return tuple(a for a in t.args if isinstance(a, Expr))

    if not isinstance(e, (Add, Mul, Pow)) or e.is_commutative is not False:
        return e
    if isinstance(e, Pow):
        if deep:
            return pow(nc_simplify(e.base), e.exp)
        return e
    if isinstance(e, Add):
        return add(*[nc_simplify(a, deep=deep) for a in e.args if isinstance(a, Expr)])
    # the commutative part (SymPy's args_cnc: the commutative prefix)
    factors = [a for a in e.args if isinstance(a, Expr)]
    split = next((i for i, f in enumerate(factors) if f.is_commutative is not True), len(factors))
    c_args, nc_args = factors[:split], factors[split:]
    com_coeff = mul(*c_args)
    if not (isinstance(com_coeff, Integer) and com_coeff.p == 1):
        return com_coeff * nc_simplify(e / com_coeff, deep=deep)
    inv_tot, args = _reduce_inverses(tuple(nc_args))
    # with mostly negative powers the inverse is simplified
    # (a**-1*b*a**-1*c**-1 is (c*a*b**-1*a)**-1)
    invert = False
    if inv_tot > len(args) / 2:
        invert = True
        args = tuple(pow(a, Integer(-1)) for a in args[::-1])
    if deep:
        args = tuple(nc_simplify(a) for a in args)
    m = _overlaps(args)
    # {subterm: end}: the end of a sequence of repetitions of the subterm
    simps: dict[tuple[Expr, ...], int] = {}
    post: Expr = Integer(1)
    pre: Expr = Integer(1)
    # the number of arguments a contraction removes (a*b*a*b to (a*b)**2
    # removes 3): the largest one is chosen
    max_simp_coeff = 0
    simp_found: Optional[tuple[int, Expr, Expr, int, int]] = None
    rep: list[int] = [0]
    for i in range(1, len(args)):
        simp_coeff = 0
        l = 0
        p: Expr = Integer(0)
        if i < len(args) - 1:
            rep = m[i][0]
        start = i
        end = i + 1
        subterm: tuple[Expr, ...]
        if i == len(args) - 1 or rep == [0]:
            # no repeated subterm in the arguments, maybe with the powers
            arg_i = args[i]
            if isinstance(arg_i, Pow) and not isinstance(arg_i.base, Symbol):
                subterm = base_args(arg_i.base)
                l = len(subterm)
                if args[i-l:i] == subterm:
                    # a*b in a*b*(a*b)**2
                    p = p + 1
                    start -= l
                if args[i+1:i+1+l] == subterm:
                    # a*b in (a*b)**2*a*b
                    p = p + 1
                    end += l
            if not (isinstance(p, Integer) and p.p == 0) and isinstance(arg_i, Pow):
                p = p + arg_i.exp
            else:
                continue
        else:
            l = rep[0]  # the longest repeated subterm here
            start -= l - 1
            subterm = args[start:end]
            p = Integer(2)
            end += l
        if subterm in simps and simps[subterm] >= start:
            # part of a sequence already considered
            continue
        # the number of repetitions
        while end < len(args):
            if l in m[end-1][0]:
                p = p + 1
                end += l
            else:
                at = args[end]
                if isinstance(at, Pow) and base_args(at.base) == subterm:
                    # a*b*a*b*(a*b)**2*a*b
                    p = p + at.exp
                    end += 1
                else:
                    break
        # another match: b*a**2 in b*a**2*b*a**3, a*b in a**2*b*a*b
        pre_exp: Expr = Integer(0)
        pre_arg: Expr = Integer(1)
        exp: Expr = Integer(1)
        if start - l >= 0 and args[start-l+1:start] == subterm[1:]:
            first = subterm[0]
            if isinstance(first, Pow):
                pre_arg, exp = first.base, first.exp
            else:
                pre_arg, exp = first, Integer(1)
            before = args[start-l]
            if isinstance(before, Pow) and before.base == pre_arg:
                pre_exp = before.exp - exp
                start -= l
                p = p + 1
            elif before == pre_arg:
                pre_exp = 1 - exp
                start -= l
                p = p + 1
        post_exp: Expr = Integer(0)
        post_arg: Expr = Integer(1)
        if end + l - 1 < len(args) and args[end:end+l-1] == subterm[:-1]:
            last = subterm[-1]
            if isinstance(last, Pow):
                post_arg, exp = last.base, last.exp
            else:
                post_arg, exp = last, Integer(1)
            after = args[end+l-1]
            if isinstance(after, Pow) and after.base == post_arg:
                post_exp = after.exp - exp
                end += l
                p = p + 1
            elif after == post_arg:
                post_exp = 1 - exp
                end += l
                p = p + 1
        # in a*b*a**2*b*a**2*b*a both b*a**2 and a*b*a repeat: (a*b*a)**3
        # is simpler than a*(b*a**2)**3*a**-1
        if not _is_zero_number(post_exp) and _is_even_number(exp) and start > 0:
            exp = exp / 2
            _pre_exp: Expr = Integer(1)
            _post_exp: Expr = Integer(1)
            before = args[start-1]
            if isinstance(before, Pow) and before.base == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = before.exp - exp
            elif before == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = 1 - exp
            if _is_zero_number(_pre_exp) or _is_zero_number(_post_exp):
                if _is_zero_number(pre_exp):
                    start -= 1
                post_exp = _post_exp
                pre_exp = _pre_exp
                pre_arg = post_arg
                subterm = (pow(post_arg, exp),) + subterm[:-1] + (pow(post_arg, exp),)
        simp_coeff += end - start
        if not _is_zero_number(post_exp):
            simp_coeff -= 1
        if not _is_zero_number(pre_exp):
            simp_coeff -= 1
        simps[subterm] = end
        if simp_coeff > max_simp_coeff:
            max_simp_coeff = simp_coeff
            simp_found = (start, mul(*subterm), p, end, l)
            pre = pow(pre_arg, pre_exp)
            post = pow(post_arg, post_exp)
    result: Expr
    if simp_found is not None:
        s_start, s_term, s_power, s_end, _ = simp_found
        sub = pow(nc_simplify(s_term, deep=deep), s_power)
        pre = nc_simplify(mul(*args[:s_start]) * pre, deep=deep)
        post = post * nc_simplify(mul(*args[s_end:]), deep=deep)
        result = pre * sub * post
        if pre != Integer(1) or post != Integer(1):
            # maybe new simplifications, without the arguments again
            result = nc_simplify(result, deep=False)
    else:
        result = mul(*args)
    if invert:
        result = pow(result, Integer(-1))
    # factor_nc may give a better simplification
    f_expr = factor_nc(e)
    if f_expr != e:
        alt = nc_simplify(f_expr, deep=deep)
        result = compare(result, alt)
    return result


def _is_even_number(e: Expr) -> bool:
    return isinstance(e, Integer) and e.p % 2 == 0


# PORTED FROM SYMPY: sympy/simplify/simplify.py:besselsimp
def besselsimp(expr: ExprLike) -> Expr:
    """Bessel functions simplified: the factors ``I`` and ``-1`` of the
    arguments taken out (``besselj(nu, I*z)`` is ``I**nu*besseli(nu, z)``),
    the half-integer orders written with trigonometric functions, the
    integer orders expanded, and the recurrence
    ``B(a, x) = 2*(a + 1)/x*B(a + 1, x) - B(a + 2, x)`` used to remove the
    lowest orders; the result factored when it changed.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, I, pi, S
    >>> from sympy_extras.functions import besselj, besseli
    >>> from sympy_extras.simplify.simplify import besselsimp
    >>> z, nu = Symbol('z'), Symbol('nu')
    >>> besselsimp(besselj(nu, z*I)), besselsimp(besseli(nu, z*(-1)))
    (besselj(nu, I*z), (-z)**nu*besseli(nu, z)/z**nu)
    >>> besselsimp(besselj(S(1)/2, z))
    sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    """
    from sympy_extras.core.numbers import I, Integer, pi
    from sympy_extras.core.mod import mod
    from sympy_extras.functions.elementary.complexes import re, unpolarify
    from sympy_extras.functions.elementary.exponential import exp, exp_polar
    from sympy_extras.functions.special.bessel import (BesselI, BesselJ, BesselK, BesselY, SphericalJ, besseli,
                                                        besselj, besselk, bessely)
    from .trigsimp import trigsimp
    e = to_expr(expr)
    BesselFunction = Callable[[Expr, Expr], Expr]

    def replacer(fro: BesselFunction, to: BesselFunction, factors: list[Expr]) -> BesselFunction:
        found = set(factors)

        def repl(nu: Expr, z: Expr) -> Expr:
            zfactors = set(z.args) if isinstance(z, Mul) else {z}
            if found.intersection(zfactors):
                return to(nu, z)
            return fro(nu, z)
        return repl

    def torewrite(fro: BesselFunction, to: type) -> BesselFunction:
        def tofunc(nu: Expr, z: Expr) -> Expr:
            return fro(nu, z).rewrite(to)
        return tofunc

    def tominus(fro: BesselFunction) -> BesselFunction:
        def tofunc(nu: Expr, z: Expr) -> Expr:
            return exp(I*pi*nu) * fro(nu, exp_polar(-I*pi) * z)
        return tofunc

    def replace(t: Expr, cls: type, f: BesselFunction) -> Expr:
        def value(*args: Expr) -> Expr:
            return f(args[0], args[1])
        return _expr(t.replace(cls, value))

    orig = e
    ifactors = [I, exp_polar(I*pi/2), exp_polar(-I*pi/2)]
    e = replace(e, BesselJ, replacer(besselj, torewrite(besselj, BesselI), ifactors))
    e = replace(e, BesselI, replacer(besseli, torewrite(besseli, BesselJ), ifactors))
    minusfactors = [Integer(-1), exp_polar(I*pi)]
    e = replace(e, BesselJ, replacer(besselj, tominus(besselj), minusfactors))
    e = replace(e, BesselI, replacer(besseli, tominus(besseli), minusfactors))
    z0 = Dummy('z')

    def expander(fro: BesselFunction) -> BesselFunction:
        def repl(nu: Expr, z: Expr) -> Expr:
            if mod(nu, Integer(1)) == Rational(1, 2):
                rewritten = fro(nu, z0).rewrite(BesselJ).rewrite(SphericalJ).expand(func=True)
                return simplify(trigsimp(unpolarify(rewritten))).subs(z0, z)
            if isinstance(nu, Integer) and nu.p > 1:
                return fro(nu, z).expand(func=True)
            return fro(nu, z)
        return repl

    e = replace(e, BesselJ, expander(besselj))
    e = replace(e, BesselY, expander(bessely))
    e = replace(e, BesselI, expander(besseli))
    e = replace(e, BesselK, expander(besselk))

    def use_recursion(cls: type, make: BesselFunction, t: Expr) -> Expr:
        while True:
            found = [b for b in t.atoms(cls) if isinstance(b, Expr)]
            keys = [re(_expr(b.args[0])) for b in found]
            if len(found) > 1 and not all(k.is_comparable for k in keys):
                return t  # SymPy's sorted raises TypeError: no ordering
            ordered_found = found if len(found) <= 1 else \
                [b for _, b in sorted(zip(keys, found), key=lambda kb: float(kb[0]))]
            for ba in ordered_found:
                a, x = _expr(ba.args[0]), _expr(ba.args[1])
                bap1, bap2 = make(a + 1, x), make(a + 2, x)
                if t.has(bap1) and t.has(bap2):
                    t = t.subs(ba, 2*(a + 1)/x*bap1 - bap2)
                    break
            else:
                return t

    if e.has(BesselJ):
        e = use_recursion(BesselJ, besselj, e)
    if e.has(BesselY):
        e = use_recursion(BesselY, bessely, e)
    if e != orig:
        e = e.factor()
    return e
