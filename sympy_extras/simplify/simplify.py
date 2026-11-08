"""``logcombine`` and ``posify`` of SymPy's ``simplify`` module.

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
# PORTED FROM SYMPY: sympy/simplify/simplify.py (logcombine, posify, nsimplify,
# _real_to_rational) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; posify of a single expression (map it over a list); the
# formulas of mpmath.identify read by a small typed parser (no parse_expr).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Number, Rational
from sympy_extras.core.power import pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.core.traversal import bottom_up

__all__ = ['logcombine', 'posify', 'nsimplify']


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
