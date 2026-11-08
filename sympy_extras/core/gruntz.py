"""Limits by Gruntz's algorithm: ``gruntz``, ``limitinf``, ``mrv``.

The limit of ``e`` at infinity is found from its most rapidly varying
subexpressions (the ``mrv`` set, exponentials compared by the limits of
the quotients of their logarithms): they are rewritten as powers of a
single one ``w`` going to zero, and the leading term ``c0*w**e0`` of the
series in ``w`` decides (``e0 > 0``: the limit is 0, ``e0 < 0``: infinite
with the sign of ``c0``, ``e0 = 0``: the limit of ``c0``, recursively).
The functions are first rewritten in their tractable form (exponentials
and logarithms).

References
==========

.. [1] D. Gruntz, *On Computing Limits in a Symbolic Manipulation
       System*, PhD thesis, ETH Zurich, 1996.
.. [2] SymPy Development Team, *SymPy documentation: Gruntz algorithm*,
       https://docs.sympy.org/latest/modules/series/series.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/gruntz.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; SubsSet is a class with an ordered mapping rather than
# a dict subclass; sign returns an expression.
# ---------------------------------------------------------------------
from __future__ import annotations

from functools import reduce
from typing import Optional

from .add import Add
from .basic import Basic
from .cache import cacheit
from .coreerrors import PoleError
from .expr import Expr, factors_of
from .function import AppliedUndef, Application, Derivative
from .intfunc import ilcm
from .mul import Mul
from .numbers import I, Integer, Rational, oo
from .power import Pow
from .symbol import Dummy, Symbol, Wild

__all__ = ['gruntz', 'limitinf', 'mrv', 'mrv_leadterm', 'sign', 'compare', 'SubsSet']


def _exp(e: Expr) -> Expr:
    from sympy_extras.functions.elementary.exponential import exp
    return exp(e)


def _log(e: Expr) -> Expr:
    from sympy_extras.functions.elementary.exponential import log
    return log(e)


def _is_exp(e: Basic) -> bool:
    from sympy_extras.functions.elementary.exponential import Exp
    from .numbers import Exp1
    return isinstance(e, Exp) or (isinstance(e, Pow) and isinstance(e.base, Exp1))


def _exponent(e: Expr) -> Expr:
    """The exponent of ``exp(a)`` (or of ``E**a``)."""
    from sympy_extras.functions.elementary.exponential import Exp
    if isinstance(e, Exp):
        return e.args[0]
    if isinstance(e, Pow):
        return e.exp
    raise TypeError("an exponential is expected, got %s" % (e,))


def _is_number(e: Expr, value: int) -> bool:
    return e.is_Number and e == value


# PORTED FROM SYMPY: sympy/series/gruntz.py:compare
def compare(a: Expr, b: Expr, x: Symbol) -> str:
    """``'<'``, ``'='`` or ``'>'``: the growth of ``a`` against ``b`` (by
    the limit of ``log(a)/log(b)``)."""
    la = _exponent(a) if _is_exp(a) else _log(a)
    lb = _exponent(b) if _is_exp(b) else _log(b)
    c = limitinf(la / lb, x)
    if _is_number(c, 0):
        return "<"
    if c.is_infinite is True:
        return ">"
    return "="


# PORTED FROM SYMPY: sympy/series/gruntz.py:SubsSet
class SubsSet:
    """The mrv set: the most rapidly varying subexpressions, each with the
    dummy replacing it (in insertion order), and the rewrites of the
    dummies of exponentials (``rewrites``)."""

    __slots__ = ('_items', 'rewrites')

    def __init__(self) -> None:
        self._items: dict[Expr, Dummy] = {}
        self.rewrites: dict[Dummy, Expr] = {}

    def __repr__(self) -> str:
        return repr(self._items) + ', ' + repr(self.rewrites)

    def get(self, key: Expr) -> Dummy:
        """The dummy of ``key``, created when missing (SymPy's
        ``__getitem__``)."""
        if key not in self._items:
            self._items[key] = Dummy()
        return self._items[key]

    def set(self, key: Expr, var: Dummy) -> None:
        self._items[key] = var

    def items(self) -> list[tuple[Expr, Dummy]]:
        return list(self._items.items())

    def keys(self) -> list[Expr]:
        return list(self._items)

    def __contains__(self, key: Expr) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def do_subs(self, e: Expr) -> Expr:
        """``e`` with the dummies replaced by their expressions."""
        for expr, var in self._items.items():
            rule: dict[Basic, Basic] = {var: expr}
            e = e.xreplace(rule)
        return e

    def meets(self, s2: SubsSet) -> bool:
        """Whether the two sets have a common expression."""
        return bool(set(self._items).intersection(s2._items))

    def union(self, s2: SubsSet, exps: Optional[Expr] = None) -> tuple[SubsSet, Optional[Expr]]:
        """The union, with the dummies of ``s2`` for the common expressions
        replaced in ``exps`` and in the rewrites."""
        res = self.copy()
        tr: dict[Basic, Basic] = {}
        for expr, var in s2._items.items():
            if expr in self._items:
                if exps is not None:
                    rule: dict[Basic, Basic] = {var: res.get(expr)}
                    exps = exps.xreplace(rule)
                tr[var] = res.get(expr)
            else:
                res._items[expr] = var
        for var, rewr in s2.rewrites.items():
            res.rewrites[var] = rewr.xreplace(tr)
        return res, exps

    def copy(self) -> SubsSet:
        r = SubsSet()
        r.rewrites = dict(self.rewrites)
        for expr, var in self._items.items():
            r._items[expr] = var
        return r


# PORTED FROM SYMPY: sympy/series/gruntz.py:mrv
def mrv(e: Expr, x: Symbol) -> tuple[SubsSet, Expr]:
    """The mrv set of ``e`` and ``e`` with its elements replaced by their
    dummies."""
    from sympy_extras.functions.elementary.exponential import Log, exp
    from sympy_extras.simplify.powsimp import powsimp
    e = powsimp(e, deep=True, combine='exp')
    if not e.has(x):
        return SubsSet(), e
    if e == x:
        s = SubsSet()
        return s, s.get(x)
    if isinstance(e, (Mul, Add)):
        i, d = e.as_independent(x)  # the terms free of x thrown away
        if type(d) is not type(e):
            s, expr = mrv(d, x)
            return s, _rebuild_assoc(e, [i, expr])
        a, b = d.as_two_terms()
        s1, e1 = mrv(a, x)
        s2, e2 = mrv(b, x)
        return mrv_max1(s1, s2, _rebuild_assoc(e, [i, e1, e2]), x)
    if isinstance(e, Pow) and not _is_exp(e):
        ep: Expr = Integer(1)
        b1: Expr = e
        while isinstance(e, Pow):
            b1 = e.base
            ep = ep * e.exp
            e = b1
        if _is_number(b1, 1):
            return SubsSet(), b1
        if ep.has(x):
            return mrv(exp(ep * _log(b1)), x)
        s, expr = mrv(b1, x)
        return s, expr**ep
    if isinstance(e, Log):
        s, expr = mrv(e.args[0], x)
        return s, _log(expr)
    if _is_exp(e):
        exponent = _exponent(e)
        if isinstance(exponent, Log):
            return mrv(exponent.args[0], x)
        li = limitinf(exponent, x)
        if any(f.is_infinite is True for f in factors_of(li)):
            s1 = SubsSet()
            e1v = s1.get(e)
            s2, e2 = mrv(exponent, x)
            su = s1.union(s2)[0]
            su.rewrites[e1v] = exp(e2)
            return mrv_max3(s1, e1v, s2, exp(e2), su, e1v, x)
        s, expr = mrv(exponent, x)
        return s, exp(expr)
    if isinstance(e, AppliedUndef):
        raise ValueError("MRV set computation for UndefinedFunction is not allowed")
    if isinstance(e, Application):
        found = [mrv(a, x) for a in e.args]
        nonempty = [s for (s, _) in found if not s.is_empty()]
        if len(nonempty) != 1:
            raise NotImplementedError("MRV set computation for functions in several variables not implemented.")
        s, ss = nonempty[0], SubsSet()
        args = [ss.do_subs(r) for _, r in found]
        return s, e._rebuilt(args)
    if isinstance(e, Derivative):
        raise NotImplementedError("MRV set computation for derivatives not implemented yet.")
    raise NotImplementedError("Don't know how to calculate the mrv of '%s'" % (e,))


def _rebuild_assoc(e: Expr, args: list[Expr]) -> Expr:
    """``e.func(*args)`` (evaluated) for a sum or a product."""
    from .add import add
    from .mul import mul
    return add(*args) if isinstance(e, Add) else mul(*args)


# PORTED FROM SYMPY: sympy/series/gruntz.py:mrv_max3
def mrv_max3(f: SubsSet, expsf: Expr, g: SubsSet, expsg: Expr, union: SubsSet, expsboth: Expr,
             x: Symbol) -> tuple[SubsSet, Expr]:
    """The maximum of the mrv sets ``f`` and ``g`` (their union when they
    are comparable) with the corresponding expression."""
    if f.is_empty():
        return g, expsg
    if g.is_empty():
        return f, expsf
    if f.meets(g):
        return union, expsboth
    c = compare(f.keys()[0], g.keys()[0], x)
    if c == ">":
        return f, expsf
    if c == "<":
        return g, expsg
    if c != "=":
        raise ValueError("c should be =")
    return union, expsboth


# PORTED FROM SYMPY: sympy/series/gruntz.py:mrv_max1
def mrv_max1(f: SubsSet, g: SubsSet, exps: Expr, x: Symbol) -> tuple[SubsSet, Expr]:
    u, b = f.union(g, exps)
    if b is None:
        raise ValueError("the union lost the expression")
    return mrv_max3(f, g.do_subs(exps), g, f.do_subs(exps), u, b, x)


# PORTED FROM SYMPY: sympy/series/gruntz.py:sign
@cacheit
def sign(e: Expr, x: Symbol) -> Expr:
    """The sign of ``e`` for ``x`` large enough (``1``, ``-1``, ``0``, or
    the sign of a constant)."""
    from sympy_extras.functions.elementary.complexes import sign as _sign
    from sympy_extras.functions.elementary.exponential import Exp, Log
    if e.is_positive is True:
        return Integer(1)
    if e.is_negative is True:
        return Integer(-1)
    if e.is_zero is True:
        return Integer(0)
    if not e.has(x):
        from sympy_extras.simplify.simplify import logcombine
        return _sign(logcombine(e))
    if e == x:
        return Integer(1)
    if isinstance(e, Mul):
        a, b = e.as_two_terms()
        sa = sign(a, x)
        if _is_number(sa, 0):
            return Integer(0)
        return sa * sign(b, x)
    if isinstance(e, Exp):
        return Integer(1)
    if isinstance(e, Pow):
        if _is_exp(e):
            return Integer(1)
        s = sign(e.base, x)
        if _is_number(s, 1):
            return Integer(1)
        if isinstance(e.exp, Integer):
            return s**e.exp
    elif isinstance(e, Log) and e.args[0].is_positive is True:
        return sign(e.args[0] - 1, x)
    c0, _ = mrv_leadterm(e, x)
    return sign(c0, x)


# PORTED FROM SYMPY: sympy/series/gruntz.py:limitinf
@cacheit
def limitinf(e: Expr, x: Symbol) -> Expr:
    """The limit of ``e`` as ``x`` goes to ``oo``."""
    from .order import Order
    from .rewriting import rewrite_tractable
    from sympy_extras.simplify.powsimp import powdenest
    old = e
    if not e.has(x):
        return e  # a constant
    if e.has(Order):
        e = e.expand().removeO()
    if x.is_positive is not True or x.is_integer is True:
        # a positive variable makes the algorithm simpler
        p = Dummy('p', positive=True)
        e = e.subs(x, p)
        x = p
    rewritten = rewrite_tractable(e, x)
    if not isinstance(rewritten, Expr):
        raise TypeError("an expression is expected, got %s" % (rewritten,))
    e = powdenest(rewritten)
    from .accumbounds import AccumBounds
    if isinstance(e, AccumBounds):
        if mrv_leadterm(e.min, x) != mrv_leadterm(e.max, x):
            raise NotImplementedError
        c0, e0 = mrv_leadterm(e.min, x)
    else:
        c0, e0 = mrv_leadterm(e, x)
    sig = sign(e0, x)
    if _is_number(sig, 1):
        return Integer(0)  # e0 > 0: the limit is 0
    if _is_number(sig, -1):
        # e0 < 0: the limit is infinite, with the sign of c0
        a = Wild("a", exclude=[I])
        if c0.match(I * a) is not None:
            return c0 * oo
        s = sign(c0, x)
        if _is_number(s, 0):
            raise ValueError("Leading term should not be 0")
        return s * oo
    if _is_number(sig, 0):
        if c0 == old:
            c0 = c0.cancel()
        return limitinf(c0, x)  # e0 = 0: the limit of c0
    raise ValueError("{} could not be evaluated".format(sig))


# PORTED FROM SYMPY: sympy/series/gruntz.py:moveup2
def moveup2(s: SubsSet, x: Symbol) -> SubsSet:
    r = SubsSet()
    up: dict[Basic, Basic] = {x: _exp(x)}
    for expr, var in s.items():
        r.set(expr.xreplace(up), var)
    for var, expr in s.rewrites.items():
        r.rewrites[var] = expr.xreplace(up)
    return r


# PORTED FROM SYMPY: sympy/series/gruntz.py:mrv_leadterm
@cacheit
def mrv_leadterm(e: Expr, x: Symbol) -> tuple[Expr, Expr]:
    """``(c0, e0)``: the leading term ``c0*w**e0`` of ``e`` in the most
    rapidly varying ``w`` going to zero."""
    from .order import order
    if not e.has(x):
        return (e, Integer(0))
    omega, exps = mrv(e, x)
    if omega.is_empty():
        return exps, Integer(0)
    if x in omega:
        # move the whole omega up (exponentiate each term)
        omega = moveup2(omega, x)
        exps = exps.xreplace({x: _exp(x)})
    w = Dummy("w", positive=True)
    f, logw = rewrite(exps, omega, x, w)
    replaced = f.replace(lambda t: isinstance(t, Pow) and t.has(x),
                         lambda t: _exp(_log(_pow_base(t)) * _pow_exp(t)))
    if not isinstance(replaced, Expr):
        raise TypeError("an expression is expected")
    f = replaced
    lt: tuple[Expr, Expr]
    try:
        lt = f.leadterm(w, logx=logw)
    except (NotImplementedError, PoleError, ValueError):
        n0 = 1
        _series: Expr = order(Integer(1))
        incr = 1
        while _series.is_Order:
            _series = f._eval_nseries(w, n0 + incr, logw, Integer(0))
            incr *= 2
        series = _series.expand().removeO()
        try:
            lt = series.leadterm(w, logx=logw)
        except (NotImplementedError, PoleError, ValueError):
            lt = f.as_coeff_exponent(w)
            if lt[0].has(w):
                base = f.as_base_exp()[0].as_coeff_exponent(w)
                ex = f.as_base_exp()[1]
                lt = (base[0]**ex, base[1] * ex)
    return (lt[0].subs(_log(w), logw), lt[1])


def _pow_base(t: Basic) -> Expr:
    if not isinstance(t, Pow):
        raise TypeError("a power is expected")
    return t.base


def _pow_exp(t: Basic) -> Expr:
    if not isinstance(t, Pow):
        raise TypeError("a power is expected")
    return t.exp


# PORTED FROM SYMPY: sympy/series/gruntz.py:build_expression_tree
def _heights(omega: list[tuple[Expr, Dummy]], rewrites: dict[Dummy, Expr]) -> dict[Dummy, int]:
    """The height of each element in the tree of the rewrites (an element
    whose rewrite contains another comes before it)."""
    before: dict[Dummy, list[Dummy]] = {v: [] for _, v in omega}
    for _, v in omega:
        if v in rewrites:
            r = rewrites[v]
            for _, v2 in omega:
                if r.has(v2):
                    before[v].append(v2)

    def ht(v: Dummy) -> int:
        return reduce(lambda a, b: a + b, [ht(u) for u in before[v]], 1)
    return {v: ht(v) for _, v in omega}


# PORTED FROM SYMPY: sympy/series/gruntz.py:rewrite
def rewrite(e: Expr, Omega: SubsSet, x: Symbol, wsym: Expr) -> tuple[Expr, Expr]:
    """``e`` rewritten in terms of the most rapidly varying ``wsym``, with
    the value of ``log(wsym)``."""
    from sympy_extras.simplify.powsimp import powsimp
    from .accumbounds import AccumBounds
    from .traversal import bottom_up
    if len(Omega) == 0:
        raise ValueError("Length cannot be 0")
    for t in Omega.keys():
        if not _is_exp(t):
            raise ValueError("Value should be exp")
    rewrites = Omega.rewrites
    omega = Omega.items()
    heights = _heights(omega, rewrites)
    omega.sort(key=lambda item: heights[item[1]], reverse=True)
    # all the elements of omega must be exponentials with a known sign
    sig: Expr = Integer(0)
    g: Expr = omega[0][0]
    for g, _ in omega:
        sig = sign(_exponent(g), x)
        if not _is_number(sig, 1) and not _is_number(sig, -1) and not sig.has(AccumBounds):
            raise NotImplementedError('Result depends on the sign of %s' % sig)
    if _is_number(sig, 1):
        wsym = 1 / wsym  # g goes to oo: 1/w substituted
    g_exp = _exponent(g)
    O2: list[tuple[Dummy, Expr]] = []
    denominators: list[int] = []
    for f, var in omega:
        c = limitinf(_exponent(f) / g_exp, x)
        if isinstance(c, Rational):
            denominators.append(c.q)
        arg = _exponent(f)
        if var in rewrites:
            if not _is_exp(rewrites[var]):
                raise ValueError("Value should be exp")
            rewritten_arg = rewrites[var].args[0]
            if not isinstance(rewritten_arg, Expr):
                raise TypeError("an expression is expected")
            arg = rewritten_arg
        O2.append((var, _exp(arg - c * g_exp) * wsym**c))
    # the elements of omega replaced by their expressions in w (whose
    # rewrites contain other elements of omega come first: sorted above)
    f = powsimp(e, deep=True, combine='exp')
    for a, b in O2:
        rule_w: dict[Basic, Basic] = {a: b}
        f = f.xreplace(rule_w)
    for _, var in omega:
        if f.has(var):
            raise ValueError("the rewrite left %s" % (var,))
    logw = g_exp
    if _is_number(sig, 1):
        logw = -logw  # log(w) -> log(1/w) = -log(w)
    # the powers of w with rational exponents made integral
    exponent = reduce(ilcm, denominators, 1)
    f = f.subs({wsym: wsym**exponent})
    logw = logw / exponent
    normalised = bottom_up(f, lambda t: t.normal() if isinstance(t, Expr) else t)
    if not isinstance(normalised, Expr):
        raise TypeError("an expression is expected")
    return normalised, logw


# PORTED FROM SYMPY: sympy/series/gruntz.py:gruntz
def gruntz(e: Expr, z: Symbol, z0: Expr, dir: str = "+") -> Expr:
    """The limit of ``e`` as ``z`` goes to ``z0`` from ``dir`` by Gruntz's
    algorithm (``limit`` calls it when the simpler methods fail).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.core.gruntz import gruntz
    >>> from sympy_extras.functions import exp, log, sin
    >>> x = Symbol('x')
    >>> gruntz(sin(x)/x, x, 0), gruntz(exp(x)/x**100, x, oo), gruntz(log(x)/x, x, oo)
    (1, oo, 0)
    """
    from .numbers import NegativeInfinity, Infinity
    from .rewriting import rewrite_tractable
    if not isinstance(z, Symbol):
        raise NotImplementedError("Second argument must be a Symbol")
    e0: Expr
    if isinstance(z0, Infinity) or z0 == I * oo:
        e0 = e
    elif isinstance(z0, NegativeInfinity) or z0 == -I * oo:
        e0 = e.subs(z, -z)
    elif dir == "-":
        e0 = e.subs(z, z0 - 1 / z)
    elif dir == "+":
        e0 = e.subs(z, z0 + 1 / z)
    else:
        raise NotImplementedError("dir must be '+' or '-'")
    r = limitinf(e0, z)
    back = rewrite_tractable(r, None, intractable=True)
    if not isinstance(back, Expr):
        raise TypeError("an expression is expected")
    return back
