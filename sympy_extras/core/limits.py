"""Limits: the unevaluated ``Limit`` and the function ``limit``.

``limit(e, z, z0, dir)`` is the limit of ``e`` as ``z`` goes to ``z0``
from the right (``'+'``), the left (``'-'``) or both (``'+-'``): the
leading term of a meromorphic expression, the leading term of the series
at the point, then Gruntz's algorithm, then heuristics (the limits of the
arguments combined). ``Limit(e, z, z0, dir)`` is the unevaluated node,
evaluated by ``doit``; ``limit`` returns it when nothing finds the limit.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: limits*,
       https://docs.sympy.org/latest/modules/series/series.html#limits
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/limits.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the direction is a string argument.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional

from .basic import Basic, NodePrinter
from .coreerrors import PoleError
from .expr import Expr, ExprLike, as_expr, to_expr
from .kind import Kind, NumberKind
from .numbers import ComplexInfinity, Float, I, Infinity, Integer, NaN, NegativeInfinity, nan, oo, zoo
from .sorting import ClassKey
from .symbol import Dummy, Symbol

__all__ = ['Limit', 'limit', 'heuristics']


def _is_number(e: Expr, value: int) -> bool:
    return e.is_Number and e == value


def _positive_number(e: Expr) -> bool:
    """``e > 0`` of SymPy for a number (``TypeError`` when undecided)."""
    answer = e.is_extended_positive
    if answer is None:
        raise TypeError("cannot determine truth value of %s > 0" % (e,))
    return answer


# PORTED FROM SYMPY: sympy/series/limits.py:limit
def limit(e: ExprLike, z: Symbol, z0: ExprLike, dir: str = "+") -> Expr:
    """The limit of ``e`` as ``z`` goes to ``z0`` (from the right with
    ``dir='+'``, from the left with ``'-'``, both with ``'+-'``); the
    unevaluated ``Limit`` when it is not found.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy for what is not ported (core.bridge)
    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.core.limits import limit
    >>> from sympy_extras.functions import cos, exp, sin
    >>> x = Symbol('x')
    >>> limit(sin(x)/x, x, 0), limit(1/x, x, 0, '-'), limit((1 + 1/x)**x, x, oo)
    (1, -oo, E)
    >>> limit((1 - cos(x))/x**2, x, 0), limit(x*exp(-x), x, oo), limit(1/x, x, 0, '+-')
    (1/2, 0, zoo)
    """
    return Limit(e, z, z0, dir).doit(deep=False)


# PORTED FROM SYMPY: sympy/series/limits.py:heuristics
def heuristics(e: Expr, z: Symbol, z0: Expr, dir: str) -> Optional[Expr]:
    """The limit from the limits of the arguments (``None`` when they do
    not decide it)."""
    from .add import Add
    from .exprtools import factor_terms
    from .function import AppliedUndef, Application
    from .mul import Mul
    from .power import Pow
    rv: Optional[Expr] = None
    if isinstance(z0, Infinity):
        found = limit(e.subs(z, 1 / z), z, Integer(0), "+")
        if isinstance(found, Limit):
            return None
        rv = found
    elif isinstance(e, (Mul, Add, Pow)) or (isinstance(e, Application) and not isinstance(e, AppliedUndef)):
        r: list[Expr] = []
        from sympy_extras.polys.rationaltools import together
        for a in e.args:
            if not isinstance(a, Expr):
                return None
            lim = limit(a, z, z0, dir)
            if lim.has(oo) and lim.is_finite is None:
                if isinstance(e, Add):
                    m = factor_terms(e)
                    if not isinstance(m, Mul):  # together then
                        m = together(m)
                    if not isinstance(m, Mul):  # factor when the others failed
                        m = e.factor()
                    if isinstance(m, Mul):
                        return heuristics(m, z, z0, dir)
                    return None
                return None
            if isinstance(lim, Limit):
                return None
            if isinstance(lim, NaN):
                return None
            r.append(lim)
        if r:
            from .accumbounds import AccumBounds
            rebuilt = e._evaluated(tuple(r))
            rv = as_expr(rebuilt)
            if isinstance(rv, NaN) and isinstance(e, Mul) and any(isinstance(rr, AccumBounds) for rr in r):
                # the bounds apart, the limit of the rest simplified
                r2: list[Expr] = []
                e2: list[Expr] = []
                for ii, rval in enumerate(r):
                    if isinstance(rval, AccumBounds):
                        r2.append(rval)
                    else:
                        arg_ii = e.args[ii]
                        if isinstance(arg_ii, Expr):
                            e2.append(arg_ii)
                if e2:
                    from .mul import mul
                    e3 = mul(*e2).simplify()
                    lim3 = limit(e3, z, z0, dir)
                    rv = lim3 * mul(*r2)
            if isinstance(rv, NaN):
                from sympy_extras.polys.polyerrors import PolynomialError
                from sympy_extras.simplify.ratsimp import ratsimp
                try:
                    rat_e = ratsimp(e)
                except PolynomialError:
                    return None
                if isinstance(rat_e, NaN) or rat_e == e:
                    return None
                return limit(rat_e, z, z0, dir)
    return rv


class Limit(Expr):
    """The unevaluated limit of ``e`` as ``z`` goes to ``z0``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.core.limits import Limit
    >>> from sympy_extras.functions import sin
    >>> x = Symbol('x')
    >>> Limit(sin(x)/x, x, 0), Limit(sin(x)/x, x, 0).doit(), Limit(1/x, x, oo).args
    (Limit(sin(x)/x, x, 0, dir='+'), 1, (1/x, x, oo, -))
    """

    __slots__ = ('_expr', '_z', '_z0', '_dir')

    def __init__(self, e: ExprLike, z: Symbol, z0: ExprLike, dir: str = "+") -> None:
        self._expr: Final[Expr] = to_expr(e)
        point = to_expr(z0)
        if isinstance(point, Infinity) or point == I * oo:
            dir = "-"
        elif isinstance(point, NegativeInfinity) or point == I * (-oo):
            dir = "+"
        if point.has(z):
            raise NotImplementedError("Limits approaching a variable point are not supported (%s -> %s)"
                                      % (z, point))
        if dir not in ('+', '-', '+-'):
            raise ValueError("direction must be one of '+', '-' or '+-', not %s" % dir)
        self._z: Final[Symbol] = z
        self._z0: Final[Expr] = point
        self._dir: Final[str] = dir
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._expr, self._z, self._z0, Symbol(self._dir))

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def variable(self) -> Symbol:
        return self._z

    @property
    def point(self) -> Expr:
        return self._z0

    @property
    def direction(self) -> str:
        return self._dir

    @property
    def kind(self) -> Kind:
        return NumberKind

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Limit')

    @property
    def free_symbols(self) -> set[Symbol]:
        symbols = set(self._expr.free_symbols)
        symbols.discard(self._z)
        return symbols | self._z0.free_symbols

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        e, z, z0, d = args
        if not (isinstance(e, Expr) and isinstance(z, Symbol) and isinstance(z0, Expr) and isinstance(d, Symbol)):
            raise TypeError("a limit of an expression is expected")
        return Limit(e, z, z0, d.name)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self._rebuild(args)

    def _eval_facts(self) -> dict[str, bool]:
        return {}

    def _sympystr(self, printer: NodePrinter) -> str:
        return "Limit(%s, %s, %s, dir='%s')" % (printer.doprint(self._expr), printer.doprint(self._z),
                                                printer.doprint(self._z0), self._dir)

    # PORTED FROM SYMPY: sympy/series/limits.py:Limit.pow_heuristics
    def pow_heuristics(self, e: Expr) -> Optional[Expr]:
        from .power import Pow
        from sympy_extras.functions.elementary.exponential import exp, log
        if not isinstance(e, Pow):
            return None
        z, z0 = self._z, self._z0
        b1, e1 = e.base, e.exp
        if not b1.has(z):
            return exp(limit(e1 * log(b1), z, z0))
        ex_lim = limit(e1, z, z0)
        base_lim = limit(b1, z, z0)
        if _is_number(base_lim, 1):
            if isinstance(ex_lim, (Infinity, NegativeInfinity)):
                return exp(limit(e1 * (b1 - 1), z, z0))
        if isinstance(base_lim, NegativeInfinity) and isinstance(ex_lim, Infinity):
            return zoo
        return None

    # PORTED FROM SYMPY: sympy/series/limits.py:Limit.doit
    def doit(self, deep: bool = True) -> Expr:
        """The limit (the node itself when it is not found)."""
        from .exprtools import factor_terms
        from .gruntz import gruntz
        from .mul import Mul
        from .order import Order, order
        from .power import Pow
        from .containers import Tuple
        from sympy_extras.functions.elementary.complexes import abs_, re, sign
        from sympy_extras.functions.elementary.exponential import Exp
        from sympy_extras.functions.special.gamma_functions import Gamma
        from sympy_extras.functions.combinatorial.factorials import Factorial
        from .numbers import Exp1
        e, z, z0, dir = self._expr, self._z, self._z0, self._dir
        if dir == '+-':
            r = limit(e, z, z0, dir='+')
            left = limit(e, z, z0, dir='-')
            if isinstance(r, Limit) and isinstance(left, Limit):
                if r.expr == left.expr:
                    return self
            if r == left:
                return left
            if r.is_infinite is True and left.is_infinite is True:
                return zoo
            raise ValueError("The limit does not exist since left hand limit = %s and right hand limit = %s"
                             % (left, r))
        if isinstance(z0, ComplexInfinity):
            raise NotImplementedError("Limits at complex infinity are not implemented")
        if z0.is_infinite is True:
            cdir_inf = sign(z0)
            cdir_inf = cdir_inf / abs_(cdir_inf)
            e = e.subs(z, cdir_inf * z)
            dir = "-"
            z0 = oo
        if deep:
            e = e.doit()
            z0 = z0.doit()
        if e == z:
            return z0
        if not e.has(z):
            return e
        if isinstance(z0, NaN):
            return nan
        if e.has(nan, oo, -oo, zoo):
            return self
        if isinstance(e, Order):
            return order(limit(e.expr, z, z0), *[Tuple(v, p) for v, p in zip(e.variables, e.point)])
        cdir: Expr = Integer(0)
        if dir == "+":
            cdir = Integer(1)
        elif dir == "-":
            cdir = Integer(-1)

        if e.has(Float):
            from sympy_extras.simplify.simplify import nsimplify
            e = nsimplify(e)
        e = _set_signs(e, z, z0, dir)

        if e.is_meromorphic(z, z0):
            if isinstance(z0, Infinity):
                newe = e.subs(z, 1 / z)
                cdir = -cdir
            else:
                newe = e.subs(z, z + z0)
            try:
                coeff, ex = newe.leadterm(z, cdir=cdir)
            except ValueError:
                pass
            else:
                if _positive_number(ex):
                    return Integer(0)
                if _is_number(ex, 0):
                    return coeff
                if _is_number(cdir, 1) or not (_int_of(ex) & 1):
                    return oo * sign(coeff)
                if _is_number(cdir, -1):
                    return -oo * sign(coeff)
                return zoo

        newz: Symbol
        if isinstance(z0, Infinity):
            if isinstance(e, Mul):
                e = factor_terms(e)
            assumptions = {k: v for k, v in (('positive', z.is_positive), ('negative', z.is_negative),
                                              ('real', z.is_real)) if v is not None}
            dummy = Dummy('z', **assumptions)
            newe = e.subs(z, 1 / dummy)
            cdir = -cdir
            newz = dummy
        else:
            newe = e.subs(z, z + z0)
            newz = z
        try:
            coeff, ex = newe.leadterm(newz, cdir=cdir)
        except (ValueError, NotImplementedError, PoleError):
            from sympy_extras.simplify.powsimp import powsimp
            e = powsimp(e)
            if isinstance(e, Pow):
                found = self.pow_heuristics(e)
                if found is not None:
                    return found
            try:
                coeff = newe.as_leading_term(newz, cdir=cdir)
                if coeff != newe and (coeff.has(Exp) or coeff.has(Exp1)):
                    return gruntz(coeff, newz, Integer(0), "-" if re(cdir).is_negative is True else "+")
            except (ValueError, NotImplementedError, PoleError):
                pass
        else:
            from .accumbounds import AccumBounds
            if isinstance(coeff, AccumBounds) and _is_number(ex, 0):
                return coeff
            if coeff.has(oo, -oo, zoo, nan):
                return self
            if not coeff.has(newz):
                if ex.is_positive is True:
                    return Integer(0)
                if _is_number(ex, 0):
                    return coeff
                if ex.is_negative is True:
                    if _is_number(cdir, 1):
                        return oo * sign(coeff)
                    if _is_number(cdir, -1):
                        return -oo * sign(coeff) * Integer(-1)**(Integer(1) + ex)
                    return zoo
                raise NotImplementedError("Not sure of sign of %s" % (ex,))

        if z0.is_extended_nonnegative is True:
            e = as_expr(e.rewrite(Factorial, Gamma))
        try:
            r = gruntz(e, z, z0, dir)
            if isinstance(r, NaN):
                raise PoleError()
        except (PoleError, ValueError):
            found_h = heuristics(e, z, z0, dir)
            if found_h is None:
                return self
            return found_h
        return r


def _int_of(e: Expr) -> int:
    from .numbers import Rational
    if isinstance(e, Integer):
        return e.p
    if isinstance(e, Rational):
        return int(e.p / e.q)
    raise TypeError("an integer is expected, got %s" % (e,))


# PORTED FROM SYMPY: sympy/series/limits.py:Limit.doit (set_signs)
def _set_signs(expr: Expr, z: Symbol, z0: Expr, dir: str) -> Expr:
    """``Abs``, ``sign`` and ``arg`` of arguments of known sign at the
    point replaced (``Abs(x)`` is ``x`` as ``x`` goes to ``0+``)."""
    from sympy_extras.functions.elementary.complexes import Abs, Arg, Sign
    from sympy_extras.functions.special.delta_functions import Heaviside
    from .numbers import pi
    if not expr.args:
        return expr
    newargs = tuple(_set_signs(a, z, z0, dir) if isinstance(a, Expr) else a for a in expr.args)
    if newargs != expr.args:
        expr = as_expr(expr._evaluated(newargs))
    abs_flag = isinstance(expr, Abs)
    arg_flag = isinstance(expr, Arg)
    sign_flag = isinstance(expr, Sign)
    # a step function too, 0 or 1 by the sign of its argument (not in
    # SymPy: the bug, SymPy's too, was that limit(Heaviside(x - 1), x, 0,
    # '+') did not finish, the Taylor terms of the step at -1 being zero
    # however far gruntz expanded it)
    step_flag = isinstance(expr, Heaviside)
    if abs_flag or sign_flag or arg_flag or step_flag:
        inner = expr.args[0]
        if not isinstance(inner, Expr):
            return expr
        try:
            sig = limit(inner, z, z0, dir)
            if sig.is_zero is True:
                sig = limit(1 / inner, z, z0, dir)
        except NotImplementedError:
            return expr
        if sig.is_extended_real is True:
            if sig.is_extended_negative is True:
                return -inner if abs_flag else Integer(-1) if sign_flag else Integer(0) if step_flag else pi
            if sig.is_extended_positive is True:
                return inner if abs_flag else Integer(1) if sign_flag or step_flag else Integer(0)
    return expr
