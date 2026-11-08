"""Powers: the class ``Pow`` and the evaluating function ``pow``.

``Pow(base, exp)`` is an unevaluated node. The function ``pow(base, exp)``
applies the rules of SymPy's ``Pow`` [1]_: ``b**0`` is ``1``, ``b**1`` is
``b``, ``1**b`` is ``1``, a power of a power combines when the outer
exponent is an integer or the inner one is a rational in ``(-1, 1)`` and
the outer one real, an integer power of a product is distributed, a
positive numerical coefficient is taken out of a product under a rational
exponent, and the powers of numbers are evaluated: integer powers of
rationals exactly, rational powers of rationals by extracting the perfect
powers of the prime factors (``sqrt(12)`` is ``2*sqrt(3)``, ``(1/3)**(1/2)``
is ``sqrt(3)/3``), powers of ``-1`` reduced to an exponent in ``(0, 1]``
(``(-1)**(3/2)`` is ``-I``), floats through mpmath, and the infinities as
SymPy does.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Pow*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.power.Pow

The content-primitive split is ported from SymPy (``sympy/core/power.py``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/power.py (Pow.as_content_primitive) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the rest of the module is original code.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from math import floor
from typing import Final, Optional, TYPE_CHECKING

import mpmath
from sympy_extras.ntheory.factor_ import factorint

from .add import Add, add
from .basic import Basic
from .expr import Expr, ExprLike, factors_of, to_expr
from .mul import Mul, mul
from .numbers import (ComplexInfinity, Exp1, Float, ImaginaryUnit, Infinity, Integer, I, NaN,
    NegativeInfinity, Number, Rational, nan, number_from_fraction, oo, zoo)
from .parameters import global_parameters
from .sorting import ClassKey

if TYPE_CHECKING:
    from .printing import StrPrinter
    from .symbol import Symbol

__all__ = ['Pow', 'pow', 'sqrt', 'cbrt', 'root', 'number_pow', 'integer_root_parts', 'minus_one_pow']

#: numbers below this size are factored completely for the extraction of
#: perfect powers; larger ones only by trial division up to ``_TRIAL``
_FULL_FACTORING_BITS = 64
_TRIAL = 2**16


class Pow(Expr):
    """An unevaluated power.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Pow
    >>> x = Symbol('x')
    >>> p = Pow(x, 0)
    >>> p, p.args, p.doit()
    (x**0, (x, 0), 1)
    >>> Pow(x, 2).base, Pow(x, 2).exp
    (x, 2)
    """

    __slots__ = ('_base', '_exp')

    is_Pow = True

    def __init__(self, base: ExprLike, exp: ExprLike) -> None:
        self._base: Final[Expr] = to_expr(base)
        self._exp: Final[Expr] = to_expr(exp)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Expr, ...]:
        return (self._base, self._exp)

    @property
    def base(self) -> Expr:
        return self._base

    @property
    def exp(self) -> Expr:
        return self._exp

    @classmethod
    def class_key(cls) -> ClassKey:
        return (3, 2, 'Pow')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        b, e = args
        if not (isinstance(b, Expr) and isinstance(e, Expr)):
            raise TypeError("a power of expressions is expected")
        return pow(b, e)

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import pow_facts
        return pow_facts(self._base, self._exp)

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # (1/q)**e is seen as q**(-e), as in SymPy
        if isinstance(self._base, Rational) and self._base.p == 1 and self._base.q > 1:
            return (Integer(self._base.q), mul(Integer(-1), self._exp))
        return (self._base, self._exp)

    def _sympystr(self, printer: StrPrinter) -> str:
        return printer.print_pow(self._base, self._exp)

    # PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        # A generalised power series with coefficients in C(log, log(x))
        # (gruntz needs it): b**e = f**e*(1 + g)**e with the leading term f
        # of b, g of positive order, and the binomial series of (1 + g)**e.
        from .add import _ceiling_int
        from .coreerrors import PoleError
        from .function import expand_mul, expand_multinomial
        from .order import order
        from .series import _less
        from .symbol import Symbol as _Symbol, Wild
        from sympy_extras.functions.elementary.exponential import exp, log, Log
        from sympy_extras.functions.combinatorial.factorials import factorial, ff
        from sympy_extras.functions.elementary.complexes import im
        from sympy_extras.functions.special.gamma_functions import PolyGamma
        from sympy_extras.simplify.powsimp import powdenest, powsimp
        from .numbers import EulerGamma

        def mexpand(e: Expr) -> Expr:
            return expand_mul(expand_multinomial(e))

        if isinstance(self._base, Exp1):
            e_series = self._exp.nseries(x, n=n, logx=logx)
            if e_series.is_Order:
                return add(Integer(1), e_series)
            e0 = e_series.removeO().limit(x, Integer(0))
            if isinstance(e0, NegativeInfinity):
                return order(pow(x, Integer(n)), x)
            if isinstance(e0, Infinity):
                return self
            t = add(e_series, mul(Integer(-1), e0))
            exp_series = term = exp(e0)
            # the series of exp(e0 + t) in t
            for i in range(1, n):
                term = mul(term, t, pow(Integer(i), Integer(-1)))
                term = term.nseries(x, n=n, logx=logx)
                exp_series = add(exp_series, term)
            exp_series = add(exp_series, order(pow(t, Integer(n)), x))
            return powsimp(exp_series, deep=True, combine='exp')
        me = powdenest(self, force=True).trigsimp()
        b, e = me.as_base_exp()
        if e.has(nan, oo, mul(Integer(-1), oo), zoo):
            raise PoleError()
        if e.has(x):
            return exp(mul(e, log(b)))._eval_nseries(x, n, logx, cdir)
        if logx is not None and b.has(Log):
            wc, wex = Wild('c', exclude=[x]), Wild('ex', exclude=[x])
            b = as_expr_basic(b.replace(log(mul(wc, pow(x, wex))), add(log(wc), mul(wex, logx))))
            me = pow(b, e)
        b = b.removeO()
        m: Expr
        try:
            if b.has(PolyGamma, EulerGamma) and logx is not None:
                raise ValueError()
            _, m = b.leadterm(x)
        except (ValueError, NotImplementedError, PoleError):
            b = b._eval_nseries(x, max(2, n), logx, cdir).removeO()
            if b.has(nan, ComplexInfinity):
                raise NotImplementedError()
            _, m = b.leadterm(x)
        if e.has(Log):
            from sympy_extras.simplify.simplify import logcombine
            e = logcombine(e).cancel()
        if not (m.is_zero is True or (e.is_number and e.is_real is True)):
            if me == me._eval_as_leading_term(x, logx, cdir):
                logarithmic = exp(mul(e, log(b)))
                via_log = logarithmic._eval_nseries(x, n, logx, cdir)
                if via_log == logarithmic:
                    return me
                return via_log
        f = b.as_leading_term(x, logx=logx)
        g = add(mexpand(b), mul(Integer(-1), f)).cancel()
        g = mul(g, pow(f, Integer(-1)))
        if not m.is_number:
            raise NotImplementedError()
        maxpow = add(Integer(n), mul(Integer(-1), m, e))
        if maxpow.has(_Symbol):
            maxpow = Integer(n)
        if maxpow.is_negative is True:
            return order(pow(x, mul(m, e)), x)
        if g.is_zero is True:
            r = pow(f, e)
            if r != me:
                r = add(r, order(pow(x, Integer(n)), x))
            return r

        def coeff_exp(term: Expr) -> tuple[Expr, Expr]:
            coeff: Expr = Integer(1)
            exponent: Expr = Integer(0)
            for factor in factors_of(term):
                if factor.has(x):
                    base, exponent = factor.as_base_exp()
                    if base != x:
                        try:
                            return term.leadterm(x)
                        except ValueError:
                            return term, Integer(0)
                else:
                    coeff = mul(coeff, factor)
            return coeff, exponent

        def product(d1: dict[Expr, Expr], d2: dict[Expr, Expr]) -> dict[Expr, Expr]:
            res: dict[Expr, Expr] = {}
            for e1 in d1:
                for e2 in d2:
                    ex = add(e1, e2)
                    if _less(ex, maxpow):
                        res[ex] = add(res.get(ex, Integer(0)), mul(d1[e1], d2[e2]))
            return res

        try:
            c0, d = g.leadterm(x, logx=logx)
        except (ValueError, NotImplementedError):
            if mul(g, pow(pow(x, maxpow), Integer(-1))).limit(x, Integer(0)) == Integer(0):
                # g has a zero of higher order: the first term of the binomial series
                fe = pow(f, e)
                return add(fe, mul(e, fe, g))
            raise NotImplementedError()
        if isinstance(c0, Float) and d == Integer(0):
            # floats like 0.5 made exact, against rounding errors in d
            from .numbers import number_from_fraction

            def exact(t: Basic) -> Basic:
                if isinstance(t, Float):
                    value = t.exact_value()
                    if value is not None:
                        return number_from_fraction(value)
                return t
            g = as_expr_basic(g.replace(lambda t: isinstance(t, Float), exact))
            _, d = g.leadterm(x, logx=logx)
        if d.is_positive is not True:
            g = g.simplify()
            if g.is_zero is True:
                return pow(f, e)
            _, d = g.leadterm(x, logx=logx)
            if d.is_positive is not True:
                g = mul(add(b, mul(Integer(-1), f)), pow(f, Integer(-1))).expand()
                _, d = g.leadterm(x, logx=logx)
                if d.is_positive is not True:
                    raise NotImplementedError()
        gpoly = g._eval_nseries(x, _ceiling_int(maxpow), logx, cdir).removeO()
        gterms: dict[Expr, Expr] = {}
        for term in (gpoly.args if isinstance(gpoly, Add) else (gpoly,)):
            if not isinstance(term, Expr):
                continue
            co1, e1 = coeff_exp(term)
            gterms[e1] = add(gterms.get(e1, Integer(0)), co1)
        k = 1
        terms: dict[Expr, Expr] = {Integer(0): Integer(1)}
        tk = gterms
        while add(mul(Integer(k), d), mul(Integer(-1), maxpow)).is_negative is True:
            coeff = mul(ff(e, Integer(k)), pow(factorial(Integer(k)), Integer(-1)))
            for ex in tk:
                terms[ex] = add(terms.get(ex, Integer(0)), mul(coeff, tk[ex]))
            tk = product(tk, gterms)
            k += 1
        inco: Expr
        inex: Expr
        if e.is_integer is not True and m.is_zero is True and f.is_negative is True:
            ndir = add(b, mul(Integer(-1), f)).dir(x, cdir)
            if im(ndir).is_negative is True:
                inco, inex = coeff_exp(mul(pow(f, e), pow(Integer(-1), mul(Integer(-2), e))))
            elif im(ndir).is_zero is True:
                inco, inex = coeff_exp(exp(mul(e, log(b))).as_leading_term(x, logx=logx, cdir=cdir))
            else:
                inco, inex = coeff_exp(pow(f, e))
        else:
            inco, inex = coeff_exp(pow(f, e))
        res: Expr = Integer(0)
        for e1 in terms:
            ex = add(e1, inex)
            res = add(res, mul(terms[e1], inco, pow(x, ex)))
        if not (e.is_integer is True and e.is_positive is True
                and add(mul(e, d), Integer(-n)).is_nonpositive is True and res == mexpand(me)):
            try:
                res = add(res, order(pow(x, Integer(n)), x))
            except NotImplementedError:
                return exp(mul(e, log(b)))._eval_nseries(x, n, logx, cdir)
        return res

    # PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        from .coreerrors import PoleError
        from sympy_extras.functions.elementary.complexes import im
        from sympy_extras.functions.elementary.exponential import exp, log
        e = self._exp
        b = self._base
        if isinstance(b, Exp1):
            arg = e.as_leading_term(x, logx=logx)
            arg0 = arg.subs(x, Integer(0))
            if isinstance(arg0, NaN):
                arg0 = arg.limit(x, Integer(0))
            if arg0.is_infinite is False:
                return pow(b, arg0)
            raise PoleError("Cannot expand %s around 0" % (self,))
        if e.has(x):
            return exp(mul(e, log(b))).as_leading_term(x, logx=logx, cdir=cdir)
        try:
            f = b.as_leading_term(x, logx=logx, cdir=cdir)
        except PoleError:
            return self
        if e.is_integer is not True and f.is_negative is True and not f.has(x):
            ndir = add(b, mul(Integer(-1), f)).dir(x, cdir)
            if im(ndir).is_negative is True:
                # on the branch cut: exp(e*(log(f) - 2*pi*I)) == f**e*(-1)**(-2*e)
                return mul(Pow(f, e).doit(), pow(Integer(-1), mul(Integer(-2), e)))
            if im(ndir).is_zero is True:
                log_leadterm = log(b)._eval_as_leading_term(x, logx, cdir)
                if log_leadterm.is_infinite is False:
                    return exp(mul(e, log_leadterm))
        return pow(f, e)

    # PORTED FROM SYMPY: sympy/core/power.py:Pow.as_content_primitive
    def as_content_primitive(self, radical: bool = False, clear: bool = True) -> tuple[Number, Expr]:
        from .mul import _keep_coeff
        b0, e0 = self.as_base_exp()
        b = _keep_coeff(*b0.as_content_primitive(radical=radical, clear=clear))
        ce, pe = e0.as_content_primitive(radical=radical, clear=clear)
        if isinstance(b, Rational):
            h, t = pe.as_coeff_Add()
            if isinstance(h, Rational) and b.p != 0:
                ceh = number_from_fraction(Fraction(ce.p * h.p, ce.q * h.q)) if isinstance(ce, Rational) else None
                if ceh is not None and isinstance(ce, Rational):
                    c = pow(b, ceh)
                    r = Integer(0)
                    if not isinstance(c, Rational):
                        iceh, rem = divmod(ceh.p, ceh.q)
                        c = pow(b, Integer(iceh))
                        r = Integer(rem)
                    if isinstance(c, Rational):
                        shift = mul(r, pow(ce, Integer(-1)), Rational(1, ceh.q))
                        return (c, pow(b, _keep_coeff(ce, add(t, shift))))
        e = _keep_coeff(ce, pe)
        if isinstance(e, Rational) and isinstance(b, Mul):
            h2, t2 = b.as_content_primitive(radical=radical, clear=clear)
            c2, m2 = pow(h2, e).as_coeff_Mul()
            m3, me = m2.as_base_exp()
            if isinstance(c2, Number) and ((isinstance(m3, Integer) and m3.p == 1) or me == e):
                return (c2, pow(_keep_coeff(m3, t2), e))
        return (Integer(1), pow(b, e))

    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        if not (isinstance(old, Expr) and isinstance(new, Expr)):
            return None
        return power_subs(self._base, self._exp, old, new)

    # PORTED FROM SYMPY: sympy/core/power.py:Pow.as_numer_denom
    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """The numerator and the denominator: the base split when its
        denominator is known real and the exponent an integer, or the
        denominator known positive (negated when nonpositive); a negative
        exponent swaps them."""
        if self.is_commutative is False:
            return (self, Integer(1))
        base, exp = self.as_base_exp()
        n, d = base.as_numer_denom()
        neg_exp = exp.is_negative is True
        if isinstance(exp, Mul) and not neg_exp and exp.is_positive is not True:
            neg_exp = exp.could_extract_minus_sign()
        int_exp = exp.is_integer is True
        if not (d.is_extended_real is True or int_exp):
            n, d = base, Integer(1)
        dnonpos = d.is_nonpositive
        if dnonpos is True:
            n, d = mul(Integer(-1), n), mul(Integer(-1), d)
        elif dnonpos is None and not int_exp:
            n, d = base, Integer(1)
        if neg_exp:
            n, d = d, n
            exp = mul(Integer(-1), exp)
        if exp.is_infinite is True:
            n_one = isinstance(n, Integer) and n.p == 1
            d_one = isinstance(d, Integer) and d.p == 1
            if n_one and not d_one:
                return (n, pow(d, exp))
            if not n_one and d_one:
                return (pow(n, exp), d)
        return (pow(n, exp), pow(d, exp))


def pow(b: ExprLike, e: ExprLike) -> Expr:
    """The evaluated power ``b**e``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, pow, Rational, Integer
    >>> x, y = symbols('x y')
    >>> pow(x, 0), pow(x, 1), pow(1, x)
    (1, x, 1)
    >>> pow(Integer(2), 10), pow(Integer(8), Rational(1, 3)), pow(Integer(12), Rational(1, 2))
    (1024, 2, 2*sqrt(3))
    >>> pow(pow(x, 2), 3), pow(2*x, 2), pow(4*x, Rational(1, 2))
    (x**6, 4*x**2, 2*sqrt(x))
    >>> pow(Integer(-1), Rational(1, 2)), pow(Integer(-8), Rational(1, 3))
    (I, 2*(-1)**(1/3))
    """
    base, exp = to_expr(b), to_expr(e)
    if not global_parameters.evaluate:
        return Pow(base, exp)
    if base.is_Order or exp.is_Order:
        import sympy
        from .interop import expr_from_sympy, to_sympy
        return expr_from_sympy(sympy.Pow(to_sympy(base), to_sympy(exp)))
    if isinstance(base, Number) and isinstance(exp, Number):
        return number_pow(base, exp)
    if isinstance(exp, Number):
        if isinstance(exp, NaN):
            return nan
        if isinstance(exp, Rational):
            if exp.p == 0:
                return Integer(1)
            if exp.p == 1 and exp.q == 1:
                return base
    if isinstance(base, Number):
        if isinstance(base, NaN):
            return nan
        if isinstance(base, Rational) and base.p == 1 and base.q == 1:
            return Integer(1)
        if isinstance(base, Integer) and base.p == -1 and not isinstance(exp, Number):
            # PORTED FROM SYMPY: sympy/core/numbers.py:NegativeOne._eval_power
            # (the parity of a symbolic exponent)
            if exp.is_odd is True:
                return Integer(-1)
            if exp.is_even is True:
                return Integer(1)
        if isinstance(base, Rational) and base.p == 0:
            # PORTED FROM SYMPY: sympy/core/numbers.py:Zero._eval_power
            if exp.is_extended_positive:
                return Integer(0)
            if exp.is_extended_negative:
                return zoo
            if exp.is_extended_real is False:
                return nan
            if exp.is_zero:
                return Integer(1)
            # the leading number of the exponent is dropped: 0**(-x) is
            # zoo**x even when x is 0, and 0**(a/2) is 0**a
            c, rest = exp.as_coeff_Mul()
            if c.is_negative:
                return pow(zoo, rest)
            if not (isinstance(c, Rational) and c.p == 1 and c.q == 1):
                return pow(base, rest)
            return Pow(base, exp)
    if isinstance(base, Add) and len(base.args) == 2 and any(t.is_infinite for t in base.args):
        # PORTED FROM SYMPY: sympy/core/add.py:Add._eval_power: a literal
        # a + I*b with real a, b and an infinite term is 0 to a negative
        # power and zoo to a positive one (other than 1)
        if exp.is_zero is False and exp != Integer(1):
            a, b = base.args
            if a.coeff(I) != 0:
                a, b = b, a
            imaginary = b.coeff(I)
            if imaginary != 0 and imaginary.is_extended_real and a.is_extended_real:
                if exp.is_extended_negative:
                    return Integer(0)
                if exp.is_extended_positive:
                    return zoo
    if isinstance(base, Mul) and isinstance(exp, Rational) and exp.q != 1:
        # SymPy's Mul._eval_power with expand_power_base: an infinite
        # coefficient times a direction c, (c*oo)**e = oo*c**e for e > 0 and
        # 0 for e < 0 ((-oo*I)**(1/2) is oo*sqrt(-I)), on the principal branch
        # since arg(c*t) = arg(c) for t > 0
        coefficient, direction = base.as_coeff_Mul()
        if isinstance(coefficient, (Infinity, NegativeInfinity)) and direction.is_finite and direction.is_zero is False:
            if exp.p < 0:
                return Integer(0)
            sign = Integer(1) if isinstance(coefficient, Infinity) else Integer(-1)
            return mul(oo, pow(mul(sign, direction), exp))
    if not isinstance(base, Number):
        own = base._eval_power_node(exp)
        if own is not None:
            return own
    if isinstance(base, Infinity) and not isinstance(exp, Number):
        # PORTED FROM SYMPY: sympy/core/numbers.py:Infinity._eval_power
        if exp.is_extended_positive:
            return oo
        if exp.is_extended_negative:
            return Integer(0)
        if exp.is_extended_real is False and exp.is_number:
            from sympy_extras.functions.elementary.complexes import re
            real = re(exp)
            if real.is_positive:
                return zoo
            if real.is_negative:
                return Integer(0)
            if real.is_zero:
                return nan
    if isinstance(base, Exp1):
        # E**a is exp(a) (E**0 is 1, E**(I*pi) is -1)
        from sympy_extras.functions.elementary.exponential import exp as exponential
        return exponential(exp)
    if isinstance(base, ImaginaryUnit) and isinstance(exp, Rational):
        if exp.q == 1:
            return _imaginary_unit_power(exp.p)
        reduced = exp.as_fraction() % 2
        k = int((exp.as_fraction() - reduced) / 2)
        power = Pow(I, number_from_fraction(reduced))
        return power if k % 2 == 0 else Mul(Integer(-1), power)
    if isinstance(base, Add) and isinstance(exp, Rational) and exp.p == -1 and exp.q == 1:
        inverse = _gaussian_inverse(base)
        if inverse is not None:
            return inverse
    if base.is_Pow or _is_exp(base):
        inner_base, inner_exp = base.as_base_exp()
        denested = _power_of_power(inner_base, inner_exp, exp)
        if denested is not None:
            return denested
    if isinstance(base, Mul):
        distributed = _power_of_product(base, exp)
        if distributed is not None:
            return distributed
    if base.is_commutative is False and not (isinstance(exp, Rational) and exp.q == 1):
        return Pow(base, exp)
    return Pow(base, exp)


def _numeric2(e: Expr) -> Optional[Expr]:
    """SymPy's ``_n2``: ``e`` evaluated to 2 digits when it is a number."""
    try:
        value = e.evalf(2)
    except (ValueError, ArithmeticError):
        return None
    return value if isinstance(value, Number) else None


def _half(e: Expr) -> bool:
    if isinstance(e, Rational) and e.q == 2:
        return True
    n, d = e.as_numer_denom()
    return n.is_integer is True and d == Integer(2)


# PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_power
def _power_of_power(b: Expr, e: Expr, expt: Expr) -> Optional[Expr]:
    """``(b**e)**expt`` as ``s*b**(e*expt)`` when the factor ``s`` (the
    branch correction) is known: ``s = 1`` for an integer ``expt``, a polar
    base, ``|e| < 1``, a nonnegative base, ``re(b) >= 0`` with ``|e| < 2``;
    the half-integer and complex exponents by the floor formula;
    ``None`` otherwise."""
    from sympy_extras.functions.elementary.complexes import abs_, arg, im, re
    from sympy_extras.functions.elementary.exponential import exp as exponential, log
    from sympy_extras.functions.elementary.integers import floor
    from .numbers import pi
    s: Optional[Expr] = None
    if expt.is_integer is True:
        s = Integer(1)
    elif b.is_polar is True:
        s = Integer(1)
    elif e.is_extended_real is not None:
        if e.is_extended_real:
            if e == Integer(-1):
                if _half(expt):
                    if b.is_negative is True:
                        return mul(pow(Integer(-1), expt), pow(mul(Integer(-1), b), mul(e, expt)))
                    if b.is_negative is False:
                        return pow(b, mul(Integer(-1), expt))
            elif e.is_even is True:
                if b.is_extended_real is True:
                    b = abs_(b)
                if b.is_imaginary is True:
                    b = mul(abs_(im(b)), I)
            if _lt(abs_(e), Integer(1)) or e == Integer(1):
                s = Integer(1)
            elif b.is_extended_nonnegative is True:
                s = Integer(1)
            elif re(b).is_extended_nonnegative is True and _lt(abs_(e), Integer(2)):
                s = Integer(1)
            elif _half(expt):
                s = exponential(mul(Integer(2), pi, I, expt,
                                    floor(add(Rational(1, 2), mul(Integer(-1), e, arg(b), pow(mul(Integer(2), pi), Integer(-1)))))))
                s = _real_sign(s)
        else:
            s = exponential(mul(Integer(2), I, pi, expt,
                                floor(add(Rational(1, 2), mul(Integer(-1), im(mul(e, log(b))),
                                                                pow(mul(Integer(2), pi), Integer(-1)))))))
            s = _real_sign(s)
    if s is not None:
        return mul(s, pow(b, mul(e, expt)))
    return None


def _real_sign(s: Expr) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import sign
    if s.is_extended_real is True:
        sg = sign(s)
        difference = _numeric2(add(sg, mul(Integer(-1), s)))
        if difference is not None and difference == 0:
            return sg
    return None


def _lt(a: Expr, b: Expr) -> bool:
    """SymPy's ``(a < b) == True``."""
    from .relational import Lt
    return Lt(a, b).doit() is true_value()


def true_value() -> Basic:
    from .logic import true
    return true


def _is_exp(e: Expr) -> bool:
    """Whether ``e`` is an ``exp(...)`` of SymPy (a power of ``E``)."""
    import sympy
    from sympy_extras.functions.elementary.exponential import Exp
    from .function import AppliedSymPyFunction
    return isinstance(e, Exp) or (isinstance(e, AppliedSymPyFunction) and e.sympy_class is sympy.exp)


def _imaginary_unit_power(n: int) -> Expr:
    return (Integer(1), I, Integer(-1), mul(Integer(-1), I))[n % 4]


def _gaussian_inverse(base: Add) -> Optional[Expr]:
    """``1/(a + b*I)`` as ``(a - b*I)/(a**2 + b**2)`` for rational ``a``, ``b``."""
    if len(base.args) != 2:
        return None
    real: Optional[Rational] = None
    imaginary: Optional[Rational] = None
    for term in base.args:
        if isinstance(term, Rational):
            real = term
        elif isinstance(term, ImaginaryUnit):
            imaginary = Integer(1)
        elif isinstance(term, Mul) and len(term.args) == 2 and isinstance(term.args[0], Rational) \
                and isinstance(term.args[1], ImaginaryUnit):
            imaginary = term.args[0]
    if real is None or imaginary is None:
        return None
    a, b = real.as_fraction(), imaginary.as_fraction()
    norm = number_from_fraction(1 / (a * a + b * b))
    conjugate = Add(real, Mul(number_from_fraction(-b), I)) if b != -1 else Add(real, I)
    return Mul(norm, conjugate)


def _power_of_product(base: Mul, exp: Expr) -> Optional[Expr]:
    """``(a*b)**e`` distributed when the rules allow it."""
    commutative = [f for f in base.args if f.is_commutative is not False]
    noncommutative = [f for f in base.args if f.is_commutative is False]
    if isinstance(exp, Rational) and exp.q == 1:
        parts = [pow(f, exp) for f in commutative]
        if noncommutative:
            parts.append(Pow(mul(*noncommutative), exp))
        return mul(*parts)
    if noncommutative or not isinstance(exp, Rational):
        return None
    if exp.q == 2 and base.is_imaginary is True:
        # PORTED FROM SYMPY: sympy/core/mul.py:Mul._eval_power (sqrt(2*I) is 1 + I)
        imaginary = base.as_real_imag()[1]
        if isinstance(imaginary, Rational):
            from math import isqrt
            half = abs(imaginary.as_fraction() / 2)
            n, d = isqrt(half.numerator), isqrt(half.denominator)
            if n * n == half.numerator and d * d == half.denominator:
                from sympy_extras.simplify.radsimp import unevaluated_mul
                unit = Integer(1 if imaginary.p > 0 else -1)
                return unevaluated_mul(pow(number_from_fraction(Fraction(n, d)), Integer(exp.p)),
                                       pow(add(Integer(1), mul(unit, I)), Integer(exp.p)))
    coefficient, rest = base.as_coeff_Mul()
    positive = [f for f in factors_of(rest) if f.is_positive is True]
    others = [f for f in factors_of(rest) if f.is_positive is not True]
    if coefficient.is_finite_number and coefficient._sign() != 0 and not (
            isinstance(coefficient, Rational) and coefficient.p == 1 and coefficient.q == 1):
        if coefficient._sign() > 0 or not others:
            parts = [pow(coefficient, exp)] + [pow(f, exp) for f in positive]
            parts.append(pow(mul(*others), exp) if others else Integer(1))
            return mul(*parts)
        negated = coefficient.number_mul(Integer(-1))
        if isinstance(negated, Rational) and negated.p == 1 and negated.q == 1 and not positive:
            return None  # (-x)**e stays
        parts = [pow(negated, exp)] + [pow(f, exp) for f in positive]
        parts.append(pow(mul(Integer(-1), *others), exp))
        return mul(*parts)
    if positive and others:
        return mul(*[pow(f, exp) for f in positive], pow(mul(*others), exp))
    if positive and not others:
        return mul(*[pow(f, exp) for f in positive])
    return None


def sqrt(x: ExprLike) -> Expr:
    """The square root ``x**(1/2)``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, sqrt
    >>> x = Symbol('x')
    >>> sqrt(x), sqrt(8), sqrt(-4), sqrt(x**2)
    (sqrt(x), 2*sqrt(2), 2*I, sqrt(x**2))
    """
    return pow(x, Rational(1, 2))


def cbrt(x: ExprLike) -> Expr:
    """The cube root ``x**(1/3)``."""
    return pow(x, Rational(1, 3))


def root(x: ExprLike, n: int) -> Expr:
    """The ``n``-th root ``x**(1/n)``."""
    return pow(x, Rational(1, n))


# ----------------------------------------------------------------------
# powers of numbers


def number_pow(b: Number, e: Number) -> Expr:
    """``b**e`` for two numbers (the result need not be a number).

    Examples
    ========

    >>> from sympy_extras.core import Integer, Rational, Float, oo
    >>> from sympy_extras.core.power import number_pow
    >>> number_pow(Rational(2, 3), Integer(-2))
    9/4
    >>> number_pow(Integer(8), Rational(-1, 2))
    sqrt(2)/4
    >>> number_pow(Float(2), Rational(1, 2))
    1.41421356237310
    >>> number_pow(Integer(2), oo), number_pow(Integer(0), Integer(0))
    (oo, 1)
    """
    if isinstance(e, NaN):
        return nan
    if isinstance(e, Rational) and e.p == 0:
        return Integer(1)
    if isinstance(b, NaN):
        return nan
    if isinstance(e, Rational) and e.p == 1 and e.q == 1:
        return b
    if not b.is_finite_number or not e.is_finite_number:
        return _infinite_pow(b, e)
    vb, ve = b.exact_value(), e.exact_value()
    if vb is None or ve is None:
        return nan
    if vb == 1 and not isinstance(b, Float):
        return Integer(1)
    if vb == 0:
        if ve > 0:
            return Integer(0) if not (isinstance(b, Float) and isinstance(e, Float)) else Float(0)
        if global_parameters.divide_error:
            raise ZeroDivisionError("division by zero")
        return zoo
    if isinstance(b, Float) or isinstance(e, Float):
        return _float_pow(b, e)
    assert isinstance(b, Rational) and isinstance(e, Rational)
    if e.q == 1:
        return number_from_fraction(vb ** e.p)
    return _rational_pow(b, e)


def _infinite_pow(b: Number, e: Number) -> Expr:
    """Powers involving ``oo``, ``-oo`` or ``zoo``."""
    if isinstance(e, Rational) or isinstance(e, Float):
        sign = e._sign()
        if isinstance(b, Infinity):
            return oo if sign > 0 else Integer(0)
        if isinstance(b, NegativeInfinity):
            if sign < 0:
                return Integer(0)
            if isinstance(e, Rational) and e.q == 1:
                return oo if e.p % 2 == 0 else NegativeInfinity()
            if isinstance(e, Float) and e.as_fraction().denominator == 1:
                return oo if int(e.as_fraction()) % 2 == 0 else NegativeInfinity()
            return mul(oo, number_pow(Integer(-1), e))
        if isinstance(b, ComplexInfinity):
            return zoo if sign > 0 else Integer(0)
        return nan
    if isinstance(e, ComplexInfinity):
        return nan
    if not b.is_finite_number:
        if isinstance(b, Infinity):
            return oo if isinstance(e, Infinity) else Integer(0)
        if isinstance(b, NegativeInfinity):
            return nan if isinstance(e, Infinity) else Integer(0)
        return nan
    vb = b.exact_value()
    if vb is None:
        return nan
    if isinstance(e, Infinity):
        if vb > 1:
            return oo
        if vb == 1 or vb == -1:
            return nan
        if -1 < vb < 1:
            return Integer(0)
        return zoo
    if isinstance(e, NegativeInfinity):
        if vb > 1 or vb < -1:
            return Integer(0)
        if vb == 1 or vb == -1:
            return nan
        if vb == 0:
            return zoo
        return oo if vb > 0 else zoo
    return nan


def _float_pow(b: Number, e: Number) -> Expr:
    prec = max(b.precision if isinstance(b, Float) else 0, e.precision if isinstance(e, Float) else 0)
    vb, ve = b.exact_value(), e.exact_value()
    if vb is None or ve is None:
        return nan
    if ve.denominator == 1 and not isinstance(e, Float):
        return Float(number_from_fraction(vb ** ve.numerator), precision=prec)
    with mpmath.workprec(prec + 10):
        base = mpmath.mpf(vb.numerator) / vb.denominator
        exponent = mpmath.mpf(ve.numerator) / ve.denominator
        value = mpmath.power(base, exponent)
        real = _mpf_to_fraction(mpmath.re(value))
        imag = _mpf_to_fraction(mpmath.im(value))
    result: Expr = Float(number_from_fraction(real), precision=prec)
    if imag != 0:
        result = add(result, mul(Float(number_from_fraction(imag), precision=prec), I))
    return result


def _mpf_to_fraction(x: mpmath.mpf) -> Fraction:
    sign, man, exp, _ = x._mpf_
    value = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -value if int(sign) else value


def minus_one_pow(e: Fraction) -> Expr:
    """``(-1)**e`` with the exponent reduced to ``(0, 1]``, as SymPy."""
    reduced = e - 2 * floor((e + 1) / 2)  # in (-1, 1]
    sign = 1
    if reduced < 0:
        reduced += 1
        sign = -1
    if reduced == 0:
        return Integer(sign)
    if reduced == 1:
        return Integer(-sign)
    if reduced == Fraction(1, 2):
        return I if sign > 0 else mul(Integer(-1), I)
    base = Pow(Integer(-1), number_from_fraction(reduced))
    return base if sign > 0 else Mul(Integer(-1), base)


def integer_root_parts(p: int, e: Fraction) -> tuple[int, int, int]:
    """``p**e`` for a positive integer ``p`` and ``0 < e < 1`` as
    ``(r, t, g)`` with ``p**e == r * t**(g/n)``, ``n`` the denominator of
    ``e``, ``g/n < 1`` and ``t**g`` free of ``n``-th powers in the
    factorisation used (complete below 64 bits, by trial division above).

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.core.power import integer_root_parts
    >>> integer_root_parts(12, Fraction(1, 2))
    (2, 3, 1)
    >>> integer_root_parts(48, Fraction(1, 3))
    (2, 6, 1)
    >>> integer_root_parts(32, Fraction(1, 6))
    (1, 2, 5)
    >>> integer_root_parts(72, Fraction(1, 3))
    (2, 3, 2)
    """
    from math import gcd
    m, n = e.numerator, e.denominator
    if p.bit_length() <= _FULL_FACTORING_BITS:
        factors = factorint(p)
    else:
        factors = factorint(p, limit=_TRIAL)
    r = 1
    remainders: list[tuple[int, int]] = []
    for prime, multiplicity in factors.items():
        q, rem = divmod(int(multiplicity) * m, n)
        r *= int(prime) ** q
        if rem:
            remainders.append((int(prime), rem))
    g = 0
    for _, rem in remainders:
        g = gcd(g, rem)
    t = 1
    for prime, rem in remainders:
        t *= prime ** (rem // g)
    return (r, t, g) if remainders else (r, 1, 0)


def _rational_pow(b: Rational, e: Rational) -> Expr:
    """``b**e`` for a non-integer rational exponent."""
    from .mul import Mul
    vb, ve = b.as_fraction(), e.as_fraction()
    if vb == 1:
        return Integer(1)
    if ve < 0:
        # b**(-m/n) = b**(k - m/n) * b**(-k) with k = ceil(m/n): valid on the
        # principal branch for every b, while (1/b)**(m/n) is not for b < 0
        k = -floor(ve)
        return mul(_rational_pow(b, Rational(k * ve.denominator + ve.numerator, ve.denominator)),
                   number_from_fraction(vb ** (-k)))
    if b.q > 1:
        k = -floor(-ve)
        parts = [_rational_pow(Integer(b.p), e) if b.p != 1 else Integer(1),
                 _rational_pow(Integer(b.q), number_from_fraction(k - ve)) if k != ve else Integer(1),
                 Rational(1, b.q ** k)]
        return mul(*parts)
    integer_part = floor(ve)
    fraction = ve - integer_part
    r, t, g = integer_root_parts(abs(b.p), fraction)
    coefficient = Integer(b.p ** integer_part * r)
    if vb > 0:
        if t == 1:
            return coefficient
        radical = Pow(Integer(t), number_from_fraction(Fraction(g, fraction.denominator)))
        if coefficient.p == 1:
            return radical
        return Mul(coefficient, radical)
    # a negative base: (-1)**f is I for a half-integer exponent, otherwise
    # the sign stays with the n-th power free part when it is not a perfect
    # power (SymPy writes (-2)**(1/3) and 2*(-1)**(1/3) but (-1)**(1/3)*2**(2/3))
    unit = minus_one_pow(fraction)
    radical_parts: list[Expr]
    if isinstance(unit, Pow) and g == fraction.numerator and t > 1:
        radical_parts = [Pow(Integer(-t), number_from_fraction(fraction))]
    elif t == 1:
        radical_parts = [unit]
    else:
        radical_parts = [unit, Pow(Integer(t), number_from_fraction(Fraction(g, fraction.denominator)))]
    return mul(coefficient, *radical_parts)


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def _independent_of_symbols(e: Expr) -> tuple[Expr, Expr]:
    """SymPy's ``e.as_independent(Symbol, as_Add=False)``: the factors free
    of symbols and the others."""
    from .symbol import Symbol
    if isinstance(e, Mul):
        free = [f for f in e.args if isinstance(f, Expr) and not f.has(Symbol)]
        dependent = [f for f in e.args if isinstance(f, Expr) and f.has(Symbol)]
        return mul(*free), mul(*dependent)
    if e.has(Symbol):
        return Integer(1), e
    return e, Integer(1)


def _combines(coeff1: Expr, coeff2: Expr, old: Expr) -> tuple[bool, Expr]:
    """SymPy's ``_check`` of ``Pow._eval_subs`` for equal terms: the power
    ``coeff1/coeff2`` and whether the substitution may use it."""
    p = mul(coeff1, pow(coeff2, Integer(-1)))
    exact = p.exact_value() if isinstance(p, Float) else None
    if isinstance(p, Integer) or (exact is not None and exact.denominator == 1):
        return True, p
    b, e = old.as_base_exp()
    combines = (b.is_positive is True and e.is_real is True) or (b.is_nonnegative is True and e.is_nonnegative is True)
    return combines, p


def _power_view(e: Expr) -> Optional[tuple[Expr, Expr]]:
    """``(base, exponent)`` of a power or of ``exp(a)`` (``E**a``)."""
    if isinstance(e, Pow):
        return (e.base, e.exp)
    from sympy_extras.functions.elementary.exponential import Exp
    if isinstance(e, Exp):
        arg = e.args[0]
        if isinstance(arg, Expr):
            return (E_number(), arg)
    return None


def E_number() -> Expr:
    from .numbers import E
    return E


# PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_subs (the commutative cases)
def power_subs(base: Expr, exponent: Expr, old: Expr, new: Expr, old_is_power: Optional[bool] = None) -> Optional[Expr]:
    """``(base**exponent).subs(old, new)`` with SymPy's rules for powers:
    ``x**4`` with ``x**2 -> y`` is ``y**2``, ``exp(3*x + 1)`` with
    ``exp(x) -> y`` is ``E*y**3``; ``None`` for the plain substitution in
    the arguments."""
    from sympy_extras.functions.elementary.exponential import Exp, log
    if old == base:
        return pow(new, _as_expr(exponent._subs(old, new)))
    view = _power_view(old) if old_is_power is not False else None
    if view is not None and (isinstance(old, Pow) or base == E_number()):
        ob, oe = view
        if exponent == oe:
            l = log(base, ob)
            if isinstance(l, Number):
                return pow(new, l)
        if base == ob:
            if not isinstance(exponent, Add):
                c1, t1 = _independent_of_symbols(exponent)
                c2, t2 = _independent_of_symbols(oe)
                if t1 == t2:
                    ok, p = _combines(c1, c2, old)
                    if ok:
                        return pow(new, p)
            else:
                c2n, t2n = oe.as_coeff_mul()
                new_l: list[Expr] = []
                others: list[Expr] = []
                for a in exponent.args:
                    if not isinstance(a, Expr):
                        return None
                    newa = _as_expr(a._subs(old, new))
                    c1n, t1n = newa.as_coeff_mul()
                    if t1n == t2n:
                        ok, p = _combines(c1n, c2n, old)
                        if ok:
                            new_l.append(pow(new, p))
                            continue
                    others.append(newa)
                if new_l:
                    expo = add(*others)
                    new_l.append(Pow(base, expo) if not (isinstance(expo, Integer) and expo.p == 1) else base)
                    return mul(*new_l)
    old_view = _power_view(old)
    if (isinstance(old, Exp) or (old_view is not None and old_view[0] == E_number())) and old_view is not None \
            and exponent.is_extended_real is True and base.is_positive is True:
        c1, t1 = _independent_of_symbols(old_view[1])
        c2, t2 = _independent_of_symbols(mul(exponent, log(base)))
        if t1 == t2:
            ok, p = _combines(c1, c2, old)
            if ok:
                return pow(new, p)
    return None


def as_expr_basic(b: Basic) -> Expr:
    """``b`` as an expression (the result of ``replace``)."""
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b
