"""The exponential and the logarithm on the typed core.

``Exp(x)`` and ``Log(x)`` are the unevaluated nodes, ``exp`` and ``log``
evaluate, as ``Add`` and ``add`` do in :mod:`sympy_extras.core`. The
evaluation rules, derivatives and assumptions are ported from SymPy's
``sympy.functions.elementary.exponential`` (``exp.eval``, ``log.eval``
and their ``_eval_is_*`` methods), restricted to what the typed core
provides; the rules which need ``logcombine``, ``re``/``im``,
matrices or set expressions are not ported yet and leave
the node unevaluated:

* ``exp(oo*z)`` for a number ``z`` which is not real (SymPy decides it
  by the signs of ``re(z)`` and ``im(z)``);
* ``exp(c*log(x)*...)`` when the log factor is only found by
  ``logcombine`` (``exp(log(x) + log(y))`` is found term by term);
* ``log(exp(z))`` for a number ``z`` which is not real (SymPy reduces
  the imaginary part into ``(-pi, pi]``);
* ``log`` of algebraic numbers ``r + i*I`` with ``r != 0`` through the
  arctangent table (the purely imaginary ones are ported).

Examples
========

>>> from sympy_extras.core import Symbol, Rational, I, pi, E
>>> from sympy_extras.functions.elementary.exponential import exp, log, Exp
>>> x, y = Symbol('x'), Symbol('y')
>>> r = Symbol('r', real=True)
>>> exp(0), exp(1), exp(log(x)), exp(2*log(x)), exp(x + log(y))
(1, E, x, x**2, y*exp(x))
>>> exp(3*pi*I/2), exp(Rational(7, 3)*pi*I)
(-I, exp(I*pi/3))
>>> log(1), log(E), log(-3), log(3*I), log(Rational(1, 7)), log(exp(r))
(0, 1, log(3) + I*pi, log(3) + I*pi/2, -log(7), r)
>>> log(24, 2), log(Rational(1, 8), 2)
(log(3)/log(2) + 3, -3)
>>> Exp(x).diff(x), log(x).diff(x), exp(r).is_positive, log(x).is_real
(exp(x), 1/x, True, None)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes Exp/Log with evaluating functions exp/log,
# strict typing, the subset of the rules the typed core supports.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence, Union

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, ExpandHints, expand_mul
from sympy_extras.core.kind import Kind, NumberKind
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, E, Exp1, I, Infinity, Integer, NaN, NegativeInfinity, Rational,
                                       nan, number_from_fraction, oo, pi, zoo)
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol

__all__ = ['Exp', 'Log', 'LambertW', 'ExpPolar', 'exp', 'log', 'ln', 'lambertw', 'exp_polar']


def _coefficient_of_pi_i(arg: Expr) -> Optional[Expr]:
    """``c`` with ``arg = c*pi*I``, ``c`` free of ``pi`` and ``I``."""
    return arg.as_coefficient(mul(pi, I))


def _multiplicity(base: int, n: int) -> int:
    """The largest ``k`` with ``base**k`` dividing ``n`` (``base > 1``,
    ``n != 0``)."""
    k = 0
    while n % base == 0:
        n //= base
        k += 1
    return k


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp
class Exp(DefinedFunction):
    """The exponential function ``exp(x)``, unevaluated (``exp`` evaluates).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, E
    >>> from sympy_extras.functions.elementary.exponential import Exp
    >>> x = Symbol('x')
    >>> Exp(x).as_base_exp(), Exp(-x).as_numer_denom()
    ((E, x), (1, exp(x)))
    """

    unbranched = True

    function_name = 'exp'
    # PORTED FROM SYMPY: _singularities
    singularities: Union[None, bool, tuple[Expr, ...]] = (zoo,)
    nargs = frozenset({1})

    @property
    def exp(self) -> Expr:
        """The argument (``exp(x)`` is ``E**x``)."""
        return self.args[0]

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        if isinstance(arg, NaN):
            return nan
        if isinstance(arg, Rational) and arg.p == 0:
            return Integer(1)
        if isinstance(arg, Rational) and arg == 1:
            return E
        if isinstance(arg, Infinity):
            return oo
        if isinstance(arg, NegativeInfinity):
            return Integer(0)
        from sympy_extras.core.accumbounds import AccumBounds, accum_bounds
        if isinstance(arg, AccumBounds):
            # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp.eval (AccumBounds)
            return accum_bounds(exp(arg.min), exp(arg.max))
        if isinstance(arg, ComplexInfinity):
            return nan
        if isinstance(arg, Log):
            return arg.args[0]
        if isinstance(arg, Mul):
            coeff = _coefficient_of_pi_i(arg)
            if coeff is not None:
                twice = mul(Integer(2), coeff)
                if twice.is_integer:
                    if coeff.is_even:
                        return Integer(1)
                    if coeff.is_odd:
                        return Integer(-1)
                    shifted = add(coeff, Rational(1, 2))
                    if shifted.is_even:
                        return mul(Integer(-1), I)
                    if shifted.is_odd:
                        return I
                elif isinstance(coeff, Rational):
                    reduced = coeff.as_fraction() % 2  # into [0, 2)
                    if reduced > 1:  # into (-1, 1]
                        reduced -= 2
                    if reduced != coeff.as_fraction():
                        return cls.apply(mul(number_from_fraction(reduced), pi, I))
            # a single log factor: exp(c*log(x)) is x**c
            number, terms = arg.as_coeff_Mul()
            if isinstance(number, (Infinity, NegativeInfinity)):
                # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp.eval (oo times a number)
                if terms.is_number:
                    from .complexes import im, re
                    if isinstance(number, NegativeInfinity):
                        terms = mul(Integer(-1), terms)
                    real = re(terms)
                    if real.is_zero and not (isinstance(terms, Rational) and terms.p == 0):
                        return nan
                    if real.is_positive and not (isinstance(im(terms), Rational) and im(terms) == 0):
                        return zoo
                    if real.is_negative:
                        return Integer(0)
                return None
            coefficients: list[Expr] = [number]
            log_term: Optional[Expr] = None
            for term in (terms.args if isinstance(terms, Mul) else (terms,)):
                if isinstance(term, Log):
                    if log_term is not None:
                        return None
                    log_term = term.args[0]
                elif term.is_comparable:
                    coefficients.append(term)
                else:
                    return None
            return pow(log_term, mul(*coefficients)) if log_term is not None else None
        if isinstance(arg, Add):
            out: list[Expr] = []
            kept: list[Expr] = []
            changed = False
            for a in arg.args:
                if isinstance(a, Rational) and a == 1:
                    kept.append(a)
                    continue
                value = cls.apply(a)
                if isinstance(value, Exp):
                    if value.args[0] != a:
                        kept.append(value.args[0])
                        changed = True
                    else:
                        kept.append(a)
                else:
                    out.append(value)
            if out or changed:
                if not kept:
                    return mul(*out)
                return mul(*out, Exp(add(*kept)))
        if arg.is_zero:
            return Integer(1)
        return None

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:ExpBase._eval_conjugate
    def _eval_conjugate(self) -> Optional[Expr]:
        from .complexes import conjugate
        return exp(conjugate(self.args[0]))

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp.as_real_imag
    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from .complexes import as_real_imag
        from .trigonometric import cos, sin
        parts = as_real_imag(self.args[0])
        if parts is None:
            return None
        real, imaginary = parts
        if deep:
            from sympy_extras.core.function import expand_with_hints
            real, imaginary = expand_with_hints(real, hints), expand_with_hints(imaginary, hints)
        return (mul(exp(real), cos(imaginary)), mul(exp(real), sin(imaginary)))

    def fdiff(self, argindex: int) -> Expr:
        return self

    @property
    def kind(self) -> Kind:
        return NumberKind

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.exp(values[0]))

    def as_base_exp(self) -> tuple[Expr, Expr]:
        return (E, self.args[0])

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:ExpBase._eval_transpose, _eval_adjoint
    def _eval_transpose_node(self) -> Optional[Expr]:
        return exp(self.args[0].transpose())

    def _eval_adjoint_node(self) -> Optional[Expr]:
        return exp(self.args[0].adjoint())

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp.taylor_term
    def taylor_term(self, n: int, x: Expr, previous: Optional[Expr] = None) -> Expr:
        if n < 0:
            return Integer(0)
        if n == 0:
            return Integer(1)
        if previous is not None:
            return mul(previous, x, pow(Integer(n), Integer(-1)))
        from sympy_extras.functions.combinatorial.factorials import factorial
        return mul(pow(x, Integer(n)), pow(factorial(Integer(n)), Integer(-1)))

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp._taylor
    def _taylor(self, x: Symbol, n: int) -> Expr:
        terms: list[Expr] = []
        g: Optional[Expr] = None
        for i in range(n):
            g = self.taylor_term(i, self.args[0], g)
            g = g.nseries(x, n=n)
            terms.append(g.removeO())
        return add(*terms)

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        from sympy_extras.core.add import _ceiling_int
        from sympy_extras.core.coreerrors import PoleError
        from sympy_extras.core.function import expand_complex
        from sympy_extras.core.order import order
        from sympy_extras.core.series import _less
        from sympy_extras.core.symbol import Dummy, Wild
        from sympy_extras.functions.elementary.complexes import Sign
        from sympy_extras.simplify.powsimp import powsimp
        arg = self.args[0]
        arg_series = arg._eval_nseries(x, n, logx, Integer(0))
        if arg_series.is_Order:
            return add(Integer(1), arg_series)
        arg0 = arg_series.removeO().limit(x, Integer(0))
        if isinstance(arg0, NegativeInfinity):
            return order(pow(x, Integer(n)), x)
        if isinstance(arg0, Infinity):
            return self
        if arg0.is_infinite is True:
            raise PoleError("Cannot expand %s around 0" % (self,))
        # undecided sign terms in arg0
        if any(isinstance(a, Sign) for a in arg0.args):
            return self
        t = Dummy("t")
        nterms = n
        cf: Optional[Expr]
        try:
            cf = order(arg.as_leading_term(x, logx=logx), x).getn()
        except (NotImplementedError, PoleError):
            cf = Integer(0)
        positive_cf = cf is not None and not (cf.is_Number and cf == 0) and _less(Integer(0), cf)
        if cf is not None and positive_cf:
            nterms = _ceiling_int(mul(Integer(n), pow(cf, Integer(-1))))
        exp_series = Exp(t)._taylor(t, nterms)
        shift = add(arg_series, mul(Integer(-1), arg0))
        r = mul(exp(arg0), exp_series.subs(t, shift))
        if logx is not None:
            if r.subs(logx, log(x)) == self:
                return r
        elif r == self:
            return r
        if cf is not None and positive_cf and _less(Integer(1), cf):
            r = add(r, mul(order(pow(shift, Integer(n)), x),
                           pow(x, mul(Integer(-1), add(cf, Integer(-1)), Integer(n)))))
        else:
            r = add(r, order(pow(shift, Integer(n)), x))
        r = r.expand()
        r = powsimp(r, deep=True, combine='exp')
        # powsimp may introduce unexpanded (-1)**Rational (SymPy's PR #17201)
        w = Wild('w', properties=[lambda e: isinstance(e, Rational) and e.q in (3, 4, 6)])
        replaced = r.replace(pow(Integer(-1), w), expand_complex(pow(Integer(-1), w)))
        return replaced if isinstance(replaced, Expr) else r

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        from sympy_extras.core.coreerrors import PoleError
        from sympy_extras.core.accumbounds import AccumBounds
        from sympy_extras.functions.elementary.complexes import re
        arg = self.args[0].cancel().as_leading_term(x, logx=logx)
        arg0 = arg.subs(x, Integer(0))
        if isinstance(arg, NaN):
            return nan
        if isinstance(arg0, AccumBounds):
            # a corner case of SymPy's (its test_issue_18473): arg0 is 0,
            # AccumBounds(-oo, 0) or AccumBounds(-oo, oo)
            if re(cdir).is_negative is True:
                return exp(-arg0)
            return exp(arg0)
        if isinstance(arg0, NaN):
            arg0 = arg.limit(x, Integer(0))
        if arg0.is_infinite is False:
            return exp(arg0)
        raise PoleError("Cannot expand %s around 0" % (self,))

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp._eval_subs
    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        """``exp(a)`` as ``E**a`` for SymPy's power substitution
        (``exp(2*x)`` with ``exp(x) -> y`` is ``y**2``)."""
        from sympy_extras.core.power import Pow, power_subs
        if not (isinstance(old, Expr) and isinstance(new, Expr)):
            return None
        target: Expr = old
        if isinstance(old, Pow):
            rewritten = exp(mul(old.exp, log(old.base)))
            if not isinstance(rewritten, Exp):
                return None
            target = rewritten
        if isinstance(target, Exp) or target == E:
            return power_subs(E, self.args[0], target, new)
        return None

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        arg = self.args[0]
        if arg.is_negative is True or arg.could_extract_minus_sign():
            return (Integer(1), exp(mul(Integer(-1), arg)))
        return (self, Integer(1))

    def _eval_facts(self) -> dict[str, bool]:
        arg = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        # exp is zero only at -oo (SymPy's _eval_is_zero)
        facts['zero'] = isinstance(self.args[0], NegativeInfinity)
        if arg.get('extended_real'):
            facts['extended_real'] = True
            if not isinstance(self.args[0], NegativeInfinity):
                facts['extended_positive'] = True
        if arg.get('real'):
            facts.update(real=True, positive=True)
        if arg.get('finite'):
            facts.update(finite=True, zero=False)
        if arg.get('complex'):
            facts['complex'] = True
        # Lindemann-Weierstrass: exp of a nonzero algebraic number is
        # transcendental; exp(q*pi*I) for a rational q is a root of unity
        if arg.get('zero') is False:
            if arg.get('algebraic'):
                facts['algebraic'] = False
            if arg.get('rational'):
                facts['rational'] = False
        coefficient = _coefficient_of_pi_i(self.args[0]) if isinstance(self.args[0], Mul) else None
        if coefficient is not None and coefficient.is_rational:
            facts['algebraic'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log
class Log(DefinedFunction):
    """The natural logarithm ``log(x)`` (principal branch), unevaluated
    (``log`` evaluates; ``log(x, b)`` is ``log(x)/log(b)``).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions.elementary.exponential import Log
    >>> p = Symbol('p', positive=True)
    >>> Log(p).is_real, Log(p + 1).is_positive
    (True, True)
    """

    function_name = 'log'
    # PORTED FROM SYMPY: _singularities
    singularities: Union[None, bool, tuple[Expr, ...]] = (Integer(0), zoo)
    nargs = frozenset({1})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        if isinstance(arg, Rational):
            if arg.p == 0:
                return zoo
            if arg == 1:
                return Integer(0)
            if arg.p == 1:
                return mul(Integer(-1), cls.apply(Integer(arg.q)))
        if isinstance(arg, (Infinity, NegativeInfinity)):
            return oo
        if isinstance(arg, NaN):
            return nan
        if isinstance(arg, Exp) and arg.args[0].is_extended_real:
            return arg.args[0]
        if isinstance(arg, ExpPolar):
            # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log.eval (exp_polar)
            from sympy_extras.functions.elementary.complexes import unpolarify
            return unpolarify(arg.args[0])
        from sympy_extras.core.accumbounds import AccumBounds, accum_bounds
        if isinstance(arg, AccumBounds):
            # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log.eval (AccumBounds)
            if arg.min.is_positive is True:
                return accum_bounds(cls.apply(arg.min), cls.apply(arg.max))
            if arg.min.is_zero is True:
                return accum_bounds(-oo, cls.apply(arg.max))
            return nan
        if arg.is_number:
            if arg.is_negative:
                return add(mul(pi, I), cls.apply(mul(Integer(-1), arg)))
            if isinstance(arg, ComplexInfinity):
                return zoo
            if isinstance(arg, Exp1):
                return Integer(1)
        if arg.is_zero:
            return zoo
        if not isinstance(arg, Add):
            coeff = arg.as_coefficient(I)
            if coeff is not None:
                if isinstance(coeff, (Infinity, NegativeInfinity)):
                    return oo
                if isinstance(coeff, Rational):
                    if coeff.p >= 0:
                        return add(mul(pi, I, Rational(1, 2)), cls.apply(coeff))
                    return add(mul(Integer(-1), pi, I, Rational(1, 2)), cls.apply(mul(Integer(-1), coeff)))
        if arg.is_number and arg.is_algebraic:
            # arg = coeff*(r + i*I) with coeff > 0: the purely imaginary case
            # (SymPy's further case r != 0 uses the arctangent table, not ported)
            coefficient, rest = arg.as_independent(I, as_Add=False)
            if coefficient.is_negative:
                coefficient, rest = mul(Integer(-1), coefficient), mul(Integer(-1), rest)
            rest = expand_mul(rest)
            real_part, imaginary_term = rest.as_independent(I, as_Add=True)
            imaginary_part = imaginary_term.as_coefficient(I)
            if coefficient.is_real and imaginary_part is not None and not _is_zero(imaginary_part) \
                    and imaginary_part.is_real and real_part.is_real:
                if real_part.is_zero:
                    if imaginary_part.is_positive:
                        return add(mul(pi, I, Rational(1, 2)), cls.apply(mul(coefficient, imaginary_part)))
                    if imaginary_part.is_negative:
                        return add(mul(Integer(-1), pi, I, Rational(1, 2)),
                                   cls.apply(mul(coefficient, mul(Integer(-1), imaginary_part))))
                else:
                    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log.eval
                    # (the arguments with rational multiples of pi)
                    return _log_of_angle(cls, coefficient, rest, real_part, imaginary_part)
        return None

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log.as_real_imag
    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log.as_real_imag
        from sympy_extras.core.function import expand_with_hints
        from .complexes import abs_, arg
        z = expand_with_hints(self.args[0], hints) if deep else self.args[0]
        modulus = abs_(z)
        if modulus == z:
            return (self, Integer(0))
        if hints is not None and hints.log:
            return (expand_with_hints(log(modulus), hints, complex=False), arg(z))
        return (log(modulus), arg(z))

    def fdiff(self, argindex: int) -> Expr:
        return pow(self.args[0], Integer(-1))

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log.taylor_term
    def taylor_term(self, n: int, x: Expr, previous: Optional[Expr] = None) -> Expr:
        """The ``n``-th term of the series of ``log(1 + x)``."""
        from sympy_extras.simplify.powsimp import powsimp
        if n < 0:
            return Integer(0)
        if n == 0:
            return x
        if previous is not None:
            return powsimp(mul(Integer(-n), previous, x, pow(Integer(n + 1), Integer(-1))), deep=True, combine='exp')
        return mul(Integer(1 - 2 * (n % 2)), pow(x, Integer(n + 1)), pow(Integer(n + 1), Integer(-1)))

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        from sympy_extras.core.add import _ceiling_int
        from sympy_extras.core.coreerrors import PoleError
        from sympy_extras.core.expr import factors_of
        from sympy_extras.core.function import _expand_logs
        from sympy_extras.core.order import Order, order
        from sympy_extras.core.series import _less
        from sympy_extras.core.symbol import Dummy, Wild
        from sympy_extras.functions.elementary.complexes import im
        from sympy_extras.functions.special.delta_functions import heaviside
        from sympy_extras.simplify.simplify import logcombine
        arg = self.args[0]
        if arg == x:
            return log(x) if logx is None else logx
        t = Dummy('t', positive=True)
        if cdir.is_Number and cdir == 0:
            cdir = Integer(1)
        z = arg.subs(x, mul(cdir, t))
        wk, wl = Wild("k"), Wild("l")
        matched = z.match(mul(wk, pow(t, wl)))
        if matched is not None and wk in matched and wl in matched:
            k, l = _expr_of(matched[wk]), _expr_of(matched[wl])
            if l != Integer(0) and not l.has(t) and not k.has(t):
                r = mul(l, log(x) if logx is None else logx)
                return add(r, log(k), mul(Integer(-1), l, log(cdir)))

        def coeff_exp(term: Expr) -> tuple[Expr, Expr]:
            coeff: Expr = Integer(1)
            exponent: Expr = Integer(0)
            for factor in factors_of(term):
                if factor.has(t):
                    base, exponent = factor.as_base_exp()
                    if base != t:
                        try:
                            return term.leadterm(t)
                        except ValueError:
                            return term, Integer(0)
                else:
                    coeff = mul(coeff, factor)
            return coeff, exponent

        nn: Expr = Integer(n)
        a: Expr
        b: Expr
        try:
            a, b = z.leadterm(t, logx=logx, cdir=1)
        except (ValueError, NotImplementedError, PoleError):
            m = n
            s = z._eval_nseries(t, m, logx, Integer(1))
            while s.is_Order:
                m += 1
                s = z._eval_nseries(t, m, logx, Integer(1))
            nn = Integer(m)
            try:
                a, b = s.removeO().leadterm(t, cdir=1)
            except ValueError:
                a, b = s.removeO().as_leading_term(t, cdir=1), Integer(0)
        # the operators, as SymPy (the accumulation bounds absorb real factors)
        p = (z / (a * t**b) - 1).cancel()
        p = p._eval_nseries(t, _ceiling_int(nn), logx, Integer(1))
        if p.has(Exp):
            p = logcombine(p)
        if isinstance(p, Order):
            found = p.getn()
            if found is not None:
                nn = found
        _, d = coeff_exp(p)
        logx_ = log(x) if logx is None else logx
        if d.is_positive is not True:
            res = add(log(a), mul(Integer(-1), b, log(cdir)), mul(b, logx_))
            expr = _expand_logs(self, True, True, False)
            if not a.could_extract_minus_sign() and logx_.could_extract_minus_sign():
                _res = _expand_logs(res.subs(mul(Integer(-1), logx_), mul(Integer(-1), log(x))), True, True, False)
            else:
                _res = _expand_logs(res.subs(logx_, log(x)), True, True, False)
            if _res == expr:
                return res
            return add(res, order(pow(x, nn), x))

        def product(d1: dict[Expr, Expr], d2: dict[Expr, Expr]) -> dict[Expr, Expr]:
            res: dict[Expr, Expr] = {}
            for e1 in d1:
                for e2 in d2:
                    ex = add(e1, e2)
                    if _less(ex, nn):
                        res[ex] = add(res.get(ex, Integer(0)), mul(d1[e1], d2[e2]))
            return res

        pterms: dict[Expr, Expr] = {}
        removed = p.removeO()
        for term in (removed.args if isinstance(removed, Add) else (removed,)):
            if not isinstance(term, Expr):
                continue
            co1, e1 = coeff_exp(term)
            pterms[e1] = add(pterms.get(e1, Integer(0)), co1)
        kk = 1
        terms: dict[Expr, Expr] = {}
        pk = pterms
        while _less(mul(Integer(kk), d), nn):
            coeff = mul(Integer(-1), pow(Integer(-1), Integer(kk)), pow(Integer(kk), Integer(-1)))
            for ex in pk:
                terms[ex] = add(terms.get(ex, Integer(0)), mul(coeff, pk[ex]))
            pk = product(pk, pterms)
            kk += 1
        res = add(log(a), mul(Integer(-1), b, log(cdir)), mul(b, logx_))
        for ex in terms:
            res = add(res, mul(terms[ex].cancel(), pow(t, ex)))
        if a.is_negative is True and im(z) != Integer(0):
            res = add(res, _branch_jump(z, t))
        res = res.subs(t, mul(x, pow(cdir, Integer(-1))))
        return add(res, order(pow(x, nn), x))

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        from sympy_extras.core.coreerrors import PoleError
        from sympy_extras.core.symbol import Dummy
        from sympy_extras.functions.elementary.complexes import im
        # the steps of SymPy's PR #23592
        arg0 = self.args[0].together()
        t = Dummy('t', positive=True)
        if cdir.is_Number and cdir == 0:
            cdir = Integer(1)
        z = arg0.subs(x, mul(cdir, t))
        try:
            c, e = z.leadterm(t, logx=logx, cdir=1)
        except ValueError:
            return log(arg0.as_leading_term(x, logx=logx, cdir=cdir))
        if c.has(t):
            c = c.subs(t, mul(x, pow(cdir, Integer(-1))))
            if e != Integer(0):
                raise PoleError("Cannot expand %s around 0" % (self,))
            return log(c)
        if c == Integer(1) and e == Integer(0):
            return add(arg0, Integer(-1)).as_leading_term(x, logx=logx)
        res = add(log(c), mul(Integer(-1), e, log(cdir)), mul(e, log(x) if logx is None else logx))
        if c.is_negative is True and im(z) != Integer(0):
            res = add(res, _branch_jump(z, t))
        return res

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[0] == 0:
            return None
        return mpmath.mpc(mpmath.log(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        arg = self.args[0]
        a = facts_of(arg)
        facts: dict[str, bool] = {'commutative': True}
        if a.get('extended_positive') is True:
            facts['extended_real'] = True
        elif a.get('extended_positive') is False:
            facts['extended_real'] = False
        if a.get('positive') and a.get('finite'):
            facts['real'] = True
        if a.get('complex') and a.get('zero') is False:
            facts['complex'] = True
        if a.get('zero'):
            facts['finite'] = False
        elif a.get('finite') and a.get('zero') is False:
            facts['finite'] = True
        shifted = facts_of(add(arg, Integer(-1)))
        if shifted.get('extended_positive') is not None:
            facts['extended_positive'] = shifted['extended_positive']
        if shifted.get('zero') is not None:
            facts['zero'] = shifted['zero']
        if shifted.get('extended_nonnegative') is not None:
            facts['extended_nonnegative'] = shifted['extended_nonnegative']
        # log(1) = 0 is rational; the log of another rational (algebraic)
        # number is irrational (transcendental)
        if shifted.get('zero') is True:
            facts.update(rational=True, algebraic=True)
        elif shifted.get('zero') is False:
            if a.get('rational'):
                facts['rational'] = False
            if a.get('algebraic'):
                facts['algebraic'] = False
        return facts


def _is_zero(e: Expr) -> bool:
    return e.is_Number and e == 0


_LOG_ATAN: dict[Expr, Expr] = {}


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:_log_atan_table
def _log_atan_table() -> dict[Expr, Expr]:
    """``tan(a) -> a`` for the angles of the first quadrant with algebraic
    tangents."""
    if not _LOG_ATAN:
        from sympy_extras.core.power import sqrt
        s2, s3, s5 = sqrt(Integer(2)), sqrt(Integer(3)), sqrt(Integer(5))
        one = Integer(1)
        entries: list[tuple[Expr, Expr]] = [
            (s3, pi / 3), (one, pi / 4), (sqrt(5 - 2 * s5), pi / 5),
            (s2 * sqrt(5 - s5) / (1 + s5), pi / 5), (sqrt(5 + 2 * s5), pi * Rational(2, 5)),
            (s2 * sqrt(s5 + 5) / (-1 + s5), pi * Rational(2, 5)), (s3 / 3, pi / 6), (s2 - 1, pi / 8),
            (sqrt(2 - s2) / sqrt(s2 + 2), pi / 8), (s2 + 1, pi * Rational(3, 8)),
            (sqrt(s2 + 2) / sqrt(2 - s2), pi * Rational(3, 8)), (sqrt(1 - 2 * s5 / 5), pi / 10),
            ((-s2 + sqrt(Integer(10))) / (2 * sqrt(s5 + 5)), pi / 10), (sqrt(1 + 2 * s5 / 5), pi * Rational(3, 10)),
            ((s2 + sqrt(Integer(10))) / (2 * sqrt(5 - s5)), pi * Rational(3, 10)), (2 - s3, pi / 12),
            ((-1 + s3) / (1 + s3), pi / 12), (2 + s3, pi * Rational(5, 12)),
            ((1 + s3) / (-1 + s3), pi * Rational(5, 12))]
        _LOG_ATAN.update(entries)
    return _LOG_ATAN


def _log_of_angle(cls: type[Log], coefficient: Expr, rest: Expr, real_part: Expr,
                  imaginary_part: Expr) -> Optional[Expr]:
    """``log(coefficient*(r + i*I)) = log(|...|) + I*a`` when ``i/r`` is the
    tangent of a rational multiple ``a`` of pi in the table."""
    from sympy_extras.functions.elementary.complexes import abs_
    from sympy_extras.simplify.ratsimp import ratsimp
    t = (imaginary_part / real_part).cancel()
    t1 = (-t).cancel()
    table = _log_atan_table()
    if t in table:
        modulus = ratsimp(coefficient * abs_(rest))
        if real_part.is_positive is True:
            return cls.apply(modulus) + I * table[t]
        return cls.apply(modulus) + I * (table[t] - pi)
    if t1 in table:
        modulus = ratsimp(coefficient * abs_(rest))
        if real_part.is_positive is True:
            return cls.apply(modulus) + I * (-table[t1])
        return cls.apply(modulus) + I * (pi - table[t1])
    return None


def _expr_of(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log._eval_nseries (the branch cut)
def _branch_jump(z: Expr, t: Symbol) -> Expr:
    """-2*I*pi*Heaviside(-im(c), 0) for the first non-real term c*t**e
    among the first five of the series of z (the jump of log across
    its branch cut), 0 when they are real."""
    from sympy_extras.core.numbers import pi
    from sympy_extras.functions.elementary.complexes import im
    from sympy_extras.functions.special.delta_functions import heaviside
    term: Expr = Integer(0)
    i = 0
    for i, term in enumerate(z.lseries(t)):
        if term.is_real is not True or i == 5:
            break
    if i < 5:
        coeff, _ = term.as_coeff_exponent(t)
        return mul(Integer(-2), I, pi, heaviside(mul(Integer(-1), im(coeff)), Integer(0)))
    return Integer(0)


def exp(x: ExprLike) -> Expr:
    """The exponential of ``x``, evaluated."""
    return Exp.apply(x)


def log(x: ExprLike, base: Optional[ExprLike] = None) -> Expr:
    """The natural logarithm of ``x``, evaluated; ``log(x, b)`` is
    ``log(x)/log(b)``, with the powers of an integer base taken out of a
    rational argument (``log(24, 2) = 3 + log(3)/log(2)``)."""
    arg = to_expr(x)
    if base is None:
        return Log.apply(arg)
    b = to_expr(base)
    if isinstance(b, Rational) and b == 1:
        return nan if isinstance(arg, Rational) and arg == 1 else zoo
    if isinstance(b, Exp1):
        return Log.apply(arg)
    if isinstance(b, Rational) and b.q == 1 and b.p > 1 and isinstance(arg, Rational) and arg.p != 0:
        n = _multiplicity(b.p, abs(arg.p)) - _multiplicity(b.p, arg.q)
        if n:
            rest = mul(arg, pow(b, Integer(-n)))
            return add(Integer(n), mul(Log.apply(rest), pow(Log.apply(b), Integer(-1))))
    return mul(Log.apply(arg), pow(Log.apply(b), Integer(-1)))


def _fuzzy_not(v: Optional[bool]) -> Optional[bool]:
    return None if v is None else not v


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:LambertW
class LambertW(DefinedFunction):
    """The Lambert W function ``LambertW(x)`` (principal branch) and its
    branch ``LambertW(x, k)``, unevaluated (``lambertw`` evaluates): the
    inverse of ``w*exp(w)``.

    Examples
    ========

    >>> from sympy_extras.core import E, Integer, Symbol, pi
    >>> from sympy_extras.functions.elementary.exponential import lambertw, log
    >>> x = Symbol('x')
    >>> lambertw(E), lambertw(-1/E), lambertw(-pi/2), lambertw(0, 1), lambertw(x, 0), lambertw(-2*E**-2, -1)
    (1, -1, I*pi/2, -oo, LambertW(x), -2)
    >>> lambertw(x).diff(x)
    LambertW(x)/(x*(LambertW(x) + 1))
    """

    function_name = 'LambertW'
    # PORTED FROM SYMPY: _singularities
    singularities: Union[None, bool, tuple[Expr, ...]] = (-pow(E, Integer(-1)), zoo)
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        x = args[0]
        k: Expr = args[1] if len(args) > 1 else Integer(0)
        if len(args) > 1 and k.is_zero:
            return lambertw(x)
        minus_one_over_e = mul(Integer(-1), pow(E, Integer(-1)))
        if k.is_zero:
            if x.is_zero:
                return Integer(0)
            if isinstance(x, Exp1):
                return Integer(1)
            if x == minus_one_over_e:
                return Integer(-1)
            log2 = log(Integer(2))
            if x == mul(Rational(-1, 2), log2):
                return mul(Integer(-1), log2)
            if x == mul(Integer(2), log2):
                return log2
            if x == mul(Rational(-1, 2), pi):
                return mul(I, pi, Rational(1, 2))
            if x == exp(add(Integer(1), E)):
                return E
            if isinstance(x, Infinity):
                return x
        if _fuzzy_not(k.is_zero):
            if x.is_zero:
                return NegativeInfinity()
        if k == Integer(-1):
            if x == mul(Rational(-1, 2), pi):
                return mul(Rational(-1, 2), I, pi)
            if x == minus_one_over_e:
                return Integer(-1)
            if x == mul(Integer(-2), exp(Integer(-2))):
                return Integer(-2)
        return None

    def fdiff(self, argindex: int) -> Expr:
        if argindex == 1:
            w = self
            return mul(w, pow(mul(self.args[0], add(Integer(1), w)), Integer(-1)))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        x = values[0]
        k = values[1] if len(values) > 1 else mpmath.mpc(0)
        if mpmath.im(k) != 0 or mpmath.re(k) != int(mpmath.re(k)):
            return None
        if x == 0 and k != 0:
            return None
        return mpmath.mpc(mpmath.lambertw(x, int(mpmath.re(k))))

    def _eval_facts(self) -> dict[str, bool]:
        x = self.args[0]
        k: Expr = self.args[1] if len(self.args) > 1 else Integer(0)
        facts: dict[str, bool] = {'commutative': True}
        shifted = add(x, pow(E, Integer(-1)))
        real: Optional[bool] = None
        if k.is_zero:
            if shifted.is_positive:
                real = True
            elif shifted.is_nonpositive:
                real = False
        elif add(k, Integer(1)).is_zero:
            if x.is_negative and shifted.is_positive:
                real = True
            elif x.is_nonpositive or shifted.is_nonnegative:
                real = False
        elif _fuzzy_not(k.is_zero) and _fuzzy_not(add(k, Integer(1)).is_zero):
            if x.is_extended_real:
                real = False
        if real is not None:
            facts['extended_real'] = real
        a = facts_of(x)
        if a.get('finite') is not None:
            facts['finite'] = a['finite']
        if len(self.args) == 1:
            if a.get('zero') is not None:
                facts['zero'] = a['zero']
        else:
            kz = facts_of(k).get('zero')
            if a.get('zero') is False or kz is False:
                facts['zero'] = False
            elif a.get('zero') and kz:
                facts['zero'] = True
        if a.get('zero') is False and a.get('algebraic'):
            facts['algebraic'] = False
        return facts


def lambertw(x: ExprLike, k: Optional[ExprLike] = None) -> Expr:
    """``LambertW(x)`` or the branch ``LambertW(x, k)``, evaluated."""
    if k is None:
        return LambertW.apply(to_expr(x))
    return LambertW.apply(to_expr(x), to_expr(k))


#: SymPy's alias of ``log``
ln = log



# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp_polar
class ExpPolar(DefinedFunction):
    """The exponential on the Riemann surface of the logarithm,
    ``exp_polar(z)``: it never evaluates (``exp_polar(2*I*pi)`` is not
    ``1``), its powers are ``exp_polar(z*b)``, its modulus is
    ``exp(re(z))``; numerically it is ``exp(z)`` for ``-pi < im(z) <= pi``."""

    function_name = 'exp_polar'
    # SymPy's exp_polar takes any number of arguments (nargs Naturals0)
    sympy_variadic = True
    # PORTED FROM SYMPY: _singularities
    singularities: Union[None, bool, tuple[Expr, ...]] = (zoo,)
    unbranched = True

    def _eval_power_node(self, exponent: Expr) -> Optional[Expr]:
        return exp_polar(mul(self.args[0], exponent))

    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp_polar.as_base_exp
    # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:ExpBase._eval_transpose, _eval_adjoint
    def _eval_transpose_node(self) -> Optional[Expr]:
        return exp_polar(self.args[0].transpose())

    def _eval_adjoint_node(self) -> Optional[Expr]:
        return exp_polar(self.args[0].adjoint())

    def as_base_exp(self) -> tuple[Expr, Expr]:
        """``(exp_polar(1), z)``, so that products collect the exponents
        (``exp_polar(1)*exp_polar(2)`` is ``exp_polar(3)``, ``exp_polar(x)*
        exp_polar(-x)`` is ``1``); ``(self, 1)`` for ``exp_polar(0)``."""
        if self.args[0].is_zero:
            return (self, Integer(1))
        return (ExpPolar(Integer(1)), self.args[0])

    # SymPy's exp_polar has no fdiff: its derivative stays unevaluated
    # (Derivative(exp_polar(x), x))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        z = values[0]
        if not -mpmath.pi < mpmath.im(z) <= mpmath.pi:
            return None
        value = mpmath.exp(z)
        if mpmath.im(z) > 0 and mpmath.im(value) < 0:
            return mpmath.mpc(mpmath.re(value))
        return mpmath.mpc(value)

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True, 'polar': True, 'zero': False}
        argument = facts_of(self.args[0])
        if argument.get('extended_real'):
            facts['extended_real'] = True
        if argument.get('real') and self.args[0].is_number:
            # e**r > 0 for a real number r (SymPy finds it numerically:
            # exp_polar(0).is_positive is True; the conditions of the
            # integration theorems ask whether a polar argument is positive)
            facts['positive'] = True
        return facts


def exp_polar(z: ExprLike) -> Expr:
    """``exp_polar(z)`` (never evaluates)."""
    return ExpPolar.apply(to_expr(z))
