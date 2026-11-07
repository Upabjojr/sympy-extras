"""The exponential and the logarithm on the typed core.

``Exp(x)`` and ``Log(x)`` are the unevaluated nodes, ``exp`` and ``log``
evaluate, as ``Add`` and ``add`` do in :mod:`sympy_extras.core`. The
evaluation rules, derivatives and assumptions are ported from SymPy's
``sympy.functions.elementary.exponential`` (``exp.eval``, ``log.eval``
and their ``_eval_is_*`` methods), restricted to what the typed core
provides; the rules which need ``logcombine``, ``re``/``im``,
``AccumBounds``, matrices or set expressions are not ported yet and leave
the node unevaluated:

* ``exp(oo*z)`` for a number ``z`` which is not real (SymPy decides it
  by the signs of ``re(z)`` and ``im(z)``);
* ``exp(c*log(x)*...)`` when the log factor is only found by
  ``logcombine`` (``exp(log(x) + log(y))`` is found term by term);
* ``log(exp(z))`` for a number ``z`` which is not real (SymPy reduces
  the imaginary part into ``(-pi, pi]``);
* ``log`` of algebraic numbers ``r + i*I`` through the arctangent table.

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

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.kind import Kind, NumberKind
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, E, Exp1, I, Infinity, Integer, NaN, NegativeInfinity, Rational,
                                       nan, number_from_fraction, oo, pi, zoo)
from sympy_extras.core.power import pow

__all__ = ['Exp', 'Log', 'exp', 'log']


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

    function_name = 'exp'
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

    def fdiff(self, argindex: int) -> Expr:
        return self

    @property
    def kind(self) -> Kind:
        return NumberKind

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.exp(values[0]))

    def as_base_exp(self) -> tuple[Expr, Expr]:
        return (E, self.args[0])

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
        return None

    def fdiff(self, argindex: int) -> Expr:
        return pow(self.args[0], Integer(-1))

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
