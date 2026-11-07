"""The trigonometric functions on the typed core: ``Sin``, ``Cos``, ``Tan``,
``Cot``, ``Sec``, ``Csc`` (unevaluated nodes) and ``sin``, ``cos``, ``tan``,
``cot``, ``sec``, ``csc`` (evaluating).

The evaluation rules, derivatives and facts are ported from SymPy's
``sympy.functions.elementary.trigonometric``: odd and even symmetry, the
exchange with the hyperbolic functions on imaginary arguments, the exact
values at rational multiples of ``pi`` (SymPy's reductions to ``[0,
pi/2]``, its product formulas for the denominators 12, 15, 20, 24, 30,
40, 60, 120, the Chebyshev values at ``pi/3`` and ``pi/5``, the
half-angle square roots, the table of ``tan(k*pi/10)``), and the
integer and half-integer multiples of ``pi`` peeled off sums. Not ported
yet, the node stays unevaluated:

* ``sin(oo)``, ``cos(oo)``, ``tan(oo)`` (SymPy returns ``AccumBounds``);
* the compositions with the inverse functions (``sin(asin(x))``);
* ``Float`` multiples of ``pi`` (SymPy turns binary fractions into
  rationals).

Examples
========

>>> from sympy_extras.core import Symbol, Rational, I, pi
>>> from sympy_extras.functions.elementary.trigonometric import sin, cos, tan
>>> x = Symbol('x')
>>> sin(pi/6), cos(pi/4), tan(pi/3), cos(pi/5), sin(-x), cos(-x)
(1/2, sqrt(2)/2, sqrt(3), 1/4 + sqrt(5)/4, -sin(x), cos(x))
>>> sin(x + pi), cos(x + pi/2), sin(I*x), cos(I*x)
(-sin(x), -sin(x), I*sinh(x), cosh(x))
>>> sin(x).diff(x), tan(x).diff(x)
(cos(x), tan(x)**2 + 1)
>>> from sympy_extras.functions.elementary.trigonometric import cot, sec, csc
>>> tan(x + pi/2), cot(pi/6), sec(pi/3), csc(-x), sec(x + pi)
(-cot(x), sqrt(3), 2, -csc(x), -sec(x))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes Sin/Cos/Tan with evaluating functions,
# strict typing, the subset of the rules the typed core supports.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, Float, I, Infinity, Integer, NaN, NegativeInfinity, Rational,
                                       nan, number_from_fraction, pi, zoo)
from sympy_extras.core.power import pow, sqrt

__all__ = ['Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'pi_coeff',
           'peeloff_pi']

HALF = Fraction(1, 2)


def imaginary_coefficient(arg: Expr) -> Optional[Expr]:
    """``c`` with ``arg = c*I`` (not for floats)."""
    if isinstance(arg, Float):
        return None
    return arg.as_coefficient(I)


def _table2() -> dict[int, tuple[int, int]]:
    """The denominators whose cosines come from a product formula, with
    the two denominators of the factors."""
    return {12: (3, 4), 20: (4, 5), 30: (5, 6), 15: (6, 10), 24: (6, 8), 40: (8, 10), 60: (20, 30), 120: (40, 60)}


def _pi_part(term: Expr) -> Optional[Expr]:
    """The coefficient of ``pi`` in a term: ``1`` for ``pi``, the other
    factors of a product containing ``pi``, ``None`` otherwise."""
    if term == pi:
        return Integer(1)
    if isinstance(term, Mul) and pi in term.args:
        others = list(term.args)
        others.remove(pi)
        return mul(*others)
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:_peeloff_pi
def peeloff_pi(arg: Expr) -> tuple[Expr, Fraction]:
    """``(rest, m)`` with ``arg = rest + m*pi``, ``m`` a multiple of 1/2
    taken from the rational multiples of ``pi`` among the terms of the sum.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions.elementary.trigonometric import peeloff_pi
    >>> x, y = Symbol('x'), Symbol('y')
    >>> peeloff_pi(x + pi/2), peeloff_pi(x + 2*pi/3 + pi*y)
    ((x, Fraction(1, 2)), (x + pi*y + pi/6, Fraction(1, 2)))
    """
    coefficient = Fraction(0)
    rest: list[Expr] = []
    for term in (arg.args if isinstance(arg, Add) else (arg,)):
        k = _pi_part(term)
        if isinstance(k, Rational):
            coefficient += k.as_fraction()
        else:
            rest.append(term)
    if coefficient == 0:
        return arg, Fraction(0)
    m1 = coefficient % HALF
    m2 = coefficient - m1
    return add(*rest, mul(number_from_fraction(m1), pi)), m2


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:_pi_coeff
def pi_coeff(arg: Expr) -> Optional[Expr]:
    """When ``arg`` is a number times ``pi``, the number normalised into
    ``[0, 2]`` (an integer multiple of a symbol of known parity becomes
    ``0`` or ``1``, of unknown parity ``2``), else the coefficient of
    ``pi`` in a product, else ``None``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Rational, pi
    >>> from sympy_extras.functions.elementary.trigonometric import pi_coeff
    >>> x, n = Symbol('x'), Symbol('n', integer=True)
    >>> pi_coeff(3*x*pi), pi_coeff(11*pi/7), pi_coeff(-11*pi/7), pi_coeff(4*pi), pi_coeff(5*pi), pi_coeff(2*n*pi)
    (3*x, 11/7, 3/7, 0, 1, 2)
    >>> pi_coeff(2 + pi) is None
    True
    """
    if arg == pi:
        return Integer(1)
    if isinstance(arg, Rational) and arg.p == 0:
        return Integer(0)
    if isinstance(arg, Mul):
        cx = arg.as_coefficient(pi)
        if cx is not None:
            c, x = cx.as_coeff_Mul()
            if not isinstance(c, Rational):
                return cx
            if x.is_integer:
                c2 = c.as_fraction() % 2
                if c2 == 1:
                    return x
                if c2 == 0:
                    if x.is_even is not None:
                        return Integer(0)
                    return Integer(2)
                return mul(number_from_fraction(c2), x)
            return cx
    if arg.is_zero:
        return Integer(0)
    return None


def _chebyshev_t(n: int, x: Expr) -> Expr:
    """The Chebyshev polynomial ``T_n`` at ``x``, expanded."""
    previous: Expr = Integer(1)
    current: Expr = x
    if n == 0:
        return previous
    for _ in range(n - 1):
        previous, current = current, add(mul(Integer(2), x, current), mul(Integer(-1), previous))
    return expand(current)


def _trig_facts(arg: Expr) -> dict[str, bool]:
    """The facts shared by the trigonometric functions: a nonzero rational
    (algebraic) argument gives an irrational (transcendental) value, a
    rational multiple of ``pi`` an algebraic one."""
    a = facts_of(arg)
    facts: dict[str, bool] = {'commutative': True}
    if a.get('zero') is False:
        if a.get('rational'):
            facts['rational'] = False
        if a.get('algebraic'):
            facts['algebraic'] = False
    coefficient = pi_coeff(arg)
    if coefficient is not None and coefficient.is_rational:
        facts['algebraic'] = True
    return facts


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sin
class Sin(DefinedFunction):
    """The sine, unevaluated (``sin`` evaluates)."""

    function_name = 'sin'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .hyperbolic import sinh
        [arg] = args
        if isinstance(arg, NaN) or isinstance(arg, ComplexInfinity):
            return nan
        if isinstance(arg, Rational) and arg.p == 0:
            return Integer(0)
        if isinstance(arg, (Infinity, NegativeInfinity)):
            return None
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return mul(I, sinh(i_coeff))
        coefficient = pi_coeff(arg)
        if coefficient is not None:
            if coefficient.is_integer:
                return Integer(0)
            if mul(Integer(2), coefficient).is_integer and coefficient.is_even is False:
                return pow(Integer(-1), add(coefficient, Rational(-1, 2)))
            if not isinstance(coefficient, Rational):
                narg = mul(coefficient, pi)
                return cls(narg) if narg != arg else None
            c = coefficient.as_fraction()
            x = c % 2
            if x > 1:
                return mul(Integer(-1), cls.apply(mul(number_from_fraction(x % 1), pi)))
            if 2*x > 1:
                return cls.apply(mul(number_from_fraction(1 - x), pi))
            narg = mul(number_from_fraction((c + Fraction(3, 2)) % 2), pi)
            result = cos(narg)
            if not isinstance(result, Cos):
                return result
            if mul(coefficient, pi) != arg:
                return cls(mul(coefficient, pi))
            return None
        if isinstance(arg, Add):
            rest, m = peeloff_pi(arg)
            if m:
                multiple = mul(number_from_fraction(m), pi)
                return add(mul(sin(multiple), cos(rest)), mul(cos(multiple), sin(rest)))
        if arg.is_zero:
            return Integer(0)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return cos(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.sin(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _trig_facts(self.args[0])
        a = facts_of(self.args[0])
        if a.get('extended_real'):
            facts.update(extended_real=True, finite=True)
        if a.get('real'):
            facts['real'] = True
        if a.get('extended_real') or a.get('complex'):
            facts['complex'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cos
class Cos(DefinedFunction):
    """The cosine, unevaluated (``cos`` evaluates)."""

    function_name = 'cos'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .hyperbolic import cosh
        [arg] = args
        if isinstance(arg, NaN) or isinstance(arg, ComplexInfinity):
            return nan
        if isinstance(arg, Rational) and arg.p == 0:
            return Integer(1)
        if isinstance(arg, (Infinity, NegativeInfinity)):
            return None
        if arg.could_extract_minus_sign():
            return cls.apply(mul(Integer(-1), arg))
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return cosh(i_coeff)
        coefficient = pi_coeff(arg)
        if coefficient is not None:
            if coefficient.is_integer:
                return pow(Integer(-1), coefficient)
            if mul(Integer(2), coefficient).is_integer and coefficient.is_even is False:
                return Integer(0)
            if not isinstance(coefficient, Rational):
                narg = mul(coefficient, pi)
                return cls(narg) if narg != arg else None
            c = coefficient.as_fraction()
            q = c.denominator
            p = c.numerator % (2*q)
            if p > q:
                return mul(Integer(-1), cls.apply(mul(number_from_fraction(c - 1), pi)))
            if 2*p > q:
                return mul(Integer(-1), cls.apply(mul(number_from_fraction(1 - c), pi)))
            table2 = _table2()
            if q in table2:
                a, b = table2[q]
                angle_a = mul(number_from_fraction(Fraction(p, a)), pi)
                angle_b = mul(number_from_fraction(Fraction(p, b)), pi)
                value_a, value_b = cls.apply(angle_a), cls.apply(angle_b)
                complement_a = cls.apply(add(mul(Rational(1, 2), pi), mul(Integer(-1), angle_a)))
                complement_b = cls.apply(add(mul(Rational(1, 2), pi), mul(Integer(-1), angle_b)))
                if any(isinstance(v, Cos) for v in (value_a, value_b, complement_a, complement_b)):
                    return None
                return add(mul(value_a, value_b), mul(complement_a, complement_b))
            if q > 12:
                return None
            if q == 3:
                return _chebyshev_t(c.numerator, Rational(1, 2))
            if q == 5:
                return _chebyshev_t(c.numerator, mul(add(sqrt(Integer(5)), Integer(1)), Rational(1, 4)))
            if q % 2 == 0:
                doubled = cls.apply(mul(number_from_fraction(2*c), pi))
                if isinstance(doubled, Cos):
                    return None
                x = (2*c + 1)/2
                sign = (-1)**((-1 if x < 0 else 1)*int(abs(x)))
                return mul(Integer(sign), sqrt(mul(add(Integer(1), doubled), Rational(1, 2))))
            return None
        if isinstance(arg, Add):
            rest, m = peeloff_pi(arg)
            if m:
                multiple = mul(number_from_fraction(m), pi)
                return add(mul(cos(multiple), cos(rest)), mul(Integer(-1), sin(multiple), sin(rest)))
        if arg.is_zero:
            return Integer(1)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(Integer(-1), sin(self.args[0]))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.cos(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _trig_facts(self.args[0])
        a = facts_of(self.args[0])
        if a.get('extended_real'):
            facts.update(extended_real=True, finite=True)
        if a.get('real'):
            facts['real'] = True
        if a.get('extended_real') or a.get('complex'):
            facts['complex'] = True
        return facts


_TAN_TENTHS: dict[int, Expr] = {}


def _tan_table10(n: int) -> Expr:
    """``tan(n*pi/10)`` for ``n = 1, ..., 4`` (SymPy's table)."""
    if not _TAN_TENTHS:
        five = sqrt(Integer(5))
        _TAN_TENTHS[1] = sqrt(add(Integer(1), mul(Rational(-2, 5), five)))
        _TAN_TENTHS[2] = sqrt(add(Integer(5), mul(Integer(-2), five)))
        _TAN_TENTHS[3] = sqrt(add(Integer(1), mul(Rational(2, 5), five)))
        _TAN_TENTHS[4] = sqrt(add(Integer(5), mul(Integer(2), five)))
    return _TAN_TENTHS[n]


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:tan
class Tan(DefinedFunction):
    """The tangent, unevaluated (``tan`` evaluates)."""

    function_name = 'tan'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .hyperbolic import tanh
        [arg] = args
        if isinstance(arg, NaN) or isinstance(arg, ComplexInfinity):
            return nan
        if isinstance(arg, Rational) and arg.p == 0:
            return Integer(0)
        if isinstance(arg, (Infinity, NegativeInfinity)):
            return None
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return mul(I, tanh(i_coeff))
        coefficient = pi_coeff(arg)
        if coefficient is not None:
            if coefficient.is_integer:
                return Integer(0)
            if not isinstance(coefficient, Rational):
                narg = mul(coefficient, pi)
                return cls(narg) if narg != arg else None
            c = coefficient.as_fraction()
            q = c.denominator
            p = c.numerator % q
            if q in (5, 10):
                n = 10*p//q
                if n > 5:
                    return mul(Integer(-1), _tan_table10(10 - n))
                return _tan_table10(n)
            if q % 2 == 0:
                narg = mul(number_from_fraction(2*c), pi)
                c_value, s_value = cos(narg), cos(add(narg, mul(Rational(-1, 2), pi)))
                if not isinstance(c_value, Cos) and not isinstance(s_value, Cos):
                    if s_value == 0:
                        return zoo
                    inverse = pow(s_value, Integer(-1))
                    return add(inverse, mul(Integer(-1), c_value, inverse))
            table2 = _table2()
            if q in table2:
                a, b = table2[q]
                value_a = cls.apply(mul(number_from_fraction(Fraction(p, a)), pi))
                value_b = cls.apply(mul(number_from_fraction(Fraction(p, b)), pi))
                if isinstance(value_a, Tan) or isinstance(value_b, Tan):
                    return None
                return mul(add(value_a, mul(Integer(-1), value_b)),
                           pow(add(Integer(1), mul(value_a, value_b)), Integer(-1)))
            reduced = ((c + HALF) % 1 - HALF)
            narg = mul(number_from_fraction(reduced), pi)
            c_value, s_value = cos(narg), cos(add(narg, mul(Rational(-1, 2), pi)))
            if not isinstance(c_value, Cos) and not isinstance(s_value, Cos):
                if c_value == 0:
                    return zoo
                return mul(s_value, pow(c_value, Integer(-1)))
            if narg != arg:
                return cls(narg)
            return None
        if isinstance(arg, Add):
            rest, m = peeloff_pi(arg)
            if m:
                # tan(x + n*pi) = tan(x), tan(x + pi/2 + n*pi) = -cot(x)
                return cls.apply(rest) if m.denominator == 1 else mul(Integer(-1), cot(rest))
        if arg.is_zero:
            return Integer(0)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return add(Integer(1), pow(self, Integer(2)))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.cos(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.tan(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _trig_facts(self.args[0])
        arg = self.args[0]
        a = facts_of(arg)
        if a.get('extended_real') is not None:
            facts['extended_real'] = a['extended_real']
        off_pole = facts_of(add(mul(arg, pow(pi, Integer(-1))), Rational(-1, 2))).get('integer') is False
        if a.get('real') and off_pole:
            facts.update(real=True, finite=True, complex=True)
        if a.get('imaginary'):
            facts['finite'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cot
class Cot(DefinedFunction):
    """The cotangent, unevaluated (``cot`` evaluates)."""

    function_name = 'cot'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .hyperbolic import coth
        [arg] = args
        if isinstance(arg, NaN) or isinstance(arg, ComplexInfinity):
            return nan
        if isinstance(arg, Rational) and arg.p == 0:
            return zoo
        if isinstance(arg, (Infinity, NegativeInfinity)):
            return None
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return mul(Integer(-1), I, coth(i_coeff))
        coefficient = pi_coeff(arg)
        if coefficient is not None:
            if coefficient.is_integer:
                return zoo
            if not isinstance(coefficient, Rational):
                narg = mul(coefficient, pi)
                return cls(narg) if narg != arg else None
            c = coefficient.as_fraction()
            q = c.denominator
            if q in (5, 10):
                return tan(add(mul(Rational(1, 2), pi), mul(Integer(-1), arg)))
            if q > 2 and q % 2 == 0:
                narg = mul(number_from_fraction(2*c), pi)
                c_value, s_value = cos(narg), cos(add(narg, mul(Rational(-1, 2), pi)))
                if not isinstance(c_value, Cos) and not isinstance(s_value, Cos):
                    inverse = pow(s_value, Integer(-1))
                    return add(inverse, mul(c_value, inverse))
            p = c.numerator % q
            table2 = _table2()
            if q in table2:
                a, b = table2[q]
                value_a = cls.apply(mul(number_from_fraction(Fraction(p, a)), pi))
                value_b = cls.apply(mul(number_from_fraction(Fraction(p, b)), pi))
                if isinstance(value_a, Cot) or isinstance(value_b, Cot):
                    return None
                return mul(add(Integer(1), mul(value_a, value_b)),
                           pow(add(value_b, mul(Integer(-1), value_a)), Integer(-1)))
            reduced = ((c + HALF) % 1 - HALF)
            narg = mul(number_from_fraction(reduced), pi)
            c_value, s_value = cos(narg), cos(add(narg, mul(Rational(-1, 2), pi)))
            if not isinstance(c_value, Cos) and not isinstance(s_value, Cos):
                if s_value == 0:
                    return zoo
                return mul(c_value, pow(s_value, Integer(-1)))
            if narg != arg:
                return cls(narg)
            return None
        if isinstance(arg, Add):
            rest, m = peeloff_pi(arg)
            if m:
                # cot(x + n*pi) = cot(x), cot(x + pi/2 + n*pi) = -tan(x)
                return cls.apply(rest) if m.denominator == 1 else mul(Integer(-1), tan(rest))
        if arg.is_zero:
            return zoo
        return None

    def fdiff(self, argindex: int) -> Expr:
        return add(Integer(-1), mul(Integer(-1), pow(self, Integer(2))))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.sin(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.cot(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return _trig_facts(self.args[0])


def _reciprocal_value(t: Expr) -> Expr:
    """``1/t``, written with ``sec`` when ``t`` or ``-t`` is a cosine and
    with ``csc`` when it is a sine (SymPy's ``(1/t).rewrite(sec)`` and
    ``(1/t).rewrite(csc)``)."""
    negated = mul(Integer(-1), t)
    for partner, reciprocal in ((Cos, Sec), (Sin, Csc)):
        if isinstance(t, partner):
            return reciprocal(t.args[0])
        if isinstance(negated, partner):
            return mul(Integer(-1), reciprocal(negated.args[0]))
    return pow(t, Integer(-1))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:ReciprocalTrigonometricFunction.eval
def _reciprocal_eval(cls: type[DefinedFunction], partner: type[DefinedFunction], even: bool, arg: Expr) -> Optional[Expr]:
    """The evaluation of ``sec`` (``partner`` = ``Cos``, even) and ``csc``
    (``Sin``, odd) through the evaluation of the partner."""
    if arg.could_extract_minus_sign():
        flipped = cls.apply(mul(Integer(-1), arg))
        return flipped if even else mul(Integer(-1), flipped)
    coefficient = pi_coeff(arg)
    if coefficient is not None and isinstance(coefficient, Rational) and not mul(Integer(2), coefficient).is_integer:
        c = coefficient.as_fraction()
        q = c.denominator
        p = c.numerator % (2*q)
        if p > q:
            return mul(Integer(-1), cls.apply(mul(number_from_fraction(c - 1), pi)))
        if 2*p > q:
            narg = mul(number_from_fraction(1 - c), pi)
            return mul(Integer(-1), cls.apply(narg)) if even else cls.apply(narg)
    t = partner.eval(arg)
    if t is None:
        return None
    return _reciprocal_value(t)


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sec
class Sec(DefinedFunction):
    """The secant, unevaluated (``sec`` evaluates)."""

    function_name = 'sec'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        return _reciprocal_eval(cls, Cos, True, arg)

    def fdiff(self, argindex: int) -> Expr:
        return mul(tan(self.args[0]), self)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.cos(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.sec(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return _trig_facts(self.args[0])


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:csc
class Csc(DefinedFunction):
    """The cosecant, unevaluated (``csc`` evaluates)."""

    function_name = 'csc'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        return _reciprocal_eval(cls, Sin, False, arg)

    def fdiff(self, argindex: int) -> Expr:
        return mul(Integer(-1), cot(self.args[0]), self)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.sin(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.csc(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return _trig_facts(self.args[0])


def sin(x: ExprLike) -> Expr:
    """The sine of ``x``, evaluated."""
    return Sin.apply(to_expr(x))


def cos(x: ExprLike) -> Expr:
    """The cosine of ``x``, evaluated."""
    return Cos.apply(to_expr(x))


def tan(x: ExprLike) -> Expr:
    """The tangent of ``x``, evaluated."""
    return Tan.apply(to_expr(x))


def cot(x: ExprLike) -> Expr:
    """The cotangent of ``x``, evaluated."""
    return Cot.apply(to_expr(x))


def sec(x: ExprLike) -> Expr:
    """The secant of ``x``, evaluated."""
    return Sec.apply(to_expr(x))


def csc(x: ExprLike) -> Expr:
    """The cosecant of ``x``, evaluated."""
    return Csc.apply(to_expr(x))
