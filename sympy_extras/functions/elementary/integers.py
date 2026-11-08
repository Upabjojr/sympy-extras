"""Integer parts on the typed core: ``Floor``, ``Ceiling``, ``Frac``
(unevaluated nodes) and ``floor``, ``ceiling``, ``frac`` (evaluating).

Ported from SymPy's ``sympy.functions.elementary.integers``: numbers,
the number symbols (``floor(pi) = 3``), integers and infinities left as
they are, imaginary arguments, the integer terms of a sum taken out and
the numerical part evaluated when it is independent of the symbolic one,
the bounds ``0 <= x < 1`` and ``1 <= x < 2`` (``-1 <= x < 0``, ...)
decided from the signs; ``frac(x) = x - floor(x)`` for numbers. The
numerical integer part (SymPy's ``get_integer_part``) is computed with
mpmath at 100 and 200 digits; a value too close to an integer to decide
leaves the node; ``frac(oo)`` is ``AccumBounds(0, 1)``. Not ported: the comparisons ``frac(x) < 1`` and the series.

Examples
========

>>> from sympy_extras.core import Symbol, Rational, I, pi, E
>>> from sympy_extras.functions.elementary.integers import floor, ceiling, frac, Floor
>>> x, n = Symbol('x'), Symbol('n', integer=True)
>>> floor(Rational(7, 2)), ceiling(Rational(7, 2)), floor(-pi), ceiling(E), floor(n + Rational(5, 2))
(3, 4, -4, 3, n + 2)
>>> floor(x + 3), floor(2*I + Rational(1, 2)), frac(Rational(7, 3)), frac(n), Floor(1)
(floor(x) + 3, 2*I, 1/3, 0, floor(1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/integers.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing,
# the numerical integer part with mpmath at two precisions.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Derivative, DefinedFunction
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import (ComplexInfinity, Float, I, Infinity, Integer, NaN, NegativeInfinity, Number,
                                       NumberSymbol, Rational, nan)
from sympy_extras.core.symbol import Symbol

__all__ = ['Floor', 'Ceiling', 'Frac', 'floor', 'ceiling', 'frac']

ZERO = Integer(0)


def _integer_value(e: Expr) -> Optional[int]:
    """``e`` as a Python integer when it is an integral number (SymPy's
    ``int_valued``)."""
    if isinstance(e, (Rational, Float)):
        value = e.as_fraction()
        if value.denominator == 1:
            return value.numerator
    return None


def _numerical_integer_part(e: Expr, direction: int) -> Optional[tuple[int, int]]:
    """The floor (``direction = -1``) or ceiling (``1``) of the real and
    imaginary parts of the number ``e``, from its value at 100 and 200
    digits; ``None`` when a part is too close to an integer to decide."""
    from sympy_extras.core.evalf import numeric_value
    # a part known to be zero is not decided numerically (its value is noise)
    known_zero = (e.is_imaginary is True, e.is_extended_real is True)
    results: list[tuple[int, int]] = []
    for bits in (340, 680):
        value = numeric_value(e, bits)
        if value is None:
            return None
        parts: list[int] = []
        for component, zero in zip((mpmath.re(value), mpmath.im(value)), known_zero):
            if zero:
                parts.append(0)
                continue
            with mpmath.workprec(bits):
                nearest = mpmath.nint(component)
                if abs(component - nearest) < mpmath.mpf(2)**(-bits//2):
                    return None
                parts.append(int(mpmath.floor(component) if direction < 0 else mpmath.ceil(component)))
        results.append((parts[0], parts[1]))
    return results[0] if results[0] == results[1] else None


def _is_lt(a: Expr, b: Expr) -> Optional[bool]:
    return add(a, mul(Integer(-1), b)).is_extended_negative


def _is_le(a: Expr, b: Expr) -> Optional[bool]:
    return add(a, mul(Integer(-1), b)).is_extended_nonpositive


def _fuzzy_and(values: Sequence[Optional[bool]]) -> Optional[bool]:
    if any(v is False for v in values):
        return False
    if all(v is True for v in values):
        return True
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:RoundFunction.eval
def _round_eval(cls: type[DefinedFunction], arg: Expr, direction: int) -> Optional[Expr]:
    from .complexes import im
    number = _round_number(cls, arg, direction)
    if number is not None:
        return number
    constant = _round_constant(arg, direction)
    if constant is not None:
        return constant
    if arg.is_integer or arg.is_finite is False:
        return arg
    if arg.is_imaginary or mul(I, arg).is_real:
        imaginary = im(arg)
        if not imaginary.has(I):
            return mul(cls.apply(imaginary), I)
        return cls(arg)
    integral: Expr = ZERO
    numerical: Expr = ZERO
    symbolic: Expr = ZERO
    for t in (arg.args if isinstance(arg, Add) else (arg,)):
        if t.is_imaginary:
            imaginary = im(t)
            k = _integer_value(imaginary)
            if k is None and imaginary.is_integer:
                integral = add(integral, mul(imaginary, I))
                continue
            if k is not None:
                integral = add(integral, mul(Integer(k), I))
                continue
        k = _integer_value(t)
        if k is not None:
            integral = add(integral, Integer(k))
        elif t.is_integer:
            integral = add(integral, t)
        elif t.is_number:
            numerical = add(numerical, t)
        else:
            symbolic = add(symbolic, t)
    if numerical == 0 and symbolic == 0:
        return integral
    if numerical != 0 and (symbolic == 0
                           or (numerical.is_real and (symbolic.is_imaginary or mul(I, symbolic).is_real))
                           or (numerical.is_imaginary and symbolic.is_real)):
        parts = _numerical_integer_part(numerical, direction)
        if parts is not None:
            integral = add(integral, Integer(parts[0]), mul(Integer(parts[1]), I))
            numerical = ZERO
    symbolic = add(symbolic, numerical)
    if symbolic == 0:
        return integral
    if symbolic.is_imaginary or mul(I, symbolic).is_real:
        return add(integral, mul(cls(im(symbolic)), I))
    if isinstance(symbolic, (Floor, Ceiling)):
        return add(integral, symbolic)
    return add(integral, cls(symbolic))


def _round_number(cls: type[DefinedFunction], arg: Expr, direction: int) -> Optional[Expr]:
    """SymPy's ``_eval_number``: numbers, floors and ceilings, number
    symbols."""
    if isinstance(arg, NaN):
        return nan
    if isinstance(arg, (Infinity, NegativeInfinity, ComplexInfinity)):
        return arg
    if isinstance(arg, Rational):
        f = arg.as_fraction()
        q, r = divmod(f.numerator, f.denominator)
        return Integer(q if direction < 0 or r == 0 else q + 1)
    if isinstance(arg, Float):
        value = arg.as_fraction()
        q, r = divmod(value.numerator, value.denominator)
        return Integer(q if direction < 0 or r == 0 else q + 1)
    if isinstance(arg, Number):
        return None
    negated = mul(Integer(-1), arg)
    if isinstance(arg, (Floor, Ceiling)) or isinstance(negated, (Floor, Ceiling)):
        return arg
    if isinstance(arg, NumberSymbol):
        parts = _numerical_integer_part(arg, direction)
        return Integer(parts[0]) if parts is not None else None
    return None


def _round_constant(arg: Expr, direction: int) -> Optional[Expr]:
    """SymPy's ``_eval_const_number``: the bounds decided from the signs
    of a real argument."""
    if not arg.is_real:
        return None
    if arg.is_zero:
        return ZERO
    if arg.is_positive or arg.is_negative:
        num, den = arg.as_numer_denom()
        s = den.is_negative
        if s is None:
            return None
        if s:
            num, den = mul(Integer(-1), num), mul(Integer(-1), den)
        two_den = mul(Integer(2), den)
        minus_den = mul(Integer(-1), den)
        if arg.is_positive:
            if direction < 0:
                if _is_lt(num, den):
                    return ZERO
                if _fuzzy_and([_is_le(den, num), _is_lt(num, two_den)]):
                    return Integer(1)
            else:
                if _is_le(num, den):
                    return Integer(1)
                if _fuzzy_and([_is_lt(den, num), _is_le(num, two_den)]):
                    return Integer(2)
        else:
            if direction < 0:
                if _is_le(minus_den, num):
                    return Integer(-1)
                if _fuzzy_and([_is_le(mul(Integer(-1), two_den), num), _is_lt(num, minus_den)]):
                    return Integer(-2)
            else:
                if _is_lt(minus_den, num):
                    return ZERO
                if _fuzzy_and([_is_lt(mul(Integer(-1), two_den), num), _is_le(num, minus_den)]):
                    return Integer(-1)
    return None


def _round_facts(arg: Expr) -> dict[str, bool]:
    a = facts_of(arg)
    facts: dict[str, bool] = {'commutative': True}
    if a.get('finite') is not None:
        facts['finite'] = a['finite']
    if a.get('real') is not None:
        facts['real'] = a['real']
        facts['integer'] = a['real']
    return facts


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:floor
class Floor(DefinedFunction):
    """The floor, unevaluated (``floor`` evaluates)."""

    function_name = 'floor'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        return _round_eval(cls, arg, -1)

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        return Derivative(self, s)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.floor(mpmath.re(values[0])), mpmath.floor(mpmath.im(values[0])))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _round_facts(self.args[0])
        a = facts_of(self.args[0])
        if a.get('negative') is not None:
            facts['negative'] = a['negative']
        if a.get('nonnegative') is not None:
            facts['nonnegative'] = a['nonnegative']
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:ceiling
class Ceiling(DefinedFunction):
    """The ceiling, unevaluated (``ceiling`` evaluates)."""

    function_name = 'ceiling'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        return _round_eval(cls, arg, 1)

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        return Derivative(self, s)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.ceil(mpmath.re(values[0])), mpmath.ceil(mpmath.im(values[0])))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _round_facts(self.args[0])
        a = facts_of(self.args[0])
        if a.get('positive') is not None:
            facts['positive'] = a['positive']
        if a.get('nonpositive') is not None:
            facts['nonpositive'] = a['nonpositive']
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:frac
class Frac(DefinedFunction):
    """The fractional part ``x - floor(x)``, unevaluated (``frac``
    evaluates)."""

    function_name = 'frac'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .complexes import im
        [arg] = args

        def part(value: Expr) -> Optional[Expr]:
            if isinstance(value, (Infinity, NegativeInfinity)):
                from sympy_extras.core.accumbounds import AccumBounds
                return AccumBounds(ZERO, Integer(1))
            if value.is_integer:
                return ZERO
            if value.is_number:
                if isinstance(value, (NaN, ComplexInfinity)):
                    return nan
                return add(value, mul(Integer(-1), floor(value)))
            return cls(value)

        real: Expr = ZERO
        imaginary: Expr = ZERO
        for t in (arg.args if isinstance(arg, Add) else (arg,)):
            if t.is_imaginary or mul(I, t).is_real:
                i = im(t)
                if not i.has(I):
                    imaginary = add(imaginary, i)
                else:
                    real = add(real, t)
            else:
                real = add(real, t)
        real_part, imaginary_part = part(real), part(imaginary)
        if real_part is None or imaginary_part is None:
            return None
        value = add(real_part, mul(I, imaginary_part))
        return None if value == cls(arg) else value

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        return Derivative(self, s)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        z = values[0]
        return mpmath.mpc(mpmath.re(z) - mpmath.floor(mpmath.re(z)), mpmath.im(z) - mpmath.floor(mpmath.im(z)))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True, 'finite': True, 'negative': False}
        if a.get('extended_real') is not None:
            facts['real'] = a['extended_real']
        if a.get('imaginary') is not None:
            facts['imaginary'] = a['imaginary']
        if a.get('integer') is not None:
            facts['integer'] = a['integer']
        if a.get('zero') is True or a.get('integer') is True:
            facts['zero'] = True
        elif a.get('zero') is False and a.get('integer') is False:
            facts['zero'] = False
        return facts


def floor(x: ExprLike) -> Expr:
    """The floor of ``x``, evaluated."""
    return Floor.apply(to_expr(x))


def ceiling(x: ExprLike) -> Expr:
    """The ceiling of ``x``, evaluated."""
    return Ceiling.apply(to_expr(x))


def frac(x: ExprLike) -> Expr:
    """The fractional part of ``x``, evaluated."""
    return Frac.apply(to_expr(x))
