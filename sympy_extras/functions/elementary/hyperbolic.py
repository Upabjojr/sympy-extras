"""The hyperbolic functions on the typed core: ``Sinh``, ``Cosh``,
``Tanh`` (unevaluated nodes) and ``sinh``, ``cosh``, ``tanh``
(evaluating).

The evaluation rules, derivatives and facts are ported from SymPy's
``sympy.functions.elementary.hyperbolic``: the values at zero and at the
infinities, odd and even symmetry, the exchange with the trigonometric
functions on imaginary arguments, the multiples of ``I*pi`` peeled off
sums. Not ported yet (the node stays): ``tanh(x + I*pi/2)`` (SymPy:
``coth(x)``), the compositions with the inverse functions, and the facts
which need ``as_real_imag``.

Examples
========

>>> from sympy_extras.core import Symbol, I, pi, oo
>>> from sympy_extras.functions.elementary.hyperbolic import sinh, cosh, tanh
>>> x = Symbol('x')
>>> sinh(0), cosh(0), tanh(oo), sinh(-x), cosh(-x), sinh(I*x), cosh(I*pi)
(0, 1, 1, -sinh(x), cosh(x), I*sin(x), -1)
>>> sinh(x + I*pi), sinh(x).diff(x), tanh(x).diff(x)
(-sinh(x), cosh(x), 1 - tanh(x)**2)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes Sinh/Cosh/Tanh with evaluating functions,
# strict typing, the subset of the rules the typed core supports.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, I, Infinity, Integer, NaN, NegativeInfinity, Number, Rational,
                                       nan, number_from_fraction, oo, pi)
from sympy_extras.core.power import pow

from .trigonometric import imaginary_coefficient

__all__ = ['Sinh', 'Cosh', 'Tanh', 'sinh', 'cosh', 'tanh', 'peeloff_ipi']


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:_peeloff_ipi
def peeloff_ipi(arg: Expr) -> tuple[Expr, Fraction]:
    """``(rest, m)`` with ``arg = rest + m*I*pi``, ``m`` a multiple of 1/2,
    from the first term of the sum which is a rational multiple of
    ``I*pi``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, I, pi
    >>> from sympy_extras.functions.elementary.hyperbolic import peeloff_ipi
    >>> x = Symbol('x')
    >>> peeloff_ipi(x + I*pi/2), peeloff_ipi(x + 2*I*pi/3)
    ((x, Fraction(1, 2)), (x + I*pi/6, Fraction(1, 2)))
    """
    ipi = mul(pi, I)
    k: Optional[Fraction] = None
    for term in (arg.args if isinstance(arg, Add) else (arg,)):
        if term == ipi:
            k = Fraction(1)
            break
        if isinstance(term, Mul):
            first, rest = term.as_two_terms()
            if rest == ipi and isinstance(first, Rational):
                k = first.as_fraction()
                break
    if k is None:
        return arg, Fraction(0)
    m2 = k - k % Fraction(1, 2)
    return add(arg, mul(Integer(-1), number_from_fraction(m2), ipi)), m2


def _odd_facts(arg: Expr) -> dict[str, bool]:
    a = facts_of(arg)
    facts: dict[str, bool] = {'commutative': True}
    if a.get('extended_real'):
        facts['extended_real'] = True
        if a.get('positive') is not None:
            facts['positive'] = a['positive']
        if a.get('negative') is not None:
            facts['negative'] = a['negative']
    if a.get('real'):
        facts['real'] = True
    return facts


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:sinh
class Sinh(DefinedFunction):
    """The hyperbolic sine, unevaluated (``sinh`` evaluates)."""

    function_name = 'sinh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import sin
        [arg] = args
        if isinstance(arg, Number):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, Infinity):
                return oo
            if isinstance(arg, NegativeInfinity):
                return mul(Integer(-1), oo)
            if isinstance(arg, Rational) and arg.p == 0:
                return Integer(0)
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
            return None
        if isinstance(arg, ComplexInfinity):
            return nan
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return mul(I, sin(i_coeff))
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if isinstance(arg, Add):
            rest, m = peeloff_ipi(arg)
            if m:
                multiple = mul(number_from_fraction(m), pi, I)
                return add(mul(sinh(multiple), cosh(rest)), mul(cosh(multiple), sinh(rest)))
        if arg.is_zero:
            return Integer(0)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return cosh(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.sinh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _odd_facts(self.args[0])
        if facts_of(self.args[0]).get('finite'):
            facts['finite'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:cosh
class Cosh(DefinedFunction):
    """The hyperbolic cosine, unevaluated (``cosh`` evaluates)."""

    function_name = 'cosh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import cos
        [arg] = args
        if isinstance(arg, Number):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, (Infinity, NegativeInfinity)):
                return oo
            if isinstance(arg, Rational) and arg.p == 0:
                return Integer(1)
            if arg.is_negative:
                return cls.apply(mul(Integer(-1), arg))
            return None
        if isinstance(arg, ComplexInfinity):
            return nan
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return cos(i_coeff)
        if arg.could_extract_minus_sign():
            return cls.apply(mul(Integer(-1), arg))
        if isinstance(arg, Add):
            rest, m = peeloff_ipi(arg)
            if m:
                multiple = mul(number_from_fraction(m), pi, I)
                return add(mul(cosh(multiple), cosh(rest)), mul(sinh(multiple), sinh(rest)))
        if arg.is_zero:
            return Integer(1)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return sinh(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.cosh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('real') or a.get('imaginary'):
            facts['real'] = True
        if a.get('real'):
            facts['positive'] = True
        if a.get('finite'):
            facts['finite'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:tanh
class Tanh(DefinedFunction):
    """The hyperbolic tangent, unevaluated (``tanh`` evaluates)."""

    function_name = 'tanh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import tan
        [arg] = args
        if isinstance(arg, Number):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, Infinity):
                return Integer(1)
            if isinstance(arg, NegativeInfinity):
                return Integer(-1)
            if isinstance(arg, Rational) and arg.p == 0:
                return Integer(0)
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
            return None
        if isinstance(arg, ComplexInfinity):
            return nan
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            if i_coeff.could_extract_minus_sign():
                return mul(Integer(-1), I, tan(mul(Integer(-1), i_coeff)))
            return mul(I, tan(i_coeff))
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if isinstance(arg, Add):
            rest, m = peeloff_ipi(arg)
            if m and m.denominator == 1:
                return cls.apply(rest)  # tanh(x + I*pi/2) = coth(x) is not ported
        if arg.is_zero:
            return Integer(0)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return add(Integer(1), mul(Integer(-1), pow(self, Integer(2))))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.cosh(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.tanh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _odd_facts(self.args[0])
        a = facts_of(self.args[0])
        if a.get('extended_real'):
            facts['finite'] = True
        if a.get('zero'):
            facts['zero'] = True
        return facts


def sinh(x: ExprLike) -> Expr:
    """The hyperbolic sine of ``x``, evaluated."""
    return Sinh.apply(to_expr(x))


def cosh(x: ExprLike) -> Expr:
    """The hyperbolic cosine of ``x``, evaluated."""
    return Cosh.apply(to_expr(x))


def tanh(x: ExprLike) -> Expr:
    """The hyperbolic tangent of ``x``, evaluated."""
    return Tanh.apply(to_expr(x))
