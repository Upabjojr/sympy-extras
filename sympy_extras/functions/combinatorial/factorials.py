"""Factorials and binomial coefficients on the typed core: ``Factorial``,
``Binomial``, ``RisingFactorial``, ``FallingFactorial`` (unevaluated
nodes) and ``factorial``, ``binomial``, ``rf``, ``ff`` (evaluating).

Ported from SymPy's ``sympy.functions.combinatorial.factorials``: integer
values computed exactly, ``factorial`` of a negative integer is ``zoo``,
``binomial`` with ``k`` or ``n - k`` equal to 0 or 1, zero for a negative
integer ``k``, the product formula for a numerical ``n`` and an integer
``k``, ``rf(x, k)`` and ``ff(x, k)`` expanded into products for an
integer ``k``, the gamma form for a non-integer numerical ``k``
(``binomial(5, 1/2) = 512/(63*pi)``), the derivatives with ``polygamma``.

Examples
========

>>> from sympy_extras.core import Symbol, Rational
>>> from sympy_extras.functions.combinatorial.factorials import factorial, binomial, rf, ff, Factorial
>>> x, n = Symbol('x'), Symbol('n', integer=True)
>>> factorial(5), factorial(-1), binomial(10, 3), binomial(n, 1), binomial(Rational(1, 2), 2), binomial(n, -2)
(120, zoo, 120, n, -1/8, 0)
>>> rf(x, 3), ff(x, 2), rf(x, -2), ff(n, n), Factorial(3)
(x*(x + 1)*(x + 2), x*(x - 1), 1/((x - 2)*(x - 1)), factorial(n), factorial(3))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing,
# math.factorial and math.comb for the integer values; the gamma forms
# and the Poly arguments not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, expand
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import (Infinity, Integer, NaN, NegativeInfinity, Rational, nan,
                                       oo, zoo)
from sympy_extras.core.power import pow

__all__ = ['Factorial', 'Binomial', 'RisingFactorial', 'FallingFactorial', 'Subfactorial', 'Factorial2', 'factorial',
           'binomial', 'rf', 'ff', 'subfactorial', 'factorial2']

ONE = Integer(1)
ZERO = Integer(0)


def _int(e: Expr) -> Optional[int]:
    """``e`` as a Python integer when it is an ``Integer``."""
    if isinstance(e, Rational) and e.q == 1:
        return e.p
    return None


def _gamma_numeric(z: mpmath.mpc) -> Optional[mpmath.mpc]:
    """``gamma(z)``, ``None`` at the poles."""
    if mpmath.im(z) == 0 and mpmath.re(z) <= 0 and mpmath.re(z) == mpmath.floor(mpmath.re(z)):
        return None
    return mpmath.mpc(mpmath.gamma(z))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:factorial
class Factorial(DefinedFunction):
    """``n!``, unevaluated (``factorial`` evaluates)."""

    function_name = 'factorial'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        if n.is_zero:
            return ONE
        if isinstance(n, Infinity):
            return oo
        value = _int(n)
        if value is not None:
            if value < 0:
                return zoo
            return Integer(math.factorial(value))
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.special.gamma_functions import gamma, polygamma
        n = add(self.args[0], ONE)
        return mul(gamma(n), polygamma(ZERO, n))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return _gamma_numeric(values[0] + 1)

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('integer') and a.get('nonnegative'):
            facts.update(integer=True, positive=True)
            n = self.args[0]
            even = sub(n, Integer(2)).is_nonnegative
            if even is not None:
                facts['even'] = even
            composite = sub(n, Integer(3)).is_nonnegative
            if composite is not None:
                facts['composite'] = composite
        if a.get('nonnegative') or a.get('noninteger'):
            facts['real'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial
class Binomial(DefinedFunction):
    """The binomial coefficient ``binomial(n, k)``, unevaluated
    (``binomial`` evaluates)."""

    function_name = 'binomial'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def _numerical(cls, n: Expr, k: Expr) -> Optional[Expr]:
        """SymPy's ``binomial._eval`` for a number ``n`` and an integer
        ``k``."""
        kk = _int(k)
        if kk is None:
            return None
        nn = _int(n)
        if nn is not None and nn >= 0:
            return ZERO if kk > nn else Integer(math.comb(nn, kk))
        d: Expr = sub(n, k)
        result: Expr = ONE
        for _ in range(1, kk + 1):
            d = add(d, ONE)
            result = mul(result, d)
        return mul(result, pow(Integer(math.factorial(kk)), Integer(-1)))

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        n, k = args
        d = sub(n, k)
        n_nonnegative, n_integer = n.is_nonnegative, n.is_integer
        if k.is_zero or ((n_nonnegative or n_integer is False) and d.is_zero):
            return ONE
        if sub(k, ONE).is_zero or ((n_nonnegative or n_integer is False) and sub(d, ONE).is_zero):
            return n
        if k.is_integer:
            if k.is_negative or (n_nonnegative and n_integer and d.is_negative):
                return ZERO
            if n.is_number:
                result = cls._numerical(n, k)
                if result is None:
                    return None
                return expand(result) if result != 0 else result
            return None
        if n_nonnegative is False and n_integer:
            return zoo
        if k.is_number:
            from sympy_extras.functions.special.gamma_functions import gamma
            return mul(gamma(add(n, ONE)), pow(mul(gamma(add(k, ONE)), gamma(add(sub(n, k), ONE))), Integer(-1)))
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.special.gamma_functions import polygamma
        n, k = self.args
        if argindex == 1:
            return mul(binomial(n, k), sub(polygamma(ZERO, add(n, ONE)), polygamma(ZERO, add(sub(n, k), ONE))))
        return mul(binomial(n, k), sub(polygamma(ZERO, add(sub(n, k), ONE)), polygamma(ZERO, add(k, ONE))))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n, k = values
        try:
            return mpmath.mpc(mpmath.binomial(n, k))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        n, k = (facts_of(a) for a in self.args)
        facts: dict[str, bool] = {'commutative': True}
        if n.get('integer') and k.get('integer'):
            facts['integer'] = True
            if n.get('nonnegative') or k.get('negative') or k.get('even'):
                facts['nonnegative'] = True
            elif k.get('even') is False:
                facts['nonnegative'] = False
        elif k.get('integer') is False:
            facts['integer'] = False
        return facts


def _product(x: Expr, offsets: range, sign: int) -> Expr:
    """``(x + sign*i)`` multiplied over ``i`` in ``offsets``, evaluated as
    SymPy's ``reduce`` does it."""
    result: Expr = ONE
    for i in offsets:
        result = mul(result, add(x, Integer(sign*i)))
    return result


def _factorial_power(x: Expr, k: Expr, rising: bool) -> Optional[Expr]:
    """SymPy's ``RisingFactorial.eval`` (``rising``) and
    ``FallingFactorial.eval`` for an integer ``k``."""
    kk = _int(k)
    if kk is None:
        return None
    if kk == 0:
        return ONE
    if kk > 0:
        if isinstance(x, Infinity):
            return oo
        if isinstance(x, NegativeInfinity):
            return NegativeInfinity() if kk % 2 else oo
        return _product(x, range(kk), 1 if rising else -1)
    if isinstance(x, (Infinity, NegativeInfinity)):
        return oo
    return pow(_product(x, range(1, -kk + 1), -1 if rising else 1), Integer(-1))


def _factorial_power_facts(args: tuple[Expr, ...]) -> dict[str, bool]:
    x, k = (facts_of(a) for a in args)
    facts: dict[str, bool] = {'commutative': True}
    values = (x.get('integer'), k.get('integer'), k.get('nonnegative'))
    if all(v is True for v in values):
        facts['integer'] = True
    elif any(v is False for v in values):
        facts['integer'] = False
    return facts


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:RisingFactorial
class RisingFactorial(DefinedFunction):
    """The rising factorial ``x*(x + 1)*...*(x + k - 1)``, unevaluated
    (``rf`` evaluates)."""

    function_name = 'RisingFactorial'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        x, k = args
        if isinstance(x, NaN) or isinstance(k, NaN):
            return nan
        if x == ONE:
            return factorial(k)
        value = _factorial_power(x, k, True)
        if value is not None:
            return value
        if k.is_integer is False and x.is_integer and x.is_negative:
            return ZERO
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.rf(values[0], values[1]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return _factorial_power_facts(self.args)


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:FallingFactorial
class FallingFactorial(DefinedFunction):
    """The falling factorial ``x*(x - 1)*...*(x - k + 1)``, unevaluated
    (``ff`` evaluates)."""

    function_name = 'FallingFactorial'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        x, k = args
        if isinstance(x, NaN) or isinstance(k, NaN):
            return nan
        if k.is_integer and x == k:
            return factorial(x)
        return _factorial_power(x, k, False)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.ff(values[0], values[1]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return _factorial_power_facts(self.args)


def factorial(n: ExprLike) -> Expr:
    """``n!``, evaluated."""
    return Factorial.apply(to_expr(n))


def binomial(n: ExprLike, k: ExprLike) -> Expr:
    """The binomial coefficient, evaluated."""
    return Binomial.apply(to_expr(n), to_expr(k))


def rf(x: ExprLike, k: ExprLike) -> Expr:
    """The rising factorial, evaluated."""
    return RisingFactorial.apply(to_expr(x), to_expr(k))


def ff(x: ExprLike, k: ExprLike) -> Expr:
    """The falling factorial, evaluated."""
    return FallingFactorial.apply(to_expr(x), to_expr(k))




# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:subfactorial
class Subfactorial(DefinedFunction):
    """The number of derangements ``subfactorial(n)``, unevaluated
    (``subfactorial`` evaluates)."""

    function_name = 'subfactorial'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        value = _int(n)
        if value is not None and value >= 0:
            if value == 0:
                return ONE
            previous, current = 1, 0
            for i in range(2, value + 1):
                previous, current = current, (i - 1) * (current + previous)
            return Integer(current)
        if isinstance(n, NaN):
            return nan
        if isinstance(n, Infinity):
            return oo
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        try:
            return mpmath.mpc(mpmath.gammainc(n + 1, -1) / mpmath.e)
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('odd') and a.get('nonnegative'):
            facts['even'] = True
        if a.get('integer') and a.get('nonnegative'):
            facts.update(integer=True, nonnegative=True)
        if a.get('even') and a.get('nonnegative'):
            facts['odd'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:factorial2
class Factorial2(DefinedFunction):
    """The double factorial ``factorial2(n) = n*(n - 2)*(n - 4)*...``,
    unevaluated (``factorial2`` evaluates)."""

    function_name = 'factorial2'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        if not n.is_Number:
            return None
        value = _int(n)
        if value is None:
            raise ValueError("argument must be nonnegative integer or negative odd integer")
        if value >= 0:
            if value % 2 == 0:
                k = Integer(value // 2)
                return mul(pow(Integer(2), k), factorial(k))
            return div_(factorial(n), factorial2(sub(n, ONE)))
        if value % 2:
            return div_(mul(n, pow(Integer(-1), div_(sub(ONE, n), Integer(2)))), factorial2(mul(Integer(-1), n)))
        raise ValueError("argument must be nonnegative integer or negative odd integer")

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.fac2(values[0]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        n = self.args[0]
        facts: dict[str, bool] = {'commutative': True}
        if n.is_integer:
            if n.is_odd:
                facts['even'] = False
            elif n.is_even:
                if n.is_positive:
                    facts['even'] = True
                elif n.is_zero:
                    facts['even'] = False
            if add(n, ONE).is_nonnegative:
                facts.update(integer=True, positive=True)
            elif n.is_odd:
                integer = add(n, Integer(3)).is_nonnegative
                if integer is not None:
                    facts['integer'] = integer
                positive = div_(add(n, ONE), Integer(2)).is_even
                if positive is not None:
                    facts['positive'] = positive
        if n.is_odd:
            odd = add(n, Integer(3)).is_nonnegative
            if odd is not None:
                facts['odd'] = odd
        elif n.is_even:
            if n.is_positive:
                facts['odd'] = False
            elif n.is_zero:
                facts['odd'] = True
        return facts


def div_(a: Expr, b: Expr) -> Expr:
    return mul(a, pow(b, Integer(-1)))


def subfactorial(n: ExprLike) -> Expr:
    """``subfactorial(n)``, evaluated."""
    return Subfactorial.apply(to_expr(n))


def factorial2(n: ExprLike) -> Expr:
    """``factorial2(n)``, evaluated."""
    return Factorial2.apply(to_expr(n))
