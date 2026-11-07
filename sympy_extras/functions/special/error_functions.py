"""The error functions on the typed core: ``Erf``, ``Erfc``, ``Erfi``,
``Erfinv``, ``Erfcinv`` (unevaluated nodes) and ``erf``, ``erfc``,
``erfi``, ``erfinv``, ``erfcinv`` (evaluating).

Ported from SymPy's ``sympy.functions.special.error_functions``: the
values at ``0``, ``oo``, ``-oo``, along the imaginary infinities, the
compositions with the inverses, the odd symmetry (``erf(-x) =
-erf(x)``, ``erfc(-x) = 2 - erfc(x)``), the derivatives, conjugates,
SymPy's ``real_to_real_as_real_imag`` complex parts and the facts.
``erfcinv`` of a float is evaluated (as ``erfinv(1 - x)``), where SymPy,
lacking it in mpmath, leaves the node. Not
ported: ``erf2``, ``erf2inv``, the Fresnel, exponential and
trigonometric integrals of the same module, the rewrites and series.

Examples
========

>>> from sympy_extras.core import Symbol, I, oo
>>> from sympy_extras.functions.special.error_functions import erf, erfc, erfi, erfinv, erfcinv, Erf
>>> x, r = Symbol('x'), Symbol('r', real=True)
>>> erf(0), erf(oo), erf(-x), erfc(-x), erfi(-x), erf(I*oo), erfi(I*oo), erf(erfinv(x)), erfinv(erf(r))
(0, 1, -erf(x), 2 - erfc(x), -erfi(x), oo*I, I, x, r)
>>> erf(x).diff(x), erfcinv(2), Erf(0)
(2*exp(-x**2)/sqrt(pi), -oo, erf(0))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/error_functions.py (erf, erfc, erfi,
# erfinv, erfcinv, real_to_real_as_real_imag) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# erf2, erf2inv, the other integrals, the rewrites and series not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import I, Infinity, Integer, NaN, NegativeInfinity, Rational, nan, oo, pi
from sympy_extras.core.power import pow, sqrt
from sympy_extras.functions.elementary.exponential import exp

__all__ = ['Erf', 'Erfc', 'Erfi', 'Erfinv', 'Erfcinv', 'erf', 'erfc', 'erfi', 'erfinv', 'erfcinv']

ONE = Integer(1)
ZERO = Integer(0)
MINUS_ONE = Integer(-1)
HALF = Rational(1, 2)


def _is_infinity(e: Optional[Expr]) -> bool:
    return isinstance(e, (Infinity, NegativeInfinity))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:real_to_real_as_real_imag
def _real_to_real_parts(f: DefinedFunction, deep: bool) -> Optional[tuple[Expr, Expr]]:
    from sympy_extras.functions.elementary.complexes import as_real_imag
    a = f.args[0]
    if a.is_extended_real:
        return (f, ZERO)
    parts = as_real_imag(a.expand() if deep else a, deep)
    if parts is None:
        return None
    x, y = parts
    cls = type(f)
    plus, minus = cls.apply(add(x, mul(I, y))), cls.apply(sub(x, mul(I, y)))
    return (div(add(plus, minus), Integer(2)), div(sub(plus, minus), mul(Integer(2), I)))


class _ErrorFunction(DefinedFunction):
    """The common methods: conjugate and complex parts."""

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return type(self).apply(conjugate(self.args[0]))

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr]) -> Optional[tuple[Expr, Expr]]:
        return _real_to_real_parts(self, deep)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf
class Erf(_ErrorFunction):
    """The error function, unevaluated (``erf`` evaluates)."""

    function_name = 'erf'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [a] = args
        if isinstance(a, NaN):
            return nan
        if isinstance(a, Infinity):
            return ONE
        if isinstance(a, NegativeInfinity):
            return MINUS_ONE
        if a.is_zero:
            return ZERO
        if isinstance(a, Erfinv):
            return a.args[0]
        if isinstance(a, Erfcinv):
            return sub(ONE, a.args[0])
        if _is_infinity(a.extract_multiplicatively(I)):
            return a
        if a.could_extract_minus_sign():
            return mul(MINUS_ONE, erf(mul(MINUS_ONE, a)))
        return None

    def fdiff(self, argindex: int) -> Expr:
        return div(mul(Integer(2), exp(mul(MINUS_ONE, pow(self.args[0], Integer(2))))), sqrt(pi))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.erf(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('finite') or a.get('extended_real'):
            facts['finite'] = True
        if a.get('extended_real'):
            facts.update(real=True)
            for name, source in (('zero', 'zero'), ('positive', 'extended_positive'), ('negative', 'extended_negative')):
                if a.get(source) is not None:
                    facts[name] = a[source]
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfc
class Erfc(_ErrorFunction):
    """The complementary error function ``1 - erf(x)``, unevaluated
    (``erfc`` evaluates)."""

    function_name = 'erfc'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [a] = args
        if isinstance(a, NaN):
            return nan
        if isinstance(a, Infinity):
            return ZERO
        if a.is_zero:
            return ONE
        if isinstance(a, Erfinv):
            return sub(ONE, a.args[0])
        if isinstance(a, Erfcinv):
            return a.args[0]
        if _is_infinity(a.extract_multiplicatively(I)):
            return mul(MINUS_ONE, a)
        if a.could_extract_minus_sign():
            return sub(Integer(2), erfc(mul(MINUS_ONE, a)))
        return None

    def fdiff(self, argindex: int) -> Expr:
        return div(mul(Integer(-2), exp(mul(MINUS_ONE, pow(self.args[0], Integer(2))))), sqrt(pi))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.erfc(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('extended_real'):
            facts['real'] = True
        elif a.get('imaginary'):
            facts['real'] = False
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfi
class Erfi(_ErrorFunction):
    """The imaginary error function ``-I*erf(I*x)``, unevaluated (``erfi``
    evaluates)."""

    function_name = 'erfi'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, NaN):
            return nan
        if z.is_zero:
            return ZERO
        if isinstance(z, Infinity):
            return oo
        if z.could_extract_minus_sign():
            return mul(MINUS_ONE, erfi(mul(MINUS_ONE, z)))
        rotated = z.extract_multiplicatively(I)
        if rotated is not None:
            if isinstance(rotated, Infinity):
                return I
            if isinstance(rotated, Erfinv):
                return mul(I, rotated.args[0])
            if isinstance(rotated, Erfcinv):
                return mul(I, sub(ONE, rotated.args[0]))
        return None

    def fdiff(self, argindex: int) -> Expr:
        return div(mul(Integer(2), exp(pow(self.args[0], Integer(2)))), sqrt(pi))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.erfi(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('extended_real'):
            facts['extended_real'] = True
        if a.get('zero') is not None:
            facts['zero'] = a['zero']
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfinv
class Erfinv(DefinedFunction):
    """The inverse of ``erf`` on ``(-1, 1)``, unevaluated (``erfinv``
    evaluates)."""

    function_name = 'erfinv'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, NaN):
            return nan
        if z == MINUS_ONE:
            return NegativeInfinity()
        if z.is_zero:
            return ZERO
        if z == ONE:
            return oo
        if isinstance(z, Erf) and z.args[0].is_extended_real:
            return z.args[0]
        negated = z.extract_multiplicatively(MINUS_ONE)
        if isinstance(negated, Erf) and negated.args[0].is_extended_real:
            return mul(MINUS_ONE, negated.args[0])
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(sqrt(pi), exp(pow(erfinv(self.args[0]), Integer(2))), HALF)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        z = values[0]
        if mpmath.im(z) != 0 or not -1 < mpmath.re(z) < 1:
            return None
        return mpmath.mpc(mpmath.erfinv(mpmath.re(z)))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('zero') is not None:
            facts['zero'] = a['zero']
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfcinv
class Erfcinv(DefinedFunction):
    """The inverse of ``erfc`` on ``(0, 2)``, unevaluated (``erfcinv``
    evaluates)."""

    function_name = 'erfcinv'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, NaN):
            return nan
        if z.is_zero:
            return oo
        if z == ONE:
            return ZERO
        if z == Integer(2):
            return NegativeInfinity()
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(MINUS_ONE, sqrt(pi), exp(pow(erfcinv(self.args[0]), Integer(2))), HALF)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        z = values[0]
        if mpmath.im(z) != 0 or not 0 < mpmath.re(z) < 2:
            return None
        return mpmath.mpc(mpmath.erfinv(1 - mpmath.re(z)))

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        zero = sub(self.args[0], ONE).is_zero
        if zero is not None:
            facts['zero'] = zero
        return facts


def erf(x: ExprLike) -> Expr:
    """The error function of ``x``, evaluated."""
    return Erf.apply(to_expr(x))


def erfc(x: ExprLike) -> Expr:
    """``erfc(x)``, evaluated."""
    return Erfc.apply(to_expr(x))


def erfi(x: ExprLike) -> Expr:
    """``erfi(x)``, evaluated."""
    return Erfi.apply(to_expr(x))


def erfinv(x: ExprLike) -> Expr:
    """``erfinv(x)``, evaluated."""
    return Erfinv.apply(to_expr(x))


def erfcinv(x: ExprLike) -> Expr:
    """``erfcinv(x)``, evaluated."""
    return Erfcinv.apply(to_expr(x))
