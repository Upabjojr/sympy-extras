"""The Mathieu functions on the typed core: ``MathieuS``, ``MathieuC``,
``MathieuSPrime``, ``MathieuCPrime`` (unevaluated) and ``mathieus``,
``mathieuc``, ``mathieusprime``, ``mathieucprime`` (evaluating).

Ported from SymPy's ``sympy.functions.special.mathieu_functions``: at
``q = 0`` the trigonometric limits (``mathieus(a, 0, z) =
sin(sqrt(a)*z)``), the parity in ``z``, the derivatives in ``z`` and the
conjugates. No numerical values: neither SymPy nor mpmath has them.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.mathieu_functions import mathieus, mathieuc, mathieusprime
>>> a, q, z = Symbol('a'), Symbol('q'), Symbol('z')
>>> mathieus(a, 0, z), mathieuc(a, q, -z), mathieus(a, q, -z), mathieusprime(a, q, z).diff(z)
(sin(sqrt(a)*z), mathieuc(a, q, z), -mathieus(a, q, z), (-a + 2*q*cos(2*z))*mathieus(a, q, z))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/mathieu_functions.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Number
from sympy_extras.core.power import sqrt

__all__ = ['MathieuBase', 'MathieuS', 'MathieuC', 'MathieuSPrime', 'MathieuCPrime', 'mathieus', 'mathieuc',
           'mathieusprime', 'mathieucprime']

MINUS_ONE = Integer(-1)


def _sin(x: Expr) -> Expr:
    from sympy_extras.functions.elementary.trigonometric import sin
    return sin(x)


def _cos(x: Expr) -> Expr:
    from sympy_extras.functions.elementary.trigonometric import cos
    return cos(x)


def _q_zero(q: Expr) -> bool:
    return isinstance(q, Number) and q.is_zero is True


# PORTED FROM SYMPY: sympy/functions/special/mathieu_functions.py:MathieuBase
class MathieuBase(DefinedFunction):
    """The common conjugate of the Mathieu functions ``f(a, q, z)``."""

    nargs = frozenset({3})
    call_nargs = frozenset({3})

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return type(self).apply(*[conjugate(x) for x in self.args])

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def _second_derivative_factor(a: Expr, q: Expr, z: Expr) -> Expr:
    """``2*q*cos(2*z) - a``."""
    return sub(mul(Integer(2), q, _cos(mul(Integer(2), z))), a)


# PORTED FROM SYMPY: sympy/functions/special/mathieu_functions.py:mathieus
class MathieuS(MathieuBase):
    """The Mathieu sine ``mathieus(a, q, z)``, unevaluated (``mathieus``
    evaluates)."""

    unbranched = True

    function_name = 'mathieus'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        a, q, z = args
        if _q_zero(q):
            return _sin(mul(sqrt(a), z))
        if z.could_extract_minus_sign():
            return mul(MINUS_ONE, mathieus(a, q, mul(MINUS_ONE, z)))
        return None

    def fdiff(self, argindex: int) -> Expr:
        if argindex == 3:
            return mathieusprime(*self.args)
        return DefinedFunction.fdiff(self, argindex)


# PORTED FROM SYMPY: sympy/functions/special/mathieu_functions.py:mathieuc
class MathieuC(MathieuBase):
    """The Mathieu cosine ``mathieuc(a, q, z)``, unevaluated (``mathieuc``
    evaluates)."""

    unbranched = True

    function_name = 'mathieuc'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        a, q, z = args
        if _q_zero(q):
            return _cos(mul(sqrt(a), z))
        if z.could_extract_minus_sign():
            return mathieuc(a, q, mul(MINUS_ONE, z))
        return None

    def fdiff(self, argindex: int) -> Expr:
        if argindex == 3:
            return mathieucprime(*self.args)
        return DefinedFunction.fdiff(self, argindex)


# PORTED FROM SYMPY: sympy/functions/special/mathieu_functions.py:mathieusprime
class MathieuSPrime(MathieuBase):
    """The derivative ``mathieusprime(a, q, z)`` of ``mathieus``,
    unevaluated (``mathieusprime`` evaluates)."""

    unbranched = True

    function_name = 'mathieusprime'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        a, q, z = args
        if _q_zero(q):
            return mul(sqrt(a), _cos(mul(sqrt(a), z)))
        if z.could_extract_minus_sign():
            return mathieusprime(a, q, mul(MINUS_ONE, z))
        return None

    def fdiff(self, argindex: int) -> Expr:
        if argindex == 3:
            a, q, z = self.args
            return mul(_second_derivative_factor(a, q, z), mathieus(a, q, z))
        return DefinedFunction.fdiff(self, argindex)


# PORTED FROM SYMPY: sympy/functions/special/mathieu_functions.py:mathieucprime
class MathieuCPrime(MathieuBase):
    """The derivative ``mathieucprime(a, q, z)`` of ``mathieuc``,
    unevaluated (``mathieucprime`` evaluates)."""

    unbranched = True

    function_name = 'mathieucprime'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        a, q, z = args
        if _q_zero(q):
            return mul(MINUS_ONE, sqrt(a), _sin(mul(sqrt(a), z)))
        if z.could_extract_minus_sign():
            return mul(MINUS_ONE, mathieucprime(a, q, mul(MINUS_ONE, z)))
        return None

    def fdiff(self, argindex: int) -> Expr:
        if argindex == 3:
            a, q, z = self.args
            return mul(_second_derivative_factor(a, q, z), mathieuc(a, q, z))
        return DefinedFunction.fdiff(self, argindex)


def mathieus(a: ExprLike, q: ExprLike, z: ExprLike) -> Expr:
    """``mathieus(a, q, z)``, evaluated."""
    return MathieuS.apply(to_expr(a), to_expr(q), to_expr(z))


def mathieuc(a: ExprLike, q: ExprLike, z: ExprLike) -> Expr:
    """``mathieuc(a, q, z)``, evaluated."""
    return MathieuC.apply(to_expr(a), to_expr(q), to_expr(z))


def mathieusprime(a: ExprLike, q: ExprLike, z: ExprLike) -> Expr:
    """``mathieusprime(a, q, z)``, evaluated."""
    return MathieuSPrime.apply(to_expr(a), to_expr(q), to_expr(z))


def mathieucprime(a: ExprLike, q: ExprLike, z: ExprLike) -> Expr:
    """``mathieucprime(a, q, z)``, evaluated."""
    return MathieuCPrime.apply(to_expr(a), to_expr(q), to_expr(z))
