"""The singularity (Macaulay bracket) functions on the typed core:
``SingularityFunctionNode`` (unevaluated) and ``SingularityFunction``
(evaluating; the function keeps SymPy's name).

Ported from SymPy's ``sympy.functions.special.singularity_functions``:
``<x - a>**n`` is ``0`` left of the offset, ``(x - a)**n`` right of it
for ``n >= 0``, ``0`` away from the offset and ``oo`` at it for
``n = -1, ..., -4``; non-real arguments, imaginary exponents and
exponents below ``-4`` are refused; the derivative lowers the exponent.
Not ported: the rewrites into ``Heaviside`` and ``DiracDelta`` and the
series.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.singularity_functions import SingularityFunction
>>> x, a, n = Symbol('x', real=True), Symbol('a', real=True), Symbol('n', nonnegative=True)
>>> SingularityFunction(5, 3, 2), SingularityFunction(2, 3, 2), SingularityFunction(3, 3, -1), SingularityFunction(x, a, n)
(4, 0, oo, SingularityFunction(x, a, n))
>>> SingularityFunction(x, a, 3).diff(x), SingularityFunction(x, a, 0).diff(x)
(3*SingularityFunction(x, a, 2), SingularityFunction(x, a, -1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/singularity_functions.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating class with an evaluating function, strict
# typing; the rewrites and series not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, NaN, nan, oo
from sympy_extras.core.power import pow

__all__ = ['SingularityFunctionNode', 'SingularityFunction']

ONE = Integer(1)


# PORTED FROM SYMPY: sympy/functions/special/singularity_functions.py:SingularityFunction
class SingularityFunctionNode(DefinedFunction):
    """The singularity function ``SingularityFunction(x, a, n)``,
    unevaluated (``SingularityFunction`` evaluates)."""

    function_name = 'SingularityFunction'
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import im
        x, a, n = args
        shift = sub(x, a)
        if im(shift).is_zero is False:
            raise ValueError("Singularity Functions are defined only for Real Numbers.")
        if im(n).is_zero is False:
            raise ValueError("Singularity Functions are not defined for imaginary exponents.")
        if isinstance(shift, NaN) or isinstance(n, NaN):
            return nan
        if add(n, Integer(4)).is_negative:
            raise ValueError("Singularity Functions are not defined for exponents less than -4.")
        if shift.is_extended_negative:
            return Integer(0)
        if n.is_nonnegative:
            if shift.is_zero:
                return pow(Integer(0), n)
            if shift.is_extended_nonnegative:
                return pow(shift, n)
        if n in (Integer(-1), Integer(-2), Integer(-3), Integer(-4)):
            if shift.is_negative or shift.is_extended_positive:
                return Integer(0)
            if shift.is_zero:
                return oo
        return None

    def fdiff(self, argindex: int) -> Expr:
        x, a, n = self.args
        if argindex == 1:
            if n in (Integer(0), Integer(-1), Integer(-2), Integer(-3)):
                return SingularityFunction(x, a, sub(n, ONE))
            if n.is_positive:
                return mul(n, SingularityFunction(x, a, sub(n, ONE)))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        x, a, n = values
        shift = x - a
        if mpmath.im(shift) != 0 or mpmath.im(n) != 0:
            return None
        if mpmath.re(shift) < 0:
            return mpmath.mpc(0)
        if mpmath.re(n) >= 0:
            return mpmath.mpc(shift**n)
        return mpmath.mpc(0) if shift != 0 else None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def SingularityFunction(x: ExprLike, a: ExprLike, n: ExprLike) -> Expr:
    """``SingularityFunction(x, a, n)``, evaluated (the class is
    :class:`SingularityFunctionNode`)."""
    return SingularityFunctionNode.apply(to_expr(x), to_expr(a), to_expr(n))
