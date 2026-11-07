"""The spherical harmonics on the typed core: ``YnmNode``, ``ZnmNode``
(unevaluated) and ``Ynm``, ``Znm``, ``Ynm_c`` (evaluating; the functions
keep SymPy's names).

Ported from SymPy's ``sympy.functions.special.spherical_harmonics``: the
minus signs of ``m``, ``theta`` and ``phi`` taken out
(``Ynm(n, -m, theta, phi) = (-1)**m*exp(-2*I*m*phi)*Ynm(n, m, theta,
phi)``), the derivatives in ``theta`` and ``phi``, the conjugate
``(-1)**m*Ynm(n, -m, theta, phi)``, the real harmonics ``Znm`` and
mpmath values. Not ported: the expansion into trigonometric polynomials
(``expand(func=True)``), the rewrites and ``as_real_imag``.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.spherical_harmonics import Ynm, Znm, Ynm_c
>>> n, m, t, p = Symbol('n'), Symbol('m'), Symbol('theta'), Symbol('phi')
>>> Ynm(n, m, -t, p), Ynm(n, m, t, p).diff(p), Ynm_c(n, m, t, p)
(Ynm(n, m, theta, phi), I*m*Ynm(n, m, theta, phi), (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the trigonometric expansion, the rewrites and as_real_imag not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import I, Integer
from sympy_extras.core.power import pow, sqrt

__all__ = ['YnmNode', 'ZnmNode', 'Ynm', 'Znm', 'Ynm_c']

ONE = Integer(1)
MINUS_ONE = Integer(-1)


def _exp(x: Expr) -> Expr:
    from sympy_extras.functions.elementary.exponential import exp
    return exp(x)


# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Ynm
class YnmNode(DefinedFunction):
    """The spherical harmonic ``Ynm(n, m, theta, phi)``, unevaluated
    (``Ynm`` evaluates)."""

    function_name = 'Ynm'
    nargs = frozenset({4})
    call_nargs = frozenset({4})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        n, m, theta, phi = args
        if m.could_extract_minus_sign():
            m = mul(MINUS_ONE, m)
            return mul(mul(pow(MINUS_ONE, m), _exp(mul(Integer(-2), I, m, phi))), Ynm(n, m, theta, phi))
        if theta.could_extract_minus_sign():
            return Ynm(n, m, mul(MINUS_ONE, theta), phi)
        if phi.could_extract_minus_sign():
            phi = mul(MINUS_ONE, phi)
            return mul(_exp(mul(Integer(-2), I, m, phi)), Ynm(n, m, theta, phi))
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import cot
        n, m, theta, phi = self.args
        if argindex == 3:
            return add(mul(mul(m, cot(theta)), Ynm(n, m, theta, phi)),
                       mul(mul(sqrt(mul(sub(n, m), add(n, m, ONE))), _exp(mul(MINUS_ONE, I, phi))),
                           Ynm(n, add(m, ONE), theta, phi)))
        if argindex == 4:
            return mul(mul(I, m), Ynm(n, m, theta, phi))
        return DefinedFunction.fdiff(self, argindex)

    def _eval_conjugate(self) -> Optional[Expr]:
        n, m, theta, phi = self.args
        return mul(pow(MINUS_ONE, m), Ynm(n, mul(MINUS_ONE, m), theta, phi))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        # mpmath wants real orders and angles where they are real
        n, m, theta, phi = (mpmath.re(v) if mpmath.im(v) == 0 else v for v in values)
        try:
            return mpmath.mpc(mpmath.spherharm(n, m, theta, phi))
        except (ValueError, ZeroDivisionError, TypeError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Znm
class ZnmNode(DefinedFunction):
    """The real spherical harmonic ``Znm(n, m, theta, phi)``, unevaluated
    (``Znm`` evaluates, into ``Ynm`` for a known sign of ``m``)."""

    function_name = 'Znm'
    nargs = frozenset({4})
    call_nargs = frozenset({4})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        n, m, theta, phi = args
        if m.is_positive:
            return div(add(Ynm(n, m, theta, phi), Ynm_c(n, m, theta, phi)), sqrt(Integer(2)))
        if m.is_zero:
            return Ynm(n, m, theta, phi)
        if m.is_negative:
            return div(sub(Ynm(n, m, theta, phi), Ynm_c(n, m, theta, phi)), mul(sqrt(Integer(2)), I))
        return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def Ynm(n: ExprLike, m: ExprLike, theta: ExprLike, phi: ExprLike) -> Expr:
    """``Ynm(n, m, theta, phi)``, evaluated (the class is :class:`YnmNode`)."""
    return YnmNode.apply(to_expr(n), to_expr(m), to_expr(theta), to_expr(phi))


# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Ynm_c
def Ynm_c(n: ExprLike, m: ExprLike, theta: ExprLike, phi: ExprLike) -> Expr:
    """The conjugate ``conjugate(Ynm(n, m, theta, phi))``."""
    from sympy_extras.functions.elementary.complexes import conjugate
    return conjugate(Ynm(n, m, theta, phi))


def Znm(n: ExprLike, m: ExprLike, theta: ExprLike, phi: ExprLike) -> Expr:
    """``Znm(n, m, theta, phi)``, evaluated (the class is :class:`ZnmNode`)."""
    return ZnmNode.apply(to_expr(n), to_expr(m), to_expr(theta), to_expr(phi))
