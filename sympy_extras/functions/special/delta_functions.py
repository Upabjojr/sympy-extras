"""The Dirac delta and the Heaviside step on the typed core:
``DiracDelta``, ``Heaviside`` (unevaluated nodes) and ``dirac_delta``,
``heaviside`` (evaluating).

Ported from SymPy's ``sympy.functions.special.delta_functions``:
``DiracDelta(x, k)`` (the ``k``-th derivative, ``k`` a nonnegative
integer, ``DiracDelta(x, 0)`` printed ``DiracDelta(x)``) is zero at a
nonzero argument and takes out a minus sign (``DiracDelta(-x, k) =
(-1)**k*DiracDelta(x, k)``); ``Heaviside(x, H0)`` (``H0 = 1/2`` by
default, as in SymPy, and not printed) is ``0`` at a negative argument,
``1`` at a positive one, ``H0`` at zero. Both refuse an argument known to
be non-real. The derivatives: ``Heaviside' = DiracDelta``,
``DiracDelta(x, k)' = DiracDelta(x, k + 1)``. Not ported: the
simplifications of the delta of a polynomial (``expand(diracdelta=True)``)
and the rewrites.

Examples
========

>>> from sympy_extras.core import Symbol, Integer
>>> from sympy_extras.functions.special.delta_functions import dirac_delta, heaviside, Heaviside
>>> x, p = Symbol('x', real=True), Symbol('p', positive=True)
>>> heaviside(p), heaviside(-p), heaviside(0), heaviside(0, 1), heaviside(x), heaviside(x).diff(x)
(1, 0, 1/2, 1, Heaviside(x), DiracDelta(x))
>>> dirac_delta(p), dirac_delta(-x, 1), dirac_delta(x, 0), dirac_delta(x).diff(x), Heaviside(1)
(0, -DiracDelta(x, 1), DiracDelta(x), DiracDelta(x, 1), Heaviside(1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions (named
# dirac_delta and heaviside), strict typing; the expansion of the delta
# of a polynomial and the rewrites not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, NaN, Number, Rational, nan
from sympy_extras.core.printing import StrPrinter

__all__ = ['DiracDelta', 'Heaviside', 'dirac_delta', 'heaviside']

ZERO = Integer(0)
ONE = Integer(1)
HALF = Rational(1, 2)

_REAL_FACTS = {'commutative': True, 'real': True, 'extended_real': True, 'finite': True, 'complex': True,
               'imaginary': False}


def _refuse_complex(arg: Expr) -> None:
    from sympy_extras.functions.elementary.complexes import im
    if im(arg).is_zero is False:
        raise ValueError("Function defined only for Real Values. Complex part: %s  found in %s ." % (im(arg), arg))


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:DiracDelta
class DiracDelta(DefinedFunction):
    """The Dirac delta ``DiracDelta(x)`` and its derivatives
    ``DiracDelta(x, k)``, unevaluated (``dirac_delta`` evaluates)."""

    function_name = 'DiracDelta'
    generic_nseries = True
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        arg = args[0]
        k = args[1] if len(args) > 1 else ZERO
        if not (isinstance(k, Rational) and k.q == 1) or k.p < 0:
            raise ValueError("Error: the second argument of DiracDelta must be a non-negative integer, %s given "
                             "instead." % (k,))
        if isinstance(arg, NaN):
            return nan
        if arg.is_nonzero:
            return ZERO
        _refuse_complex(arg)
        # SymPy's args_cnc splits a negative numerical coefficient into -1
        # and its absolute value
        coefficient, _ = arg.as_coeff_Mul()
        if isinstance(coefficient, Number) and coefficient.is_extended_negative:
            negated = mul(Integer(-1), arg)
            if k.p % 2:
                return mul(Integer(-1), dirac_delta(negated, k))
            return dirac_delta(negated, k) if k.p else dirac_delta(negated)
        if k.p == 0 and len(args) > 1:
            return cls(arg)
        return None

    def fdiff(self, argindex: int) -> Expr:
        k = self.args[1] if len(self.args) > 1 else ZERO
        return dirac_delta(self.args[0], add(k, ONE))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(0) if values[0] != 0 else None

    def _eval_facts(self) -> dict[str, bool]:
        return dict(_REAL_FACTS)


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:Heaviside
class Heaviside(DefinedFunction):
    """The Heaviside step ``Heaviside(x, H0)`` (``H0`` its value at zero,
    ``1/2`` when not given, as in SymPy), unevaluated (``heaviside``
    evaluates)."""

    function_name = 'Heaviside'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({1, 2})

    def __init__(self, arg: ExprLike, h0: ExprLike = HALF) -> None:
        # SymPy's __new__: a Heaviside of one argument as H0 means 1/2
        value = to_expr(h0)
        if isinstance(value, Heaviside):
            value = HALF
        DefinedFunction.__init__(self, arg, value)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        arg = args[0]
        h0 = args[1] if len(args) > 1 else HALF
        if arg.is_extended_negative:
            return ZERO
        if arg.is_extended_positive:
            return ONE
        if arg.is_zero:
            return h0
        if isinstance(arg, NaN):
            return nan
        _refuse_complex(arg)
        return None

    @property
    def pargs(self) -> tuple[Expr, ...]:
        """The arguments without the default ``H0 = 1/2`` (SymPy's)."""
        return self.args[:1] if self.args[1] == HALF else self.args

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Heaviside(%s)' % ', '.join(printer.doprint(a) for a in self.pargs)

    def fdiff(self, argindex: int) -> Expr:
        return dirac_delta(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        x, h0 = values
        if mpmath.im(x) != 0:
            return None
        if mpmath.re(x) == 0:
            return h0
        return mpmath.mpc(1 if mpmath.re(x) > 0 else 0)

    def _eval_facts(self) -> dict[str, bool]:
        return dict(_REAL_FACTS)


def dirac_delta(x: ExprLike, k: ExprLike = 0) -> Expr:
    """``DiracDelta(x, k)``, evaluated."""
    if isinstance(k, int) and k == 0:
        return DiracDelta.apply(to_expr(x))
    return DiracDelta.apply(to_expr(x), to_expr(k))


def heaviside(x: ExprLike, h0: ExprLike = HALF) -> Expr:
    """``Heaviside(x, H0)``, evaluated."""
    return Heaviside.apply(to_expr(x), to_expr(h0))

