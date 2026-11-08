"""The beta function on the typed core: ``Beta`` (unevaluated) and
``beta`` (evaluating).

Ported from SymPy's ``sympy.functions.special.beta_functions.beta``:
``beta(x)`` is ``beta(x, x)``; for numbers SymPy's ``doit`` rules apply
(``zoo`` at a zero argument, ``1/x`` and ``1/y`` at the other argument
``1``, ``1/(x*y*catalan(x))`` when ``y = x + 1``, ``0`` when ``x + y`` is
a negative integer and ``x``, ``y`` are not integers); the derivatives
with ``digamma``, conjugate, facts and mpmath values; the generalised
incomplete beta functions ``betainc(a, b, x1, x2)`` and
``betainc_regularized`` (nodes, as in SymPy, with their derivatives in
the limits).

Examples
========

>>> from sympy_extras.core import Rational, Symbol
>>> from sympy_extras.functions.special.beta_functions import beta, Beta
>>> x, y = Symbol('x'), Symbol('y')
>>> beta(2, 3), beta(Rational(1, 2), Rational(3, 2)), beta(5, 1), beta(x), beta(x, y).diff(x), Beta(2, 3)
(1/12, 4/(3*catalan(1/2)), 1/5, beta(x, x), (polygamma(0, x) - polygamma(0, x + y))*beta(x, y), beta(2, 3))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:beta (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating class with an evaluating function, strict
# typing, SymPy's doit rules for numbers in eval; betainc not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import Integer, Number, zoo

__all__ = ['Beta', 'BetaInc', 'BetaIncRegularized', 'beta', 'betainc', 'betainc_regularized']

ONE = Integer(1)


# PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:beta.doit
def _beta_numbers(x: Expr, y: Expr) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.numbers import catalan
    if y.is_zero or x.is_zero:
        return zoo
    if y == ONE:
        return div(ONE, x)
    if x == ONE:
        return div(ONE, y)
    if y == add(x, ONE):
        return div(ONE, mul(x, y, catalan(x)))
    s = add(x, y)
    if s.is_integer and s.is_negative and x.is_integer is False and y.is_integer is False:
        return Integer(0)
    return None


# PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:beta
class Beta(DefinedFunction):
    """The Euler beta function ``beta(x, y)``, unevaluated (``beta``
    evaluates)."""

    unbranched = True

    function_name = 'beta'
    generic_nseries = True
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        if len(args) == 1:
            return beta(args[0], args[0])
        x, y = args
        if isinstance(x, Number) and isinstance(y, Number):
            return _beta_numbers(x, y)
        return None

    def fdiff(self, argindex: int) -> Expr:
        from .gamma_functions import digamma
        x, y = self.args
        own = x if argindex == 1 else y
        return mul(beta(x, y), sub(digamma(own), digamma(add(x, y))))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return beta(*[conjugate(a) for a in self.args])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        x, y = values[0], values[-1]
        try:
            return mpmath.mpc(mpmath.beta(x, y))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        real = [a.is_real for a in self.args]
        if all(v is True for v in real):
            facts['real'] = True
        elif real[0] is False or (real[0] is True and real[-1] is False):
            facts['real'] = False
        return facts


def beta(x: ExprLike, y: Optional[ExprLike] = None) -> Expr:
    """``beta(x, y)`` (``beta(x, x)`` for one argument), evaluated."""
    if y is None:
        return Beta.apply(to_expr(x))
    return Beta.apply(to_expr(x), to_expr(y))



class _IncompleteBeta(DefinedFunction):
    """The common methods of ``betainc`` and ``betainc_regularized``."""

    nargs = frozenset({4})
    call_nargs = frozenset({4})
    regularized: bool = False

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.core.power import pow
        a, b, x1, x2 = self.args
        if argindex not in (3, 4):
            return DefinedFunction.fdiff(self, argindex)
        x = x1 if argindex == 3 else x2
        value = mul(pow(sub(ONE, x), sub(b, ONE)), pow(x, sub(a, ONE)))
        if argindex == 3:
            value = mul(Integer(-1), value)
        return div(value, beta(a, b)) if self.regularized else value

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return type(self).apply(*[conjugate(x) for x in self.args])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        a, b, x1, x2 = values
        try:
            return mpmath.mpc(mpmath.betainc(a, b, x1, x2, regularized=self.regularized))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if all(x.is_real for x in self.args):
            facts['real'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:betainc
class BetaInc(_IncompleteBeta):
    """The generalised incomplete beta function ``betainc(a, b, x1, x2)``
    (no values of its own, as in SymPy)."""

    unbranched = True

    function_name = 'betainc'
    generic_nseries = True


# PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:betainc_regularized
class BetaIncRegularized(_IncompleteBeta):
    """The regularised incomplete beta function
    ``betainc_regularized(a, b, x1, x2)``."""

    unbranched = True

    function_name = 'betainc_regularized'
    generic_nseries = True
    regularized = True


def betainc(a: ExprLike, b: ExprLike, x1: ExprLike, x2: ExprLike) -> Expr:
    """``betainc(a, b, x1, x2)``."""
    return BetaInc.apply(to_expr(a), to_expr(b), to_expr(x1), to_expr(x2))


def betainc_regularized(a: ExprLike, b: ExprLike, x1: ExprLike, x2: ExprLike) -> Expr:
    """``betainc_regularized(a, b, x1, x2)``."""
    return BetaIncRegularized.apply(to_expr(a), to_expr(b), to_expr(x1), to_expr(x2))
