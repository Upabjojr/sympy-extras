"""Combinatorial numbers on the typed core: ``Catalan`` (the Catalan
numbers as a function, unevaluated) and ``catalan`` (evaluating).

Ported from SymPy's ``sympy.functions.combinatorial.numbers.catalan``:
``catalan(n) = 4**n*gamma(n + 1/2)/(gamma(1/2)*gamma(n + 2))`` at the
nonnegative integers and the negative non-integers, ``-1/2`` at ``-1``,
``0`` at the integers below ``-1``; the derivative, facts and mpmath
values. The other numbers of the module (Fibonacci, Bernoulli as
functions, Bell, Euler, ...) are not ported; the Bernoulli numbers
themselves are in :mod:`sympy_extras.functions.special.zeta_functions`.

Examples
========

>>> from sympy_extras.core import Rational, Symbol
>>> from sympy_extras.functions.combinatorial.numbers import catalan
>>> catalan(5), catalan(0), catalan(-1), catalan(-3), catalan(Rational(-1, 2)), catalan(Rational(1, 2))
(42, 1, -1/2, 0, zoo, catalan(1/2))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:catalan (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating class with an evaluating function, strict
# typing; the other numbers of the module not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import pow

__all__ = ['Catalan', 'catalan']

HALF = Rational(1, 2)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:catalan
class Catalan(DefinedFunction):
    """The Catalan numbers ``catalan(n)``, unevaluated (``catalan``
    evaluates). Not the constant ``Catalan`` of the core, which is
    :data:`sympy_extras.core.Catalan`."""

    function_name = 'catalan'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.special.gamma_functions import gamma
        [n] = args
        if (isinstance(n, Rational) and n.q == 1 and n.p >= 0) or (n.is_noninteger and n.is_negative):
            return div(mul(pow(Integer(4), n), gamma(add(n, HALF))), mul(gamma(HALF), gamma(add(n, Integer(2)))))
        if n.is_integer and n.is_negative:
            if add(n, Integer(1)).is_negative:
                return Integer(0)
            if add(n, Integer(1)).is_zero:
                return Rational(-1, 2)
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.elementary.exponential import log
        from sympy_extras.functions.special.gamma_functions import polygamma
        n = self.args[0]
        return mul(catalan(n), add(sub(polygamma(Integer(0), add(n, HALF)), polygamma(Integer(0), add(n, Integer(2)))),
                                   log(Integer(4))))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        try:
            return mpmath.mpc(4**n*mpmath.gamma(n + 0.5)/(mpmath.gamma(0.5)*mpmath.gamma(n + 2)))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        n = self.args[0]
        a = facts_of(n)
        facts: dict[str, bool] = {'commutative': True}
        if a.get('integer') and a.get('nonnegative'):
            facts['integer'] = True
        if a.get('nonnegative'):
            facts['positive'] = True
        if a.get('integer') and sub(n, Integer(3)).is_positive:
            facts['composite'] = True
        return facts


def catalan(n: ExprLike) -> Expr:
    """The Catalan number ``catalan(n)``, evaluated."""
    return Catalan.apply(to_expr(n))
