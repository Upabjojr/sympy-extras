"""The Kronecker delta and the Levi-Civita symbol on the typed core:
``KroneckerDeltaNode``, ``LeviCivitaNode`` (unevaluated) and
``KroneckerDelta``, ``LeviCivita`` (evaluating; the functions keep
SymPy's names).

Ported from SymPy's ``sympy.functions.special.tensor_functions``: the
delta is ``1`` when the difference of its indices is zero, ``0`` when it
is known nonzero, and its indices are put in canonical order; the Levi-Civita symbol of integers is
the product formula, ``0`` with a repeated index. Not ported: the
optional index range of ``KroneckerDelta`` (a ``Tuple`` argument, which
the functions of the core, taking expressions, cannot hold), the Fermi
level assumptions (``above_fermi``, ``below_fermi``), the powers
(``KroneckerDelta(i, j)**2``) and the rewrites.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.tensor_functions import KroneckerDelta, LeviCivita
>>> i, j = Symbol('i', integer=True), Symbol('j', integer=True)
>>> KroneckerDelta(1, 1), KroneckerDelta(1, 2), KroneckerDelta(j, i), KroneckerDelta(i, i + 1)
(1, 0, KroneckerDelta(i, j), 0)
>>> LeviCivita(1, 2, 3), LeviCivita(2, 1, 3), LeviCivita(1, 1, 2), LeviCivita(i, j, i)
(1, -1, 0, 0)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/tensor_functions.py (KroneckerDelta,
# LeviCivita, eval_levicivita) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the Fermi level assumptions, powers and rewrites not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Optional

from sympy_extras.core.add import sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.sorting import default_sort_key

__all__ = ['KroneckerDeltaNode', 'LeviCivitaNode', 'KroneckerDelta', 'LeviCivita', 'eval_levicivita']

_INTEGER_FACTS = {'commutative': True, 'integer': True, 'rational': True, 'real': True, 'extended_real': True,
                  'finite': True, 'complex': True, 'hermitian': True, 'algebraic': True, 'imaginary': False,
                  'infinite': False, 'noninteger': False, 'irrational': False, 'transcendental': False}


# PORTED FROM SYMPY: sympy/functions/special/tensor_functions.py:KroneckerDelta
class KroneckerDeltaNode(DefinedFunction):
    """The Kronecker delta ``KroneckerDelta(i, j)``, unevaluated
    (``KroneckerDelta`` evaluates)."""

    function_name = 'KroneckerDelta'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        i, j = args
        difference = sub(i, j)
        if difference.is_zero:
            return Integer(1)
        if difference.is_zero is False:
            return Integer(0)
        if default_sort_key(j) < default_sort_key(i):
            return KroneckerDelta(j, i)
        return None

    def _eval_facts(self) -> dict[str, bool]:
        return dict(_INTEGER_FACTS)


# PORTED FROM SYMPY: sympy/functions/special/tensor_functions.py:eval_levicivita
def eval_levicivita(*args: int) -> Expr:
    """The Levi-Civita symbol of integers:
    ``prod(prod(a_j - a_i, j > i)/i!)``."""
    value = Fraction(1)
    for i in range(len(args)):
        inner = 1
        for j in range(i + 1, len(args)):
            inner *= args[j] - args[i]
        value *= Fraction(inner, factorial(i))
    return number_from_fraction(value)


# PORTED FROM SYMPY: sympy/functions/special/tensor_functions.py:LeviCivita
class LeviCivitaNode(DefinedFunction):
    """The Levi-Civita symbol ``LeviCivita(i, j, k, ...)``, unevaluated
    (``LeviCivita`` evaluates)."""

    function_name = 'LeviCivita'
    variadic = True
    nargs = frozenset()
    call_nargs = frozenset()

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        integers = [a.p for a in args if isinstance(a, Rational) and a.q == 1]
        if len(integers) == len(args):
            return eval_levicivita(*integers)
        if len(set(args)) != len(args):
            return Integer(0)
        return None

    def _eval_facts(self) -> dict[str, bool]:
        return dict(_INTEGER_FACTS)


def KroneckerDelta(i: ExprLike, j: ExprLike) -> Expr:
    """``KroneckerDelta(i, j)``, evaluated (the class is
    :class:`KroneckerDeltaNode`)."""
    return KroneckerDeltaNode.apply(to_expr(i), to_expr(j))


def LeviCivita(*args: ExprLike) -> Expr:
    """``LeviCivita(i, j, ...)``, evaluated (the class is
    :class:`LeviCivitaNode`)."""
    return LeviCivitaNode.apply(*[to_expr(a) for a in args])
