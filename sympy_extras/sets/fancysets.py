"""The named sets of numbers on the typed core: the natural numbers (from 1
and from 0), the integers, the rationals and the complex numbers
(:data:`~sympy_extras.sets.sets.Reals` is in :mod:`~sympy_extras.sets.sets`).

Ported from SymPy's ``sympy.sets.fancysets``, as far as the memberships
go: each set is a singleton which decides the membership of an
expression from its assumptions (``contains(2, Integers)`` is ``True``,
``contains(x, Integers)`` stays unevaluated).

Examples
========

>>> from sympy_extras.core import Symbol, Rational, sqrt
>>> from sympy_extras.sets import Integers, Naturals0, Rationals, contains
>>> x = Symbol('x')
>>> contains(2, Integers), contains(Rational(1, 2), Integers), contains(x, Integers)
(True, False, Contains(x, Integers))
>>> contains(0, Naturals0), contains(sqrt(2), Rationals)
(True, False)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/sets/fancysets.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; only the singletons and their memberships (no ranges,
# images, iteration); Complexes decides by is_complex instead of the real
# and imaginary parts of a CartesianComplexRegion.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.core.numbers import Integer, NegativeInfinity, oo
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.sorting import ClassKey

from .sets import Set

__all__ = ['NaturalsClass', 'Naturals0Class', 'IntegersClass', 'RationalsClass', 'ComplexesClass',
           'Naturals', 'Naturals0', 'Integers', 'Rationals', 'Complexes']


def _tfn(value: Optional[bool]) -> Optional[Boolean]:
    """SymPy's ``tfn``: ``true``, ``false`` or ``None``."""
    if value is None:
        return None
    return true if value else false


class _NamedSet(Set):
    """A singleton set of numbers, printed by its name."""

    __slots__ = ()

    #: the printed name (SymPy's)
    set_name: str = ''

    @property
    def args(self) -> tuple[Basic, ...]:
        return ()

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, cls.set_name)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self

    def _sympystr(self, printer: StrPrinter) -> str:
        return self.set_name

    @property
    def is_empty(self) -> Optional[bool]:
        return False

    @property
    def is_finite_set(self) -> Optional[bool]:
        return False


# PORTED FROM SYMPY: sympy/sets/fancysets.py:Naturals
class NaturalsClass(_NamedSet):
    """The natural numbers ``1, 2, 3, ...`` (the singleton :data:`Naturals`)."""

    __slots__ = ()

    set_name = 'Naturals'

    def _contains(self, other: Expr) -> Optional[Boolean]:
        if other.is_positive and other.is_integer:
            return true
        if other.is_integer is False or other.is_positive is False:
            return false
        return None

    def _inf(self) -> Expr:
        return Integer(1)

    def _sup(self) -> Expr:
        return oo


# PORTED FROM SYMPY: sympy/sets/fancysets.py:Naturals0
class Naturals0Class(_NamedSet):
    """The natural numbers with zero (the singleton :data:`Naturals0`)."""

    __slots__ = ()

    set_name = 'Naturals0'

    def _contains(self, other: Expr) -> Optional[Boolean]:
        if other.is_integer and other.is_nonnegative:
            return true
        if other.is_integer is False or other.is_nonnegative is False:
            return false
        return None

    def _inf(self) -> Expr:
        return Integer(0)

    def _sup(self) -> Expr:
        return oo


# PORTED FROM SYMPY: sympy/sets/fancysets.py:Integers
class IntegersClass(_NamedSet):
    """The integers (the singleton :data:`Integers`)."""

    __slots__ = ()

    set_name = 'Integers'

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return _tfn(other.is_integer)

    def _inf(self) -> Expr:
        return NegativeInfinity()

    def _sup(self) -> Expr:
        return oo


# PORTED FROM SYMPY: sympy/sets/fancysets.py:Rationals
class RationalsClass(_NamedSet):
    """The rational numbers (the singleton :data:`Rationals`)."""

    __slots__ = ()

    set_name = 'Rationals'

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return _tfn(other.is_rational)

    def _inf(self) -> Expr:
        return NegativeInfinity()

    def _sup(self) -> Expr:
        return oo


# PORTED FROM SYMPY: sympy/sets/fancysets.py:Complexes
class ComplexesClass(_NamedSet):
    """The complex numbers (the singleton :data:`Complexes`)."""

    __slots__ = ()

    set_name = 'Complexes'

    def _contains(self, other: Expr) -> Optional[Boolean]:
        return _tfn(other.is_complex)


Naturals: Final[NaturalsClass] = NaturalsClass()
Naturals0: Final[Naturals0Class] = Naturals0Class()
Integers: Final[IntegersClass] = IntegersClass()
Rationals: Final[RationalsClass] = RationalsClass()
Complexes: Final[ComplexesClass] = ComplexesClass()
