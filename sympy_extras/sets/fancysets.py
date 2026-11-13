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
# Changes: typed; the singletons and their memberships, and ImageSet
# without its rules of membership and iteration (no ranges); the
# function of an ImageSet is its variables and its image (a tuple of
# expressions for a function with tuple values); Complexes decides by
# is_complex instead of the real and imaginary parts of a
# CartesianComplexRegion.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional

from sympy_extras.core.basic import Basic, NodePrinter
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.fuzzy import fuzzy_or
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.core.numbers import Integer, NegativeInfinity, oo
from sympy_extras.core.sorting import ClassKey
from sympy_extras.core.symbol import Symbol

from .sets import Set

__all__ = ['NaturalsClass', 'Naturals0Class', 'IntegersClass', 'RationalsClass', 'ComplexesClass',
           'Naturals', 'Naturals0', 'Integers', 'Rationals', 'Complexes', 'ImageSet', 'image_set']


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

    def _sympystr(self, printer: NodePrinter) -> str:
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


# PORTED FROM SYMPY: sympy/sets/fancysets.py:ImageSet
class ImageSet(Set):
    """The image of sets under a function: ``variables`` (a tuple of
    symbols, one for each base set) mapped to ``image``, an expression or
    a tuple of expressions (SymPy's ``ImageSet(Lambda(variables, image),
    *base_sets)``; unevaluated, :func:`image_set` evaluates)."""

    __slots__ = ('_variables', '_image', '_base_sets')

    def __init__(self, variables: Tuple, image: Basic, *base_sets: Set) -> None:
        if not all(isinstance(v, Symbol) for v in variables.args):
            raise TypeError('the variables of an ImageSet are symbols, got %s' % (variables,))
        if not (isinstance(image, Expr) or (isinstance(image, Tuple)
                                            and all(isinstance(e, Expr) for e in image.args))):
            raise TypeError('the image of an ImageSet is an expression or a tuple of them, got %s' % (image,))
        self._variables: Final[Tuple] = variables
        self._image: Final[Basic] = image
        self._base_sets: Final[tuple[Set, ...]] = base_sets
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._variables, self._image) + self._base_sets

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        variables, image, base_sets = _image_set_args(args)
        return ImageSet(variables, image, *base_sets)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        variables, image, base_sets = _image_set_args(args)
        return image_set(variables, image, *base_sets)

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return tuple(v for v in self._variables.args if isinstance(v, Symbol))

    @property
    def image(self) -> Basic:
        """The value of the function: an expression or a tuple."""
        return self._image

    @property
    def base_sets(self) -> tuple[Set, ...]:
        return self._base_sets

    @property
    def free_symbols(self) -> set[Symbol]:
        result = self._image.free_symbols - set(self.variables)
        for s in self._base_sets:
            result |= s.free_symbols
        return result

    # PORTED FROM SYMPY: sympy/printing/str.py:StrPrinter._print_ImageSet
    def _sympystr(self, printer: NodePrinter) -> str:
        variables = self.variables
        shown = printer.doprint(variables[0]) if len(variables) == 1 else \
            '(%s)' % ', '.join(printer.doprint(v) for v in variables)
        return 'ImageSet(Lambda(%s, %s), %s)' % (shown, printer.doprint(self._image),
                                                 ', '.join(printer.doprint(s) for s in self._base_sets))

    def _contains(self, other: Expr) -> Optional[Boolean]:
        # SymPy solves the equation of the image: not ported
        return None


def _image_set_args(args: tuple[Basic, ...]) -> tuple[Tuple, Basic, list[Set]]:
    variables, image = args[0], args[1]
    if not isinstance(variables, Tuple):
        raise TypeError('the variables of an ImageSet are a tuple, got %s' % (variables,))
    base_sets = [s for s in args[2:] if isinstance(s, Set)]
    if len(base_sets) != len(args) - 2:
        raise TypeError('the base sets of an ImageSet are sets')
    return variables, image, base_sets


# PORTED FROM SYMPY: sympy/sets/fancysets.py:ImageSet.__new__
def image_set(variables: Tuple, image: Basic, *base_sets: Set) -> Set:
    """The image of the base sets, evaluated as SymPy's ``ImageSet``
    constructor does: the base set itself for the identity, a single
    point for a constant function."""
    from .sets import EmptySet, finite_set, point_set
    if len(variables.args) != len(base_sets):
        raise ValueError('Incompatible signature')
    if len(base_sets) == 1 and variables.args == (image,):
        return base_sets[0]
    if not (set(variables.args) & image.free_symbols):
        is_empty = fuzzy_or(s.is_empty for s in base_sets)
        if is_empty is True:
            return EmptySet
        if is_empty is False:
            if isinstance(image, Tuple):
                return point_set(image)
            if isinstance(image, Expr):
                return finite_set(image)
    return ImageSet(variables, image, *base_sets)
