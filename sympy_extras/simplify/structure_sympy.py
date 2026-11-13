"""The SymPy face of the structure theorem
(:mod:`sympy_extras.simplify.structure`).

:func:`canonical_form`, :func:`is_zero`, :func:`equal` and
:class:`ElementaryTower` take and return SymPy expressions (the
assumptions as SymPy Booleans), converted by :mod:`sympy_extras.interop`
around the native tower of :mod:`sympy_extras.simplify.structure`: the
elements are those of the native tower, wrapped, and the generators show
their meanings as SymPy expressions.

Examples
========

>>> from sympy import symbols, exp, log, sin, cos
>>> from sympy_extras.simplify.structure_sympy import ElementaryTower, canonical_form, is_zero
>>> x, y = symbols('x y')
>>> canonical_form(exp(x + y) - exp(x)*exp(y) + sin(x)**2 + cos(x)**2)
1
>>> is_zero(log(x*y) - log(x) - log(y), [x > 0, y > 0]), is_zero(log(x*y) - log(x) - log(y))
(True, False)
>>> tower = ElementaryTower()
>>> u = tower.element(exp(x/2) + log(x))
>>> [(g.kind, g.expression) for g in tower.generators], tower.certified
([('variable', x), ('exp', exp(x/2)), ('log', log(x))], True)
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.symbol import Symbol

from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr

from . import structure as native
from .structure import EXPONENTIAL, LOGARITHM, NotElementary, ROOT, VARIABLE

__all__ = ['ElementaryTower', 'Element', 'Generator', 'NotElementary', 'canonical_form', 'is_zero', 'equal',
           'VARIABLE', 'EXPONENTIAL', 'LOGARITHM', 'ROOT']


# ---------------------------------------------------------------------------
# the face

def _fraction(r: Rational) -> Fraction:
    return Fraction(int(r.p), int(r.q))


class Element:
    """An element of an :class:`ElementaryTower`: the element of the
    native tower (``native``), combined with ``+``, ``-``, ``*``, ``/`` and
    ``**`` with an integer."""

    __slots__ = ('tower', 'native')

    def __init__(self, tower: ElementaryTower, element: native.Element) -> None:
        self.tower: ElementaryTower = tower
        self.native: native.Element = element

    def _other(self, other: Element) -> native.Element:
        if other.tower is not self.tower:
            raise ValueError("an element of another tower")
        return other.native

    def __add__(self, other: Element) -> Element:
        return Element(self.tower, self.native + self._other(other))

    def __sub__(self, other: Element) -> Element:
        return Element(self.tower, self.native - self._other(other))

    def __mul__(self, other: Element) -> Element:
        return Element(self.tower, self.native * self._other(other))

    def __truediv__(self, other: Element) -> Element:
        return Element(self.tower, self.native / self._other(other))

    def __neg__(self) -> Element:
        return Element(self.tower, -self.native)

    def __pow__(self, n: int) -> Element:
        return Element(self.tower, self.native**n)

    def times(self, r: Rational) -> Element:
        """``r`` times the element, ``r`` a rational number."""
        return Element(self.tower, self.native.times(_fraction(r)))

    def __repr__(self) -> str:
        return "Element(%s)" % (self.tower.to_expr(self),)


class Generator:
    """One generator of the tower, with its meaning as a SymPy expression
    (:class:`sympy_extras.simplify.structure.Generator`)."""

    def __init__(self, tower: ElementaryTower, generator: native.Generator) -> None:
        self.tower: ElementaryTower = tower
        self.native: native.Generator = generator

    def _element(self, a: Optional[native.Element]) -> Optional[Element]:
        return None if a is None else Element(self.tower, a)

    @property
    def kind(self) -> str:
        return self.native.kind

    @property
    def expression(self) -> Expr:
        return sympy_expr(self.native.expression)

    @property
    def certified(self) -> bool:
        return self.native.certified

    @property
    def index(self) -> int:
        return self.native.index

    @property
    def source(self) -> int:
        return self.native.source

    @property
    def argument(self) -> Optional[Element]:
        return self._element(self.native.argument)

    @property
    def radicand(self) -> Optional[Element]:
        return self._element(self.native.radicand)

    @property
    def scale(self) -> Optional[Element]:
        return self._element(self.native.scale)

    @property
    def eliminated(self) -> Optional[Element]:
        return self._element(self.native.eliminated)

    def __repr__(self) -> str:
        return "Generator(%s, %s)" % (self.kind, self.expression)


class ElementaryTower:
    """The tower of :class:`sympy_extras.simplify.structure.ElementaryTower`
    on SymPy expressions.

    Parameters
    ==========

    assumptions : Boolean or list of Booleans, optional
        The region on which the identities are to hold.

    Examples
    ========

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.simplify.structure_sympy import ElementaryTower
    >>> x = symbols('x')
    >>> tower = ElementaryTower(x > 0)
    >>> tower.to_expr(tower.element(log(x**2) - 2*log(x)))
    0
    >>> tower = ElementaryTower()
    >>> u = tower.element(log(x**2) - 2*log(x))
    >>> tower.to_expr(u), tower.vanishes(u), tower.certified
    (-2*log(x) + log(x**2), False, False)
    """

    def __init__(self, assumptions: Assumptions = None) -> None:
        self.native: native.ElementaryTower = native.ElementaryTower(core_assumptions(assumptions))

    def _wrap(self, a: native.Element) -> Element:
        return Element(self, a)

    def _unwrap(self, a: Element) -> native.Element:
        if a.tower is not self:
            raise ValueError("an element of another tower")
        return a.native

    @property
    def generators(self) -> list[Generator]:
        return [Generator(self, g) for g in self.native.generators]

    @property
    def zero(self) -> Element:
        return self._wrap(self.native.zero)

    @property
    def one(self) -> Element:
        return self._wrap(self.native.one)

    def constant(self, value: Rational) -> Element:
        """The rational number ``value`` as an element."""
        return self._wrap(self.native.constant(_fraction(value)))

    def variable(self, symbol: Symbol) -> Element:
        """The generator of a free symbol."""
        return self._wrap(self.native.variable(core_symbol(symbol)))

    def element(self, expression: object) -> Element:
        """The element of the tower which ``expression`` is; the tower
        grows by the generators it needs.

        Raises :class:`NotElementary` for an expression outside the
        elementary functions."""
        return self._wrap(self.native.element(core_expr(as_expr(expression))))

    def to_expr(self, a: Element) -> Expr:
        """The element as a SymPy expression in the meanings of the
        generators."""
        return sympy_expr(self.native.to_expr(self._unwrap(a)))

    def normal(self, a: Element) -> Element:
        """``a`` in normal form."""
        return self._wrap(self.native.normal(self._unwrap(a)))

    def vanishes(self, a: Element) -> bool:
        """Whether the normal form of ``a`` is zero."""
        return self.native.vanishes(self._unwrap(a))

    def as_number(self, a: Element) -> Optional[Expr]:
        """The algebraic number which ``a`` is, or ``None`` when it
        involves a generator."""
        found = self.native.as_number(self._unwrap(a))
        return None if found is None else sympy_expr(found)

    def derivative(self, a: Element, variable: int) -> Element:
        """The derivative of ``a`` with respect to the variable generator
        at position ``variable``."""
        return self._wrap(self.native.derivative(self._unwrap(a), variable))

    def is_constant(self, a: Element) -> bool:
        """Whether every derivative of ``a`` vanishes."""
        return self.native.is_constant(self._unwrap(a))

    def exponential(self, a: Element) -> Element:
        """``exp(a)`` in the tower."""
        return self._wrap(self.native.exponential(self._unwrap(a)))

    def logarithm(self, v: Element) -> Element:
        """``log(v)`` in the tower."""
        return self._wrap(self.native.logarithm(self._unwrap(v)))

    def certifies(self, a: Element) -> bool:
        """Whether the generators which ``a`` involves are independent."""
        return self.native.certifies(self._unwrap(a))

    @property
    def certified(self) -> bool:
        """Whether the generators are algebraically independent."""
        return self.native.certified


def canonical_form(expression: object, assumptions: Assumptions = None) -> Expr:
    """The canonical form of an elementary expression
    (:func:`sympy_extras.simplify.structure.canonical_form`).

    Raises :class:`NotElementary` when the expression is not elementary.

    Examples
    ========

    >>> from sympy import symbols, exp, log, sqrt, sin, cos, I, pi
    >>> from sympy_extras.simplify.structure_sympy import canonical_form
    >>> x, y = symbols('x y')
    >>> canonical_form(exp(x + y) - exp(x)*exp(y))
    0
    >>> canonical_form(sin(2*x) - 2*sin(x)*cos(x))
    0
    >>> canonical_form(exp(I*pi) + 1)
    0
    >>> canonical_form((sqrt(x) + 1)**2 - x - 1)
    2*sqrt(x)
    >>> canonical_form(log(x*y) - log(x) - log(y), [x > 0, y > 0])
    0
    """
    return sympy_expr(native.canonical_form(core_expr(as_expr(expression)), core_assumptions(assumptions)))


def is_zero(expression: object, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether an elementary expression vanishes identically on the
    region of the assumptions (:func:`sympy_extras.simplify.structure.is_zero`):
    ``True`` is a proof, ``False`` means that it does not vanish
    identically, ``None`` that neither was established.

    Examples
    ========

    >>> from sympy import symbols, exp, log, sqrt, cosh, sinh
    >>> from sympy_extras.simplify.structure_sympy import is_zero
    >>> x = symbols('x')
    >>> is_zero(cosh(x)**2 - sinh(x)**2 - 1)
    True
    >>> is_zero(exp(x)*exp(-x) - 1), is_zero(exp(x) - x - 1)
    (True, False)
    >>> is_zero(sqrt(x)**2 - x), is_zero(sqrt(x**2) - x), is_zero(sqrt(x**2) - x, x > 0)
    (True, False, True)
    >>> is_zero(log(x**2) - 2*log(x)), is_zero(log(x**2) - 2*log(x), x > 0)
    (False, True)
    """
    return native.is_zero(core_expr(as_expr(expression)), core_assumptions(assumptions))


def equal(a: object, b: object, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether two elementary expressions are equal on the region of the
    assumptions, by :func:`is_zero` of their difference.

    >>> from sympy import symbols, tan, sin, cos, sqrt
    >>> from sympy_extras.simplify.structure_sympy import equal
    >>> x = symbols('x')
    >>> equal(tan(x), sin(x)/cos(x)), equal(sqrt(x)*sqrt(x), x), equal(sin(x), x)
    (True, True, False)
    """
    return is_zero(as_expr(a) - as_expr(b), assumptions)


def _witness(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is clearly not zero at a sample point of the region
    (:func:`sympy_extras.simplify.structure._witness`)."""
    return native._witness(core_expr(e), core_assumptions(assumptions))

