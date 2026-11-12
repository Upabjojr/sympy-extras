"""The SymPy face of the region integrals with a rotational symmetry
(:mod:`sympy_extras.integrals.axisymmetric`), for the integrators which
are still written on SymPy (:mod:`.regions`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back. The step of the native
module which stays on SymPy is here too, on core expressions: the
integral over the profile by
:func:`~sympy_extras.integrals.regions.integrate_by_ranges`, not ported
yet (:func:`integrate_by_ranges_of_core`).

Examples
========

>>> from sympy import symbols, Eq, S
>>> from sympy_extras.integrals.axisymmetric_sympy import axisymmetric_integral
>>> x, y, z = symbols('x y z')
>>> axisymmetric_integral(1, (x**2 + y**2 + z**2 < 1) & ((x - 1)**2 + y**2 + z**2 < 1), [x, y, z])
5*pi/12
>>> axisymmetric_integral(1, Eq(x**2 + y**2 + z**2, 1) & (z > S(1)/2), [x, y, z], measure='hausdorff')
pi
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_symbol
from sympy_extras.assumptions.ask_sympy import _core_symbol, _sympy_symbol, core_boolean, sympy_boolean
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import to_sympy

from . import axisymmetric as native

__all__ = ['axisymmetric_integral', 'rotation_group', 'rotation_invariant', 'aligned_frame', 'Frame',
           'integrate_by_ranges_of_core']


def _core_symbols(names: Sequence[Symbol]) -> list[CoreSymbol]:
    return [_core_symbol(v) for v in names]


def rotation_invariant(e: ExprLike, group: Sequence[Symbol], samples: int = 3) -> bool:
    """:func:`sympy_extras.integrals.axisymmetric.rotation_invariant` on
    SymPy objects: whether ``e`` takes the same value at random rational
    points and at their images under exact rotations in the coordinate
    planes of ``group``.

    >>> from sympy import symbols, Heaviside
    >>> from sympy_extras.integrals.axisymmetric_sympy import rotation_invariant
    >>> x, y = symbols('x y')
    >>> rotation_invariant(x**2 + y**2, [x, y]), rotation_invariant(Heaviside(x), [x, y])
    (True, False)
    """
    return native.rotation_invariant(core_expr(sympify(e)), _core_symbols(group), samples)


def _invariant(e: ExprLike, group: Sequence[Symbol]) -> bool:
    """The private test of :mod:`sympy_extras.integrals.axisymmetric`
    (infinitesimal and finite rotations) on SymPy objects, for the
    tests."""
    return native._invariant(core_expr(sympify(e)), _core_symbols(group))


def rotation_group(f: ExprLike, formula: Boolean, names: Sequence[Symbol]) -> Optional[tuple[Symbol, ...]]:
    """:func:`sympy_extras.integrals.axisymmetric.rotation_group` on SymPy
    objects: the largest group of at least two variables through whose
    sum of squares alone the integrand and every relation of the
    condition depend on them; ``None`` when there is none.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.axisymmetric_sympy import rotation_group
    >>> x, y, z = symbols('x y z')
    >>> rotation_group(z, (x**2 + y**2 + z**2 < 1) & (z > x**2 + y**2), [x, y, z])
    (x, y)
    >>> rotation_group(x*y, x**2 + y**2 + z**2 < 1, [x, y, z]) is None
    True
    """
    found = native.rotation_group(core_expr(sympify(f)), core_boolean(formula), _core_symbols(names))
    return None if found is None else tuple(_sympy_symbol(v) for v in found)


class Frame:
    """:class:`sympy_extras.integrals.axisymmetric.Frame` on SymPy objects:
    the integrand, the condition and the variables in the frame of
    :func:`aligned_frame`; the similarity has ratio ``1/ratio``."""

    __slots__ = ('integrand', 'condition', 'variables', 'ratio')

    def __init__(self, integrand: Expr, condition: Boolean, variables: list[Symbol], ratio: Expr) -> None:
        self.integrand: Final[Expr] = integrand
        self.condition: Final[Boolean] = condition
        self.variables: Final[list[Symbol]] = variables
        self.ratio: Final[Expr] = ratio

    @staticmethod
    def from_core(frame: native.Frame) -> Frame:
        """A frame of the core on SymPy objects."""
        return Frame(sympy_expr(frame.integrand), sympy_boolean(frame.condition),
                     [_sympy_symbol(v) for v in frame.variables], sympy_expr(frame.ratio))


def aligned_frame(f: ExprLike, formula: Boolean, names: Sequence[Symbol]) -> Optional[Frame]:
    """:func:`sympy_extras.integrals.axisymmetric.aligned_frame` on SymPy
    objects: the problem in the frame whose first axis is the common
    normal of the relations of degree one, scaled by its length; ``None``
    when there is no such frame.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.axisymmetric_sympy import aligned_frame
    >>> x, y, z = symbols('x y z')
    >>> frame = aligned_frame(1, (x**2 + y**2 + z**2 < 1) & (x + y + z > 1), [x, y, z])
    >>> Y1, Y2, Y3 = frame.variables
    >>> frame.condition == ((Y1 - 1 > 0) & (Y1**2/3 + Y2**2/3 + Y3**2/3 - 1 < 0)), frame.ratio
    (True, sqrt(3))
    """
    found = native.aligned_frame(core_expr(sympify(f)), core_boolean(formula), _core_symbols(names))
    return None if found is None else Frame.from_core(found)


def axisymmetric_integral(f: ExprLike, formula: Boolean, names: Sequence[Symbol],
                          assumptions: Sequence[Boolean] = (), measure: str = 'lebesgue') -> Optional[Expr]:
    """:func:`sympy_extras.integrals.axisymmetric.axisymmetric_integral` on
    SymPy objects: the integral of ``f`` over the region ``formula`` of
    the variables ``names`` through the profile of its rotational
    symmetry; ``None`` when there is no group of variables to rotate or
    the profile integral is not evaluated.

    >>> from sympy import symbols, Rational
    >>> from sympy_extras.integrals.axisymmetric_sympy import axisymmetric_integral
    >>> x, y, z = symbols('x y z')
    >>> axisymmetric_integral(1, (x**2 + y**2 + z**2 < 1) & (x**2 + y**2 < Rational(1, 4)), [x, y, z])
    pi*(16 - 9*sqrt(3))/12 + sqrt(3)*pi/4
    """
    found = native.axisymmetric_integral(core_expr(sympify(f)), core_boolean(formula), _core_symbols(names),
                                         [core_boolean(a) for a in assumptions], measure)
    return None if found is None else sympy_expr(found)


# ---------------------------------------------------------------------------
# the step of the native module which stays on SymPy

def integrate_by_ranges_of_core(f: CoreExpr, condition: CoreBoolean, variables: Sequence[CoreSymbol],
                                assumptions: Sequence[CoreBoolean], measure: str,
                                seconds: Optional[float]) -> Optional[CoreExpr]:
    """:func:`~sympy_extras.integrals.regions.integrate_by_ranges` of ``f``
    over ``condition`` within ``seconds``, on core expressions (not ported
    yet); ``None`` when it does not finish, leaves an ``IntegralByRanges``
    or its value cannot be converted."""
    from .regions import IntegralByRanges, integrate_by_ranges
    integrand, region = sympy_expr(f), sympy_boolean(condition)
    order = [as_symbol(to_sympy(v)) for v in variables]
    outer: list[Boolean] = [sympy_boolean(a) for a in assumptions]
    found = attempt(lambda: integrate_by_ranges(integrand, region, order, outer, measure=measure), seconds)
    if found is None or found.has(IntegralByRanges):
        return None
    try:
        return core_expr(found)
    except (TypeError, ValueError, NotImplementedError):
        return None
