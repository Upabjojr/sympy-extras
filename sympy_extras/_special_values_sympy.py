"""The SymPy face of the isolated values of the parameters at which a
closed form fails (:mod:`sympy_extras._special_values`).

The functions take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native ones, the callbacks too:
``compute`` receives the point and the assumptions at the point as SymPy
objects and returns a SymPy expression, ``defined`` receives a SymPy
point and ``degenerate`` a SymPy expression. The integrators and the sums
written on SymPy (:mod:`sympy_extras.integrals.definite`,
:mod:`sympy_extras.concrete.karr_sympy`) use this module; the native code
(:mod:`sympy_extras.solvers._special_values`, the summation algorithms of
:mod:`sympy_extras.concrete`) calls :mod:`sympy_extras._special_values`
directly. Not to be confused with
:mod:`sympy_extras.solvers._special_values_sympy`, the face of the cases of
the ODE solutions.

The callbacks run under the native time limits of the module, which do
not prepare SymPy's tables; ``compute`` and ``degenerate``, written on
SymPy, prepare them first (:func:`~sympy_extras._timeout.prepare_sympy`),
as a limit without ``native`` does.

Examples
========

>>> from sympy import symbols, log, Integer
>>> from sympy_extras._special_values_sympy import special_points, with_special_values
>>> y, k, x = symbols('y k x')
>>> special_points((y**2 - 1)/log(y), [y])
[{y: 0}, {y: 1}]
>>> with_special_values((y**2 - 1)/log(y), [y], None, lambda point, assumptions: Integer(2))
Piecewise((2, Eq(y, 1)), ((y**2 - 1)/log(y), True))
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean

from sympy_extras._timeout import prepare_sympy
from sympy_extras._typing_sympy import as_boolean, as_expr, as_symbol
from sympy_extras.assumptions.ask import Assumptions as CoreAssumptions
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, core_boolean
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.logic import to_boolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import _special_values as native

__all__ = ['Point', 'Compute', 'special_points', 'with_special_values', 'point_condition', 'condition_point',
           'point_assumptions', 'undefined_at', 'equality_points', 'defined_problem']

#: the values of some parameters, ``{a: b, b: 0}``
Point = dict[Symbol, Expr]

#: a problem solved at a point, under the assumptions at the point: its
#: value, or ``None`` when it cannot be computed
Compute = Callable[[Point, Assumptions], Optional[Expr]]


# ---------------------------------------------------------------------------
# the boundary with the core

def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_expr(e: CoreBasic) -> Expr:
    return as_expr(to_sympy(e))


def _sympy_boolean(b: CoreBasic) -> Boolean:
    return as_boolean(to_sympy(b))


def _core_point(point: Point) -> native.Point:
    return {_core_symbol(s): _core_expr(v) for s, v in point.items()}


def _sympy_point(point: native.Point) -> Point:
    return {as_symbol(to_sympy(s)): _sympy_expr(v) for s, v in point.items()}


def _sympy_assumptions(assumptions: CoreAssumptions) -> Assumptions:
    if assumptions is None:
        return None
    if isinstance(assumptions, (CoreBoolean, bool)):
        return _sympy_boolean(to_boolean(assumptions))
    return [_sympy_boolean(to_boolean(a)) for a in assumptions]


def _sympy_assumption_list(assumptions: Optional[list[CoreBoolean]]) -> Optional[list[Boolean]]:
    return None if assumptions is None else [_sympy_boolean(a) for a in assumptions]


# ---------------------------------------------------------------------------
# the face

def special_points(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions = None,
                   seconds: Optional[float] = None) -> list[Point]:
    """:func:`sympy_extras._special_values.special_points` on SymPy objects.

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras._special_values_sympy import special_points
    >>> a, b, k = symbols('a b k')
    >>> special_points((exp(k) - 1)/k, [k])
    [{k: 0}]
    >>> special_points(1/(a**2 - b**2), [a, b])
    [{a: -b}, {a: b}]
    >>> special_points(1/(a**2 + 1), [a]), special_points(1/sin(a), [a])
    ([], [])
    """
    found = native.special_points(_core_expr(value), [_core_symbol(p) for p in parameters],
                                  core_assumptions(assumptions), seconds)
    return [_sympy_point(point) for point in found]


def point_condition(point: Point) -> Boolean:
    """:func:`sympy_extras._special_values.point_condition` on SymPy objects.

    >>> from sympy import symbols
    >>> from sympy_extras._special_values_sympy import point_condition
    >>> a, b = symbols('a b')
    >>> point_condition({a: 0, b: 0})
    Eq(a, 0) & Eq(b, 0)
    """
    return _sympy_boolean(native.point_condition({_core_symbol(s): _core_expr(as_expr(v)) for s, v in point.items()}))


def point_assumptions(point: Point, assumptions: Assumptions) -> Optional[list[Boolean]]:
    """:func:`sympy_extras._special_values.point_assumptions` on SymPy
    objects.

    >>> from sympy import symbols
    >>> from sympy_extras._special_values_sympy import point_assumptions
    >>> a, b = symbols('a b')
    >>> point_assumptions({a: b}, [a > 0, b < 2])
    [b > 0, b < 2]
    >>> point_assumptions({a: 0}, a > 0) is None
    True
    """
    found = native.point_assumptions({_core_symbol(s): _core_expr(as_expr(v)) for s, v in point.items()},
                                     core_assumptions(assumptions))
    return _sympy_assumption_list(found)


def undefined_at(e: Expr, point: Point) -> bool:
    """:func:`sympy_extras._special_values.undefined_at` on SymPy objects.

    >>> from sympy import symbols, sin
    >>> from sympy_extras._special_values_sympy import undefined_at
    >>> a = symbols('a')
    >>> undefined_at(sin(a)/a, {a: 0}), undefined_at(sin(a)/a, {a: 1})
    (True, False)
    """
    return native.undefined_at(_core_expr(e), {_core_symbol(s): _core_expr(as_expr(v)) for s, v in point.items()})


def equality_points(value: Expr, parameters: Sequence[Symbol]) -> list[Point]:
    """:func:`sympy_extras._special_values.equality_points` on SymPy objects.

    >>> from sympy import symbols, Piecewise, Eq, log
    >>> from sympy_extras._special_values_sympy import equality_points
    >>> y, c = symbols('y c')
    >>> equality_points(Piecewise((2, Eq(y, 1)), ((y**2 - 1)/log(y), True)), [y])
    [{y: 1}]
    >>> equality_points(Piecewise((-1, Eq(c, 0) | Eq(c, 1)), (c, c > 1)), [c])
    [{c: 0}, {c: 1}]
    """
    found = native.equality_points(_core_expr(value), [_core_symbol(p) for p in parameters])
    return [_sympy_point(point) for point in found]


def condition_point(condition: Boolean) -> Optional[Point]:
    """:func:`sympy_extras._special_values.condition_point` on SymPy
    objects.

    >>> from sympy import symbols, Eq
    >>> from sympy_extras._special_values_sympy import condition_point
    >>> a, b = symbols('a b')
    >>> condition_point(Eq(a, b) & Eq(b, 0)), condition_point(a > 0)
    ({a: 0, b: 0}, None)
    """
    found = native.condition_point(core_boolean(condition))
    return None if found is None else _sympy_point(found)


def defined_problem(integrand: Expr, variables: Sequence[Symbol]) -> bool:
    """:func:`sympy_extras._special_values.defined_problem` on SymPy
    objects.

    >>> from sympy import symbols, DiracDelta, Integer
    >>> from sympy_extras._special_values_sympy import defined_problem
    >>> a, b, z = symbols('a b z')
    >>> defined_problem((a**(b*z)/z).subs(a, 0), [z]), defined_problem((a**(b*z)/z).subs(a, 1), [z])
    (False, True)
    >>> defined_problem(DiracDelta(a*z).subs(a, 0), [z]), defined_problem(Integer(0)**2*z, [z])
    (False, True)
    """
    return native.defined_problem(_core_expr(integrand), [_core_symbol(v) for v in variables])


def with_special_values(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions,
                        compute: Compute, defined: Optional[Callable[[Point], bool]] = None,
                        sympy_style: bool = False, budget: Optional[float] = None,
                        keep_unevaluated: bool = True,
                        degenerate: Optional[Callable[[Expr], list[Expr]]] = None) -> Expr:
    """:func:`sympy_extras._special_values.with_special_values` on SymPy
    objects, the callbacks taking and returning SymPy objects. A value
    which gets no case is returned as given (not converted back from the
    core).

    >>> from sympy import symbols, exp, Integer
    >>> from sympy_extras._special_values_sympy import with_special_values
    >>> k = symbols('k')
    >>> with_special_values((exp(k) - 1)/k, [k], None, lambda point, assumptions: Integer(1))
    Piecewise((1, Eq(k, 0)), ((exp(k) - 1)/k, True))
    >>> with_special_values(exp(k)/k, [k], None, lambda point, assumptions: Integer(1), sympy_style=True)
    Piecewise((exp(k)/k, Ne(k, 0)), (1, True))
    """
    if not parameters:
        return value
    converted = _core_expr(value)

    def at_point(point: native.Point, at: CoreAssumptions) -> Optional[CoreExpr]:
        prepare_sympy()
        found = compute(_sympy_point(point), _sympy_assumptions(at))
        return None if found is None else _core_expr(found)

    core_defined: Optional[Callable[[native.Point], bool]] = None
    if defined is not None:
        defined_ = defined

        def core_defined_(point: native.Point) -> bool:
            return defined_(_sympy_point(point))
        core_defined = core_defined_
    core_degenerate: Optional[Callable[[CoreExpr], list[CoreExpr]]] = None
    if degenerate is not None:
        degenerate_ = degenerate

        def core_degenerate_(e: CoreExpr) -> list[CoreExpr]:
            prepare_sympy()
            return [_core_expr(u) for u in degenerate_(_sympy_expr(e))]
        core_degenerate = core_degenerate_
    result = native.with_special_values(converted, [_core_symbol(p) for p in parameters],
                                        core_assumptions(assumptions), at_point, core_defined, sympy_style, budget,
                                        keep_unevaluated, core_degenerate)
    # no case: the value itself, not its conversion back
    return value if result is converted else _sympy_expr(result)
