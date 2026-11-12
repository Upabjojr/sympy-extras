"""The SymPy face of the integrals over regions
(:mod:`sympy_extras.integrals.regions`).

:class:`IntegralByRanges` is the unevaluated integral on SymPy objects
and :func:`integrate_by_ranges` computes it: the arguments are converted
by :mod:`sympy_extras.interop`, the native code runs and the value is
converted back (the algebraic numbers of the bounds as ``CRootOf``). The
step of the native module which stays on SymPy is here too, on core
expressions: the one-dimensional integrals by
:func:`~sympy_extras.integrals.definite.definite_integral`, not ported
yet (:func:`definite_integral_of_core`, the default integrator of the
native module).

Examples
========

>>> from sympy import symbols, pi, exp, Eq
>>> from sympy_extras.integrals.regions_sympy import IntegralByRanges, integrate_by_ranges
>>> x, y, z, r = symbols('x y z r')
>>> IntegralByRanges(1, x**2 + y**2 < 1)
IntegralByRanges(1, x**2 + y**2 < 1, (x, y))
>>> IntegralByRanges(1, x**2 + y**2 < 1).doit()
pi
>>> integrate_by_ranges(x*y, (x > 0) & (y > 0) & (x + y < 1))
1/24
>>> integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y], r > 0)
pi*r**2
>>> integrate_by_ranges(exp(-x**2 - y**2), x**2 + y**2 < 1)
-pi*exp(-1) + pi
>>> integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y < exp(x)))
-1 + E
>>> a, b = symbols('a b', positive=True)
>>> integrate_by_ranges(1, x**2/a**2 + y**2/b**2 < 1, [x, y])
pi*a*b
>>> integrate_by_ranges(1, (x**2 + y**2 < 1) & (z > 0) & (z < x**2 + y**2))
pi/2
>>> integrate_by_ranges(1, (x > 0) & (y > 0) & (y < exp(-x)))
1

A polynomial over a polytope, and a radial integrand in a symbolic
dimension:

>>> integrate_by_ranges((x + y)**4, (x > 0) & (y > 0) & (x + y < 1) & (x < 2*y))
1/9
>>> n = symbols('n', positive=True, integer=True)
>>> integrate_by_ranges(1, r < 1, [r], dimension=n)
pi**(n/2)/gamma(n/2 + 1)
>>> integrate_by_ranges(exp(-r**2), True, [r], dimension=n)
pi**(n/2)

The length of the unit circle and the area of the unit sphere, with the
Hausdorff measure:

>>> integrate_by_ranges(1, Eq(x**2 + y**2, 1), measure='hausdorff')
2*pi
>>> integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1), measure='hausdorff')
4*pi

The length of the circle `z = 1` on the paraboloid `z = x^2 + y^2`, a
curve in space given by two equations, and the area under the cubic
`y = x^3 - x` between its root and `x = 0`, with the variables ordered
so that the boundary root is in the last variable:

>>> integrate_by_ranges(1, Eq(z, x**2 + y**2) & Eq(z, 1), measure='hausdorff')
2*pi
>>> integrate_by_ranges(1, (x > 0) & (y > 0) & (y < 1) & (x**3 - x - y < 0), [y, x])
-CRootOf(x**3 - x - 1, 0)**4/4 - 1/4 + CRootOf(x**3 - x - 1, 0)**2/2 + CRootOf(x**3 - x - 1, 0)
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.symbol import Str, Symbol
from sympy.logic.boolalg import Boolean
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import ComplexRootOf

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike, as_boolean, as_expr, as_symbol, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, core_boolean, sympy_boolean
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.polys.cad.indexedroot import real_root
from sympy_extras.polys.cad.samplepoints_sympy import core_real
from sympy_extras.polys.polytools import Poly as CorePoly

from . import regions as native

__all__ = ['IntegralByRanges', 'integrate_by_ranges', 'definite_integral_of_core']

#: the measures of integration (see :mod:`sympy_extras.integrals.regions`)
_MEASURES = ('lebesgue', 'hausdorff')


class IntegralByRanges(Expr):
    """:class:`sympy_extras.integrals.regions.IntegralByRanges` on SymPy
    objects: the integral of ``integrand`` over the region described by
    ``condition``, a Boolean combination of polynomial inequalities and
    equations in the integration ``variables`` (and parameters).

    The constructor does not evaluate: :meth:`doit` (or
    :func:`integrate_by_ranges`) computes the integral.

    ``measure='hausdorff'`` integrates with respect to the
    `(n-1)`-dimensional Hausdorff measure on the hypersurface given by
    the equation of the condition (arc length, surface area), see the
    module documentation of :mod:`sympy_extras.integrals.regions`.

    Examples
    ========

    >>> from sympy import symbols, Eq
    >>> from sympy_extras.integrals.regions_sympy import IntegralByRanges
    >>> x, y = symbols('x y')
    >>> region = IntegralByRanges(x*y, (x > 0) & (x < y) & (y < 1))
    >>> region.integrand, region.condition, region.variables
    (x*y, (x > 0) & (x < y) & (y < 1), (x, y))
    >>> region.doit()
    1/8

    The integration variables are those of the condition, less the
    ``parameters``; or they are given as ``variables``:

    >>> a = symbols('a')
    >>> IntegralByRanges(1, x**2 + y**2 < a**2, parameters=[a]).variables
    (x, y)
    >>> circle = IntegralByRanges(1, Eq(x**2 + y**2, 1), measure='hausdorff')
    >>> circle.measure == 'hausdorff'
    True
    >>> circle.doit()
    2*pi
    """

    def __new__(cls, integrand: ExprLike, condition: object,
                variables: Optional[Sequence[Symbol]] = None,
                measure: str = 'lebesgue',
                dimension: Optional[ExprLike] = None,
                parameters: Optional[Sequence[Symbol]] = None) -> IntegralByRanges:
        integrand_ = as_expr(integrand)
        condition_ = as_boolean(condition)
        excluded = [as_symbol(p) for p in parameters] if parameters is not None else []
        if variables is None:
            names = [v for v in sorted_symbols(free_symbols(condition_)) if v not in excluded]
            if dimension is not None:
                names = [_radial_symbol(names)]
        else:
            names = [as_symbol(v) for v in variables]
            clash = [v for v in names if v in excluded]
            if clash:
                raise ValueError("%s cannot be both an integration variable and a parameter" % (clash[0],))
        if not names:
            raise ValueError("no integration variable: the condition has no symbols")
        if isinstance(measure, Str):
            measure = str(measure)                          # rebuilt from its arguments (subs, copy, pickle)
        if measure not in _MEASURES:
            raise ValueError("measure must be one of %s, got %r" % (", ".join(_MEASURES), measure))
        args: list[Basic] = [integrand_, condition_, Tuple(*names)]
        if measure != _MEASURES[0] or dimension is not None:
            args.append(Str(measure))
        if dimension is not None:
            if len(names) != 1:
                raise ValueError("with a dimension the integral is radial, in one variable")
            args.append(as_expr(dimension))
        obj = Expr.__new__(cls, *args)
        if not isinstance(obj, IntegralByRanges):
            raise TypeError("unexpected construction of IntegralByRanges")
        return obj

    @property
    def integrand(self) -> Expr:
        return as_expr(self.args[0])

    @property
    def condition(self) -> Boolean:
        return as_boolean(self.args[1])

    @property
    def variables(self) -> tuple[Symbol, ...]:
        names = self.args[2]
        if not isinstance(names, Tuple):
            raise TypeError("the variables of IntegralByRanges must be a Tuple")
        return tuple(as_symbol(v) for v in names.args)

    @property
    def measure(self) -> str:
        """``'lebesgue'`` or ``'hausdorff'``."""
        if len(self.args) < 4:
            return _MEASURES[0]
        name = self.args[3]
        if not isinstance(name, Str):
            raise TypeError("the measure of IntegralByRanges must be a Str")
        return str(name)

    @property
    def dimension(self) -> Optional[Expr]:
        """The dimension of the space of a radial integral, ``None`` for
        an integral in the variables themselves."""
        if len(self.args) < 5:
            return None
        return as_expr(self.args[4])

    @property
    def free_symbols(self) -> set[Basic]:
        """The symbols of the integrand and the condition other than the
        integration variables, which are bound."""
        symbols: set[Basic] = set(self.integrand.free_symbols) | set(self.condition.free_symbols)
        if self.dimension is not None:
            symbols |= set(self.dimension.free_symbols)
        return symbols - set(self.variables)

    def _eval_subs(self, old: Basic, new: Basic) -> Optional[Basic]:
        # a bound variable is not substituted
        return self if old in self.variables else None

    def doit(self, assumptions: Assumptions = None, **hints: object) -> Expr:
        """The value, or the integral unchanged when it cannot be computed."""
        return integrate_by_ranges(self.integrand, self.condition, self.variables, assumptions,
                                   self.measure, self.dimension)


def _radial_symbol(names: Sequence[Symbol]) -> Symbol:
    """The radial variable among the symbols of a condition: the one
    named ``r``, or the only one."""
    for name in names:
        if name.name == 'r':
            return name
    if len(names) == 1:
        return names[0]
    if not names:
        return Symbol('r')
    raise ValueError("the radial variable of a condition in %s must be given" % (list(names),))


def _core_assumptions(assumptions: Assumptions) -> list[CoreBoolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [core_boolean(assumptions)]
    return [core_boolean(a) for a in assumptions]


def _core_symbols(names: Sequence[Symbol]) -> list[CoreSymbol]:
    return [_core_symbol(v) for v in names]


def integrate_by_ranges(integrand: ExprLike, condition: object,
                        variables: Optional[Sequence[Symbol]] = None,
                        assumptions: Assumptions = None,
                        measure: str = 'lebesgue',
                        dimension: Optional[ExprLike] = None,
                        parameters: Optional[Sequence[Symbol]] = None) -> Expr:
    """:func:`sympy_extras.integrals.regions.integrate_by_ranges` on SymPy
    objects: the integral of ``integrand`` over the region ``condition``,
    a ``Piecewise`` over the cases of the parameters, or the unevaluated
    :class:`IntegralByRanges` when a bound has no explicit form or an
    inner integral is not computed. See the native function for the
    parameters.

    Examples
    ========

    >>> from sympy import symbols, Eq
    >>> from sympy_extras.integrals.regions_sympy import integrate_by_ranges
    >>> x, y, z, r = symbols('x y z r')
    >>> integrate_by_ranges(x**2, x**2 + y**2 < 1)
    pi/4
    >>> integrate_by_ranges(1/(1 + x**2 + y**2), x**2 + y**2 < 3)
    2*pi*log(2)
    >>> integrate_by_ranges(1, x**2 + y**2 + z**2 < 1)
    4*pi/3
    >>> integrate_by_ranges(1, (0 < x) & (x < y) & (y < x**2) & (x < 2))
    5/6

    Without the assumption the cells ``r < 0``, ``r = 0`` and ``r > 0`` of
    the parameter space are integrated apart, and their values agree
    (``pi*r**2`` is ``0`` at ``r = 0``), so the answer is one expression;
    a parameter is told from the variables by ``parameters`` as well:

    >>> integrate_by_ranges(1, x**2 + y**2 < r**2, parameters=[r])
    pi*r**2
    >>> integrate_by_ranges(1, x**2 + y**2 < r**2)
    oo

    The cases stay apart where the values differ:

    >>> integrate_by_ranges(1, (x > 0) & (x < r), parameters=[r])
    Piecewise((0, Eq(r, 0) | (r < 0)), (r, True))

    The length of the arc of the parabola ``y = x**2`` over ``0 < x < 1``
    and the area of the paraboloid ``z = x**2 + y**2`` below ``z = 1``:

    >>> integrate_by_ranges(1, Eq(y, x**2) & (x > 0) & (x < 1), measure='hausdorff')
    asinh(2)/4 + sqrt(5)/2
    >>> integrate_by_ranges(1, Eq(z, x**2 + y**2) & (z < 1), measure='hausdorff')
    pi*(-1 + 5*sqrt(5))/6

    The volume of the cube ``|x|, |y|, |z| < 1`` cut by ``x + y + z < 1``
    (a polytope, without the decomposition) and the volume of the ball
    of radius ``R`` in ``n`` dimensions:

    >>> cube = (x > -1) & (x < 1) & (y > -1) & (y < 1) & (z > -1) & (z < 1)
    >>> integrate_by_ranges(1, cube & (x + y + z < 1))
    20/3
    >>> n, R = symbols('n R', positive=True)
    >>> integrate_by_ranges(1, r < R, [r], dimension=n)
    pi**(n/2)*R**n/gamma(n/2 + 1)
    """
    node = IntegralByRanges(integrand, condition, variables, measure, dimension, parameters)
    # the variables of the node (those of the condition less the parameters by default)
    found = native.integrate_by_ranges(core_expr(node.integrand), core_boolean(node.condition),
                                       _core_symbols(node.variables), _core_assumptions(assumptions), node.measure,
                                       None if node.dimension is None else core_expr(node.dimension))
    if found.has(native.IntegralByRanges):
        return node
    return sympy_expr(found)


# ---------------------------------------------------------------------------
# the step of the native module which stays on SymPy

def definite_integral_of_core(f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                              assumptions: Sequence[CoreBoolean]) -> Optional[CoreExpr]:
    """:func:`~sympy_extras.integrals.definite.definite_integral` of
    ``Integral(f, (x, a, b))`` under the assumptions, without the cases of
    the special values of the parameters, on core expressions (not ported
    yet): the default integrator of
    :func:`sympy_extras.integrals.regions.integrate_by_ranges`. ``None``
    when the integral is not computed (it keeps an ``Integral``) or its
    value cannot be converted; the ``ValueError`` of bounds which
    contradict the assumptions goes on to the caller (an empty cell)."""
    from .definite import definite_integral
    from .integral import is_unevaluated
    value = definite_integral(f, (x, a, b), list(assumptions), special_values=False)
    if is_unevaluated(value) or value.has(native.IntegralByRanges):
        return None
    return value


# ---------------------------------------------------------------------------
# the private steps of the native module on SymPy objects, for the tests

def _core_list(assumptions: Sequence[Union[Boolean, bool]]) -> list[CoreBoolean]:
    return [core_boolean(a) for a in assumptions]


def _optional(e: Optional[CoreExpr]) -> Optional[Expr]:
    return None if e is None else sympy_expr(e)


def _split_roots(expr: ExprLike, assumptions: Sequence[Union[Boolean, bool]]) -> Expr:
    """:func:`sympy_extras.integrals.regions._split_roots` on SymPy objects."""
    return sympy_expr(native._split_roots(core_expr(as_expr(expr)), _core_list(assumptions)))


def _solved_bounds(f: ExprLike, formula: Boolean, names: Sequence[Symbol],
                   assumptions: Sequence[Union[Boolean, bool]]) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.regions._solved_bounds` on SymPy objects."""
    return _optional(native._solved_bounds(core_expr(as_expr(f)), core_boolean(formula), _core_symbols(names),
                                           _core_list(assumptions), native._definite))


def _decomposed(node: IntegralByRanges, f: ExprLike, formula: Boolean, names: Sequence[Symbol],
                assumptions: Assumptions, extra: Sequence[Union[Boolean, bool]]) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.regions._decomposed` on SymPy objects
    (``node`` is the integral computed, as the old private function took
    it)."""
    facts = _core_list(extra) + _core_assumptions(assumptions)
    return _optional(native._decomposed(core_expr(as_expr(f)), core_boolean(formula), _core_symbols(names),
                                        facts, native._definite))


def _solved_bound(atom: Boolean, y: Symbol, assumptions: Sequence[Union[Boolean, bool]]
                  ) -> Optional[list[tuple[bool, Expr]]]:
    """:func:`sympy_extras.integrals.regions._solved_bound` on SymPy objects."""
    relation = core_boolean(atom)
    from sympy_extras.core.relational import Relational
    if not isinstance(relation, Relational):
        raise TypeError("a relation is expected, got %s" % (atom,))
    found = native._solved_bound(relation, _core_symbol(y), _core_list(assumptions))
    return None if found is None else [(upper, sympy_expr(bound)) for upper, bound in found]


def _trigonometric_roots(poly: Poly, x: Symbol) -> Optional[list[Expr]]:
    """:func:`sympy_extras.integrals.regions._trigonometric_roots` on SymPy objects."""
    variable = _core_symbol(x)
    found = native._trigonometric_roots(CorePoly(core_expr(as_expr(poly.as_expr())), variable), variable)
    return None if found is None else [sympy_expr(e) for e in found]


def _polytope(f: ExprLike, formula: Boolean, names: Sequence[Symbol],
              assumptions: Sequence[Union[Boolean, bool]]) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.regions._polytope` on SymPy objects."""
    return _optional(native._polytope(core_expr(as_expr(f)), core_boolean(formula), _core_symbols(names),
                                      _core_list(assumptions)))


def _radical_form(root: ExprLike) -> Expr:
    """:func:`sympy_extras.integrals.regions._radical_form` on a SymPy
    ``CRootOf`` (other expressions are returned unchanged)."""
    e = as_expr(root)
    if not isinstance(e, ComplexRootOf) or not e.is_real:
        return e
    symbols = sorted_symbols(free_symbols(as_expr(e.expr)))
    if len(symbols) != 1:
        return e
    return sympy_expr(native._radical_form(real_root(core_real(e), symbols[0].name)))


def _radial_integrand(f: ExprLike, names: Sequence[Symbol], rho: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.regions._radial_integrand` on SymPy objects."""
    return _optional(native._radial_integrand(core_expr(as_expr(f)), _core_symbols(names), _core_symbol(rho)))


def _reality_conditions(e: ExprLike) -> Optional[list[Boolean]]:
    """:func:`sympy_extras.integrals.regions._reality_conditions` on SymPy objects."""
    found = native._reality_conditions(core_expr(as_expr(e)))
    return None if found is None else [sympy_boolean(c) for c in found]


def _full_measure(formula: Boolean, names: Sequence[Symbol]) -> Boolean:
    """:func:`sympy_extras.integrals.regions._full_measure` on SymPy objects."""
    return sympy_boolean(native._full_measure(core_boolean(formula), _core_symbols(names)))


def _conditional(value: ExprLike, bound: ExprLike, assumptions: Sequence[Union[Boolean, bool]]) -> Expr:
    """:func:`sympy_extras.integrals.regions._conditional` on SymPy objects.

    >>> from sympy import symbols, pi
    >>> from sympy_extras.integrals.regions_sympy import _conditional
    >>> a = symbols('a')
    >>> _conditional(pi*a**2, a**2, []), _conditional(a, a, [])
    (pi*a**2, Piecewise((a, a > 0), (0, True)))
    """
    return sympy_expr(native._conditional(core_expr(as_expr(value)), core_expr(as_expr(bound)),
                                          _core_list(assumptions)))
