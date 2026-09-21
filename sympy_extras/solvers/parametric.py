"""Polynomial systems with parameters: the cases of the parameters.

``a*x = b`` has the solution ``b/a`` for ``a != 0``, every ``x`` for
``a = b = 0`` and none otherwise. SymPy's solvers (and Mathematica's
``Solve``) give the first case, the generic one; Mathematica's ``Reduce``
gives them all. They are read here from a triangular decomposition in the
sense of Lazard (:func:`~sympy_extras.polys.regularchains.triangularize`)
with the unknowns greater than the parameters: the complex zeros of the
system are the union of the quasi-components of the regular chains,

.. math::

    V(F) = \\bigcup_i W(T_i), \\qquad W(T) = V(T) \\setminus V(h_T),

`h_T` the product of the initials of `T` [ChenMorenoMaza]_. In a chain, the
polynomials whose main variable is a parameter are the equations of the
case, the initials (and the inequations of the system) which do not vanish
are its inequations, and the polynomials whose main variable is an unknown
give the unknowns one after the other, the smallest first: a polynomial of
degree one by a division (its initial does not vanish in the case), one of
degree two by the quadratic formula, which holds over the complex numbers
whatever the values of the parameters. A chain with a polynomial of higher
degree in an unknown is a case whose solutions are left as the equations
of the chain. An unknown which is not the main variable of a polynomial of
the chain is free in the case, and stands for itself in the solutions, as
in :func:`sympy.solvers.solveset.nonlinsolve`.

An inequation in the unknowns which the other conditions of its case
imply is not written (``y != 0`` with ``a*y**2 = 1`` and ``a != 0``): it
is one when the chain, the vanishing of the inequation and the other
inequations have no common solution, which is again a triangular
decomposition.

The cases are not disjoint in general, and their number is not minimal:
the decomposition depends on the order of the variables.

References
==========

.. [ChenMorenoMaza] C. Chen, M. Moreno Maza, *Algorithms for computing
   triangular decomposition of polynomial systems*, J. Symbolic Comput. 47
   (2012); C. Chen, O. Golubitsky, F. Lemaire, M. Moreno Maza, W. Pan,
   *Comprehensive triangular decomposition*, CASC 2007 (the parameters as
   the smallest variables).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import And, Boolean
from sympy.polys.polytools import Poly, cancel, factor_list
from sympy.sets.sets import FiniteSet

from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.polys.regularchains import RegularChain, triangularize

__all__ = ['ParametricCase', 'parametric_cases']


class ParametricCase:
    """A case of a system with parameters.

    Attributes
    ==========

    condition : Boolean
        The equations and inequations of the case. They are in the
        parameters, but for an inequation in the unknowns which the others
        do not imply.
    equations : list of Expr
        The polynomials of the chain in the unknowns, which vanish: with
        the condition, they are the case.
    solutions : FiniteSet or None
        Their solutions, tuples of the unknowns in terms of the parameters;
        an unknown which is free in the case stands for itself. ``None``
        when an unknown is a root of a polynomial of degree three or more.
    """

    __slots__ = ('condition', 'solutions', 'equations')

    def __init__(self, condition: Boolean, solutions: Optional[FiniteSet], equations: list[Expr]) -> None:
        self.condition = condition
        self.solutions = solutions
        self.equations = equations

    def __repr__(self) -> str:
        shown = self.solutions if self.solutions is not None else [Eq(e, 0) for e in self.equations]
        return "ParametricCase(%s, %s)" % (self.condition, shown)

    def as_formula(self) -> Boolean:
        """The case as a formula in polynomials of the unknowns and the
        parameters."""
        return as_boolean(And(self.condition, *[Eq(e, 0) for e in self.equations]))


def _factors(h: Expr) -> list[Expr]:
    """The irreducible factors of a polynomial which is not zero: it does
    not vanish when none of them does."""
    return [as_expr(f) for f, _ in factor_list(h)[1]]


def _kept_inequations(chain: RegularChain, inequations: list[Expr], symbols: Sequence[Symbol],
                      unknowns: Sequence[Symbol]) -> list[Expr]:
    """The inequations in the parameters, and those in the unknowns which
    the chain and the others do not imply. One in the parameters stays
    even when it is implied: the chain ``(a + 1)*y - 1, (a + 1)*x - 1``
    has no zero with ``a = -1``, but its solution ``1/(a + 1)`` is only
    written for the other values."""
    kept = list(inequations)
    for h in list(inequations):
        if not h.has(*unknowns):
            continue
        others = [other for other in kept if other != h]
        if not triangularize(list(chain.polys) + [h], *symbols, inequations=others):
            kept = others
    return kept


def parametric_cases(equations: Iterable[Expr], unknowns: Sequence[Symbol],
                     parameters: Optional[Sequence[Symbol]] = None,
                     inequations: Iterable[Expr] = ()) -> list[ParametricCase]:
    """The complex solutions of polynomial ``equations`` with rational
    coefficients in the ``unknowns``, for every value of the
    ``parameters`` (the other symbols of the system, by default): the
    cases of the parameters, each with its solutions. The ``inequations``
    are polynomials which do not vanish.

    Examples
    ========

    >>> from sympy.abc import a, b, c, x, y
    >>> from sympy_extras.solvers.parametric import parametric_cases
    >>> parametric_cases([a*x - b], [x])
    [ParametricCase(Ne(a, 0), {(b/a,)}), ParametricCase(Eq(a, 0) & Eq(b, 0), {(x,)})]
    >>> for case in parametric_cases([a*x**2 + b*x + c], [x]):
    ...     print(case)
    ParametricCase(Ne(a, 0), {(-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a),), (-b/(2*a) + sqrt(-4*a*c + b**2)/(2*a),)})
    ParametricCase(Eq(a, 0) & Ne(b, 0), {(-c/b,)})
    ParametricCase(Eq(a, 0) & Eq(b, 0) & Eq(c, 0), {(x,)})
    >>> parametric_cases([a*x + y - 1, x + a*y - 1], [x, y])
    [ParametricCase(Eq(a, 1), {(1 - y, y)}), ParametricCase(Ne(a + 1, 0), {(1/(a + 1), 1/(a + 1))})]
    """
    given = [as_expr(e) for e in equations]
    excluded = [as_expr(h) for h in inequations]
    names = list(unknowns)
    if parameters is None:
        found: set[Symbol] = set()
        for e in given + excluded:
            found |= free_symbols(e)
        parameters = sorted_symbols(found - set(names))
    symbols = names + [p for p in parameters if p not in names]
    cases: list[ParametricCase] = []
    for chain in triangularize(given, *symbols, inequations=excluded):
        nonzero: list[Expr] = []
        for h in list(chain.initials) + excluded:
            for factor in _factors(as_expr(h)):
                if factor.free_symbols and factor not in nonzero:
                    nonzero.append(factor)
        nonzero = _kept_inequations(chain, nonzero, symbols, names)
        conditions: list[Boolean] = []
        in_unknowns: list[tuple[Expr, Symbol]] = []
        for p, v in zip(chain.polys, chain.main_variables):
            if v in names:
                in_unknowns.append((as_expr(p), v))
            else:
                conditions.append(_solved_parameter(as_expr(p), v))
        conditions.extend(Ne(h, 0) for h in nonzero)
        solutions = _families(in_unknowns, names)
        cases.append(ParametricCase(as_boolean(And(*conditions)), solutions, [p for p, _ in in_unknowns]))
    return cases


def _solved_parameter(p: Expr, v: Symbol) -> Boolean:
    """The equation of a case, solved for its main variable when that one
    occurs alone in it with a constant coefficient (``a = 0``, ``b = 2``)."""
    poly = Poly(p, v)
    if poly.degree() == 1 and not as_expr(poly.LC()).free_symbols and not as_expr(poly.TC()).free_symbols:
        return as_boolean(Eq(v, -as_expr(poly.TC()) / as_expr(poly.LC())))
    return as_boolean(Eq(p, 0))


def _families(polys: Sequence[tuple[Expr, Symbol]], unknowns: Sequence[Symbol]) -> Optional[FiniteSet]:
    """The unknowns from the polynomials of a chain, the smallest main
    variable first; ``None`` when one has degree three or more."""
    families: list[dict[Symbol, Expr]] = [{}]
    for p, v in polys:
        poly = Poly(p, v)
        if poly.degree() > 2:
            return None
        extended: list[dict[Symbol, Expr]] = []
        for family in families:
            coefficients = [as_expr(as_expr(c).xreplace(family)) for c in poly.all_coeffs()]
            if len(coefficients) == 2:
                values = [as_expr(cancel(-coefficients[1] / coefficients[0]))]
            else:
                a, b, c = coefficients
                root = sqrt(as_expr(cancel(b**2 - 4 * a * c)))
                values = [as_expr(-b / (2 * a) - root / (2 * a)), as_expr(-b / (2 * a) + root / (2 * a))]
            for value in values:
                longer = dict(family)
                longer[v] = value
                extended.append(longer)
        families = extended
    return FiniteSet(*[Tuple(*[family.get(s, s) for s in unknowns]) for family in families])
