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
in SymPy's ``nonlinsolve``.

An inequation in the unknowns which the other conditions of its case
imply is not written (``y != 0`` with ``a*y**2 = 1`` and ``a != 0``): it
is one when the chain, the vanishing of the inequation and the other
inequations have no common solution, which is again a triangular
decomposition.

The cases are not disjoint in general, and their number is not minimal:
the decomposition depends on the order of the variables.

The module works on objects of the core and imports no SymPy
(:mod:`sympy_extras.solvers.parametric_sympy` is its face on SymPy
objects).

References
==========

.. [ChenMorenoMaza] C. Chen, M. Moreno Maza, *Algorithms for computing
   triangular decomposition of polynomial systems*, J. Symbolic Comput. 47
   (2012); C. Chen, O. Golubitsky, F. Lemaire, M. Moreno Maza, W. Pan,
   *Comprehensive triangular decomposition*, CASC 2007 (the parameters as
   the smallest variables).
"""
from __future__ import annotations

from typing import Final, Iterable, Optional, Sequence

from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, and_
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import sqrt
from sympy_extras.core.relational import eq, ne
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.polytools import Poly, cancel, factor_list
from sympy_extras.polys.regularchains import RegularChain, triangularize
from sympy_extras.sets.sets import Set, point_set

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
    solutions : Set or None
        Their solutions, a finite set of tuples of the unknowns in terms of
        the parameters; an unknown which is free in the case stands for
        itself. ``None`` when an unknown is a root of a polynomial of
        degree three or more.
    """

    __slots__ = ('condition', 'solutions', 'equations')

    def __init__(self, condition: Boolean, solutions: Optional[Set], equations: list[Expr]) -> None:
        self.condition: Final[Boolean] = condition
        self.solutions: Final[Optional[Set]] = solutions
        self.equations: Final[list[Expr]] = equations

    def __repr__(self) -> str:
        shown = str(self.solutions) if self.solutions is not None else str([eq(e, 0) for e in self.equations])
        return "ParametricCase(%s, %s)" % (self.condition, shown)

    def as_formula(self) -> Boolean:
        """The case as a formula in polynomials of the unknowns and the
        parameters."""
        return and_(self.condition, *[eq(e, 0) for e in self.equations])


def _factors(h: Expr) -> list[Expr]:
    """The irreducible factors of a polynomial which is not zero: it does
    not vanish when none of them does."""
    return [f for f, _ in factor_list(h)[1]]


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

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.parametric import parametric_cases
    >>> a, b, c, x, y = symbols('a b c x y')
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
    given = list(equations)
    excluded = list(inequations)
    names = list(unknowns)
    if parameters is None:
        found: set[Symbol] = set()
        for e in given + excluded:
            found |= e.free_symbols
        parameters = sorted(found - set(names), key=lambda s: s.name)
    symbols = names + [p for p in parameters if p not in names]
    cases: list[ParametricCase] = []
    for chain in triangularize(given, *symbols, inequations=excluded):
        nonzero: list[Expr] = []
        for h in list(chain.initials) + excluded:
            for factor in _factors(h):
                if factor.free_symbols and factor not in nonzero:
                    nonzero.append(factor)
        nonzero = _kept_inequations(chain, nonzero, symbols, names)
        conditions: list[Boolean] = []
        in_unknowns: list[tuple[Expr, Symbol]] = []
        for p, v in zip(chain.polys, chain.main_variables):
            if v in names:
                in_unknowns.append((p, v))
            else:
                conditions.append(_solved_parameter(p, v))
        conditions.extend(ne(h, 0) for h in nonzero)
        solutions = _families(in_unknowns, names)
        cases.append(ParametricCase(and_(*conditions), solutions, [p for p, _ in in_unknowns]))
    return cases


def _solved_parameter(p: Expr, v: Symbol) -> Boolean:
    """The equation of a case, solved for its main variable when that one
    occurs alone in it with a constant coefficient (``a = 0``, ``b = 2``)."""
    poly = Poly(p, v)
    if poly.degree() == 1 and not poly.LC().free_symbols and not poly.TC().free_symbols:
        return eq(v, -poly.TC() / poly.LC())
    return eq(p, 0)


def _families(polys: Sequence[tuple[Expr, Symbol]], unknowns: Sequence[Symbol]) -> Optional[Set]:
    """The unknowns from the polynomials of a chain, the smallest main
    variable first; ``None`` when one has degree three or more."""
    families: list[dict[Symbol, Expr]] = [{}]
    for p, v in polys:
        poly = Poly(p, v)
        if poly.degree() > 2:
            return None
        extended: list[dict[Symbol, Expr]] = []
        for family in families:
            coefficients = [_substituted(c, family) for c in poly.all_coeffs()]
            if len(coefficients) == 2:
                values = [cancel(-coefficients[1] / coefficients[0])]
            else:
                a, b, c = coefficients
                root = sqrt(cancel(b**2 - 4 * a * c))
                two = Integer(2)
                values = [-b / (two * a) - root / (two * a), -b / (two * a) + root / (two * a)]
            for value in values:
                longer = dict(family)
                longer[v] = value
                extended.append(longer)
        families = extended
    return point_set(*[Tuple(*[family.get(s, s) for s in unknowns]) for family in families])


def _substituted(e: Expr, values: dict[Symbol, Expr]) -> Expr:
    """``e`` with the values of the unknowns found so far, evaluated."""
    if not values:
        return e
    rule: dict[Basic, Basic] = {s: value for s, value in values.items()}
    result = e._xreplace_evaluated(rule)
    if not isinstance(result, Expr):
        raise TypeError("expected an expression, got %s" % (result,))
    return result
