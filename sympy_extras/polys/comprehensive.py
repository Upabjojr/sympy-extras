"""Reduction over the complex numbers: comprehensive Gröbner systems and
quantifier elimination for polynomial equations and inequations.

Over the complex numbers, a system of polynomial equations `f_i = 0` and
inequations `g_j \\ne 0` in the unknowns `x` with parameters `p` has a
solution for exactly those `p` which form a *constructible* set, again
described by equations and inequations in `p`. The description is found
with a **comprehensive Gröbner system** (Weispfenning): a finite family of
branches, each a constructible set of parameters together with polynomials
which specialise to a Gröbner basis of the system on the whole branch. On
a branch the system is solvable if and only if that Gröbner basis contains
no nonzero constant. The branches are built with the algorithm of Kapur,
Sun and Wang: a Gröbner basis for a block order with the unknowns in the
first block is split according to the vanishing of the leading
coefficients (in the unknowns) of its elements. Inequations are turned
into equations by Rabinowitsch's trick (`g \\ne 0 \\iff \\exists z\\; z g = 1`),
and a universal quantifier is `\\neg\\exists\\neg`. Quantifier-free formulas
are simplified with Gröbner bases: the equations of a conjunction are
replaced by the reduced basis of their ideal and the inequations reduced
modulo it (with radical membership deciding whether an inequation
contradicts the equations).

References
==========

.. [KapurSunWang] D. Kapur, Y. Sun, D. Wang, A new algorithm for
   computing comprehensive Gröbner systems, ISSAC 2010.
.. [Weispfenning] V. Weispfenning, Comprehensive Gröbner bases, Journal of
   Symbolic Computation 14 (1992).
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.relational import Relational, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or, Not, true, false,
    to_dnf)
from sympy.polys.domains import QQ
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import monomial_divides
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, factor_list
from sympy.polys.rings import ring as _ring, PolyElement

from sympy_extras._typing import QuantifierPrefix, as_boolean, as_expr, free_symbols, sorted_symbols

from .ideals import Ideal
from .orderings import elimination_order

__all__ = ['Branch', 'comprehensive_groebner_system', 'exists_complex',
           'complex_quantifier_elimination', 'reduce_complex']


class Branch:
    """A branch of a comprehensive Gröbner system: on the parameters
    where the ``equations`` vanish and the ``nonzero`` polynomials do
    not, the ``basis`` specialises to a Gröbner basis of the system.

    ``basis == [1]`` means that the system has no solution on the branch,
    ``basis == []`` that every point is a solution.
    """

    def __init__(self, equations: list[Expr], nonzero: list[Expr], basis: list[Expr]) -> None:
        self.equations = equations
        self.nonzero = nonzero
        self.basis = basis

    def condition(self) -> Boolean:
        """The constructible set of the branch, as a formula."""
        return And(*[Eq(e, 0) for e in self.equations], *[Ne(n, 0) for n in self.nonzero])

    @property
    def solvable(self) -> bool:
        """Whether the system has solutions on the branch."""
        return not any(g.is_number and g != 0 for g in self.basis)

    def __repr__(self) -> str:
        return "Branch(%s, %s, %s)" % (self.equations, self.nonzero, self.basis)


def _consistent(equations: list[Expr], nonzero: list[Expr], parameters: list[Symbol]) -> bool:
    """Whether some parameter value satisfies the conditions: the product
    of the nonzero polynomials is not in the radical of the ideal of the
    equations."""
    if not equations:
        return True
    product = as_expr(Mul(*nonzero)) if nonzero else S.One
    return not Ideal(equations, *parameters).radical_contains(product)


def _irreducible_factors(e: Expr, symbols: Sequence[Symbol]) -> list[Expr]:
    """The nonconstant irreducible factors of ``e`` (without repetition)."""
    _, factors = factor_list(e, *symbols)
    result: list[Expr] = []
    for f, _ in factors:
        f_ = as_expr(f)
        if free_symbols(f_) and f_ not in result:
            result.append(f_)
    return result


def _cgs(equations: list[Expr], nonzero: list[Expr], polynomials: list[Expr],
         variables: list[Symbol], parameters: list[Symbol], branches: list[Branch]) -> None:
    if not _consistent(equations, nonzero, parameters):
        return
    k = len(variables)
    ordered = variables + parameters
    R = _ring(ordered, QQ, elimination_order(k, len(ordered)))[0]
    gens = [R.from_expr(f) for f in polynomials + equations if f != 0]
    G = _groebner(gens, R) if gens else []
    if any(g.is_ground for g in G):
        branches.append(Branch(equations, nonzero, [S.One]))
        return
    parametric = [g for g in G if all(all(e == 0 for e in m[:k]) for m in g.monoms())]
    mixed = [g for g in G if g not in parametric]
    # where some polynomial of the parametric part does not vanish, the
    # specialised system contains a nonzero constant
    for g in parametric:
        if _consistent(equations, nonzero + [g.as_expr()], parameters):
            branches.append(Branch(equations, nonzero + [g.as_expr()], [S.One]))
    equations = [g.as_expr() for g in parametric]
    if not _consistent(equations, nonzero, parameters):
        return
    basis = [g.as_expr() for g in mixed]
    if not mixed:
        branches.append(Branch(equations, nonzero, []))
        return
    # the leading coefficients (in the unknowns) of the elements with
    # minimal leading monomials: where they do not vanish the basis
    # specialises to a Gröbner basis
    leading = [(g.LM[:k], g) for g in mixed]
    minimal = [g for m, g in leading
               if not any(m2 != m and monomial_divides(m2, m) for m2, _ in leading)]
    coefficients: list[Expr] = []
    for g in minimal:
        for f in _irreducible_factors(_leading_coefficient(g, k, parameters), parameters):
            if f not in coefficients:
                coefficients.append(f)
    branches.append(Branch(equations, nonzero + coefficients, basis))
    for i, h in enumerate(coefficients):
        _cgs(equations + [h], nonzero + coefficients[:i], basis, variables, parameters, branches)


def _leading_coefficient(g: PolyElement, k: int, parameters: list[Symbol]) -> Expr:
    """The coefficient, a polynomial in the parameters, of the leading
    monomial of ``g`` in the first ``k`` variables."""
    head = g.LM[:k]
    domain = g.ring.domain
    terms: list[Expr] = []
    for m, c in g.terms():
        if tuple(m[:k]) == tuple(head):
            terms.append(as_expr(domain.to_sympy(c)*Mul(*[p**e for p, e in zip(parameters, m[k:])])))
    return as_expr(sum(terms, S.Zero))


def comprehensive_groebner_system(polynomials: Sequence[Expr], variables: Sequence[Symbol],
                                  parameters: Sequence[Symbol]) -> list[Branch]:
    """The comprehensive Gröbner system of polynomials in the variables
    with coefficients depending on the parameters (Kapur, Sun and Wang):
    branches covering the parameter space, on each of which the basis
    specialises to a Gröbner basis (for the graded reverse lexicographic
    order on the variables) of the specialised system.

    Examples
    ========

    >>> from sympy.abc import a, b, x
    >>> from sympy_extras.polys.comprehensive import comprehensive_groebner_system
    >>> for branch in comprehensive_groebner_system([a*x - b], [x], [a, b]):
    ...     print(branch)
    Branch([], [a], [a*x - b])
    Branch([a], [b], [1])
    Branch([a, b], [], [])
    """
    variables_ = list(variables)
    parameters_ = list(parameters)
    branches: list[Branch] = []
    _cgs([], [], [as_expr(p) for p in polynomials], variables_, parameters_, branches)
    return branches


def exists_complex(equations: Sequence[Expr], inequations: Sequence[Expr],
                   variables: Sequence[Symbol]) -> Boolean:
    """The condition on the parameters (the other symbols) for the system
    of equations ``f = 0`` and inequations ``g != 0`` to have a solution
    in the variables over the complex numbers.

    Examples
    ========

    >>> from sympy.abc import a, b, c, x
    >>> from sympy_extras.polys.comprehensive import exists_complex
    >>> exists_complex([a*x - b], [], [x])
    Ne(a, 0) | (Eq(a, 0) & Eq(b, 0))
    >>> exists_complex([a*x**2 + b*x + c, 2*a*x + b], [a], [x])
    Ne(a, 0) & Eq(-4*a*c + b**2, 0)
    >>> exists_complex([x**2 - 2*x + 1], [x - a], [x])
    Ne(a - 1, 0)
    """
    variables_ = list(variables)
    polynomials = [as_expr(e) for e in equations]
    for g in inequations:
        z = Dummy('z')
        variables_.append(z)
        polynomials.append(as_expr(z*g - 1))
    parameters = sorted_symbols(set().union(*[free_symbols(p) for p in polynomials]) - set(variables_)) \
        if polynomials else []
    if not parameters:
        # a closed question: the ideal is the whole ring or not
        if not polynomials:
            return true
        return false if Ideal(polynomials, *variables_).is_whole_ring() else true
    conditions = [branch.condition() for branch in
                  comprehensive_groebner_system(polynomials, variables_, parameters) if branch.solvable]
    return as_boolean(Or(*conditions))


# ---------------------------------------------------------------------------
# formulas

def _polynomial_atom(atom: Relational, symbols: set[Symbol]) -> Expr:
    """``lhs - rhs`` of an equation or inequation, which must be a
    polynomial in the symbols with rational coefficients."""
    if not isinstance(atom, (Eq, Ne)):
        raise ValueError("only equations and inequations are meaningful over the complex "
                         "numbers, got %s" % (atom,))
    e = as_expr(atom.lhs - atom.rhs)
    gens = sorted_symbols(free_symbols(e))
    if not gens:
        return e
    try:
        poly = Poly(e, *gens)
    except PolynomialError:
        raise ValueError("%s is not polynomial" % (atom,))
    if not all(c.is_rational for c in poly.coeffs()) or not free_symbols(e) <= symbols:
        raise ValueError("%s is not a polynomial relation with rational coefficients" % (atom,))
    return e


def _conjunctions(formula: Boolean) -> list[tuple[list[Expr], list[Expr]]]:
    """The equations and inequations of each conjunction of the
    disjunctive normal form of the formula."""
    dnf = to_dnf(formula.to_nnf(simplify=False), simplify=False)
    if isinstance(dnf, BooleanFalse):
        return []
    result: list[tuple[list[Expr], list[Expr]]] = []
    symbols = set(s for s in dnf.free_symbols if isinstance(s, Symbol))
    for conjunction in (dnf.args if isinstance(dnf, Or) else [dnf]):
        literals = conjunction.args if isinstance(conjunction, And) else [conjunction]
        equations: list[Expr] = []
        inequations: list[Expr] = []
        consistent = True
        for literal in literals:
            if isinstance(literal, BooleanTrue):
                continue
            if isinstance(literal, BooleanFalse):
                consistent = False
                break
            if not isinstance(literal, Relational):
                raise ValueError("%s is not a polynomial relation" % (literal,))
            e = _polynomial_atom(literal, symbols)
            (equations if isinstance(literal, Eq) else inequations).append(e)
        if consistent:
            result.append((equations, inequations))
    return result


def _eliminate_block(formula: Boolean, variables: list[Symbol]) -> Boolean:
    disjuncts = [exists_complex(equations, inequations, variables)
                 for equations, inequations in _conjunctions(formula)]
    return as_boolean(Or(*disjuncts))


def complex_quantifier_elimination(formula: Boolean, prefix: QuantifierPrefix) -> Boolean:
    """Eliminate the quantifiers (outermost first in ``prefix``) of a
    Boolean combination of polynomial equations and inequations over the
    complex numbers.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, b, x, y
    >>> from sympy_extras.polys.comprehensive import complex_quantifier_elimination
    >>> complex_quantifier_elimination(Eq(x**2, a), [('exists', x)])
    True
    >>> complex_quantifier_elimination(Eq(a*x, 1), [('exists', x)])
    Ne(a, 0)
    >>> complex_quantifier_elimination(Eq(x*y, 1), [('forall', x), ('exists', y)])
    False
    >>> complex_quantifier_elimination(Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('exists', x)])
    Eq(a**2 - 4*b, 0)
    """
    current = as_boolean(formula)
    remaining = list(prefix)
    while remaining:
        kind = remaining[-1][0]
        block: list[Symbol] = []
        while remaining and remaining[-1][0] == kind:
            block.insert(0, remaining.pop()[1])
        if kind == 'exists':
            current = _eliminate_block(current, block)
        else:
            current = as_boolean(Not(_eliminate_block(as_boolean(Not(current)), block)))
    return reduce_complex(current)


def _reduce_conjunction(equations: list[Expr], inequations: list[Expr]) -> Boolean:
    symbols = sorted_symbols(set().union(*[free_symbols(e) for e in equations + inequations]))
    if not symbols:
        return as_boolean(And(*[Eq(e, 0) for e in equations], *[Ne(g, 0) for g in inequations]))
    equations = [e for e in equations if e != 0]
    if not equations:
        basis: list[Expr] = []
        ideal: Optional[Ideal] = None
    else:
        ideal = Ideal(equations, *symbols)
        if ideal.is_whole_ring():
            return false
        basis = [as_expr(g.as_expr()) for g in ideal.groebner_basis()]
    factors: list[Expr] = []
    for g in inequations:
        reduced = ideal.reduce(g) if ideal is not None else g
        if reduced == 0:
            return false
        if not free_symbols(reduced):
            continue
        for f in _irreducible_factors(reduced, symbols):
            if ideal is not None:
                if ideal.radical_contains(f):
                    return false
                if Ideal(basis + [f], *symbols).is_whole_ring():
                    # the factor does not vanish on the variety
                    continue
            if f not in factors:
                factors.append(f)
    return as_boolean(And(*[Eq(g, 0) for g in basis], *[Ne(f, 0) for f in factors]))


def reduce_complex(formula: Boolean) -> Boolean:
    """Simplify a quantifier-free Boolean combination of polynomial
    equations and inequations over the complex numbers: in each
    conjunction of its disjunctive normal form the equations are replaced
    by the reduced Gröbner basis of their ideal, and the inequations
    reduced modulo it (an inequation contradicts the equations when its
    polynomial is in the radical of the ideal).

    Examples
    ========

    >>> from sympy import Eq, Ne
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.comprehensive import reduce_complex
    >>> reduce_complex(Eq(x**2 - y, 0) & Eq(x*y - 1, 0) & Ne(x, 0))
    Eq(x*y - 1, 0) & Eq(-x + y**2, 0) & Eq(x**2 - y, 0)
    >>> reduce_complex(Eq(x**2, 0) & Ne(x, 0))
    False
    """
    formula = as_boolean(formula)
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return formula
    disjuncts = [_reduce_conjunction(equations, inequations)
                 for equations, inequations in _conjunctions(formula)]
    return as_boolean(Or(*disjuncts))
