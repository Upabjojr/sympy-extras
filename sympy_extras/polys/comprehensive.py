"""Reduction over the complex numbers: comprehensive Gröbner systems and
quantifier elimination for polynomial equations and inequations, on the
typed polynomial layer (no SymPy).

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

The polynomials are core expressions with rational coefficients, the
formulas Booleans of the core, the ideals those of
:mod:`sympy_extras.polys.ideals` over ``QQ`` (their Gröbner bases by F4),
and the factorisations those of :mod:`sympy_extras.polys.polytools`.
:mod:`sympy_extras.polys.comprehensive_sympy` is the SymPy face of this
module.

References
==========

.. [KapurSunWang] D. Kapur, Y. Sun, D. Wang, A new algorithm for
   computing comprehensive Gröbner systems, ISSAC 2010.
.. [Weispfenning] V. Weispfenning, Comprehensive Gröbner bases, Journal of
   Symbolic Computation 14 (1992).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Optional, Sequence

from sympy_extras.core.add import add, sub
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import (And, Boolean, BooleanFalse, BooleanTrue, Equivalent, ITE, Implies, Not, Or, Xor,
    and_, false, not_, or_, to_boolean, true)
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.core.relational import Eq, Ne, Relational, eq, ne
from sympy_extras.core.symbol import Dummy, Symbol

from .domains.rationalfield import QQ
from .ideals import Ideal
from .polytools import factor_list
from .polyutils import dict_from_expr
from .rings import PolyElement, elimination_order, monomial_divides

__all__ = ['Branch', 'QuantifierPrefix', 'comprehensive_groebner_system', 'exists_complex',
           'complex_quantifier_elimination', 'reduce_complex']

#: a quantifier prefix, outermost first: ``('exists', x)``, ``('forall', y)``
QuantifierPrefix = Sequence[tuple[str, Symbol]]


class Branch:
    """A branch of a comprehensive Gröbner system: on the parameters
    where the ``equations`` vanish and the ``nonzero`` polynomials do
    not, the ``basis`` specialises to a Gröbner basis of the system.

    ``basis == (1,)`` means that the system has no solution on the branch,
    ``basis == ()`` that every point is a solution.
    """

    def __init__(self, equations: Sequence[Expr], nonzero: Sequence[Expr], basis: Sequence[Expr]) -> None:
        self.equations: Final[tuple[Expr, ...]] = tuple(equations)
        self.nonzero: Final[tuple[Expr, ...]] = tuple(nonzero)
        self.basis: Final[tuple[Expr, ...]] = tuple(basis)

    def condition(self) -> Boolean:
        """The constructible set of the branch, as a formula."""
        zero = Integer(0)
        return and_(*[eq(e, zero) for e in self.equations], *[ne(n, zero) for n in self.nonzero])

    @property
    def solvable(self) -> bool:
        """Whether the system has solutions on the branch."""
        return not any(not g.free_symbols and g != Integer(0) for g in self.basis)

    def __repr__(self) -> str:
        return "Branch([%s], [%s], [%s])" % tuple(", ".join(str(e) for e in part)
                                                   for part in (self.equations, self.nonzero, self.basis))


def _ideal(polynomials: Sequence[Expr], symbols: Sequence[Symbol]) -> Ideal[Fraction]:
    return Ideal(polynomials, *symbols, domain=QQ)


def _consistent(equations: list[Expr], nonzero: list[Expr], parameters: list[Symbol]) -> bool:
    """Whether some parameter value satisfies the conditions: the product
    of the nonzero polynomials is not in the radical of the ideal of the
    equations."""
    if not equations:
        return True
    product = mul(*nonzero) if nonzero else Integer(1)
    return not _ideal(equations, parameters).radical_contains(product)


def _irreducible_factors(e: Expr, symbols: Sequence[Symbol]) -> list[Expr]:
    """The nonconstant irreducible factors of ``e`` (without repetition)."""
    _, factors = factor_list(e, *symbols)
    result: list[Expr] = []
    for f, _ in factors:
        if f.free_symbols and f not in result:
            result.append(f)
    return result


def _is_parametric(g: PolyElement[Fraction], k: int) -> bool:
    """Whether ``g`` is free of the first ``k`` variables."""
    return all(all(e == 0 for e in m[:k]) for m in g.terms)


def _cgs(equations: list[Expr], nonzero: list[Expr], polynomials: list[Expr],
         variables: list[Symbol], parameters: list[Symbol], branches: list[Branch]) -> None:
    if not _consistent(equations, nonzero, parameters):
        return
    k = len(variables)
    ordered = variables + parameters
    gens = [f for f in polynomials + equations if f != Integer(0)]
    ideal: Ideal[Fraction] = Ideal(gens, *ordered, domain=QQ, order=elimination_order(k, len(ordered)))
    G = ideal.groebner_basis() if gens else []
    if any(g.is_ground for g in G):
        branches.append(Branch(equations, nonzero, [Integer(1)]))
        return
    parametric = [g for g in G if _is_parametric(g, k)]
    mixed = [g for g in G if not _is_parametric(g, k)]
    # where some polynomial of the parametric part does not vanish, the
    # specialised system contains a nonzero constant
    for g in parametric:
        h = ideal.to_expr(g)
        if _consistent(equations, nonzero + [h], parameters):
            branches.append(Branch(equations, nonzero + [h], [Integer(1)]))
    equations = [ideal.to_expr(g) for g in parametric]
    if not _consistent(equations, nonzero, parameters):
        return
    basis = [ideal.to_expr(g) for g in mixed]
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


def _leading_coefficient(g: PolyElement[Fraction], k: int, parameters: list[Symbol]) -> Expr:
    """The coefficient, a polynomial in the parameters, of the leading
    monomial of ``g`` in the first ``k`` variables."""
    head = g.LM[:k]
    terms: list[Expr] = []
    for m, c in g.terms.items():
        if m[:k] == head:
            factors: list[Expr] = [QQ.to_expr(c)]
            factors.extend(pow(p, Integer(e)) for p, e in zip(parameters, m[k:]) if e)
            terms.append(mul(*factors))
    return add(*terms)


def comprehensive_groebner_system(polynomials: Sequence[Expr], variables: Sequence[Symbol],
                                  parameters: Sequence[Symbol]) -> list[Branch]:
    """The comprehensive Gröbner system of polynomials in the variables
    with coefficients depending on the parameters (Kapur, Sun and Wang):
    branches covering the parameter space, on each of which the basis
    specialises to a Gröbner basis (for the graded reverse lexicographic
    order on the variables) of the specialised system.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.comprehensive import comprehensive_groebner_system
    >>> a, b, x = symbols('a b x')
    >>> for branch in comprehensive_groebner_system([a*x - b], [x], [a, b]):
    ...     print(branch)
    Branch([], [a], [a*x - b])
    Branch([a], [b], [1])
    Branch([a, b], [], [])
    """
    branches: list[Branch] = []
    _cgs([], [], list(polynomials), list(variables), list(parameters), branches)
    return branches


def exists_complex(equations: Sequence[Expr], inequations: Sequence[Expr],
                   variables: Sequence[Symbol]) -> Boolean:
    """The condition on the parameters (the other symbols) for the system
    of equations ``f = 0`` and inequations ``g != 0`` to have a solution
    in the variables over the complex numbers.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.comprehensive import exists_complex
    >>> a, b, c, x = symbols('a b c x')
    >>> exists_complex([a*x - b], [], [x])
    Ne(a, 0) | (Eq(a, 0) & Eq(b, 0))
    >>> exists_complex([a*x**2 + b*x + c, 2*a*x + b], [a], [x])
    Ne(a, 0) & Eq(-4*a*c + b**2, 0)
    >>> exists_complex([x**2 - 2*x + 1], [x - a], [x])
    Ne(a - 1, 0)
    """
    variables_ = list(variables)
    polynomials = list(equations)
    for g in inequations:
        z = Dummy('z')
        variables_.append(z)
        polynomials.append(sub(mul(z, g), Integer(1)))
    found: set[Symbol] = set()
    for p in polynomials:
        found |= p.free_symbols
    parameters = sorted(found - set(variables_), key=lambda s: s.name)
    if not parameters:
        # a closed question: the ideal is the whole ring or not
        if not polynomials:
            return true
        return false if _ideal(polynomials, variables_).is_whole_ring() else true
    conditions = [branch.condition() for branch in
                  comprehensive_groebner_system(polynomials, variables_, parameters) if branch.solvable]
    return or_(*conditions)


# ---------------------------------------------------------------------------
# formulas

def _polynomial_atom(atom: Relational) -> Expr:
    """``lhs - rhs`` of an equation or inequation, which must be a
    polynomial with rational coefficients."""
    if not isinstance(atom, (Eq, Ne)):
        raise ValueError("only equations and inequations are meaningful over the complex "
                         "numbers, got %s" % (atom,))
    e = sub(atom.lhs, atom.rhs)
    names = sorted({s.name for s in e.free_symbols})
    try:
        dict_from_expr(e, names, QQ)
    except ValueError:
        raise ValueError("%s is not a polynomial relation with rational coefficients" % (atom,))
    return e


def _disjunctive_form(formula: Boolean, positive: bool = True) -> list[list[Boolean]]:
    """The literals of the conjunctions of a disjunctive normal form of the
    formula (of its negation when not ``positive``), the negations pushed
    down to the atoms (the negation of a relation is the opposite
    relation)."""
    if isinstance(formula, BooleanTrue):
        return [[]] if positive else []
    if isinstance(formula, BooleanFalse):
        return [] if positive else [[]]
    if isinstance(formula, Not):
        return _disjunctive_form(formula.arg, not positive)
    if not isinstance(formula, (And, Or, Implies, Equivalent, Xor, ITE)):
        return [[formula if positive else not_(formula)]]
    args = [to_boolean(a) for a in formula.args]
    if isinstance(formula, (And, Or)):
        if isinstance(formula, And) == positive:
            # a conjunction (or the negation of a disjunction)
            result: list[list[Boolean]] = [[]]
            for a in args:
                result = [left + right for left in result for right in _disjunctive_form(a, positive)]
            return result
        return [c for a in args for c in _disjunctive_form(a, positive)]
    if isinstance(formula, Implies):
        return _disjunctive_form(or_(not_(args[0]), args[1]), positive)
    if isinstance(formula, Equivalent):
        return _disjunctive_form(or_(and_(*args), and_(*[not_(a) for a in args])), positive)
    if isinstance(formula, Xor):
        parity = args[0]
        for a in args[1:]:
            parity = or_(and_(parity, not_(a)), and_(not_(parity), a))
        return _disjunctive_form(parity, positive)
    return _disjunctive_form(or_(and_(args[0], args[1]), and_(not_(args[0]), args[2])), positive)


def _conjunctions(formula: Boolean) -> list[tuple[list[Expr], list[Expr]]]:
    """The equations and inequations of each conjunction of the
    disjunctive normal form of the formula."""
    result: list[tuple[list[Expr], list[Expr]]] = []
    seen: list[Boolean] = []
    for literals in _disjunctive_form(formula):
        conjunction = and_(*literals)
        if isinstance(conjunction, BooleanFalse) or conjunction in seen:
            continue
        seen.append(conjunction)
        equations: list[Expr] = []
        inequations: list[Expr] = []
        for literal in (conjunction.args if isinstance(conjunction, And) else
                        (() if isinstance(conjunction, BooleanTrue) else (conjunction,))):
            if not isinstance(literal, Relational):
                raise ValueError("%s is not a polynomial relation" % (literal,))
            e = _polynomial_atom(literal)
            (equations if isinstance(literal, Eq) else inequations).append(e)
        result.append((equations, inequations))
    return result


def _eliminate_block(formula: Boolean, variables: list[Symbol]) -> Boolean:
    disjuncts = [exists_complex(equations, inequations, variables)
                 for equations, inequations in _conjunctions(formula)]
    return or_(*disjuncts)


def complex_quantifier_elimination(formula: Boolean, prefix: QuantifierPrefix) -> Boolean:
    """Eliminate the quantifiers (outermost first in ``prefix``) of a
    Boolean combination of polynomial equations and inequations over the
    complex numbers.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.polys.comprehensive import complex_quantifier_elimination
    >>> a, b, x, y = symbols('a b x y')
    >>> complex_quantifier_elimination(Eq(x**2, a), [('exists', x)])
    True
    >>> complex_quantifier_elimination(Eq(a*x, 1), [('exists', x)])
    Ne(a, 0)
    >>> complex_quantifier_elimination(Eq(x*y, 1), [('forall', x), ('exists', y)])
    False
    >>> complex_quantifier_elimination(Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('exists', x)])
    Eq(a**2 - 4*b, 0)
    """
    current = formula
    remaining = list(prefix)
    while remaining:
        kind = remaining[-1][0]
        block: list[Symbol] = []
        while remaining and remaining[-1][0] == kind:
            block.insert(0, remaining.pop()[1])
        if kind == 'exists':
            current = _eliminate_block(current, block)
        else:
            current = not_(_eliminate_block(not_(current), block))
    return reduce_complex(current)


def _reduce_conjunction(equations: list[Expr], inequations: list[Expr]) -> Boolean:
    found: set[Symbol] = set()
    for e in equations + inequations:
        found |= e.free_symbols
    symbols = sorted(found, key=lambda s: s.name)
    zero = Integer(0)
    if not symbols:
        return and_(*[eq(e, zero) for e in equations], *[ne(g, zero) for g in inequations])
    equations = [e for e in equations if e != zero]
    ideal: Optional[Ideal[Fraction]] = None
    basis: list[Expr] = []
    if equations:
        ideal = _ideal(equations, symbols)
        if ideal.is_whole_ring():
            return false
        basis = [ideal.to_expr(g) for g in ideal.groebner_basis()]
    factors: list[Expr] = []
    for g in inequations:
        reduced = g
        if ideal is not None:
            remainder = ideal.normal_form(g)
            if not remainder.terms:
                return false
            reduced = ideal.to_expr(remainder)
        elif g == zero:
            return false
        if not reduced.free_symbols:
            continue
        for f in _irreducible_factors(reduced, symbols):
            if ideal is not None:
                if ideal.radical_contains(f):
                    return false
                if _ideal(basis + [f], symbols).is_whole_ring():
                    # the factor does not vanish on the variety
                    continue
            if f not in factors:
                factors.append(f)
    return and_(*[eq(g, zero) for g in basis], *[ne(f, zero) for f in factors])


def reduce_complex(formula: Boolean) -> Boolean:
    """Simplify a quantifier-free Boolean combination of polynomial
    equations and inequations over the complex numbers: in each
    conjunction of its disjunctive normal form the equations are replaced
    by the reduced Gröbner basis of their ideal, and the inequations
    reduced modulo it (an inequation contradicts the equations when its
    polynomial is in the radical of the ideal).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq, Ne
    >>> from sympy_extras.polys.comprehensive import reduce_complex
    >>> x, y = symbols('x y')
    >>> reduce_complex(Eq(x**2 - y, 0) & Eq(x*y - 1, 0) & Ne(x, 0))
    Eq(x*y - 1, 0) & Eq(-x + y**2, 0) & Eq(x**2 - y, 0)
    >>> reduce_complex(Eq(x**2, 0) & Ne(x, 0))
    False
    """
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return formula
    disjuncts = [_reduce_conjunction(equations, inequations)
                 for equations, inequations in _conjunctions(formula)]
    return or_(*disjuncts)
