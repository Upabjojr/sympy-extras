"""The Rosenfeld-Gröbner algorithm: the radical of a differential ideal
as an intersection of regular differential ideals.

Let `P = 0`, `S \\ne 0` be a system of polynomial differential equations
and inequations (ordinary or partial) with coefficients rational in the
independent variables. The differential polynomials which vanish on all
its solutions form the radical differential ideal `\\{P\\} : S^\\infty`
(the differential Nullstellensatz). The algorithm of Boulier, Lazard,
Ollivier and Petitot [BLOP1995]_, [BLOP2009]_ writes it as

.. math:: \\{P\\} : S^\\infty = \\bigcap_k\\ [A_k] : H_k^\\infty

where each `(A_k, H_k)` is a *regular differential system* for the
ranking: `A_k` is an autoreduced set (every element is reduced with
respect to the others) which is *coherent* (the `\\Delta`-polynomials
`s_b\\,\\theta_a a - s_a\\,\\theta_b b` of the pairs with leaders
`\\theta_a v_a = \\theta_b v_b` reduce to zero: the integrability
conditions hold), and `H_k` contains the initials and the separants of
`A_k`. Then

* **Rosenfeld's lemma**: a differential polynomial `p` lies in
  `[A] : H^\\infty` exactly when its partial remainder by `A` lies in the
  ideal `(A) : H^\\infty` of a polynomial ring in finitely many
  derivatives. That ideal is computed here by a Gröbner basis
  (:class:`sympy_extras.polys.ideals.Ideal`), which decides the membership
  and detects the systems without solutions;
* **Lazard's lemma**: `(A) : H^\\infty`, hence `[A] : H^\\infty`, is
  radical.

The decomposition is obtained by Ritt's process with splittings. For a
system `(P, S)`, let `A` be an autoreduced subset of `P` of lowest rank
and `R` the nonzero remainders by `A` of the other elements of `P` and of
the `\\Delta`-polynomials of `A`. The solutions on which no initial and no
separant of `A` vanishes are those of `(A \\cup R,\\ S \\cup H_A)`, a system
of lower rank, or a regular system when `R` is empty; the other solutions
satisfy `(P \\cup \\{h\\}, S)` for an initial or separant `h` of `A`, again
of lower rank. The ranks are well-ordered, so the process ends.

What this gives:

* the membership in the radical differential ideal
  (:meth:`RadicalDifferentialIdeal.contains`): whether an equation follows
  from a system; whether a system is inconsistent;
* with an elimination ranking, the equations of each component in the
  lower functions only: the elimination of unknowns (input-output
  equations), the hidden constraints of differential-algebraic systems;
* the separation of the general solution from the singular ones (the
  components where a separant vanishes).

The components may be redundant, and their equations are regular
differential systems rather than characteristic sets of prime components:
no factorization is attempted.

References
==========

.. [BLOP1995] F. Boulier, D. Lazard, F. Ollivier, M. Petitot,
   Representation for the radical of a finitely generated differential
   ideal, ISSAC 1995, 158-166.
.. [BLOP2009] F. Boulier, D. Lazard, F. Ollivier, M. Petitot, Computing
   representations for radicals of finitely generated differential ideals,
   Appl. Algebra Engrg. Comm. Comput. 20 (2009), 73-121.
.. [Hubert] E. Hubert, Notes on triangular sets and
   triangulation-decomposition algorithms II: differential systems,
   Lecture Notes in Computer Science 2630, Springer 2003, 40-87.
.. [Rosenfeld] A. Rosenfeld, Specializations in differential algebra,
   Trans. Amer. Math. Soc. 90 (1959), 394-407.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, factor_list

from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.polys.ideals import Ideal

from .polynomial import DifferentialPolynomial
from .ring import DifferentialRing, Equation, Jet, Terms

__all__ = ['RegularDifferentialSystem', 'RadicalDifferentialIdeal', 'rosenfeld_groebner']


class _JetSymbols:
    """Symbols standing for the derivatives in the polynomial rings."""

    def __init__(self, ring: DifferentialRing) -> None:
        self.ring: DifferentialRing = ring
        self.symbols: dict[Jet, Symbol] = {}

    def symbol(self, jet: Jet) -> Symbol:
        s = self.symbols.get(jet)
        if s is None:
            a, J = jet
            s = Dummy('%s_%s' % (self.ring.functions[a].func.__name__, ''.join(str(j) for j in J)))
            self.symbols[jet] = s
        return s

    def expr(self, p: DifferentialPolynomial) -> Expr:
        parts: list[Expr] = []
        for monomial, c in p.terms.items():
            term = as_expr(self.ring.integers.to_sympy(c))
            for jet, e in monomial:
                term = term*self.symbol(jet)**e
            parts.append(term)
        return Add(*parts)

    def polynomial(self, expr: Expr) -> DifferentialPolynomial:
        """The differential polynomial of an expression in the symbols.

        (Not through the expressions of the derivatives: SymPy evaluates
        the derivative of ``a(x)`` with respect to ``y`` to zero, and that
        derivative is an element of the ring like any other.)"""
        ring = self.ring
        pairs = [(jet, s) for jet, s in self.symbols.items() if expr.has(s)]
        if not pairs:
            c = ring.coefficient(expr)
            return DifferentialPolynomial(ring, ring.integral({(): c} if c else {}))
        poly = Poly(expr, *[s for _, s in pairs], domain=ring.domain)
        terms: Terms = {}
        for exponents, c in poly.as_dict(native=True).items():
            monomial = tuple(sorted((jet, e) for (jet, _), e in zip(pairs, exponents) if e))
            terms[monomial] = c
        return DifferentialPolynomial(ring, ring.integral(terms))


class RegularDifferentialSystem:
    """A regular differential system `A = 0`, `H \\ne 0`: a component of
    the decomposition computed by :func:`rosenfeld_groebner`.

    It stands for the radical differential ideal `[A] : H^\\infty`, the
    differential polynomials which vanish on the solutions of `A = 0` on
    which no element of `H` vanishes.
    """

    def __init__(self, ring: DifferentialRing, chain: list[DifferentialPolynomial],
                 inequations: list[DifferentialPolynomial]) -> None:
        self.ring: DifferentialRing = ring
        self.chain: list[DifferentialPolynomial] = chain
        self.nonzero: list[DifferentialPolynomial] = inequations
        self._symbols: _JetSymbols = _JetSymbols(ring)
        self._basis: Optional[list[Expr]] = None

    def __repr__(self) -> str:
        return "RegularDifferentialSystem(%s, %s)" % (self.equations, self.inequations)

    @property
    def equations(self) -> list[Expr]:
        """The expressions which vanish (a coherent autoreduced set), by
        increasing leader."""
        hidden = set(self.ring.constancy())
        return [p.to_expr() for p in self.chain
                if not any(self.ring.is_derivative_of(p.leader(), jet) for jet in hidden)]

    @property
    def inequations(self) -> list[Expr]:
        """The expressions which do not vanish."""
        return [p.to_expr() for p in self.nonzero]

    @property
    def leaders(self) -> list[Expr]:
        """The leaders of the equations."""
        return [p.leader_expr() for p in self.chain]

    def saturation(self) -> list[Expr]:
        """A Gröbner basis of the ideal `(A) : H^\\infty` in the polynomial
        ring of the derivatives which occur in the system, in symbols
        standing for the derivatives."""
        if self._basis is None:
            gens = [self._symbols.expr(p) for p in self.chain]
            product = Mul(*[self._symbols.expr(h) for h in self.nonzero])
            jets: set[Jet] = set()
            for p in self.chain + self.nonzero:
                jets |= p.jets()
            ordered = [self._symbols.symbol(jet) for jet in sorted(jets, key=self.ring.rank_key, reverse=True)]
            if not self.nonzero:
                ideal = Ideal(gens, *ordered, domain=self.ring.domain)
            else:
                t = Dummy('t')
                ideal = Ideal(gens + [1 - t*product], t, *ordered, domain=self.ring.domain).eliminate([t])
            self._basis = ideal.exprs
        return self._basis

    @property
    def is_consistent(self) -> bool:
        """Whether the system has solutions: `1 \\notin (A) : H^\\infty`."""
        return not any(g.is_number and g != 0 for g in self.saturation())

    def remainder(self, expr: Equation) -> Expr:
        """Ritt's remainder of a differential polynomial by the
        equations: a product of the polynomial with initials and
        separants, minus an element of `[A]`, reduced with respect to
        `A`."""
        return DifferentialPolynomial.from_expr(self.ring, expr).remainder(self.chain).to_expr()

    def contains(self, expr: Equation) -> bool:
        """Whether the differential polynomial lies in `[A] : H^\\infty`:
        it vanishes on every solution of the system."""
        p = DifferentialPolynomial.from_expr(self.ring, expr)
        if not p.remainder(self.chain):
            return True
        partial = p.partial_remainder(self.chain)
        basis = self.saturation()
        jets: set[Jet] = set(partial.jets())
        for q in self.chain + self.nonzero:
            jets |= q.jets()
        ordered = [self._symbols.symbol(jet) for jet in sorted(jets, key=self.ring.rank_key, reverse=True)]
        return Ideal(basis, *ordered, domain=self.ring.domain).contains(self._symbols.expr(partial))

    def __contains__(self, expr: Equation) -> bool:
        return self.contains(expr)


class RadicalDifferentialIdeal:
    """The radical differential ideal of a system, as the intersection of
    the regular differential ideals of its components; see
    :func:`rosenfeld_groebner`."""

    def __init__(self, ring: DifferentialRing, components: list[RegularDifferentialSystem]) -> None:
        self.ring: DifferentialRing = ring
        self.components: list[RegularDifferentialSystem] = components

    def __repr__(self) -> str:
        return "RadicalDifferentialIdeal(%s)" % (self.components,)

    def __len__(self) -> int:
        return len(self.components)

    @property
    def is_consistent(self) -> bool:
        """Whether the system has a solution (in some differential
        extension field)."""
        return bool(self.components)

    def contains(self, expr: Equation) -> bool:
        """Whether the differential polynomial vanishes on every solution
        of the system."""
        return all(component.contains(expr) for component in self.components)

    def __contains__(self, expr: Equation) -> bool:
        return self.contains(expr)


def _characteristic_set(polynomials: Sequence[DifferentialPolynomial]) -> list[DifferentialPolynomial]:
    """An autoreduced subset of lowest rank."""
    chain: list[DifferentialPolynomial] = []
    for p in sorted(polynomials, key=lambda p: (p.rank(), len(p.terms))):
        if all(p.is_reduced(a) for a in chain):
            chain.append(p)
    return chain


def _delta_polynomials(chain: Sequence[DifferentialPolynomial]) -> list[DifferentialPolynomial]:
    """The cross derivatives of the pairs with leaders which are
    derivatives of the same function."""
    result: list[DifferentialPolynomial] = []
    for k, a in enumerate(chain):
        for b in chain[k + 1:]:
            v, w = a.leader(), b.leader()
            if v[0] != w[0]:
                continue
            common = tuple(max(i, j) for i, j in zip(v[1], w[1]))
            first = a.prolonged(tuple(c - i for c, i in zip(common, v[1])))
            second = b.prolonged(tuple(c - j for c, j in zip(common, w[1])))
            result.append(b.separant()*first - a.separant()*second)
    return result


def _distinct(polynomials: Sequence[DifferentialPolynomial]) -> list[DifferentialPolynomial]:
    result: list[DifferentialPolynomial] = []
    for p in polynomials:
        if p and p not in result:
            result.append(p)
    return result


class _Factorizer:
    """The irreducible factors of differential polynomials, remembered."""

    def __init__(self, symbols: _JetSymbols) -> None:
        self.symbols: _JetSymbols = symbols
        self._known: dict[DifferentialPolynomial, list[DifferentialPolynomial]] = {}

    def factors(self, p: DifferentialPolynomial) -> list[DifferentialPolynomial]:
        """The distinct irreducible factors which are not in the field of
        coefficients."""
        known = self._known.get(p)
        if known is not None:
            return known
        result: list[DifferentialPolynomial] = []
        if not p.is_ground:
            try:
                _, found = factor_list(self.symbols.expr(p))
            except PolynomialError:
                found = []
                result = [p]
            for f, _ in found:
                q = self.symbols.polynomial(as_expr(f)).primitive()
                if not q.is_ground and q not in result:
                    result.append(q)
        self._known[p] = result
        for q in result:
            self._known.setdefault(q, [q])
        return result

    def nonzero(self, polynomials: Sequence[DifferentialPolynomial]) -> list[DifferentialPolynomial]:
        """The factors of the polynomials: none of the polynomials
        vanishes exactly when none of the factors does."""
        result: list[DifferentialPolynomial] = []
        for h in polynomials:
            for f in self.factors(h):
                if f not in result:
                    result.append(f)
        return result


def _without_factors(symbols: _JetSymbols, a: DifferentialPolynomial,
                     nonzero: Sequence[DifferentialPolynomial]) -> DifferentialPolynomial:
    """The equation without its factors which are free of its leader and
    are known not to vanish. The ideal `[A] : H^\\infty` does not change,
    nor do the leaders and the degrees."""
    leader = a.leader()
    kept: Optional[DifferentialPolynomial] = None
    dropped = False
    expr = symbols.expr(a)
    gens = sorted(free_symbols(expr) & set(symbols.symbols.values()), key=lambda s: s.name)
    try:
        _, factors = factor_list(expr, *gens)
    except PolynomialError:
        return a
    for f, e in factors:
        q = symbols.polynomial(as_expr(f)).primitive()
        if leader not in q.jets() and (q in nonzero or q.is_ground):
            dropped = dropped or not q.is_ground
            continue
        kept = q**e if kept is None else kept*q**e
    if kept is None or not dropped:
        return a
    return kept.primitive()


def rosenfeld_groebner(equations: Sequence[Equation], functions: Sequence[AppliedUndef],
                       variables: Optional[Sequence[Symbol]] = None,
                       ranking: Optional[Sequence[Sequence[AppliedUndef]]] = None,
                       inequations: Sequence[Expr] = ()) -> RadicalDifferentialIdeal:
    """The radical of the differential ideal of a polynomial system of
    differential equations, as an intersection of regular differential
    ideals.

    Parameters
    ==========

    equations : sequence of Expr or Eq
        Expressions which vanish, or equalities, polynomial in the
        functions and their derivatives with coefficients rational in the
        variables. The symbols which are not variables are constant
        parameters, taken to be generic.
    functions : sequence of applied functions
        The unknown functions, such as ``u(x, y)``.
    variables : sequence of Symbol, optional
        The independent variables (the arguments of the functions by
        default).
    ranking : sequence of sequences of functions, optional
        The blocks of an elimination ranking, the functions to eliminate
        first; an orderly ranking by default. See
        :class:`~sympy_extras.polys.differential.ring.DifferentialRing`.
    inequations : sequence of Expr
        Expressions which do not vanish.

    Returns
    =======

    RadicalDifferentialIdeal
        Its ``components`` are the regular differential systems; there is
        none when the system has no solution.

    Examples
    ========

    The equation `u'^2 = 4u` has the general solution `(x + c)^2` and the
    singular solution `0`, on which the separant `2u'` vanishes:

    >>> from sympy import Function, symbols
    >>> from sympy_extras.polys.differential import rosenfeld_groebner
    >>> x = symbols('x')
    >>> u = Function('u')(x)
    >>> I = rosenfeld_groebner([u.diff(x)**2 - 4*u], [u])
    >>> [(c.equations, c.inequations) for c in I.components]
    [([-4*u(x) + Derivative(u(x), x)**2], [Derivative(u(x), x)]), ([u(x)], [])]
    >>> I.contains(u.diff(x)*(u.diff(x, 2) - 2))
    True
    >>> I.contains(u.diff(x, 2) - 2)
    False
    >>> I.components[0].contains(u.diff(x, 2) - 2)
    True

    A system without solutions:

    >>> rosenfeld_groebner([u.diff(x) - u, u.diff(x, 2) - u - 1], [u]).is_consistent
    False
    """
    given = list(equations)
    excluded = [as_expr(h) for h in inequations]
    chosen = list(variables) if variables is not None else None
    probe = DifferentialRing(functions, chosen, ranking)
    found: set[Symbol] = set()
    for equation in given:
        found |= free_symbols(equation)
    for excluded_expr in excluded:
        found |= free_symbols(excluded_expr)
    parameters = sorted(found - set(probe.variables), key=lambda s: s.name)
    ring = DifferentialRing(functions, chosen, ranking, parameters)
    symbols = _JetSymbols(ring)
    one = ring.integers.one
    start = [DifferentialPolynomial.from_expr(ring, e).primitive() for e in given]
    start += [DifferentialPolynomial(ring, {((jet, 1),): one}) for jet in ring.constancy()]
    avoid = [DifferentialPolynomial.from_expr(ring, e).primitive() for e in excluded]
    if any(not q for q in avoid):
        return RadicalDifferentialIdeal(ring, [])
    factorizer = _Factorizer(symbols)
    todo: list[tuple[list[DifferentialPolynomial], list[DifferentialPolynomial]]] = [
        (_distinct(start), factorizer.nonzero(avoid))]
    components: list[RegularDifferentialSystem] = []
    while todo:
        P, H = todo.pop()
        if any(p.is_ground for p in P):
            continue
        # an equation is replaced by its factors which may vanish, one at a time
        split = False
        for k, p in enumerate(P):
            factors = [f for f in factorizer.factors(p) if f not in H]
            if factors == [p]:
                continue
            before: list[DifferentialPolynomial] = []
            for f in factors:
                todo.append((_distinct(P[:k] + [f] + P[k + 1:]), _distinct(H + before)))
                before.append(f)
            split = True
            break
        if split:
            continue
        chain = _characteristic_set(P)
        others = [p for p in P if p not in chain]
        R = _distinct([p.remainder(chain) for p in others + _delta_polynomials(chain)])
        splitting = factorizer.nonzero([h for a in chain for h in (a.initial(), a.separant())])
        # the solutions on which an initial or a separant vanishes
        before = []
        for h in splitting:
            if h not in H:
                todo.append((_distinct(P + [h]), _distinct(H + before)))
                before.append(h)
        # the solutions on which none does
        if any(not h.remainder(chain) for h in H):
            continue
        if R:
            if not any(r.is_ground for r in R):
                todo.append((_distinct(chain + R), _distinct(H + splitting)))
            continue
        nonzero = [h for h in factorizer.nonzero([h.partial_remainder(chain) for h in H + splitting])
                   if not h.remainder(chain).is_ground]
        component = RegularDifferentialSystem(
            ring, sorted([_without_factors(symbols, a, nonzero) for a in chain], key=lambda p: p.rank()), nonzero)
        if component.is_consistent:
            components.append(component)
    components.sort(key=lambda c: (-len(c.chain), [p.rank() for p in c.chain]), reverse=True)
    return RadicalDifferentialIdeal(ring, components)
