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
  derivatives. The systems without solutions are detected by a Gröbner
  basis of that ideal (:class:`sympy_extras.polys.ideals.Ideal`), and the
  membership is decided by pseudo-division by the regular chains of its
  triangular decomposition (:mod:`sympy_extras.polys.regularchains`,
  :meth:`RegularDifferentialSystem.regular_chains`);
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

The module works on objects of the core and imports no SymPy: the
saturations are ideals of :mod:`sympy_extras.polys.ideals`, the regular
chains those of :mod:`sympy_extras.polys.regularchains` and the factors
those of :func:`sympy_extras.polys.polytools.factor_list`, all native.
:mod:`sympy_extras.polys.differential_sympy` is the face on SymPy
objects.
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.polytools import factor_list
from sympy_extras.polys.regularchains.regularchain import RegularChain, triangularize

from .polynomial import DifferentialPolynomial
from .ring import Coefficient, DifferentialRing, Equation, Jet

__all__ = ['RegularDifferentialSystem', 'RadicalDifferentialIdeal', 'rosenfeld_groebner']

#: the terms of a polynomial in the symbols of the derivatives, with
#: coefficients in the field
_SymbolTerms = dict[tuple[int, ...], Coefficient]


class JetSymbols:
    """Symbols standing for the derivatives in the polynomial rings."""

    def __init__(self, ring: DifferentialRing) -> None:
        self.ring: Final[DifferentialRing] = ring
        self.symbols: Final[dict[Jet, Dummy]] = {}

    def symbol(self, jet: Jet) -> Dummy:
        """The symbol of a derivative."""
        s = self.symbols.get(jet)
        if s is None:
            a, J = jet
            s = Dummy('%s_%s' % (self.ring.functions[a].name, ''.join(str(j) for j in J)))
            self.symbols[jet] = s
        return s

    def expr(self, p: DifferentialPolynomial) -> Expr:
        """The polynomial as an expression in the symbols."""
        parts: list[Expr] = []
        for monomial, c in p.terms.items():
            factors: list[Expr] = [self.ring.integer_expr(c)]
            for jet, e in monomial:
                factors.append(pow(self.symbol(jet), Integer(e)) if e > 1 else self.symbol(jet))
            parts.append(mul(*factors))
        return add(*parts) if parts else Integer(0)

    def polynomial(self, expr: Expr) -> DifferentialPolynomial:
        """The differential polynomial of an expression in the symbols.

        (Not through the expressions of the derivatives: the derivative of
        ``a(x)`` with respect to ``y`` evaluates to zero, and that
        derivative is an element of the ring like any other.)"""
        jets: dict[Basic, Jet] = {s: jet for jet, s in self.symbols.items()}
        return DifferentialPolynomial(self.ring, self.ring.integral(self.ring.terms_of(expr, jets)))

    def terms(self, p: DifferentialPolynomial, jets: Sequence[Jet], shift: int = 0) -> _SymbolTerms:
        """The terms of a polynomial in the derivatives ``jets`` (after
        ``shift`` other variables), with coefficients in the field."""
        position = {jet: shift + k for k, jet in enumerate(jets)}
        result: _SymbolTerms = {}
        for monomial, c in p.terms.items():
            exponents = [0]*(shift + len(jets))
            for jet, e in monomial:
                exponents[position[jet]] = e
            result[tuple(exponents)] = self.ring.field_element(c)
        return result


class RegularDifferentialSystem:
    """A regular differential system `A = 0`, `H \\ne 0`: a component of
    the decomposition computed by :func:`rosenfeld_groebner`.

    It stands for the radical differential ideal `[A] : H^\\infty`, the
    differential polynomials which vanish on the solutions of `A = 0` on
    which no element of `H` vanishes.
    """

    def __init__(self, ring: DifferentialRing, chain: list[DifferentialPolynomial],
                 inequations: list[DifferentialPolynomial]) -> None:
        self.ring: Final[DifferentialRing] = ring
        self.chain: Final[list[DifferentialPolynomial]] = chain
        self.nonzero: Final[list[DifferentialPolynomial]] = inequations
        self.symbols: Final[JetSymbols] = JetSymbols(ring)
        self._basis: Optional[list[Expr]] = None
        self._chains: Optional[list[RegularChain]] = None

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
            K = self.ring.domain
            jets: set[Jet] = set()
            for p in self.chain + self.nonzero:
                jets |= p.jets()
            ordered = sorted(jets, key=self.ring.rank_key, reverse=True)
            symbols: list[Symbol] = [self.symbols.symbol(jet) for jet in ordered]
            if not self.nonzero:
                ideal = Ideal.from_terms([self.symbols.terms(p, ordered) for p in self.chain], symbols, K)
            else:
                product = DifferentialPolynomial(self.ring, {(): self.ring.integers.one})
                for h in self.nonzero:
                    product = product*h
                # 1 - t*product
                last: dict[tuple[int, ...], Coefficient] = {
                    (1,) + m[1:]: K.neg(c) for m, c in self.symbols.terms(product, ordered, 1).items()}
                origin = (0,)*(len(ordered) + 1)
                last[origin] = K.add(last[origin], K.one) if origin in last else K.one
                if K.is_zero(last[origin]):
                    del last[origin]
                t = Dummy('t')
                gens = [self.symbols.terms(p, ordered, 1) for p in self.chain] + [last]
                ideal = Ideal.from_terms(gens, [t] + symbols, K).eliminate([t])
            self._basis = ideal.exprs
        return self._basis

    def _ordered(self, jets: set[Jet]) -> list[Symbol]:
        """The symbols of the derivatives by decreasing rank, then the
        variables and the parameters."""
        ordered: list[Symbol] = [self.symbols.symbol(jet) for jet in sorted(jets, key=self.ring.rank_key, reverse=True)]
        return ordered + list(self.ring.variables) + list(self.ring.parameters)

    def regular_chains(self) -> list[RegularChain]:
        """Squarefree regular chains `C_i`, in symbols standing for the
        derivatives ordered by the ranking, with `(A) : H^\\infty = \\bigcap
        \\operatorname{sat}(C_i)`: the triangular decomposition in the sense
        of Kalkbrener of the zeros of `A` on which `H` does not vanish
        identically.

        The main variables of every chain are the leaders of `A` (the ideal
        is unmixed, the other derivatives being algebraically independent
        modulo each of its primes, by Lazard's lemma), and membership in the
        ideal is decided by pseudo-division by the chains, without a Gröbner
        basis. The variables and the parameters are the smallest variables
        of the polynomial ring; a chain with a condition on them alone is
        dropped, since they are invertible in the field of coefficients.
        """
        if self._chains is None:
            jets: set[Jet] = set()
            for p in self.chain + self.nonzero:
                jets |= p.jets()
            coefficients = set(self.ring.variables) | set(self.ring.parameters)
            chains = triangularize([self.symbols.expr(p) for p in self.chain], *self._ordered(jets),
                inequations=[self.symbols.expr(h) for h in self.nonzero], mode='kalkbrener')
            self._chains = [T for T in chains if not coefficients & set(T.main_variables)]
        return self._chains

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
        # Rosenfeld's lemma: the partial remainder is in (A) : H^oo, which
        # is the intersection of the saturated ideals of the chains
        partial = p.partial_remainder(self.chain)
        jets: set[Jet] = set(partial.jets())
        for q in self.chain + self.nonzero:
            jets |= q.jets()
        target = self.symbols.expr(partial)
        symbols = self._ordered(jets)
        return all(T.with_symbols(*symbols).contains(target) for T in self.regular_chains())

    def __contains__(self, expr: Equation) -> bool:
        return self.contains(expr)


class RadicalDifferentialIdeal:
    """The radical differential ideal of a system, as the intersection of
    the regular differential ideals of its components; see
    :func:`rosenfeld_groebner`."""

    def __init__(self, ring: DifferentialRing, components: list[RegularDifferentialSystem]) -> None:
        self.ring: Final[DifferentialRing] = ring
        self.components: Final[list[RegularDifferentialSystem]] = components

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

    def __init__(self, symbols: JetSymbols) -> None:
        self.symbols: Final[JetSymbols] = symbols
        self._known: Final[dict[DifferentialPolynomial, list[DifferentialPolynomial]]] = {}

    def factors(self, p: DifferentialPolynomial) -> list[DifferentialPolynomial]:
        """The distinct irreducible factors which are not in the field of
        coefficients."""
        known = self._known.get(p)
        if known is not None:
            return known
        result: list[DifferentialPolynomial] = []
        if not p.is_ground:
            found: list[tuple[Expr, Expr]]
            try:
                _, found = factor_list(self.symbols.expr(p))
            except PolynomialError:
                found = []
                result = [p]
            for f, _ in found:
                q = self.symbols.polynomial(f).primitive()
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


def _without_factors(symbols: JetSymbols, a: DifferentialPolynomial,
                     nonzero: Sequence[DifferentialPolynomial]) -> DifferentialPolynomial:
    """The equation without its factors which are free of its leader and
    are known not to vanish. The ideal `[A] : H^\\infty` does not change,
    nor do the leaders and the degrees."""
    leader = a.leader()
    kept: Optional[DifferentialPolynomial] = None
    dropped = False
    expr = symbols.expr(a)
    known: set[Symbol] = set(symbols.symbols.values())
    gens = sorted(expr.free_symbols & known, key=lambda s: s.name)
    try:
        _, factors = factor_list(expr, *gens)
    except PolynomialError:
        return a
    for f, e in factors:
        q = symbols.polynomial(f).primitive()
        if leader not in q.jets() and (q in nonzero or q.is_ground):
            dropped = dropped or not q.is_ground
            continue
        if not isinstance(e, Integer):
            raise TypeError("an integer multiplicity is expected, got %s" % (e,))
        kept = q**e.p if kept is None else kept*q**e.p
    if kept is None or not dropped:
        return a
    return kept.primitive()


def rosenfeld_groebner(equations: Sequence[Equation], functions: Sequence[Basic],
                       variables: Optional[Sequence[Basic]] = None,
                       ranking: Optional[Sequence[Sequence[Basic]]] = None,
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

    >>> from sympy_extras.core import Function, symbols
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
    excluded = list(inequations)
    chosen = list(variables) if variables is not None else None
    probe = DifferentialRing(functions, chosen, ranking)
    found: set[Symbol] = set()
    for equation in given:
        found |= equation.free_symbols
    for excluded_expr in excluded:
        found |= excluded_expr.free_symbols
    parameters = sorted(found - set(probe.variables), key=lambda s: s.name)
    ring = DifferentialRing(functions, chosen, ranking, parameters)
    symbols = JetSymbols(ring)
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
