"""The operations on regular chains behind the triangular decompositions.

Notation. The variables are ordered, `T` is a triangular set (its elements
have distinct main variables), `T_v` its element of main variable `v`,
`T_{<v}` the elements of smaller main variable, `h_T` the product of the
initials, `\\operatorname{sat}(T) = (T) : h_T^\\infty` the *saturated ideal*
and `W(T) = V(T) \\setminus V(h_T)` the *quasi-component*, whose Zariski
closure is `V(\\operatorname{sat}(T))`. `T` is a *regular chain* when the
initial of every `T_v` is regular (not a zero divisor) modulo
`\\operatorname{sat}(T_{<v})`, and it is *squarefree* when moreover every
`T_v` is squarefree modulo `\\operatorname{sat}(T_{<v})`, which makes the
saturated ideal radical. Then `p \\in \\operatorname{sat}(T)` if and only
if the pseudo-remainder of `p` by `T` is zero, and the ring of fractions
of `\\mathbb{Q}[x] / \\operatorname{sat}(T)` is a product of fields: the
algorithms compute in it as in a field, and split the chain whenever
an element turns out to be a zero divisor (the D5 principle of Della Dora,
Dicrescenzo and Duval).

Every chain here is a squarefree regular chain. The operations follow the
specifications of [ChenMorenoMaza]_:

``regularize(p, T)``
    chains `T_i` with `W(T) \\subseteq \\bigcup W(T_i) \\subseteq
    \\overline{W(T)}` such that `p` is zero or regular modulo each
    `\\operatorname{sat}(T_i)`;

``regular_gcd(p, q, v, T)``
    pairs `(g_i, T_i)` with the same covering property, where `g_i` is a
    *regular greatest common divisor* of `p` and `q` modulo
    `\\operatorname{sat}(T_i)`: its leading coefficient is regular, it is
    in the ideal of `p` and `q` and it pseudo-divides both. It is the
    subresultant `S_j` of lowest index whose principal coefficient is
    regular, the ones below being zero: subresultants commute with the
    specializations which keep the two degrees;

``intersect(p, T)``
    chains `T_i` with `V(p) \\cap W(T) \\subseteq \\bigcup W(T_i) \\subseteq
    V(p) \\cap \\overline{W(T)}`;

``extend(T, polys)``
    the chains obtained by putting polynomials of greater main variables on
    top of `T`, where their initials do not vanish identically.

The implementation is recursive from the top of the chain rather than
incremental from the bottom as in [ChenMorenoMaza]_, and relies on two
facts. A chain `T'` of the same height as `T` with
`\\operatorname{sat}(T) \\subseteq \\operatorname{sat}(T')` has its
associated primes among those of `T` (both ideals are unmixed of the same
dimension), so that what is regular modulo `\\operatorname{sat}(T)` stays
regular, and the polynomials above can be put back unchanged. A chain of
greater height (smaller dimension) inherits nothing, and the computation is
*restarted* on it with the original polynomial; this terminates because
the height is bounded by the number of variables.

In the Kalkbrener mode the chains of height greater than the number of
equations are discarded as soon as they appear: every irreducible
component of the solutions has at most that codimension (Krull), so it is
contained in the closure of a quasi-component of a chain which is kept.

References
==========

.. [ChenMorenoMaza] C. Chen, M. Moreno Maza, Algorithms for computing
   triangular decomposition of polynomial systems, Journal of Symbolic
   Computation 47 (2012), 610-642.
.. [ALM] P. Aubry, D. Lazard, M. Moreno Maza, On the theories of
   triangular sets, Journal of Symbolic Computation 28 (1999), 105-124.
.. [Kalkbrener] M. Kalkbrener, A generalized Euclidean algorithm for
   computing triangular representations of algebraic varieties, Journal of
   Symbolic Computation 15 (1993), 143-167.
.. [BLM] F. Boulier, F. Lemaire, M. Moreno Maza, Well known theorems on
   triangular systems and the D5 principle, Transgressive Computing 2006.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from sympy.ntheory.generate import nextprime
from sympy.polys.rings import PolyElement, PolyRing

from .recursive import (coefficient, degree, distinct_factors, exact_quotient, initial, main_variable, normalize,
    primitive_part, pseudo_divide, pseudo_remainder, subresultant_chain, tail)

__all__ = ['Chain', 'Decomposer', 'Regularized', 'RegularGcd']

#: polynomials with a coefficient above this are not factored
_FACTOR_NORM = 10**6


class Chain:
    """A triangular set: at most one polynomial per main variable.

    ``polys[i]`` is the polynomial whose main variable has the index ``i``
    (the first generator of the ring is the greatest variable), or ``None``.
    """

    __slots__ = ('polys', 'height')

    def __init__(self, polys: Sequence[Optional[PolyElement]]) -> None:
        self.polys: tuple[Optional[PolyElement], ...] = tuple(polys)
        self.height: int = sum(1 for p in self.polys if p is not None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Chain) and self.polys == other.polys

    def __hash__(self) -> int:
        return hash(self.polys)

    def __repr__(self) -> str:
        return 'Chain(%s)' % ', '.join(str(p) for p in self.polys if p is not None)

    def get(self, i: int) -> Optional[PolyElement]:
        return self.polys[i]

    def below(self, i: int) -> Chain:
        """The polynomials whose main variable is smaller than the variable ``i``."""
        return Chain([None]*(i + 1) + list(self.polys[i + 1:]))

    def above(self, i: int) -> list[PolyElement]:
        """The polynomials whose main variable is greater than the variable
        ``i``, by increasing main variable."""
        return [p for p in reversed(self.polys[:i]) if p is not None]

    def with_poly(self, i: int, p: PolyElement) -> Chain:
        polys = list(self.polys)
        polys[i] = p
        return Chain(polys)

    def union(self, polys: Iterable[PolyElement]) -> Chain:
        result = list(self.polys)
        for p in polys:
            result[main_variable(p)] = p
        return Chain(result)

    def elements(self) -> list[PolyElement]:
        """The polynomials by increasing main variable."""
        return [p for p in reversed(self.polys) if p is not None]


#: a chain with whether the polynomial is zero (``True``) or regular modulo it
Regularized = tuple[Chain, bool]
#: a regular gcd (``None`` on a chain of greater height), the chain, and the
#: polynomial outside whose zeros the gcd specializes
RegularGcd = tuple[Optional[PolyElement], Chain, PolyElement]


def _distinct(chains: Iterable[Chain]) -> list[Chain]:
    seen: set[Chain] = set()
    result: list[Chain] = []
    for chain in chains:
        if chain not in seen:
            seen.add(chain)
            result.append(chain)
    return result


class Decomposer:
    """The operations on the squarefree regular chains of a polynomial ring
    over the integers, with their caches.

    ``bound`` is the greatest height of the chains which are kept (the
    Kalkbrener mode), ``None`` to keep them all.
    """

    def __init__(self, ring: PolyRing, bound: Optional[int] = None) -> None:
        self.ring: PolyRing = ring
        self.n: int = ring.ngens
        self.bound: Optional[int] = bound
        self.empty: Chain = Chain([None]*self.n)
        self._regularized: dict[tuple[PolyElement, Chain], list[Regularized]] = {}
        self._intersected: dict[tuple[PolyElement, Chain], list[Chain]] = {}
        self._subresultants: dict[tuple[PolyElement, PolyElement, int], list[PolyElement]] = {}
        self._factors: dict[PolyElement, list[PolyElement]] = {}
        #: the irreducible factors met so far, with their values at the point
        self._pool: dict[PolyElement, int] = {}
        self._point: list[int] = [nextprime(100*(i + 1)) for i in range(self.n)]

    # ------------------------------------------------------------------
    # elementary operations

    def kept(self, chain: Chain) -> bool:
        return self.bound is None or chain.height <= self.bound

    def factors(self, p: PolyElement) -> list[PolyElement]:
        """Distinct factors of ``p`` with the same zeros.

        Splitting is only an optimization, and the factorization of the
        iterated resultants met here is slow in SymPy (Hensel lifting and
        heuristic gcds with huge coefficients; Wang's algorithm looks for a
        prime above a coefficient bound which is astronomical, issue #25).
        These resultants are mostly high powers of factors which were met
        before: a polynomial with large coefficients is divided by the known
        irreducible factors, and only a small cofactor is factored.
        """
        found = self._factors.get(p)
        if found is not None:
            return found
        if p.is_ground:
            found = []
        elif p.max_norm() < _FACTOR_NORM:
            found = self._irreducible_factors(p)
        else:
            found = []
            rest = p
            value = self._value(rest)
            for q in list(self._pool):
                at = self._pool[q]
                if not at or value % at:
                    continue
                quotient, remainder = rest.div(q)
                if remainder:
                    continue
                found.append(q)
                while not remainder:
                    rest = quotient.copy()
                    quotient, remainder = rest.div(q)
                value = self._value(rest)
            rest = normalize(rest)
            if not rest.is_ground:
                # the content in the main variable is cheap to split off
                primitive = primitive_part(rest, main_variable(rest))
                parts = [primitive]
                if primitive != rest:
                    parts.append(normalize(exact_quotient(rest, primitive)))
                for part in parts:
                    if part.max_norm() < _FACTOR_NORM or part != rest:
                        found.extend(q for q in self.factors(part) if q not in found)
                    else:
                        found.append(part)
        self._factors[p] = found
        return found

    def _irreducible_factors(self, p: PolyElement) -> list[PolyElement]:
        found = distinct_factors(p)
        for q in found:
            if q not in self._pool:
                self._pool[q] = self._value(q)
        return found

    def _value(self, p: PolyElement) -> int:
        """The value at a fixed integer point: a divisor divides there."""
        value = p.evaluate(list(zip(self.ring.gens, self._point)))
        return int(value)

    def subresultants(self, f: PolyElement, g: PolyElement, i: int) -> list[PolyElement]:
        key = (f, g, i)
        found = self._subresultants.get(key)
        if found is None:
            found = subresultant_chain(f, g, i)
            self._subresultants[key] = found
        return found

    def reduce(self, p: PolyElement, chain: Chain) -> PolyElement:
        """The pseudo-remainder of ``p`` by the chain, normalized: it is zero
        if and only if ``p`` is in the saturated ideal."""
        for i, t in enumerate(chain.polys):
            if not p:
                break
            if t is not None and degree(p, i) >= degree(t, i):
                p = pseudo_remainder(p, t, i)
        return normalize(p)

    def _monomial(self, p: PolyElement, i: int, d: int) -> PolyElement:
        monom = [0]*self.n
        monom[i] = d
        result: PolyElement = p.mul_monom(tuple(monom))
        return result

    # ------------------------------------------------------------------
    # regularize

    def regularize(self, p: PolyElement, chain: Chain) -> list[Regularized]:
        """Split the chain so that ``p`` is zero or regular modulo each part."""
        if p.is_ground:
            return [(chain, not p)]
        key = (p, chain)
        found = self._regularized.get(key)
        if found is not None:
            return found
        r = self.reduce(p, chain)
        result: list[Regularized]
        if not r:
            result = [(chain, True)]
        elif r.is_ground:
            result = [(chain, False)]
        else:
            result = []
            factors = self.factors(r)
            undecided: list[tuple[Chain, Optional[bool]]]
            if len(factors) == 1:
                undecided = self._regularize_reduced(factors[0], chain)
            else:
                # a product is zero where a factor is, regular where all are
                undecided = []
                pending = [chain]
                for f in factors:
                    following: list[Chain] = []
                    for E in pending:
                        for E2, zero in self.regularize(f, E):
                            if E2.height > chain.height:
                                undecided.append((E2, None))
                            elif zero:
                                undecided.append((E2, True))
                            else:
                                following.append(E2)
                    pending = following
                undecided.extend((E, False) for E in pending)
            for E, status in undecided:
                if not self.kept(E):
                    continue
                if status is None:
                    # a smaller dimension: nothing is inherited
                    result.extend(self.regularize(p, E))
                else:
                    result.append((E, status))
        self._regularized[key] = result
        return result

    def _regularize_reduced(self, f: PolyElement, chain: Chain) -> list[tuple[Chain, Optional[bool]]]:
        """``regularize`` for ``f`` reduced with respect to the chain and not
        constant; the chains of greater height come with ``None``."""
        v = main_variable(f)
        t = chain.get(v)
        C = chain.below(v)
        above = chain.above(v)
        top = above if t is None else [t] + above
        out: list[tuple[Chain, Optional[bool]]] = []
        a = initial(f)
        for E, zero in self.regularize(a, C):
            if E.height > C.height:
                out.extend((T2, None) for T2 in self.extend(E, top))
                continue
            if zero:
                T2 = E.union(top)
                for E2, zero2 in self.regularize(tail(f), T2):
                    out.append((E2, zero2 if E2.height == chain.height else None))
                continue
            if t is None:
                # a regular leading coefficient in a free variable
                out.append((E.union(top), False))
                continue
            for g, F, lost in self.regular_gcd(t, f, v, E):
                if g is None:
                    out.extend((T2, None) for T2 in self.extend(F, top))
                    continue
                if not degree(g, v):
                    out.append((F.union(top), False))
                    continue
                _, q, rho = pseudo_divide(t, g, v)
                if self.reduce(rho, F):
                    raise AssertionError("the regular gcd does not divide the polynomial of the chain")
                q = primitive_part(self.reduce(q, F), v)
                out.append((F.with_poly(v, g).union(above), True))
                # the chain is squarefree: f is invertible modulo the cofactor
                out.append((F.with_poly(v, q).union(above), False))
                for F2 in self.intersect(lost, F):
                    out.extend((T2, None) for T2 in self.extend(F2, top))
        return out

    # ------------------------------------------------------------------
    # regular gcd

    def regular_gcd(self, p: PolyElement, q: PolyElement, v: int, chain: Chain) -> list[RegularGcd]:
        """Regular gcds of ``p`` and ``q``, of main variable ``v`` and with
        initials regular modulo the chain, whose variables are smaller."""
        if degree(p, v) < degree(q, v):
            p, q = q, p
        subresultants = self.subresultants(p, q, v)
        zero: PolyElement = self.ring.zero
        out: list[RegularGcd] = []
        pending = [chain]
        for j, S in enumerate(subresultants):
            s = coefficient(S, v, j)
            following: list[Chain] = []
            for F in pending:
                for F2, vanishes in self.regularize(s, F):
                    if F2.height > chain.height:
                        out.append((None, F2, zero))
                    elif vanishes:
                        following.append(F2)
                    else:
                        g = self.reduce(S, F2)
                        if degree(g, v) != j:
                            raise AssertionError("a regular principal subresultant coefficient was reduced to zero")
                        out.append((primitive_part(g, v), F2, coefficient(g, v, j)))
            pending = following
            if not pending:
                break
        for F in pending:
            g = self.reduce(q, F)
            d = degree(g, v)
            if d != degree(q, v):
                raise AssertionError("a regular initial was reduced to zero")
            out.append((primitive_part(g, v), F, coefficient(g, v, d)))
        return out

    # ------------------------------------------------------------------
    # building chains

    def add(self, chain: Chain, f: PolyElement) -> list[Chain]:
        """Put ``f`` on top of the chain, whose variables are smaller than
        its main variable and modulo which its initial is regular: the
        chains with the squarefree part of ``f``."""
        v = main_variable(f)
        g = self.reduce(f, chain)
        d = degree(g, v)
        if d != degree(f, v):
            raise AssertionError("a regular initial was reduced to zero")
        g = primitive_part(g, v)
        if d == 1:
            return [chain.with_poly(v, g)]
        if not chain.height:
            squarefree: PolyElement = self.ring.one
            for part in self.factors(g):
                squarefree = squarefree*(part if part in self._pool else normalize(part.sqf_part()))
            return [chain.with_poly(v, squarefree)]
        out: list[Chain] = []
        for common, F, lost in self.regular_gcd(g, g.diff(v), v, chain):
            if common is None:
                out.extend(self.extend(F, [f]))
                continue
            if not degree(common, v):
                out.append(F.with_poly(v, g))
                continue
            _, q, rho = pseudo_divide(g, common, v)
            if self.reduce(rho, F):
                raise AssertionError("the regular gcd does not divide the polynomial")
            out.append(F.with_poly(v, primitive_part(self.reduce(q, F), v)))
            for F2 in self.intersect(lost, F):
                out.extend(self.extend(F2, [f]))
        return [T for T in out if self.kept(T)]

    def extend(self, chain: Chain, polys: Sequence[PolyElement]) -> list[Chain]:
        """Put the polynomials, of increasing main variables greater than the
        ones of the chain, on top of it; the parts of the chain where an
        initial vanishes identically are dropped."""
        chains = [chain] if self.kept(chain) else []
        for f in polys:
            a = initial(f)
            following: list[Chain] = []
            for A in chains:
                for A2, zero in self.regularize(a, A):
                    if not zero:
                        following.extend(self.add(A2, f))
            chains = _distinct(following)
        return chains

    # ------------------------------------------------------------------
    # intersect

    def intersect(self, p: PolyElement, chain: Chain) -> list[Chain]:
        """The zeros of ``p`` in the quasi-component of the chain."""
        if not p:
            return [chain]
        if p.is_ground or not self.kept(chain):
            return []
        key = (p, chain)
        found = self._intersected.get(key)
        if found is not None:
            return found
        out: list[Chain] = []
        for f in self.factors(p):
            out.extend(self._intersect(f, f, chain))
        result = [T for T in _distinct(out) if self.kept(T)]
        self._intersected[key] = result
        return result

    def _intersect(self, f: PolyElement, fr: PolyElement, chain: Chain) -> list[Chain]:
        """The zeros of ``fr`` in the quasi-component, where a regular
        multiple of ``f`` is a multiple of ``fr`` modulo the saturated ideal;
        ``f`` is in the saturated ideal of every result."""
        out: list[Chain] = []
        for E, zero in self.regularize(fr, chain):
            if E.height > chain.height:
                out.extend(self.intersect(f, E))
            elif zero:
                out.append(E)
            else:
                for h in self.factors(self.reduce(fr, E)):
                    out.extend(self._intersect_regular(f, h, E))
        return out

    def _intersect_regular(self, f: PolyElement, h: PolyElement, chain: Chain) -> list[Chain]:
        """The same for ``h`` reduced, not constant and regular modulo the chain."""
        u = main_variable(h)
        t = chain.get(u)
        C = chain.below(u)
        above = chain.above(u)
        out: list[Chain] = []
        if t is not None:
            # common zeros of h and t lie over the zeros of their resultant
            resultant = self.subresultants(t, h, u)[0]
            if not resultant:
                raise AssertionError("a regular polynomial has a common factor with the chain")
            for E in self.intersect(resultant, C):
                for T2 in self.extend(E, [t] + above):
                    out.extend(self.intersect(f, T2))
            return out
        a = initial(h)
        if not a.is_ground:
            # where the initial vanishes
            rest = tail(h)
            for T2 in self.intersect(a, chain):
                if T2.height > chain.height:
                    out.extend(self.intersect(f, T2))
                else:
                    out.extend(self._intersect(f, rest, T2))
        for E, zero in self.regularize(a, C):
            if zero:
                continue
            if E.height > C.height:
                for R in self.extend(E, [h] + above):
                    out.extend(self.intersect(f, R))
                continue
            for A in self.add(E, h):
                for R in self.extend(A, above):
                    if R.height == chain.height + 1:
                        out.append(R)
                    else:
                        out.extend(self.intersect(f, R))
        return out

    # ------------------------------------------------------------------
    # decompositions

    def is_regular(self, p: PolyElement, chain: Chain) -> bool:
        """Whether ``p`` is regular modulo the saturated ideal of the chain."""
        return all(not zero for E, zero in self.regularize(p, chain) if E.height == chain.height)

    def inside(self, first: Chain, second: Chain, closures: bool = False) -> bool:
        """A sufficient condition for the quasi-component of ``first`` to be
        contained in the one of ``second`` (in its closure with ``closures``):
        ``second`` is in the saturated ideal of ``first`` and its initials
        vanish nowhere on the quasi-component of ``first`` (are regular)."""
        if second.height > first.height:
            return False
        elements = second.elements()
        if any(self.reduce(p, first) for p in elements):
            return False
        for p in elements:
            a = initial(p)
            if a.is_ground:
                continue
            if closures:
                if not self.is_regular(a, first):
                    return False
            elif self.intersect(a, first):
                return False
        return True

    def irredundant(self, chains: Sequence[Chain], closures: bool = False) -> list[Chain]:
        """Remove the chains which :meth:`inside` finds in another one."""
        kept = list(chains)
        for chain in sorted(chains, key=lambda T: -T.height):
            if any(other is not chain and self.inside(chain, other, closures) for other in kept):
                kept = [T for T in kept if T is not chain]
        return kept

    def separate_points(self, chains: Sequence[Chain]) -> list[Chain]:
        """Make the chains without free variables disjoint: the points of a
        chain which are zeros of an earlier one are removed (modulo such a
        chain a regular polynomial vanishes nowhere)."""
        result: list[Chain] = []
        for chain in chains:
            parts = [chain]
            if chain.height == self.n:
                for other in result:
                    if other.height == self.n:
                        parts = [piece for part in parts for piece in self._outside(part, other)]
            result.extend(parts)
        return result

    def _outside(self, chain: Chain, other: Chain) -> list[Chain]:
        out: list[Chain] = []
        pending = [chain]
        for p in other.elements():
            following: list[Chain] = []
            for part in pending:
                for E, zero in self.regularize(p, part):
                    (following if zero else out).append(E)
            pending = following
        return out

    def triangularize(self, equations: Sequence[PolyElement], inequations: Sequence[PolyElement] = ()) -> list[Chain]:
        """Chains whose quasi-components cover the zeros of the equations
        where no inequation vanishes (in the Kalkbrener mode, the closures
        cover them); every inequation is regular modulo every chain."""
        polys = sorted((normalize(p) for p in equations if p),
            key=lambda p: (-main_variable(p), len(p), p.sort_key()))
        chains = [self.empty]
        for p in polys:
            chains = self.irredundant(_distinct(T2 for T in chains for T2 in self.intersect(p, T)))
        for h in inequations:
            chains = _distinct(E for T in chains for E, zero in self.regularize(h, T) if not zero)
        if self.bound is not None:
            chains = self.irredundant(chains, True)
        return self.separate_points(sorted(chains, key=self._order))

    def _order(self, chain: Chain) -> tuple[int, int, list[tuple[int, str]]]:
        """The greatest dimension first, then the greatest degree."""
        elements = chain.elements()
        total = 1
        for p in elements:
            total *= degree(p, main_variable(p))
        return chain.height, -total, [(len(p), str(p)) for p in elements]
