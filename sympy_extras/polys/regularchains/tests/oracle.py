"""An oracle for regular chains, independent of their algorithms: Gröbner
bases (saturations, ideal quotients and radical membership of
:class:`~sympy_extras.polys.ideals.Ideal`)."""
from __future__ import annotations

from typing import Sequence

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol

from sympy_extras._typing import as_expr
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.regularchains import RegularChain


def saturated(chain: RegularChain) -> Ideal:
    """The saturated ideal of the chain, by a Gröbner basis."""
    return chain.saturated_ideal()


def is_squarefree_regular_chain(chain: RegularChain) -> bool:
    """Distinct main variables, and the initial and the derivative in the
    main variable of every polynomial are not zero divisors modulo the
    saturated ideal of the polynomials below it (which is not the whole
    ring)."""
    polys, variables = chain.polys, chain.main_variables
    if len(set(variables)) != len(variables):
        return False
    below: list[Expr] = []
    initials: list[Expr] = []
    for p, v, a in zip(polys, variables, chain.initials):
        ideal = Ideal(below + [p], *chain.symbols).saturate(as_expr(Mul(*initials, a)))
        if ideal.is_whole_ring():
            return False
        lower = Ideal(below if below else [Integer(0)], *chain.symbols)
        if below:
            lower = lower.saturate(as_expr(Mul(*initials)))
        if not lower.quotient(a) == lower:
            return False
        if not ideal.quotient(as_expr(p.diff(v))) == ideal:
            return False
        below.append(p)
        initials.append(a)
    return True


def lies_in(chains: Sequence[RegularChain], equations: Sequence[Expr]) -> bool:
    """Every equation vanishes on every quasi-component."""
    return all(saturated(chain).contains(f) for chain in chains for f in equations)


def same_radical(chains: Sequence[RegularChain], equations: Sequence[Expr], symbols: Sequence[Symbol]) -> bool:
    """The radical of the ideal of the equations is the intersection of the
    saturated ideals (the decomposition in the sense of Kalkbrener)."""
    system = Ideal(list(equations), *symbols)
    if not chains:
        return system.is_whole_ring()
    if not lies_in(chains, equations):
        return False
    meet = saturated(chains[0])
    for chain in chains[1:]:
        meet = meet.intersect(saturated(chain))
    return all(system.radical_contains(g) for g in meet.exprs)


def covers(chains: Sequence[RegularChain], equations: Sequence[Expr], symbols: Sequence[Symbol],
           inequations: Sequence[Expr] = ()) -> bool:
    """Every zero of the equations where no inequation vanishes is in a
    quasi-component (the decomposition in the sense of Lazard).

    A zero which is in no quasi-component is, for every chain, a zero of
    an initial or not a zero of a polynomial of the chain: every such
    choice must be inconsistent. `p \\ne 0` is written `1 - t p = 0`.
    """
    slack = [Symbol('t_%d' % k) for k in range(len(chains) + len(inequations))]
    start = [as_expr(f) for f in equations]
    for k, h in enumerate(inequations):
        start.append(1 - slack[len(chains) + k]*h)

    def uncovered(k: int, gens: list[Expr]) -> bool:
        if Ideal(gens if gens else [Integer(0)], *symbols, *slack).is_whole_ring():
            return False
        if k == len(chains):
            return True
        chain = chains[k]
        if uncovered(k + 1, gens + [as_expr(Mul(*chain.initials))]):
            return True
        return any(uncovered(k + 1, gens + [1 - slack[k]*p]) for p in chain.polys)

    return not uncovered(0, start)
