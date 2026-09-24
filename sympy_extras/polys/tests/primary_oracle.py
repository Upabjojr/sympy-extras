"""Checks of a primary decomposition which do not trust the algorithm
which computed it: the intersection of the components is the ideal (a
Gröbner basis equality), each component has a prime radical and is not
changed by the saturation by polynomials outside that prime, the primes are
different and no component can be left out, and the minimal ones among them
are the minimal primes of the ideal."""
from __future__ import annotations

from sympy import Add, Expr, Mul

from sympy_extras.polys.ideals import Ideal


def intersection(ideals: list[Ideal]) -> Ideal:
    result = ideals[0]
    for ideal in ideals[1:]:
        result = result.intersect(ideal)
    return result.reduced()


def _probes(prime: Ideal, primes: list[Ideal]) -> list[Expr]:
    """Polynomials outside ``prime``: the generators of the other primes
    and a linear form, those of them which are not in it."""
    variables = prime.symbols
    form = Add(*[(i + 2)*v for i, v in enumerate(variables)]) + 7
    candidates: list[Expr] = [form] + [g for other in primes if other is not prime for g in other.exprs]
    # the product of the generators of an embedded prime not in ``prime``
    # is what a component with a wrong embedded part would be saturated by
    for other in primes:
        outside = [g for g in other.exprs if not prime.contains(g)]
        if outside:
            candidates.append(Mul(*outside))
    return [f for f in candidates if not prime.contains(f)]


def check_decomposition(ideal: Ideal, components: list[tuple[Ideal, Ideal]]) -> None:
    """Assert that ``components`` is an irredundant primary decomposition
    of ``ideal``."""
    if ideal.is_whole_ring():
        assert components == []
        return
    assert components
    primes = [p for _, p in components]
    queries = [q for q, _ in components]
    # the intersection
    assert intersection(queries) == ideal
    # different primes, each prime
    for i, p in enumerate(primes):
        assert not any(p == other for other in primes[:i])
        assert p.is_prime()
    # each component primary with the given prime
    for q, p in components:
        assert q.radical() == p
        for f in _probes(p, primes):
            assert q.saturate(f) == q
    # irredundant
    if len(components) > 1:
        for i in range(len(components)):
            assert intersection(queries[:i] + queries[i + 1:]) != ideal
    # the minimal primes
    minimal = [p for p in primes if not any(o is not p and o.subset(p) for o in primes)]
    expected = ideal.minimal_primes()
    assert len(minimal) == len(expected) and all(any(p == e for e in expected) for p in minimal)
