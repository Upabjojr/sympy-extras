"""Primary decompositions of random ideals built from primary ideals which
are primary by construction, checked against the construction and by
:func:`primary_oracle.check_decomposition`; ``is_primary``, the independent
sets and the separating linear forms."""
from __future__ import annotations

import random
from itertools import combinations

from sympy import QQ, Expr, I, Matrix, S, Symbol
from sympy.abc import x, y, z
from sympy.testing.pytest import raises

from sympy_extras.polys.ideals import Ideal, separating_forms
from sympy_extras.polys.idealdecomposition import (associated_primes, independent_set, is_primary,
                                                   primary_decomposition)

from .primary_oracle import check_decomposition, intersection

Component = tuple[Ideal, Ideal]


def _frame(rng: random.Random, variables: tuple[Symbol, ...]) -> list[Expr]:
    """Affine forms with an invertible matrix of entries -1, 0, 1 (larger
    ones make intersections whose Gröbner bases have coefficients of
    twenty digits, and minutes of computation)."""
    n = len(variables)
    while True:
        A = Matrix(n, n, lambda i, j: rng.randint(-1, 1))
        if A.det() != 0:
            break
    return [sum((A[i, j]*variables[j] for j in range(n)), S.Zero) + rng.randint(-1, 1) for i in range(n)]


def _random_component(rng: random.Random, L: list[Expr], variables: tuple[Symbol, ...],
                      kinds: tuple[int, ...] = tuple(range(8))) -> Component:
    """A primary ideal and its prime, primary by construction: after the
    affine change of coordinates to ``L``, a monomial ideal whose variables
    all have a pure power in it, or a complete intersection whose radical
    is prime (complete intersections are unmixed)."""
    a, b, c = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 2)
    kind = rng.choice(kinds)
    V = variables
    if kind == 0:
        return Ideal([L[0]**a], *V), Ideal([L[0]], *V)
    if kind == 1:
        return Ideal([L[0], L[1]], *V)**a, Ideal([L[0], L[1]], *V)
    if kind == 2:
        return Ideal([L[0], L[1]], *V)**a + Ideal([L[2]**b], *V), Ideal(L, *V)
    if kind == 3:
        return Ideal([L[0]**a, L[1]**b], *V), Ideal([L[0], L[1]], *V)
    if kind == 4:
        q = L[0] - L[1]**2 if rng.randrange(2) else L[1]**2 - 2*L[2]**2
        return Ideal([q**c], *V), Ideal([q], *V)
    if kind == 5:
        q = L[1]**2 - 2*L[2]**2 if rng.randrange(2) else L[1] - L[2]**2
        return Ideal([L[0]**a, q**c], *V), Ideal([L[0], q], *V)
    if kind == 6:
        return Ideal([L[0]**a, L[1]**b, L[2]**c], *V), Ideal(L, *V)
    return Ideal([(L[0]**2 - 2)**c, L[1]**a, L[2]], *V), Ideal([L[0]**2 - 2, L[1], L[2]], *V)


def _pruned(components: list[Component]) -> list[Component]:
    """A minimal primary decomposition of the intersection: the components
    of one prime merged, then those which contain the intersection of the
    others removed; its primes are the associated primes."""
    merged: list[Component] = []
    for q, p in components:
        same = [i for i, (_, p2) in enumerate(merged) if p2 == p]
        if same:
            merged[same[0]] = (merged[same[0]][0].intersect(q), p)
        else:
            merged.append((q, p))
    kept = list(merged)
    for component in merged:
        others = [o for o in kept if o is not component]
        if others and intersection([o[0] for o in others]).subset(component[0]):
            kept = others
    return kept


def _random_ideals(seed: int, count: int, nested: bool) -> list[tuple[Ideal, list[Component], bool]]:
    """Random intersections (and, one time in four, products) of primary
    ideals: ``nested`` ones, two or three with the same affine frame and
    primes among the ideals of its forms, so that the primes are often
    contained in one another and the components embedded; or two of any
    kind, with frames drawn separately one time in three."""
    rng = random.Random(seed)
    V = (x, y, z)
    cases: list[tuple[Ideal, list[Component], bool]] = []
    for _ in range(count):
        L = _frame(rng, V)
        if nested:
            components = [_random_component(rng, L, V, (0, 1, 2, 3, 6)) for _ in range(rng.randint(2, 3))]
        else:
            components = [_random_component(rng, L if rng.randrange(3) else _frame(rng, V), V) for _ in range(2)]
        product = rng.randrange(4) == 0
        if product:
            ideal = components[0][0]
            for q, _ in components[1:]:
                ideal = ideal*q
        else:
            ideal = intersection([q for q, _ in components])
        cases.append((ideal.reduced(), components, product))
    return cases


def _check_random(ideal: Ideal, components: list[Component], product: bool) -> None:
    found = primary_decomposition(ideal)
    check_decomposition(ideal, found)
    primes = [p for _, p in found]
    if product:
        # the minimal primes of a product are those of its factors
        expected = [p for p in (p for _, p in components)
                    if not any(o.subset(p) and not p.subset(o) for _, o in components)]
        assert all(any(m == e for e in expected) for m in ideal.minimal_primes())
        return
    expected_components = _pruned(components)
    assert len(primes) == len(expected_components)
    assert all(any(p == e for _, e in expected_components) for p in primes)
    # the components of the minimal primes are unique
    for q, p in expected_components:
        if not any(o.subset(p) and not p.subset(o) for _, o in expected_components):
            assert any(q == q2 and p == p2 for q2, p2 in found)


def test_random_intersections_and_products() -> None:
    for ideal, components, product in _random_ideals(11, 8, nested=True):
        _check_random(ideal, components, product)
    for ideal, components, product in _random_ideals(12, 6, nested=False):
        _check_random(ideal, components, product)


def test_is_primary() -> None:
    assert is_primary(Ideal([x**2, y**3], x, y)) is True
    assert is_primary(Ideal([x**2, x*y], x, y)) is False                # an embedded point
    assert is_primary(Ideal([x*y], x, y)) is False                      # two minimal primes
    assert is_primary(Ideal([x**2, y], x, y, z)) is True
    assert is_primary(Ideal([], x, y)) is True and is_primary(Ideal([x, x + 1], x, y)) is False
    # (x, z)**2 is not primary though its radical is prime (Atiyah-Macdonald)
    assert is_primary(Ideal([x**2, x*y, x*z, z**2], x, y, z)) is False
    assert Ideal([x, z**2], x, y, z).is_primary() is True
    # every component found is primary, and the ideals are not
    J = Ideal([x*z, y**2*z, (x - 1)**2*y], x, y, z)
    assert not J.is_primary() and all(q.is_primary() for q, _ in J.primary_decomposition())
    assert associated_primes(J) == J.associated_primes() == [p for _, p in J.primary_decomposition()]


def test_trivial_ideals() -> None:
    assert primary_decomposition(Ideal([x, x + 1], x, y)) == []
    zero = Ideal([], x, y)
    assert primary_decomposition(zero) == [(zero, zero)]
    P = Ideal([x**2 - 2*y**2], x, y)
    assert primary_decomposition(P) == [(P, P)]
    # other fields: the first version raised a SympifyError on the
    # coefficients of the leading coefficients in the contraction
    gaussian = QQ.algebraic_field(I)
    raises(NotImplementedError, lambda: primary_decomposition(Ideal([x**2, x*y], x, y, domain=gaussian)))


def test_independent_sets() -> None:
    assert independent_set(Ideal([x*y, x*z], x, y, z)) == [y, z]
    assert independent_set(Ideal([x - y**2, z], x, y, z)) == [x]
    raises(ValueError, lambda: independent_set(Ideal([S.One], x, y)))
    # the variables returned are independent: the ideal meets the ring in
    # them in zero, and there are as many as the dimension
    for J in (Ideal([x*z - y**2, x**2 - y*z], x, y, z), Ideal([x**2 - y, y*z - 1], x, y, z)):
        u = independent_set(J)
        assert len(u) == J.dimension()
        assert J.eliminate([v for v in J.symbols if v not in u]).is_zero()


def test_separating_forms() -> None:
    # the search has a bound: the height (p*(p - 1)/2 + 1)//2 is reached,
    # and some form of it separates any p points
    # with one variable the first version raised ValueError (the maximum of
    # the empty tuple of other coefficients at the height 1)
    assert list(separating_forms([x], 5)) == [x]
    forms = list(separating_forms([x, y], 4))
    assert len(forms) == 1 + 2*3 and forms[:3] == [x, x + y, x - y]         # heights 0 to 3
    # the five points which defeated the five fixed forms of an earlier
    # version (sympy-extras#11), and every set of 5 points of the 3 x 3 grid: a
    # form of the list takes 5 different values
    five = [(0, 0), (1, -1), (2, -1), (3, -1), (2, 1)]
    grid = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
    for points in [tuple(five)] + list(combinations(grid, 5)):
        assert any(len({f.subs({x: p, y: q}) for p, q in points}) == 5 for f in separating_forms([x, y], 5))
