"""Sets of sets and power sets of the Rust core against
``sympy_extras.sets.sets`` and ``sympy_extras.sets.powerset``: random
``set_of_sets``, ``PowerSet`` and ``Set.powerset`` compared by structure
and printing, and on the membership of sets and expressions, inclusion,
emptiness, finiteness, the elements of a power set and the operations of
sets (a Rust path not ported yet is skipped)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def base_set(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.sets import sets as s
    x = Symbol('x')
    return rng.choice([s.EmptySet, s.finite_set(1), s.finite_set(1, 2), s.finite_set(2), s.finite_set(x),
                       s.interval(0, 1), s.Reals, s.UniversalSet, s.point_set(Tuple(Integer(1), Integer(2))),
                       s.union(s.finite_set(3), s.interval(5, 6))])


def small(a: Any) -> bool:
    """A set whose power set is small (the subsets grow as 2**n)."""
    from sympy_extras.sets.sets import finite_elements
    elements = finite_elements(a)
    return elements is None or len(elements) <= 3


def some_set(rng: random.Random, depth: int) -> Any:
    from sympy_extras.sets import sets as s
    from sympy_extras.sets.powerset import PowerSet
    if depth == 0:
        return base_set(rng)
    kind = rng.random()
    if kind < 0.4:
        return s.set_of_sets(*[some_set(rng, depth - 1) for _ in range(rng.randint(0, 3))])
    if kind < 0.6:
        return PowerSet(some_set(rng, depth - 1))
    if kind < 0.8:
        inner = some_set(rng, depth - 1)
        return inner.powerset() if small(inner) else inner
    return base_set(rng)


@pytest.mark.parametrize("seed", range(60))
def test_powerset(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.sets import sets as s
    from sympy_extras.sets.powerset import PowerSet
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("powerset", seed))
    for _ in range(4):
        a, b = some_set(rng, rng.randint(1, 2)), some_set(rng, rng.randint(0, 2))
        assert str(rc.roundtrip(a)) == str(a), a
        if small(a):
            assert outcome(lambda: rc.logic_call('powerset', [a])) == outcome(lambda: a.powerset()), a
        assert outcome(lambda: rc.logic_call('power_set', [a])) == outcome(lambda: PowerSet(a)), a
        assert outcome(lambda: rc.logic_call('set_of_sets', [a, b, a])) == outcome(lambda: s.set_of_sets(a, b, a))
        for e in (b, Symbol('x'), s.finite_set(1)):
            got = outcome(lambda: rc.logic_call('contains', [e, a]))
            if got != 'SKIP':
                assert got == outcome(lambda: a.contains(e)), (e, a)
        for name, expected in (('is_subset', lambda: a.is_subset(b)), ('is_empty', lambda: a.is_empty),
                               ('is_finite_set', lambda: a.is_finite_set)):
            args = [a, b] if name == 'is_subset' else [a]
            got = outcome(lambda: rc.logic_call(name, args))
            if got != 'SKIP':
                assert got == outcome(expected), (name, a, b)
        if isinstance(a, PowerSet) and small(a.arg):
            got = outcome(lambda: rc.logic_call('power_set_elements', [a]))
            from sympy_extras.core.containers import Tuple
            assert got == outcome(lambda: Tuple(*a)), a
        for name in ('union', 'intersection'):
            got = outcome(lambda: rc.logic_call(name, [a, b]))
            if got != 'SKIP':
                assert got == outcome(lambda: getattr(s, name)(a, b)), (name, a, b)
