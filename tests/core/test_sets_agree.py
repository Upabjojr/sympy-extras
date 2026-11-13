"""The sets of the Rust core against ``sympy_extras.sets.sets``: random
intervals (numeric, infinite and symbolic ends, open or closed), finite
sets, ``Reals``, ``EmptySet``, ``UniversalSet`` and their unions,
intersections and complements, compared by structure for the evaluating
functions, membership, subsets, emptiness, finiteness, infimum, supremum
and the condition of membership."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed


def tree(e: Any) -> Any:
    """The structure: the class and the arguments, in order."""
    args = getattr(e, 'args', None)
    if not args:
        return repr(e)
    return (type(e).__name__, tuple(tree(a) for a in args))


def outcome(f: Callable[[], Any]) -> Any:
    try:
        return tree(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def ends() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol, oo, pi, sqrt
    p, r = Symbol('p', positive=True), Symbol('r', real=True)
    return [Integer(0), Integer(1), Integer(-2), Integer(3), Rational(1, 2), sqrt(2), pi, -oo, oo, r, p, r + 1]


def elements() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    return [Integer(0), Integer(1), Integer(-2), Integer(5), Rational(1, 2), sqrt(2), I, Symbol('x'),
            Symbol('r', real=True), Symbol('p', positive=True)]


def random_set(rng: random.Random, depth: int) -> Any:
    from sympy_extras.sets import sets as s
    choice = rng.random()
    if depth == 0 or choice < 0.55:
        kind = rng.random()
        if kind < 0.55:
            try:
                return s.interval(rng.choice(ends()), rng.choice(ends()), rng.random() < 0.4, rng.random() < 0.4)
            except ValueError:
                return s.Reals
        if kind < 0.85:
            return s.finite_set(*rng.sample(elements(), rng.randint(1, 3)))
        return rng.choice([s.Reals, s.EmptySet, s.UniversalSet])
    try:
        operation = rng.choice(['union', 'intersection', 'complement'])
        if operation == 'complement':
            return s.complement(random_set(rng, depth - 1), random_set(rng, depth - 1))
        return getattr(s, operation)(*[random_set(rng, depth - 1) for _ in range(rng.randint(2, 3))])
    except Exception:
        return s.Reals


@pytest.mark.parametrize("seed", range(40))
def test_sets(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.sets import sets as s
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("sets", seed))
    x = Symbol('x', real=True)
    for _ in range(8):
        a, b, c = random_set(rng, 2), random_set(rng, 1), random_set(rng, 1)
        for name, args, expected in [
            ('union', [a, b, c], lambda: s.union(a, b, c)),
            ('intersection', [a, b], lambda: s.intersection(a, b)),
            ('complement', [a, b], lambda: s.complement(a, b)),
        ]:
            assert outcome(lambda: rc.logic_call(name, args)) == outcome(expected), (name, args)
        e = rng.choice(elements() + [x])
        assert outcome(lambda: rc.logic_call('contains', [e, a])) == outcome(lambda: s.contains(e, a)), (e, a)
        assert outcome(lambda: rc.logic_call('as_relational', [a, x])) == \
            outcome(lambda: a.as_relational(x)), a
        assert outcome(lambda: rc.logic_call('inf', [a])) == outcome(lambda: a.inf), a
        assert outcome(lambda: rc.logic_call('sup', [a])) == outcome(lambda: a.sup), a
        assert outcome(lambda: rc.logic_call('is_subset', [a, b])) == outcome(lambda: a.is_subset(b)), (a, b)
        assert outcome(lambda: rc.logic_call('is_empty', [a])) == outcome(lambda: a.is_empty), a
        assert outcome(lambda: rc.logic_call('is_finite_set', [a])) == outcome(lambda: a.is_finite_set), a


def test_singletons(rust: Any) -> None:
    from sympy_extras.core import oo
    from sympy_extras.sets import sets as s
    rc = load("sympy_extras_rs.core")
    for e in [s.Reals, s.EmptySet, s.UniversalSet, s.interval(0, 1, True), s.finite_set(1, 2)]:
        assert tree(rc.roundtrip(e)) == tree(e)
        assert repr(rc.roundtrip(e)) == repr(e)
    assert rc.logic_call('interval', [-oo, oo]) == s.Reals
