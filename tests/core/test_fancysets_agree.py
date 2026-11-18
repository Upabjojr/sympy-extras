"""``sympy_extras.sets.fancysets`` against its Rust port: the named sets
of numbers on memberships (numbers and symbols with assumptions),
``inf``, ``sup``, emptiness, finiteness, inclusions and the operations
with the other sets; ``image_set`` (identity, constant functions, empty
bases) and the printing of ``ImageSet``."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def element(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol, pi, sqrt
    return rng.choice([Integer(0), Integer(3), Integer(-2), Rational(1, 2), sqrt(2), pi, I, 1 + I,
                       Symbol('x'), Symbol('n', integer=True), Symbol('p', positive=True, integer=True),
                       Symbol('q', rational=True), Symbol('r', real=True), Symbol('k', integer=True, negative=True),
                       Symbol('m', nonnegative=True, integer=True)])


def some_set(rng: random.Random) -> Any:
    from sympy_extras.sets import fancysets as fs
    from sympy_extras.sets import sets as s
    return rng.choice([fs.Naturals, fs.Naturals0, fs.Integers, fs.Rationals, fs.Complexes, s.Reals,
                       s.interval(0, 5), s.finite_set(1, 2), s.EmptySet, s.UniversalSet])


@pytest.mark.parametrize("seed", range(40))
def test_fancysets(rust: Any, seed: int) -> None:
    from sympy_extras.sets import sets as s
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("fancysets", seed))
    for _ in range(6):
        a, b, e = some_set(rng), some_set(rng), element(rng)
        assert str(rc.roundtrip(a)) == str(a)
        assert outcome(lambda: rc.logic_call('contains', [e, a])) == outcome(lambda: s.contains(e, a)), (e, a)
        for name, expected in (('inf', lambda: a.inf), ('sup', lambda: a.sup), ('is_empty', lambda: a.is_empty),
                               ('is_finite_set', lambda: a.is_finite_set), ('is_subset', lambda: a.is_subset(b))):
            args = [a, b] if name == 'is_subset' else [a]
            got = outcome(lambda: rc.logic_call(name, args))
            if got != 'SKIP':
                assert got == outcome(expected), (name, a, b)
        for name in ('union', 'intersection', 'complement'):
            got = outcome(lambda: rc.logic_call(name, [a, b]))
            if got != 'SKIP':
                assert got == outcome(lambda: getattr(s, name)(a, b)), (name, a, b)


def test_image_set(rust: Any) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.sets import fancysets as fs
    from sympy_extras.sets import sets as s
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    cases = [(Tuple(x), x, [fs.Integers]), (Tuple(x), 2*x + 1, [fs.Integers]), (Tuple(x), Integer(3), [fs.Naturals]),
             (Tuple(x), y, [s.EmptySet]), (Tuple(x, y), Tuple(x, y + 1), [fs.Integers, s.Reals]),
             (Tuple(x), Tuple(Integer(1), Integer(2)), [s.interval(0, 1)]), (Tuple(x), x**2, [s.finite_set(1, 2)]),
             (Tuple(x, y), x, [fs.Integers])]
    for variables, image, bases in cases:
        got = outcome(lambda: rc.logic_call('image_set', [variables, image, *bases]))
        assert got == outcome(lambda: fs.image_set(variables, image, *bases)), (variables, image)
        r = outcome(lambda: str(rc.logic_call('image_set', [variables, image, *bases])))
        assert r == outcome(lambda: str(fs.image_set(variables, image, *bases)))
