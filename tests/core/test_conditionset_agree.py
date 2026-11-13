"""``ConditionSet`` of the Rust core against
``sympy_extras.sets.conditionset``: random conditions in the unknown and
a parameter over the random sets of ``test_sets_agree``, nested condition
sets, tuples of unknowns and ``Contains`` conditions, compared by
structure for the evaluation, membership, ``as_relational``, the free
symbols and the operations of the sets on them."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_sets_agree import outcome, random_set


def random_condition(rng: random.Random, x: Any, a: Any) -> Any:
    from sympy_extras.core.logic import and_, false, or_, true
    from sympy_extras.core.relational import eq, gt, le, ne
    from sympy_extras.sets import sets as s
    atoms: list[Any] = [gt(x, a), le(x**2, 2), eq(x, a + 1), ne(x, 0), gt(x, 1), true, false,
                        s.Contains(x, s.interval(0, 1))]
    first = rng.choice(atoms)
    if rng.random() < 0.4:
        return rng.choice([and_, or_])(first, rng.choice(atoms))
    return first


@pytest.mark.parametrize("seed", range(40))
def test_condition_sets(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.sets import conditionset as cs, sets as s
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("conditionset", seed))
    x, y, a = Symbol('x', real=True), Symbol('y', real=True), Symbol('a', real=True)
    for _ in range(6):
        base = random_set(rng, 1)
        if rng.random() < 0.3:
            inner = rng.choice([x, y])
            try:
                base = cs.condition_set(inner, random_condition(rng, inner, a), base)
            except Exception:
                pass
        sym: Any = x if rng.random() < 0.85 else Tuple(x, y)
        condition = random_condition(rng, x, a)
        expected = outcome(lambda: cs.condition_set(sym, condition, base))
        assert outcome(lambda: rc.logic_call('condition_set', [sym, condition, base])) == expected, \
            (sym, condition, base)
        if isinstance(expected, str):
            continue
        c = cs.condition_set(sym, condition, base)
        if isinstance(c, cs.ConditionSet):
            assert sorted(map(str, rc.roundtrip(c).free_symbols)) == sorted(map(str, c.free_symbols))
        e = rng.choice([Symbol('t', real=True), a, x])
        values: list[Any] = [e, Integer(rng.choice([0, 1, 2]))]
        for value in values:
            assert outcome(lambda: rc.logic_call('contains', [value, c])) == outcome(lambda: s.contains(value, c)), \
                (value, c)
        t = Symbol('t', real=True)
        assert outcome(lambda: rc.logic_call('as_relational', [c, t])) == outcome(lambda: c.as_relational(t)), c
        other = random_set(rng, 1)
        for name in ['union', 'intersection']:
            assert outcome(lambda: rc.logic_call(name, [c, other])) == \
                outcome(lambda: getattr(s, name)(c, other)), (name, c, other)
        assert outcome(lambda: rc.logic_call('complement', [other, c])) == \
            outcome(lambda: s.complement(other, c)), (other, c)
