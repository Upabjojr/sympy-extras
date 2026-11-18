"""``sympy_extras.logic.anf`` against its Rust port: random formulas
(symbols and relations, among them pairs with the same or complementary
canonical forms) compared on ``xor_evaluated``, ``and_evaluated``,
``is_anf``, ``to_anf`` (deep or not), ``distribute_xor_over_and``;
random truth tables on ``anf_coeffs``, ``ANFform`` and
``bool_monomial`` (or the errors)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_boolalg_agree import formula
from .test_integers_agree import outcome


def relation(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Symbol
    x = Symbol('x')
    return rng.choice([x > 1, Integer(1) < x, x <= 1, Integer(1) >= x, x >= 2, x < 2])


@pytest.mark.parametrize("seed", range(60))
def test_anf(rust: Any, seed: int) -> None:
    from sympy_extras.core.logic import false, true
    from sympy_extras.logic import anf
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("anf", seed))
    for _ in range(4):
        f = formula(rng, rng.randint(1, 2))
        args = [rng.choice([formula(rng, 1), relation(rng), true, false]) for _ in range(rng.randint(1, 4))]
        for remove_true in (True, False):
            got = outcome(lambda: rc.logic_call('xor_evaluated', [true if remove_true else false, *args]))
            assert got == outcome(lambda: anf.xor_evaluated(*args, remove_true=remove_true)), (args, remove_true)
        assert outcome(lambda: rc.logic_call('and_evaluated', args)) == outcome(lambda: anf.and_evaluated(*args)), args
        assert str(rc.logic_call('is_anf', [f])) == str(anf.is_anf(f)), f
        for deep in (True, False):
            got = outcome(lambda: rc.logic_call('to_anf', [f, true if deep else false]))
            assert got == outcome(lambda: anf.to_anf(f, deep)), (f, deep)
        g = anf.to_anf(f, False)
        assert outcome(lambda: rc.logic_call('distribute_xor_over_and', [g])) == \
            outcome(lambda: anf.distribute_xor_over_and(g)), g


@pytest.mark.parametrize("seed", range(20))
def test_truth_tables(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, symbols
    from sympy_extras.core.containers import Tuple
    from sympy_extras.logic import anf
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("anf tables", seed))
    n = rng.randint(0, 4)
    variables = list(symbols('v0:%d' % n)) if n else []
    size = 2**n if rng.random() < 0.85 else rng.randint(0, 9)
    table = [rng.randint(0, 1) for _ in range(size)]
    as_tuple = lambda ds: Tuple(*map(Integer, ds))
    got = outcome(lambda: rc.logic_call('anf_coeffs', [as_tuple(table)]))
    assert got == outcome(lambda: as_tuple(anf.anf_coeffs(table))), table
    got = outcome(lambda: rc.logic_call('ANFform', [Tuple(*variables), as_tuple(table)]))
    assert got == outcome(lambda: anf.ANFform(variables, table)), table
    k = rng.randint(-1, 2**n)
    got = outcome(lambda: rc.logic_call('bool_monomial', [Integer(k), Tuple(*variables)]))
    assert got == outcome(lambda: anf.bool_monomial(k, variables)), k
    digits = [rng.randint(0, 1) for _ in range(n)]
    got = outcome(lambda: rc.logic_call('bool_monomial', [as_tuple(digits), Tuple(*variables)]))
    assert got == outcome(lambda: anf.bool_monomial(digits, variables)), digits
