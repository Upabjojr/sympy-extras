"""``sympy_extras.logic.boolalg`` against its Rust port: random formulas
of symbols and relations under all the connectives, compared on
``simplify_logic`` (every form, with and without ``dontcare`` and
``force``), the normal forms and their predicates; ``SOPform`` and
``POSform`` of random truth tables."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def atom(rng: random.Random) -> Any:
    from sympy_extras.core import Eq, Integer, Symbol, symbols
    a, b, c, d = symbols('a b c d')
    x, y = Symbol('x'), Symbol('y')
    return rng.choice([a, b, c, d, a, b, x > 0, x <= 0, Eq(y, 1), y > 2, x + 1 > 0, Integer(1) < x])


def formula(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core.logic import and_, equivalent, implies, ite, not_, or_, xor
    if depth == 0 or rng.random() < 0.25:
        return atom(rng)
    k = rng.randint(2, 3)
    parts = [formula(rng, depth - 1) for _ in range(k)]
    op = rng.choice(['and', 'or', 'or', 'and', 'not', 'xor', 'implies', 'equivalent', 'ite'])
    if op == 'and':
        return and_(*parts)
    if op == 'or':
        return or_(*parts)
    if op == 'not':
        return not_(parts[0])
    if op == 'xor':
        return xor(*parts)
    if op == 'implies':
        return implies(parts[0], parts[1])
    if op == 'equivalent':
        return equivalent(*parts)
    return ite(parts[0], parts[1], parts[-1])


@pytest.mark.parametrize("seed", range(60))
def test_boolalg(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.logic import false, true
    from sympy_extras.logic import boolalg as ba
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("boolalg", seed))
    for _ in range(4):
        f = formula(rng, rng.randint(1, 3))
        for form in (None, 'cnf', 'dnf'):
            dontcare = formula(rng, 1) if rng.random() < 0.3 else None
            force = rng.random() < 0.3
            got = outcome(lambda: rc.logic_call('simplify_logic', [f, Symbol(form or 'none'), true if force else false,
                                                                 Integer(0) if dontcare is None else dontcare]))
            assert got == outcome(lambda: ba.simplify_logic(f, form, force, dontcare)), (f, form, force, dontcare)
        for name in ('is_literal', 'is_cnf', 'is_dnf'):
            assert str(rc.logic_call(name, [f])) == str(getattr(ba, name)(f)), (name, f)
        for flag in (True, False):
            b = true if flag else false
            assert str(rc.logic_call('is_nnf', [f, b])) == str(ba.is_nnf(f, flag)), f
            assert outcome(lambda: rc.logic_call('to_nnf', [f, b])) == outcome(lambda: ba.to_nnf(f, flag)), f
        assert outcome(lambda: rc.logic_call('eliminate_implications', [f])) == \
            outcome(lambda: ba.eliminate_implications(f)), f
        # the distribution is exponential (in both): only for small formulas
        small = len(str(ba.eliminate_implications(f))) <= 60
        for name in ('distribute_and_over_or', 'distribute_or_over_and'):
            if small:
                assert outcome(lambda: rc.logic_call(name, [f])) == outcome(lambda: getattr(ba, name)(f)), (name, f)
        for name in ('to_cnf', 'to_dnf'):
            for simplify in ((False, True) if small else (True,)):
                args = [f, true if simplify else false, false]
                assert outcome(lambda: rc.logic_call(name, args)) == \
                    outcome(lambda: getattr(ba, name)(f, simplify)), (name, f, simplify)


@pytest.mark.parametrize("seed", range(30))
def test_sop_pos(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, symbols
    from sympy_extras.core.containers import Tuple
    from sympy_extras.logic import boolalg as ba
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("sop pos", seed))
    n = rng.randint(1, 5)
    variables = list(symbols('v0:%d' % n))
    rows = [[(k >> (n - 1 - i)) & 1 for i in range(n)] for k in range(2**n)]
    rng.shuffle(rows)
    m = rng.randint(0, len(rows))
    minterms, rest = rows[:m], rows[m:]
    dontcares = rest[:rng.randint(0, len(rest))]
    as_tuple = lambda ts: Tuple(*[Tuple(*[Integer(v) for v in t]) for t in ts])
    for name in ('SOPform', 'POSform'):
        got = outcome(lambda: rc.logic_call(name, [Tuple(*variables), as_tuple(minterms), as_tuple(dontcares)]))
        assert got == outcome(lambda: getattr(ba, name)(variables, minterms, dontcares)), (name, minterms, dontcares)
