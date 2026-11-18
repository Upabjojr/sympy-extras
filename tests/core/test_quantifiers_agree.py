"""``sympy_extras.assumptions.quantifiers`` against its Rust port: random
quantified formulas (nested, with clashing bound variables) compared on
``forall``/``exists``, the nodes through the bridge, the free symbols,
substitution (of bound and free symbols) and the prenex normal form
(prefix and matrix, the renamed variables as dummies)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def atom(rng: random.Random) -> Any:
    from sympy_extras.core import Eq, symbols
    x, y, z, a = symbols('x y z a')
    return rng.choice([x > 0, x < y, Eq(y, z), z + a > 1, y <= 2, a, x*y > z])


def qformula(rng: random.Random, depth: int) -> Any:
    from sympy_extras.assumptions.quantifiers import exists, forall
    from sympy_extras.core import symbols
    from sympy_extras.core.logic import and_, equivalent, implies, ite, not_, or_, xor
    x, y, z = symbols('x y z')
    if depth == 0 or rng.random() < 0.2:
        return atom(rng)
    op = rng.choice(['forall', 'exists', 'forall', 'exists', 'and', 'or', 'not', 'implies', 'xor', 'equivalent',
                     'ite'])
    if op in ('forall', 'exists'):
        vs = rng.sample([x, y, z], rng.randint(1, 2))
        return (forall if op == 'forall' else exists)(vs, qformula(rng, depth - 1))
    parts = [qformula(rng, depth - 1) for _ in range(3)]
    if op == 'and':
        return and_(parts[0], parts[1])
    if op == 'or':
        return or_(parts[0], parts[1])
    if op == 'not':
        return not_(parts[0])
    if op == 'implies':
        return implies(parts[0], parts[1])
    if op == 'xor':
        return xor(*parts)
    if op == 'equivalent':
        return equivalent(parts[0], parts[1])
    return ite(*parts)


def shown(prefix_matrix: Any) -> str:
    prefix, matrix = prefix_matrix
    return str(([(k, str(v)) for k, v in prefix], str(matrix)))


@pytest.mark.parametrize("seed", range(60))
def test_quantifiers(rust: Any, seed: int) -> None:
    from sympy_extras.assumptions import quantifiers as qf
    from sympy_extras.core import Integer, symbols
    from sympy_extras.core.containers import Tuple
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("quantifiers", seed))
    x, y, z = symbols('x y z')
    for _ in range(5):
        f = qformula(rng, rng.randint(1, 3))
        for kind in ('forall', 'exists'):
            vs = rng.sample([x, y, z], rng.randint(1, 3))
            got = outcome(lambda: rc.logic_call(kind, [Tuple(*vs), f]))
            assert got == outcome(lambda: getattr(qf, kind)(vs, f)), (kind, vs, f)
        assert outcome(lambda: rc.logic_call('forall', [Tuple(x, x), f])) == outcome(lambda: qf.forall([x, x], f))
        assert sorted(map(str, rc.free_symbols(f))) == sorted(map(str, f.free_symbols)), f
        for old, new in ((x, Integer(2)), (y, z + 1), (z, x)):
            assert outcome(lambda: rc.subs(f, old, new)) == outcome(lambda: f.subs(old, new)), (f, old)
        r = rc.logic_call('prenex', [f])
        prefix = [(str(p.args[0]), p.args[1]) for p in r.args[0].args]
        assert shown((prefix, r.args[1])) == shown(qf.prenex(f)), f
