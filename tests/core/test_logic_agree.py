"""The boolean and relational nodes of the Rust core against
``sympy_extras.core.logic`` and ``sympy_extras.core.relational``: the
evaluating functions (``and_``, ``or_``, ``not_``, ``xor``, ``implies``,
``equivalent``, ``ite``, ``Nand``, ``Nor``, ``Xnor``, ``eq``, ``ne``,
``lt``, ``le``, ``gt``, ``ge``) on random formulas over symbols and
relations, the transformations of relations (``reversed``, ``negated``,
``reversedsign``, ``canonical``), the printing, the canonical order, the
equality of lattice nodes as sets, and the round trip through the
bridge."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed


def outcome(f: Callable[[], Any]) -> str:
    try:
        return repr(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def logic() -> Any:
    import importlib
    return importlib.import_module("sympy_extras.core.logic")


def relational() -> Any:
    import importlib
    return importlib.import_module("sympy_extras.core.relational")


def random_expression(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, sqrt
    x, p, n = Symbol('x'), Symbol('p', positive=True), Symbol('n', negative=True)
    r = Symbol('r', real=True)
    return rng.choice([x, p, n, r, x + 1, 2*x, p + 1, -p, n - 1, r**2, Integer(rng.randint(-3, 3)), Rational(1, 2),
                       oo, -oo, pi, sqrt(2), pi - 3, I, x - x, p*n])


def random_formula(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Symbol
    from sympy_extras.core.logic import false, true
    lg, rl = logic(), relational()
    a, b = Symbol('a'), Symbol('b')
    if depth == 0:
        c = rng.random()
        if c < 0.3:
            return rng.choice([a, b, true, false])
        op = rng.choice([rl.Eq, rl.Ne, rl.Lt, rl.Le, rl.Gt, rl.Ge])
        return op(random_expression(rng), random_expression(rng))
    c = rng.random()
    if c < 0.25:
        return lg.and_(*[random_formula(rng, depth - 1) for _ in range(rng.randint(1, 3))])
    if c < 0.5:
        return lg.or_(*[random_formula(rng, depth - 1) for _ in range(rng.randint(1, 3))])
    if c < 0.65:
        return lg.not_(random_formula(rng, depth - 1))
    if c < 0.75:
        return lg.xor(random_formula(rng, depth - 1), random_formula(rng, depth - 1))
    if c < 0.85:
        return lg.implies(random_formula(rng, depth - 1), random_formula(rng, depth - 1))
    if c < 0.93:
        return lg.equivalent(random_formula(rng, depth - 1), random_formula(rng, depth - 1))
    return lg.ite(random_formula(rng, depth - 1), random_formula(rng, depth - 1), random_formula(rng, depth - 1))


@pytest.mark.parametrize("seed", range(40))
def test_logic(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("logic", seed))
    lg = logic()
    compared = 0
    for _ in range(10):
        formulas = [random_formula(rng, rng.randint(0, 2)) for _ in range(3)]
        name = rng.choice(["and_", "or_", "not_", "xor", "implies", "equivalent", "ite", "Nand", "Nor", "Xnor"])
        args = formulas[:1] if name == "not_" else formulas[:2] if name == "implies" else formulas
        got = outcome(lambda: rc.logic_call(name, args))
        expected = outcome(lambda: getattr(lg, name)(*args))
        assert got == expected, (name, args)
        assert repr(rc.roundtrip(formulas[0])) == repr(formulas[0])
        compared += 1
    assert compared >= 10


@pytest.mark.parametrize("seed", range(30))
def test_relations(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("relations", seed))
    rl = relational()
    for _ in range(12):
        a, b = random_expression(rng), random_expression(rng)
        for name in ("eq", "ne", "lt", "le", "gt", "ge"):
            assert outcome(lambda: rc.logic_call(name, [a, b])) == outcome(lambda: getattr(rl, name)(a, b)), (name, a, b)
        r = getattr(rl, rng.choice(["Eq", "Ne", "Lt", "Le", "Gt", "Ge"]))(a, b)
        for name in ("reversed", "negated", "reversedsign", "canonical"):
            assert outcome(lambda: rc.logic_call(name, [r])) == outcome(lambda: getattr(r, name)), (name, r)


def test_order_and_equality(rust: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.logic import And, Or, false, true
    from sympy_extras.core.relational import Eq, Lt
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    items = [Lt(x, 1), Eq(y, 2), x, true, false, And(x, y), Or(Lt(x, 1), y), ~x, x + 1]
    assert [repr(e) for e in rc.sorted_by_key(items)] == [repr(e) for e in sorted(items, key=lambda e: e.sort_key())]
    assert rc.equal(And(y, x), And(x, y)) and rc.equal(Or(x, x, y), Or(y, x))
    assert not rc.equal(And(x, y), Or(x, y))
