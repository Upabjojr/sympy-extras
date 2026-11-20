"""``Sum`` and ``Product`` of the Rust core against
``sympy_extras.concrete.expr_with_limits``: the constructors (nested
nodes merged, ``Piecewise`` folded), the free symbols with the bound
variables, the substitution, the facts and the arithmetic with them."""
from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def pieces() -> tuple[list[Any], list[Any]]:
    from sympy_extras.core import Integer, Rational, Symbol, Tuple, oo
    from sympy_extras.functions import Piecewise, exp
    i, j, n, x = Symbol('i'), Symbol('j'), Symbol('n'), Symbol('x')
    k = Symbol('k', integer=True, positive=True)
    functions = [x**i, i, i*j, 1 / i**2, exp(i * x), Integer(0), x, i + j, Piecewise((x, x > 0), (0, True)),
                 Piecewise((i, i > 2), (1, True)), k * i]
    limits = [Tuple(i, 0, n), Tuple(i, 1, oo), Tuple(j, 0, i), Tuple(i, k, 2 * k), Tuple(j, 1, n),
              Tuple(i, Rational(1, 2), 3)]
    return functions, limits


@pytest.mark.parametrize("seed", range(30))
def test_expr_with_limits(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.concrete.expr_with_limits import product_, sum_
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("expr-with-limits", seed))
    functions, limits = pieces()
    i, j, n, x = Symbol('i'), Symbol('j'), Symbol('n'), Symbol('x')
    for _ in range(10):
        name = rng.choice(['sum_', 'product_'])
        build = sum_ if name == 'sum_' else product_
        f = rng.choice(functions)
        lims = rng.sample(limits, rng.choice([1, 2]))
        got = outcome(lambda: rc.logic_call(name, [f] + lims))
        assert got == outcome(lambda: build(f, *lims)), (name, f, lims)
        try:
            e = build(f, *lims)
        except Exception:
            continue
        r = rc.logic_call(name, [f] + lims)
        assert set(rc.free_symbols(r)) == set(e.free_symbols), e
        assert rc.facts(r) == dict(facts_of(e)), e
        for old, new in ((i, j), (n, Integer(3)), (x, x + 1), (j, n)):
            assert outcome(lambda: rc.subs(r, old, new)) == outcome(lambda: e.subs(old, new)), (e, old, new)
        other = rng.choice(functions)
        assert outcome(lambda: rc.mul(r, other)) == outcome(lambda: e * other), (e, other)
        assert outcome(lambda: rc.add(r, other)) == outcome(lambda: e + other), (e, other)
