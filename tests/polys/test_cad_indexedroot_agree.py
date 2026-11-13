"""``IndexedRoot`` of the Rust core against
``sympy_extras.polys.cad.indexedroot``: ``indexed_root`` of random
polynomials in ``t`` with rational coefficients (rational roots, roots in
radicals, canonical nodes of minimal polynomials) and with parameters
(nodes), every index, the facts, the numerical values, the substitution
of the parameters (which evaluates) and of the bound variable (which
does not), and the round trip through the bridge."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from ..core.test_bridge_agree import close


def outcome(f: Any) -> str:
    try:
        return repr(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def random_polynomial(rng: random.Random, t: Any, params: list[Any]) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    degree = rng.choice([1, 2, 2, 3, 3, 5])
    terms = [pow(t, Integer(degree))]
    for i in range(degree):
        if rng.random() < 0.6:
            c: Any = rng.choice([Integer(rng.randint(-5, 5)), Rational(rng.randint(-3, 3), 2)])
            if params and rng.random() < 0.3:
                c = mul(c, rng.choice(params))
            terms.append(mul(c, pow(t, Integer(i))))
    return add(*terms)


@pytest.mark.parametrize("seed", range(40))
def test_indexed_root(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.polys.cad.indexedroot import indexed_root
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("indexed root", seed))
    t, a = Symbol('t'), Symbol('a')
    compared = 0
    for _ in range(3):
        f = random_polynomial(rng, t, [a] if rng.random() < 0.4 else [])
        for k in range(4):
            got = outcome(lambda: rc.indexed_root(f, t, k))
            if got == 'SKIP':
                continue
            assert got == outcome(lambda: indexed_root(f, t, k)), (f, k)
            compared += 1
            e = indexed_root(f, t, k)
            r = rc.indexed_root(f, t, k)
            assert rc.facts(r) == dict(facts_of(e)), (f, k)
            for prec in (53, 120):
                assert close(rc.numeric_value(r, prec), numeric_value(e, prec), prec), (f, k, prec)
            if a in e.free_symbols:
                v = Integer(rng.randint(-3, 3))
                assert repr(rc.subs(r, a, v)) == repr(e.subs(a, v)), (f, k, v)
                assert repr(rc.subs(r, t, v)) == repr(e.subs(t, v)), (f, k, v)
            assert repr(rc.roundtrip(e)) == repr(e)
    assert compared >= 3
