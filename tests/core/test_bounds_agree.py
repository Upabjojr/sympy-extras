"""``sympy_extras.assumptions.bounds.polynomial_abstraction`` against its
Rust port: random relations and connectives of elementary functions
(nested, of polynomial and non-polynomial arguments, roots and absolute
values), compared on the abstracted formula, the constraints, the new
variables and the requirements (or the absence of an abstraction)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def term(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    from sympy_extras.functions import Abs, atan, cos, cosh, exp, log, sin, sinh, tanh
    x, y, w = Symbol('x'), Symbol('y'), Symbol('w')
    if depth == 0:
        return rng.choice([x, y, x + 1, 2*x - y, x**2, Integer(3), Rational(1, 2), x*y, w, I*x])
    inner = term(rng, depth - 1)
    f = rng.choice([exp, log, sin, cos, atan, tanh, sinh, cosh, Abs, sqrt, lambda u: u**Rational(2, 3),
                    lambda u: u**Rational(-1, 2), lambda u: u + term(rng, 0), lambda u: u*term(rng, 0)])
    return f(inner)


def relation(rng: random.Random) -> Any:
    from sympy_extras.core import Eq, Ne
    from sympy_extras.core.logic import and_, implies, not_, or_
    a, b = term(rng, rng.randint(0, 2)), term(rng, rng.randint(0, 1))
    r = rng.choice([lambda: a > b, lambda: a <= b, lambda: Eq(a, b), lambda: Ne(a, b), lambda: a >= 0])()
    if rng.random() < 0.3:
        s = term(rng, 1) < 2
        return rng.choice([and_, or_, implies])(r, s) if rng.random() < 0.8 else not_(r)
    return r


def shown(a: Any) -> str:
    if a is None:
        return 'None'
    return str((str(a.formula), [str(c) for c in a.constraints], [str(v) for v in a.variables],
                [str(c) for c in a.requirements]))


def rust_shown(t: Any) -> str:
    if not t.args:
        return 'None'
    vs, f, cs, rs = t.args
    return str((str(f), [str(c) for c in cs.args], [str(v) for v in vs.args], [str(c) for c in rs.args]))


@pytest.mark.parametrize("seed", range(60))
def test_bounds(rust: Any, seed: int) -> None:
    from sympy_extras.assumptions.bounds import polynomial_abstraction
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.containers import Tuple
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("bounds", seed))
    x, y = Symbol('x'), Symbol('y')
    for _ in range(6):
        f = relation(rng)
        real = rng.choice([[x], [x, y]])
        m = rng.choice([1, 3])
        got = outcome(lambda: rust_shown(rc.logic_call('polynomial_abstraction', [f, Tuple(*real), Integer(m)])))
        assert got == outcome(lambda: shown(polynomial_abstraction(f, set(real), m))), (f, real, m)
