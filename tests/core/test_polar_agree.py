"""The polar numbers of the Rust core against
``sympy_extras.functions.elementary``: ``exp_polar``, ``polar_lift``,
``periodic_argument``, ``unbranched_argument``, ``principal_branch``,
``unpolarify`` and ``polarify`` of random arguments (numbers, multiples of
``I*pi``, symbols with signs and polar symbols, products and powers of
polar numbers), compared by structure (or the error), facts, conjugates,
derivatives and values, and the arithmetic of the polar numbers."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome

ARITY = {'exp_polar': 1, 'polar_lift': 1, 'periodic_argument': 2, 'unbranched_argument': 1, 'principal_branch': 2,
         'unpolarify': 1, 'polarify': 1}


def module() -> dict[str, Callable[..., Any]]:
    from sympy_extras.functions.elementary import complexes as c
    from sympy_extras.functions.elementary.exponential import exp_polar
    from sympy_extras.core import Tuple

    def polarify(e: Any) -> Any:
        result, back = c.polarify(e, subs=False)
        return Tuple(result, Tuple())
    return {'exp_polar': exp_polar, 'polar_lift': c.polar_lift, 'periodic_argument': c.periodic_argument,
            'unbranched_argument': c.unbranched_argument, 'principal_branch': c.principal_branch,
            'unpolarify': c.unpolarify, 'polarify': polarify}


def values() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, sqrt
    from sympy_extras.functions.elementary.complexes import polar_lift
    from sympy_extras.functions.elementary.exponential import exp, exp_polar
    x, p, z = Symbol('x'), Symbol('p', positive=True), Symbol('z', polar=True)
    return [Integer(0), Integer(1), Integer(-1), Integer(-3), Rational(1, 2), I, -I, 1 + I, sqrt(2), pi, x, -x, p,
            -p, z, 2*z, p*z, I*p, exp_polar(I*pi), exp_polar(2*I*pi), exp_polar(5*I*pi)*p, exp_polar(-3*I*pi/2),
            exp_polar(x), exp_polar(I*pi)*x, polar_lift(x), polar_lift(x)**2, z**Rational(1, 2), exp(I*x),
            exp_polar(I*pi/3)*z, x + 1, exp_polar(2*I*pi) + x, 2*pi, oo, 3*pi]


def periods() -> list[Any]:
    from sympy_extras.core import Integer, pi, oo, Symbol
    return [2*pi, pi, oo, 4*pi, Integer(1), Symbol('q', positive=True)]


@pytest.mark.parametrize("seed", range(40))
def test_polar(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    rc = load("sympy_extras_rs.core")
    functions = module()
    rng = random.Random(stable_seed("polar", seed))
    for _ in range(12):
        name = rng.choice(list(ARITY))
        args = [rng.choice(values())]
        if ARITY[name] == 2:
            args.append(rng.choice(periods()))
        from sympy_extras.core.logic import false
        rust_args = args + [false] if name == 'polarify' else args
        got = outcome(lambda: rc.logic_call(name, rust_args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: functions[name](*args)), (name, args)
        if name == 'polarify':
            continue
        try:
            e = functions[name](*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        for v in (Symbol('x'), Symbol('p', positive=True)):
            g = outcome(lambda: rc.diff(r, (v, 1)))
            if g != 'SKIP':
                assert g == outcome(lambda: diff(e, v)), (e, v)
        if not e.free_symbols:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


@pytest.mark.parametrize("seed", range(20))
def test_arithmetic(rust: Any, seed: int) -> None:
    from sympy_extras.core.function import expand
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("polar-arithmetic", seed))
    pool = values()
    for _ in range(15):
        a, b = rng.choice(pool), rng.choice(pool)
        for op, rust_op, python_op in (('mul', rc.mul, lambda u, v: u*v), ('add', rc.add, lambda u, v: u + v),
                                       ('pow', rc.pow, lambda u, v: u**v)):
            got = outcome(lambda: rust_op(a, b))
            if got != 'SKIP':
                assert got == outcome(lambda: python_op(a, b)), (op, a, b)
        e = a*b
        got = outcome(lambda: rc.expand(e))
        if got != 'SKIP':
            assert got == outcome(lambda: expand(e)), e
