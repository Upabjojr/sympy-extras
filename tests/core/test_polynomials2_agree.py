"""The orthogonal polynomials of the Rust core (part 2) against
``sympy_extras.functions.special.polynomials``: ``jacobi``,
``gegenbauer``, ``assoc_legendre`` and ``hermite_prob`` of random orders
and parameters (the reductions ``jacobi(n, a, a, x)``, ``jacobi(n, a, -a,
x)``), compared by structure (or the error), facts, conjugates,
derivatives and values (as in part 1, a node of numerical order has no
Rust value); ``chebyshevt_root``, ``chebyshevu_root`` and
``jacobi_normalized``. ``jacobi`` of a symbolic order at ``0`` is a
``hyper``."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome
from .test_polynomials_agree import arguments, orders

NAMES = ['jacobi', 'gegenbauer', 'assoc_legendre', 'hermite_prob']
CLASSES = ['Jacobi', 'Gegenbauer', 'AssocLegendre', 'HermiteProb', 'Legendre', 'ChebyshevT', 'ChebyshevU']


def parameters(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol
    a = Symbol('a')
    return rng.choice([Integer(0), Integer(1), Integer(-1), Integer(2), Rational(1, 2), Rational(-1, 2),
                       Rational(3, 2), a, -a, a + 1, Symbol('p', positive=True)])


def small_orders(rng: random.Random) -> Any:
    from sympy_extras.core import Integer
    o = orders(rng)
    if o.is_Integer and o > 7:
        return Integer(rng.randint(0, 7))  # jacobi expands binomials in a, b
    return o


def call_args(rng: random.Random, name: str) -> list[Any]:
    from sympy_extras.core import Integer
    n, x = small_orders(rng), rng.choice(arguments())
    if name == 'jacobi':
        a = parameters(rng)
        b = rng.choice([a, -a, parameters(rng)])
        return [n, a, b, x]
    if name == 'gegenbauer':
        return [n, parameters(rng), x]
    if name == 'assoc_legendre':
        m = rng.choice([Integer(rng.randint(-3, 3)), parameters(rng)])
        return [n, m, x]
    return [n, x]


@pytest.mark.parametrize("seed", range(40))
def test_polynomials_part2(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import polynomials as op
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("polynomials2", seed))
    classes = tuple(getattr(op, c) for c in CLASSES)
    for _ in range(10):
        name = rng.choice(NAMES)
        args = call_args(rng, name)
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(op, name)(*args)), (name, args)
        try:
            e = getattr(op, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        partial = not e.free_symbols and e.has(*classes)
        if not partial:
            assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        for v in (Symbol('x'), Symbol('n'), Symbol('a'), Symbol('r', real=True)):
            g = outcome(lambda: rc.diff(r, (v, 1)))
            if g != 'SKIP':
                assert g == outcome(lambda: diff(e, v)), (e, v)
        if not e.free_symbols and not partial:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


@pytest.mark.parametrize("seed", range(10))
def test_roots_and_normalized(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.functions.special import polynomials as op
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("polynomial roots", seed))
    for _ in range(10):
        n = rng.choice([Integer(rng.randint(1, 9)), Symbol('n'), Symbol('k', integer=True, positive=True)])
        k = rng.choice([Integer(rng.randint(-1, 9)), Rational(1, 2), Symbol('j')])
        for name in ('chebyshevt_root', 'chebyshevu_root'):
            assert outcome(lambda: rc.logic_call(name, [n, k])) == outcome(lambda: getattr(op, name)(n, k)), (name, n, k)
        args = [rng.choice([Integer(rng.randint(0, 4)), Symbol('n')]), parameters(rng), parameters(rng),
                rng.choice([Symbol('x'), Rational(1, 3), Integer(1)])]
        got = outcome(lambda: rc.logic_call('jacobi_normalized', args))
        if got != 'SKIP':
            assert got == outcome(lambda: op.jacobi_normalized(*args)), args
