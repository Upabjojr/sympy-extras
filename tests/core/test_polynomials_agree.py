"""The orthogonal polynomials of the Rust core (part 1) against
``sympy_extras.functions.special.polynomials``: ``legendre``,
``chebyshevt``, ``chebyshevu``, ``hermite``, ``laguerre`` and
``assoc_laguerre`` of random orders (integers up to 12, negative ones,
non-integers, symbols) and arguments, compared by structure (or the
error), facts, conjugates, derivatives and values (a node with a
numerical non-integer order has no Rust value: there the values and the
facts Python takes from them are not compared)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome

NAMES = ['legendre', 'chebyshevt', 'chebyshevu', 'hermite', 'laguerre', 'assoc_laguerre']
CLASSES = ['Legendre', 'ChebyshevT', 'ChebyshevU', 'Hermite', 'Laguerre', 'AssocLaguerre']


def orders(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol
    n = Symbol('n')
    return rng.choice([Integer(rng.randint(0, 12)), Integer(rng.randint(-6, -1)), Rational(1, 2), Rational(-7, 3),
                       n, -n, -n - 1, -n + 3, Symbol('k', integer=True, positive=True), Symbol('t', integer=False)])


def arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, sqrt
    x = Symbol('x')
    return [Integer(0), Integer(1), Integer(-1), Rational(1, 3), sqrt(2), I, 1 + I, oo, -oo, x, -x, 2*x,
            Symbol('r', real=True)]


@pytest.mark.parametrize("seed", range(40))
def test_polynomials(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import polynomials as op
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("polynomials", seed))
    classes = tuple(getattr(op, c) for c in CLASSES)
    for _ in range(10):
        name = rng.choice(NAMES)
        args = [orders(rng), rng.choice(arguments())]
        if name == 'assoc_laguerre':
            args.insert(1, rng.choice([Integer(0), Integer(2), Rational(1, 2), Symbol('a'), -Symbol('a')]))
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
