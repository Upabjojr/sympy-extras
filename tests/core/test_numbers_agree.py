"""The combinatorial numbers of the Rust core against
``sympy_extras.functions.combinatorial.numbers``: ``catalan``,
``fibonacci``, ``lucas``, ``harmonic``, ``bernoulli``, ``euler``,
``bell``, ``genocchi`` and ``partition`` (numbers and polynomials) of
random integers, rationals, radicals, infinities and symbols with
assumptions, compared by structure (or the error), facts, derivatives
and values (against Python's at 200 more bits; mpmath's ``bell`` of a
non-integer is not ported: no value there)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def indices() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, sqrt
    n, k = Symbol('n', integer=True, nonnegative=True), Symbol('k', integer=True)
    return [Integer(0), Integer(1), Integer(2), Integer(5), Integer(8), Integer(12), Integer(-1), Integer(-2),
            Integer(-7), Rational(1, 2), Rational(-3, 2), Rational(7, 3), sqrt(2), I + 1, oo, n, k, n + 1,
            Symbol('x'), Symbol('p', positive=True), Symbol('r', noninteger=True, negative=True)]


def seconds() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    return [Integer(0), Integer(1), Integer(2), Integer(-1), Rational(1, 2), Symbol('x'), Symbol('m', integer=True)]


NAMES = ['catalan', 'fibonacci', 'lucas', 'harmonic', 'bernoulli', 'euler', 'bell', 'genocchi', 'partition']
TWO = {'fibonacci', 'harmonic', 'bernoulli', 'euler', 'bell', 'genocchi'}


@pytest.mark.parametrize("seed", range(40))
def test_numbers(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.combinatorial import numbers as nb
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("numbers", seed))
    x = Symbol('x')
    for _ in range(12):
        name = rng.choice(NAMES)
        args = [rng.choice(indices())]
        if name in TWO and rng.random() < 0.4:
            args.append(rng.choice(seconds()))
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(nb, name)(*args)), (name, args)
        try:
            e = getattr(nb, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        vr = rc.numeric_value(r, 53) if not e.free_symbols else None
        partial = name == 'bell' and vr is None and not e.free_symbols
        if not partial:
            assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.diff(r, (x, 1)))
        if g != 'SKIP':
            assert g == outcome(lambda: diff(e, x)), e
        if not e.free_symbols and not partial:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)
