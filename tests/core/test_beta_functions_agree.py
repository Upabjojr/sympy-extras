"""The beta functions of the Rust core against
``sympy_extras.functions.special.beta_functions``: ``beta`` (one and two
arguments), ``betainc`` and ``betainc_regularized`` of random numbers and
symbols, compared by structure (or the error), facts, conjugates,
derivatives and values (``beta`` against mpmath at 200 more bits; the
incomplete betas have no value in the Rust core: their values, and what
Python takes from them, are not compared)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    return [Integer(0), Integer(1), Integer(2), Integer(-3), Rational(1, 2), Rational(3, 2), Rational(-5, 2),
            Rational(1, 3), sqrt(2), I, 2 + I, Symbol('x'), Symbol('p', positive=True), Symbol('r', real=True)]


@pytest.mark.parametrize("seed", range(40))
def test_beta_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import beta_functions as bf
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("beta", seed))
    x = Symbol('x')
    for _ in range(12):
        name = rng.choice(['beta', 'beta', 'betainc', 'betainc_regularized'])
        count = rng.choice([1, 2]) if name == 'beta' else 4
        args = [rng.choice(arguments()) for _ in range(count)]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(bf, name)(*args)), (name, args)
        try:
            e = getattr(bf, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        partial = name != 'beta' and not e.free_symbols
        if not partial:
            assert rc.facts(r) == dict(facts_of(e)), e
            assert outcome(lambda: rc.conjugate(r)) == outcome(lambda: conjugate(e)), e
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
