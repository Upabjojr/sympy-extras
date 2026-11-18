"""The ``HyperRep`` functions of the Rust core against
``sympy_extras.functions.special.hyperrep``: random heads, parameters
and (non-polar) arguments, compared by structure, facts, conjugates,
derivatives and the closed forms of ``rewrite_nonrep`` (small and on
every branch), alone and inside expressions."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome

TWO = ['HyperRep_power1', 'HyperRep_power2', 'HyperRep_sqrts1', 'HyperRep_sqrts2', 'HyperRep_cosasin',
       'HyperRep_sinasin']
ONE = ['HyperRep_log1', 'HyperRep_atanh', 'HyperRep_asin1', 'HyperRep_asin2', 'HyperRep_log2']


def parameter(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol
    return rng.choice([Integer(1), Integer(-2), Integer(3), Rational(1, 2), Rational(-3, 2), Rational(1, 3),
                       Symbol('a'), Symbol('k', integer=True), Symbol('a') + 1])


def argument(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol
    z = Symbol('z')
    return rng.choice([z, 2*z, z**2, -z, Symbol('r', real=True), Rational(1, 2), Integer(4), Integer(-3), I,
                       Rational(1, 3) + I])


@pytest.mark.parametrize("seed", range(40))
def test_hyperrep(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import hyperrep as hr
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("hyperrep", seed))
    for _ in range(10):
        name = rng.choice(TWO + ONE)
        args = ([parameter(rng)] if name in TWO else []) + [argument(rng)]
        got = outcome(lambda: rc.logic_call('hyperrep', [Symbol(name)] + args))
        if got == 'SKIP':
            continue
        cls = getattr(hr, name)
        assert got == outcome(lambda: cls.apply(*args)), (name, args)
        e, r = cls.apply(*args), rc.logic_call('hyperrep', [Symbol(name)] + args)
        assert rc.facts(r) == dict(facts_of(e)), e
        assert outcome(lambda: rc.conjugate(r)) == outcome(lambda: conjugate(e)), e
        for v in (Symbol('z'), Symbol('a')):
            g = outcome(lambda: rc.diff(r, (v, 1)))
            if g != 'SKIP':
                assert g == outcome(lambda: diff(e, v)), (e, v)
        wrapped = (1 + Symbol('x')*e, 1 + Symbol('x')*r)
        for pe, pr in ((e, r), wrapped):
            assert outcome(lambda: rc.logic_call('rewrite_nonrepsmall', [pr])) == \
                outcome(lambda: hr.rewrite_nonrep(pe)), pe
            assert outcome(lambda: rc.logic_call('rewrite_nonrep', [pr])) == \
                outcome(lambda: hr.rewrite_nonrep(pe, small=False)), pe
