"""``sympy_extras.physics.dimensional`` against its Rust port: random sets of
SI quantities (and a fractional dimension), compared on the dimension
matrix, the Pi exponents and groups, ``solve_for`` (its errors included)
and ``dimension_of``."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def encode(quantities: dict[str, dict[str, Any]]) -> Any:
    from sympy_extras.core import Rational, Symbol
    from sympy_extras.core.containers import Tuple
    return Tuple(*[Tuple(Symbol(name), Tuple(*[Tuple(Symbol(b), Rational(Fraction(e).numerator, Fraction(e).denominator))
                                               for b, e in dimension.items()]))
                   for name, dimension in quantities.items()])


@pytest.mark.parametrize("seed", range(40))
def test_dimensional(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.physics import dimensional as dm
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("dimensional", seed))
    kinds = sorted(dm.SI)
    names = ['q%d' % i for i in range(rng.randint(1, 6))]
    quantities: dict[str, dict[str, Any]] = {n: dict(dm.SI[rng.choice(kinds)]) for n in names}
    if rng.random() < 0.2:
        quantities['h'] = {'L': Fraction(1, 2), 'T': -1}
    q = encode(quantities)
    assert outcome(lambda: [list(r.args) for r in rc.logic_call('pi_exponents', [q]).args]) == \
        outcome(lambda: [[Integer(e) for e in v] for v in dm.pi_exponents(quantities)])
    assert outcome(lambda: list(rc.logic_call('pi_groups', [q]).args)) == outcome(lambda: dm.pi_groups(quantities))
    bases, rows = dm.dimension_matrix(quantities)
    r = rc.logic_call('dimension_matrix', [q])
    assert [str(b) for b in r.args[0].args] == bases and [[int(x) for x in row.args] for row in r.args[1].args] == rows
    for target in list(quantities)[:3]:
        assert outcome(lambda: rc.logic_call('solve_for', [Symbol(target), q])) == \
            outcome(lambda: dm.solve_for(target, quantities)), (target, quantities)
    exponents = {n: rng.choice([1, -1, 2, Fraction(1, 2)]) for n in rng.sample(list(quantities), min(2, len(quantities)))}
    e = Tuple(*[Tuple(Symbol(n), Rational(Fraction(x).numerator, Fraction(x).denominator)) for n, x in exponents.items()])
    got = rc.logic_call('dimension_of', [e, q])
    assert {str(t.args[0]): Fraction(str(t.args[1])) for t in got.args} == dm.dimension_of(exponents, quantities)


def test_docstring(rust: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.physics.dimensional import SI, pi_groups, solve_for
    rc = load("sympy_extras_rs.core")
    flow = {'F': SI['force'], 'rho': SI['density'], 'v': SI['velocity'], 'L': SI['length'], 'mu': SI['dynamic viscosity']}
    assert list(rc.logic_call('pi_groups', [encode(flow)]).args) == pi_groups(flow)
    pendulum = {'T': SI['time'], 'l': SI['length'], 'g': SI['acceleration'], 'm': SI['mass']}
    assert rc.logic_call('solve_for', [Symbol('T'), encode(pendulum)]) == solve_for('T', pendulum)
    assert str(rc.logic_call('pi_groups', [encode(flow)]).args[0]) == 'F/(L*mu*v)'
