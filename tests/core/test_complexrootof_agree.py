"""``sympy_extras.polys.complexrootof`` against its Rust port: random
polynomials with rational coefficients (products of factors, repeated
factors, roots on the axes) compared on ``all_roots``, ``real_roots``,
``root_count`` and ``rootof`` (with and without radicals, negative
indices, the errors); the nodes on their facts, values, addresses,
equality across variables, substitution and ``real_roots_indexed``."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_integers_agree import outcome


def random_poly(rng: random.Random, x: Any) -> Any:
    from sympy_extras.core import Integer, Rational
    factors = [x**3 - x - 1, x**2 + 1, x**2 - 2, x**4 + 1, x - Rational(1, 2), x**5 - x - 1, x**2 + x + 1,
               x**3 - 2, 2*x**2 + 3, x**4 + 6*x**2 + 4, x**3 + 2*x + 5, x, x**4 - 10*x**2 + 1]
    p: Any = Integer(rng.choice([1, 1, 2, -3]))
    for _ in range(rng.randint(1, 3)):
        p = p * rng.choice(factors)**rng.choice([1, 1, 1, 2])
    if rng.random() < 0.3:
        p = p + rng.randint(-3, 3)
    return p.expand()


@pytest.mark.parametrize("seed", range(40))
def test_complexrootof(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol, true, false
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.containers import Tuple
    from sympy_extras.polys import complexrootof as cr
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("complexrootof", seed))
    x, y = Symbol('x'), Symbol('y')
    for _ in range(3):
        p = random_poly(rng, x)
        for radicals in (True, False):
            flag = true if radicals else false
            got = outcome(lambda: rc.logic_call('all_roots', [p, x, flag]))
            assert got == outcome(lambda: Tuple(*cr.all_roots(p, x, radicals))), (p, radicals)
            got = outcome(lambda: rc.logic_call('real_roots', [p, Integer(0), flag]))
            assert got == outcome(lambda: Tuple(*cr.real_roots(p, None, radicals))), (p, radicals)
        assert outcome(lambda: rc.logic_call('root_count', [p, x])) == outcome(lambda: Tuple(*map(Integer, cr.root_count(p, x)))), p
        for k in (0, 1, -1, rng.randint(-8, 8)):
            got = outcome(lambda: rc.logic_call('rootof', [p, Integer(k), x, false]))
            assert got == outcome(lambda: cr.rootof(p, k, x, radicals=False)), (p, k)
        try:
            roots = cr.all_roots(p, x, radicals=False)
        except Exception:
            continue
        for r in roots:
            if not isinstance(r, cr.ComplexRootOf):
                continue
            rr = rc.logic_call('rootof', [r.expr, Integer(r.index), x, false])
            assert rc.facts(rr) == dict(facts_of(r)), r
            vr, ve = rc.numeric_value(rr, 80), numeric_value(r, 280)
            assert ve is not None and vr is not None and close(vr, ve, 80), r
            f, i = r.address()
            assert outcome(lambda: rc.logic_call('crootof_address', [rr])) == \
                outcome(lambda: Tuple(Tuple(*map(Integer, f)), Integer(i))), r
            other = cr.ComplexRootOf(r.expr.subs(x, y), r.index)
            assert outcome(lambda: rc.logic_call('complex_root_of', [other.expr, Integer(r.index)])) == \
                outcome(lambda: other), other
            assert str(rc.subs(rr, x, Integer(2))) == str(r.subs(x, 2))
            assert outcome(lambda: rc.logic_call('real_roots_indexed', [rr + x])) == \
                outcome(lambda: cr.real_roots_indexed(r + x)), r
