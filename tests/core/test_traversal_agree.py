"""``sympy_extras.core.traversal`` and ``sympy_extras.core.fuzzy``
against their Rust ports: the traversal orders of random expressions
(with bound variables: ``Subs``, quantifiers), ``bottom_up`` and ``use``
with a fixed function, and ``fuzzy_group`` on every short sequence."""

from __future__ import annotations

import itertools
import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Integer, Subs, Symbol
    from sympy_extras.core.function import Derivative
    from sympy_extras.functions import sin
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    if depth == 0:
        return rng.choice([x, y, z, Integer(2), Integer(-3)])
    a, b = expression(rng, depth - 1), expression(rng, depth - 1)
    return rng.choice([lambda: a + b, lambda: a*b, lambda: a**2, lambda: sin(a) + b,
                       lambda: Subs(Derivative(sin(x*a), x), (x,), (b,))])()


@pytest.mark.parametrize("seed", range(40))
def test_traversal(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol, false, true
    from sympy_extras.core import traversal as tv
    from sympy_extras.core.containers import Tuple
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("traversal", seed))
    x = Symbol('x')
    f = lambda n: Integer(2) if n == x else n
    for _ in range(5):
        e = expression(rng, rng.randint(1, 3))
        if rng.random() < 0.3:
            from sympy_extras.assumptions.quantifiers import ForAll
            e = ForAll(x, e > expression(rng, 1))
        for name in ('postorder_traversal', 'iterargs', 'iterfreeargs'):
            assert outcome(lambda: rc.logic_call(name, [e])) == \
                outcome(lambda: Tuple(*getattr(tv, name)(e))), (name, e)
        for atoms in (True, False):
            got = outcome(lambda: rc.logic_call('bottom_up_x2', [e, true if atoms else false]))
            assert got == outcome(lambda: tv.bottom_up(e, f, atoms=atoms)), (e, atoms)
        for level in range(3):
            assert outcome(lambda: rc.logic_call('use_x2', [e, Integer(level)])) == \
                outcome(lambda: tv.use(e, f, level)), (e, level)


def test_fuzzy_group(rust: Any) -> None:
    from sympy_extras.core import Integer, false, true
    from sympy_extras.core.containers import Tuple
    from sympy_extras.core.fuzzy import fuzzy_group
    rc = load("sympy_extras_rs.core")
    as_node = {True: true, False: false, None: Integer(0)}
    for n in range(4):
        for values in itertools.product([True, False, None], repeat=n):
            for quick in (True, False):
                got = rc.logic_call('fuzzy_group', [Tuple(*[as_node[v] for v in values]), as_node[quick]])
                assert str(got) == str(as_node[fuzzy_group(values, quick_exit=quick)]), (values, quick)
