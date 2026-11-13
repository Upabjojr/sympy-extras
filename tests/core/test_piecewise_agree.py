"""``Piecewise`` of the Rust core against
``sympy_extras.functions.elementary.piecewise``: random pairs (relations
of linear expressions with rational coefficients, their conjunctions,
disjunctions and negations, true and false; nested ``Piecewise``
expressions), compared by the evaluated structure (or the error), the
printed form, facts, conjugates, derivatives, powers, substitutions and
``piecewise_fold`` (evaluated, unevaluated and with ``evaluate=None``)
of sums, products and functions of them."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def _relation(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.relational import eq, ge, gt, le, lt, ne
    x, y = Symbol('x'), Symbol('y', real=True)
    v = rng.choice([x, x, y])
    a = rng.choice([Integer(1), Integer(-1), Integer(2), Integer(-2), Rational(1, 3), Rational(-3, 2)])
    b = rng.choice([Integer(0), Integer(1), Integer(-1), Rational(1, 2)])
    lhs = a*v + b if rng.random() < 0.7 else v
    rhs = rng.choice([Integer(0), Integer(1), Integer(2), Integer(-1), y if v is x else x])
    return rng.choice([lt, le, gt, ge, eq, ne, lt, gt])(lhs, rhs)


def _condition(rng: random.Random, depth: int = 0) -> Any:
    from sympy_extras.core import false, true
    from sympy_extras.core.logic import and_, not_, or_
    r = rng.random()
    if r < 0.08:
        return true
    if r < 0.12:
        return false
    if depth < 2 and r < 0.3:
        return and_(_condition(rng, depth + 1), _condition(rng, depth + 1))
    if depth < 2 and r < 0.4:
        return or_(_condition(rng, depth + 1), _condition(rng, depth + 1))
    if depth < 2 and r < 0.45:
        return not_(_condition(rng, depth + 1))
    return _relation(rng)


def _expression(rng: random.Random, depth: int = 0) -> Any:
    from sympy_extras.core import I, Integer, Symbol, sqrt
    x, y = Symbol('x'), Symbol('y', real=True)
    if depth < 1 and rng.random() < 0.2:
        return _piecewise(rng, depth + 1)
    return rng.choice([x, x**2, -x, Integer(1), Integer(2), Integer(0), y, x + 1, I*x, sqrt(2), 1/x, Integer(-3)])


def _pairs(rng: random.Random, depth: int = 0) -> list[Any]:
    from sympy_extras.core import true
    from sympy_extras.functions.elementary.piecewise import ExprCondPair
    pairs = [ExprCondPair(_expression(rng, depth), _condition(rng)) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.6:
        pairs.append(ExprCondPair(_expression(rng, depth), true))
    if rng.random() < 0.15 and pairs:
        pairs.append(pairs[rng.randrange(len(pairs))])
    return pairs


def _piecewise(rng: random.Random, depth: int = 0) -> Any:
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
    pairs = _pairs(rng, depth)
    try:
        return piecewise(*pairs)
    except (ValueError, TypeError):
        return Piecewise(*pairs)


def _check(got: Any, expected: Callable[[], Any], what: Any) -> None:
    rc = load("sympy_extras_rs.core")
    g = outcome(got)
    if g == 'SKIP':
        return
    assert g == outcome(expected), what
    try:
        e = expected()
    except Exception:
        return
    assert rc.sstr(got()) == str(e), what


@pytest.mark.parametrize("seed", range(30))
def test_piecewise(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.function import diff
    from sympy_extras.core.power import pow
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.elementary.piecewise import piecewise
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("piecewise", seed))
    x, y = Symbol('x'), Symbol('y', real=True)
    for _ in range(10):
        pairs = _pairs(rng)
        _check(lambda: rc.logic_call('piecewise', pairs), lambda: piecewise(*pairs), pairs)
        try:
            e = piecewise(*pairs)
        except Exception:
            continue
        r = rc.roundtrip(e)
        assert rc.facts(r) == dict(facts_of(e)), e
        _check(lambda: rc.conjugate(r), lambda: conjugate(e), ('conjugate', e))
        for v in (x, y):
            _check(lambda: rc.diff(r, (v, 1)), lambda: diff(e, v), ('diff', e, v))
        for n in (Integer(2), Integer(-1), Rational(1, 2)):
            _check(lambda: rc.pow(r, n), lambda: pow(e, n), ('pow', e, n))
        for old, new in ((x, Integer(1)), (x, Integer(-1)), (y, Rational(1, 2)), (x, y)):
            _check(lambda: rc.subs(r, old, new), lambda: e.subs(old, new), ('subs', e, old, new))


@pytest.mark.parametrize("seed", range(30))
def test_piecewise_fold(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions.elementary.exponential import exp
    from sympy_extras.functions.elementary.piecewise import piecewise_fold
    from sympy_extras.functions.elementary.trigonometric import sin
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("piecewise-fold", seed))
    x = Symbol('x')
    for _ in range(8):
        # only the first may nest: folding nested products grows quickly
        p, q = _piecewise(rng), _piecewise(rng, 1)
        if rng.random() < 0.3:
            q = p.subs(x, x + 1) if rng.random() < 0.5 else p
        builders: list[Callable[[], Any]] = [
            lambda: x*p + 3, lambda: p + q, lambda: p*q, lambda: sin(p) + q, lambda: exp(p*q),
            lambda: p**2 + x*q, lambda: p + q + Integer(2)*p, lambda: p,
        ]
        try:
            e = rng.choice(builders)()
        except Exception:
            continue
        for name, evaluate in (('piecewise_fold', True), ('piecewise_fold_none', None),
                               ('piecewise_fold_false', False)):
            _check(lambda: rc.logic_call(name, [e]), lambda: piecewise_fold(e, evaluate), (name, e))
