"""``sympy_extras.simplify.powsimp`` (``powsimp``, ``powdenest``,
``multiplicity``) and ``logcombine``, ``posify`` of
``sympy_extras.simplify.simplify`` against their Rust ports, on random
products of powers and sums of logarithms; the outputs compared as
strings."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('powsimp', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _symbols() -> Any:
    from sympy_extras.core import Symbol
    return (Symbol('x'), Symbol('y'), Symbol('z'), Symbol('p', positive=True), Symbol('n', integer=True),
            Symbol('r', real=True), Symbol('q', positive=True))


def _products(seed: int) -> list[Any]:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.functions import exp, sqrt
    x, y, z, p, n, r, q = _symbols()
    rng = random.Random(seed)
    atoms = [x, y, z, p, Integer(2), Integer(3), Rational(1, 2), exp(x), sqrt(x), sqrt(Integer(2)), x + 1, Integer(8),
             x*y, p*q**2, Integer(-2)]
    exponents = [Integer(2), Integer(-1), Rational(1, 2), Rational(-3, 2), y, n, x + y, 2*n, Rational(1, 3), r, p]
    out = []
    for _ in range(70):
        e = mul(*[pow(rng.choice(atoms), rng.choice(exponents)) for _ in range(rng.randint(2, 4))])
        if rng.random() < 0.3:
            e = pow(e, rng.choice(exponents))
        if rng.random() < 0.25:
            e = e + mul(*[pow(rng.choice(atoms), rng.choice(exponents)) for _ in range(2)])
        out.append(e)
    return out


def _logs(seed: int) -> list[Any]:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.functions import log
    x, y, z, p, n, r, q = _symbols()
    rng = random.Random(seed)
    atoms = [log(p), log(q), log(x), log(Integer(2)), log(p*q), r, n, x, Integer(2), Rational(-1, 2), log(y)]
    return [add(*[mul(rng.choice([Integer(1), Integer(2), Integer(-1), r, x]), *[rng.choice(atoms) for _ in range(rng.randint(1, 2))])
                  for _ in range(rng.randint(1, 4))]) for _ in range(50)]


OPTIONS = [(False, 'all', False), (False, 'exp', False), (False, 'base', False), (False, 'all', True), (True, 'all', False)]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_powsimp(rust: Any, seed: int) -> None:
    from sympy_extras.core.logic import false, true
    from sympy_extras.core import Symbol
    from sympy_extras.simplify.powsimp import powsimp
    different = []
    for e in _products(seed):
        for deep, combine, force in OPTIONS:
            got = _outcome(lambda: _call('powsimp', e, true if deep else false, Symbol(combine), true if force else false))
            expected = _outcome(lambda: powsimp(e, deep=deep, combine=combine, force=force))
            if got != expected:
                different.append((e, (deep, combine, force), got, expected))
    assert not different, different[:3]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_powdenest_and_logcombine(rust: Any, seed: int) -> None:
    from sympy_extras.core.logic import false, true
    from sympy_extras.simplify.powsimp import powdenest
    from sympy_extras.simplify.simplify import logcombine
    different = []
    for e in _products(seed) + _logs(seed):
        for force in (False, True):
            f = true if force else false
            for name, got, expected in [
                    ('powdenest', _outcome(lambda: _call('powdenest', e, f, false)), _outcome(lambda: powdenest(e, force=force))),
                    ('logcombine', _outcome(lambda: _call('logcombine', e, f)), _outcome(lambda: logcombine(e, force=force)))]:
                if got != expected:
                    different.append((name, force, e, got, expected))
    assert not different, different[:3]


def test_posify_and_multiplicity(rust: Any) -> None:
    from sympy_extras.core import Rational
    from sympy_extras.simplify.powsimp import multiplicity
    from sympy_extras.simplify.simplify import posify
    for e in _products(4)[:30]:
        replaced, back = _call('posify', e).args
        expected, mapping = posify(e)
        assert str(replaced) == str(expected), e
        assert sorted(str(s) for _, s in back.args) == sorted(str(s) for s in mapping.values()), e
        assert all(d.is_positive for d, _ in back.args)
        assert str(replaced._xreplace_evaluated({d: s for d, s in back.args})) == str(expected._xreplace_evaluated(mapping)), e
    for p, n in [(Rational(2), Rational(24)), (Rational(3), Rational(24)), (Rational(1, 2), Rational(3, 8)),
                 (Rational(2), Rational(1, 16)), (Rational(2, 3), Rational(8, 27)), (Rational(2, 3), Rational(9, 4)),
                 (Rational(5), Rational(1)), (Rational(1), Rational(7)), (Rational(2), Rational(0))]:
        got = _outcome(lambda: _call('multiplicity', p, n))
        expected = _outcome(lambda: multiplicity(p, n))
        assert got == expected, (p, n)
