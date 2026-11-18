"""``sympy_extras.simplify.radsimp`` (``radsimp``, ``split_surds``,
``rad_rationalize``, ``unevaluated_mul``) and
``sympy_extras.simplify.sqrtdenest`` against their Rust ports, on random
nested radicals, denominators with surds and symbolic radicals; the
outputs compared as strings."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('radsimp', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _cases(seed: int) -> list[tuple[Any, bool]]:
    from sympy_extras.core import Integer
    from sympy_extras.core.add import add
    from sympy_extras.core.power import pow
    from sympy_extras.functions import sqrt
    rng = random.Random(seed)
    surds = [sqrt(Integer(k)) for k in (2, 3, 5, 6, 10, 15)]
    out: list[tuple[Any, bool]] = []
    for _ in range(50):
        b = rng.randint(1, 4)*rng.choice(surds)
        if rng.random() < 0.3:
            b = b + rng.choice(surds + [Integer(1), Integer(2)])
        out.append((sqrt(rng.randint(1, 9) + b), True))
        denominator = add(*[rng.randint(-3, 3)*rng.choice(surds + [Integer(1), Integer(2)]) for _ in range(rng.randint(2, 4))])
        if denominator != 0:
            out.append((pow(denominator, Integer(-1)), False))
    return out


def _symbolic() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.power import pow
    from sympy_extras.functions import sqrt
    x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
    p = Symbol('p', positive=True)
    inv = lambda e: pow(e, Integer(-1))
    return [inv(1 + sqrt(x)), inv(a + b*sqrt(x)), inv(sqrt(x) + sqrt(y)), inv(2 + sqrt(2)), inv(sqrt(2) + sqrt(3)),
            inv(1 + sqrt(2) + sqrt(3)), inv(1 + sqrt(2) + sqrt(3) + sqrt(5)), x/(sqrt(p) - 1), inv(pow(x, Rational(1, 4)) + 1),
            (1 + sqrt(2))/(3 - sqrt(2)), inv(sqrt(x) - sqrt(x + 1)), sqrt(x + 2*sqrt(x - 1)), sqrt(5 + 2*sqrt(6)),
            sqrt(3 + 2*sqrt(2)), sqrt(Integer(2)), sqrt(x**2 + 2*x*sqrt(y) + y), sqrt(2*sqrt(2) + 3) + sqrt(5 - 2*sqrt(6)),
            sqrt(10 + 2*sqrt(6) + 2*sqrt(10) + 2*sqrt(15)), sqrt(16 - 2*sqrt(29) + 2*sqrt(55 - 10*sqrt(29))),
            inv(sqrt(2) + 1 + sqrt(3)*sqrt(5)), sqrt(sqrt(2) + sqrt(3)), inv(x*sqrt(2) + y), (x + sqrt(2))**2/(1 - sqrt(3))]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer
    from sympy_extras.core.logic import true
    from sympy_extras.simplify.radsimp import radsimp
    from sympy_extras.simplify.sqrtdenest import sqrtdenest
    different = []
    for e, denesting in _cases(seed):
        if denesting:
            got, expected = _outcome(lambda: _call('sqrtdenest', e, Integer(3))), _outcome(lambda: sqrtdenest(e))
        else:
            got, expected = _outcome(lambda: _call('radsimp', e, true, Integer(4))), _outcome(lambda: radsimp(e))
        if got != expected:
            different.append((e, got, expected))
    assert not different, different[:3]


def test_symbolic(rust: Any) -> None:
    from sympy_extras.core import Integer
    from sympy_extras.core.logic import false, true
    from sympy_extras.simplify.radsimp import radsimp
    from sympy_extras.simplify.sqrtdenest import is_algebraic, sqrt_depth, sqrtdenest
    different = []
    for e in _symbolic():
        for name, got, expected in [
                ('radsimp', _outcome(lambda: _call('radsimp', e, true, Integer(4))), _outcome(lambda: radsimp(e))),
                ('radsimp_numeric', _outcome(lambda: _call('radsimp', e, false, Integer(4))),
                 _outcome(lambda: radsimp(e, symbolic=False))),
                ('radsimp_two', _outcome(lambda: _call('radsimp', e, true, Integer(2))), _outcome(lambda: radsimp(e, max_terms=2))),
                ('sqrtdenest', _outcome(lambda: _call('sqrtdenest', e, Integer(3))), _outcome(lambda: sqrtdenest(e))),
                ('sqrt_depth', _outcome(lambda: _call('sqrt_depth', e)), _outcome(lambda: sqrt_depth(e))),
                ('is_algebraic', _outcome(lambda: bool(_call('is_algebraic', e))), _outcome(lambda: is_algebraic(e)))]:
            if got != expected:
                different.append((name, e, got, expected))
    assert not different, different[:3]


def test_helpers(rust: Any) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import sqrt
    from sympy_extras.simplify.radsimp import rad_rationalize, split_surds, unevaluated_mul
    x = Symbol('x')
    s2, s3, s5, s6 = sqrt(Integer(2)), sqrt(Integer(3)), sqrt(Integer(5)), sqrt(Integer(6))
    for e in [3*s2 + s6 + 1, s2 + s3 + s6, 2*s6 + 3*s3 + s5 + 7, s2*x + s3 + 1, 5 + s5]:
        assert _outcome(lambda: tuple(str(t) for t in _call('split_surds', e).args)) == \
            _outcome(lambda: tuple(str(t) for t in split_surds(e))), e
    for num, den in [(Integer(1), 1 + s2), (x, s2 + s3), (Integer(1), 1 + s2 + s3), (Integer(2), s3), (x + 1, 2 - s5 + s6)]:
        assert _outcome(lambda: tuple(str(t) for t in _call('rad_rationalize', num, den).args)) == \
            _outcome(lambda: tuple(str(t) for t in rad_rationalize(num, den))), (num, den)
    for args in [(Integer(2), x, s2), (x + 1, Integer(3), x), (Integer(1), Integer(1)), (s3, 2*x, Integer(-1))]:
        assert str(_call('unevaluated_mul', *args)) == str(unevaluated_mul(*args)), args
