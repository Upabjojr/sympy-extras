"""``sympy_extras.polys.domains.expressiondomain`` (``EX``, ``EXR``)
against its Rust port: the arithmetic, its normal form and the
predicates on random rational functions of symbols, square roots, ``pi``
and ``exp(a)``; the outputs compared as strings."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def _call(op: str, rationalized: bool, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    from sympy_extras.core.logic import false, true
    return load("sympy_extras_rs.core").logic_call('ex_domain', [Symbol(op), true if rationalized else false, *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _elements(seed: int) -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol, pi
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.functions import exp, sqrt
    a, b = Symbol('a'), Symbol('b')
    rng = random.Random(seed)
    atoms = [a, b, sqrt(Integer(2)), sqrt(Integer(3)), pi, exp(a), Integer(2), Rational(-1, 3), a + 1, a - b]
    out = []
    for _ in range(30):
        terms = [mul(Integer(rng.randint(-3, 3)), *[pow(rng.choice(atoms), Integer(rng.choice([1, 1, 2, -1])))
                                                    for _ in range(rng.randint(1, 2))]) for _ in range(rng.randint(1, 3))]
        out.append(add(*terms))
    return out


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("rationalized", [False, True])
def test_arithmetic(rust: Any, seed: int, rationalized: bool) -> None:
    from sympy_extras.core import Integer
    from sympy_extras.polys.domains.expressiondomain import EX, EXR
    K = EXR if rationalized else EX
    xs = _elements(seed)
    rng = random.Random(seed + 10)
    different = []
    for _ in range(40):
        x, y = rng.choice(xs), rng.choice(xs)
        n = rng.randint(0, 3)
        for op, args, f in [('add', (x, y), lambda: K.add(x, y)), ('sub', (x, y), lambda: K.sub(x, y)),
                            ('mul', (x, y), lambda: K.mul(x, y)), ('neg', (x,), lambda: K.neg(x)),
                            ('exquo', (x, y), lambda: K.exquo(x, y)), ('invert', (x,), lambda: K.invert(x)),
                            ('pow', (x, Integer(n)), lambda: K.pow(x, n)), ('lcm', (x, y), lambda: K.lcm(x, y)),
                            ('gcdex', (x, y), lambda: tuple(K.gcdex(x, y))), ('is_zero', (x,), lambda: K.is_zero(x)),
                            ('is_negative', (x,), lambda: K.is_negative(x)), ('normal', (x,), lambda: K.normal(x)),
                            ('canonical_unit', (x,), lambda: K.canonical_unit(x))]:
            got = _outcome(lambda: _call(op, rationalized, *args))
            expected = _outcome(f)
            if op == 'gcdex' and got[0] == 'ok':
                got = ('ok', str(tuple(_call(op, rationalized, *args).args)))
            if op in ('is_zero', 'is_negative') and got[0] == 'ok':
                got = ('ok', str(bool(_call(op, rationalized, *args))))
            if got != expected:
                different.append((op, args, got, expected))
    assert not different, different[:3]


def test_examples(rust: Any) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import sqrt
    a = Symbol('a')
    assert str(_call('add', False, 1/a, a)) == 'a + 1/a'
    assert str(_call('mul', False, sqrt(Integer(2)), sqrt(Integer(2))/a)) == '2/a'
    assert str(_call('exquo', False, a**2 - 1, a + 1)) == 'a - 1'
    assert bool(_call('is_zero', False, _call('sub', False, sqrt(Integer(2))*sqrt(Integer(3)), sqrt(Integer(6)))))
    assert str(_call('invert', True, 3 + 2*sqrt(Integer(2)))) == '3 - 2*sqrt(2)'
