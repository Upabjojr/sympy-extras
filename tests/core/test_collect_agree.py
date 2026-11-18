"""``collect``, ``collect_dict`` and ``rcollect`` of
``sympy_extras.simplify.radsimp`` against their Rust port, on random sums
of products of powers, exponentials, sines and roots, with
``exact=False``, ``None`` and ``True``; the outputs compared as strings."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('collect', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', f())
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _cases(seed: int) -> list[tuple[Any, list[Any]]]:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.functions import exp, sin, sqrt
    x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
    n = Symbol('n', integer=True)
    rng = random.Random(seed)
    atoms = [x, y, a, b, x**2, x**n, exp(x), exp(2*x), sin(x), sqrt(x), 1/x, x*y, (x + 1)**2, Rational(1, 2),
             sqrt(Integer(2)), x**3*y, exp(x + 1)]
    patterns = [[x], [x, y], [exp(x)], [x**2], [a, b], [sin(x)], [x, exp(x)], [x*y], [x + 1], [sqrt(x)]]
    out = []
    for _ in range(60):
        e = add(*[mul(Integer(rng.randint(-3, 3)), *[rng.choice(atoms) for _ in range(rng.randint(1, 3))])
                  for _ in range(rng.randint(1, 5))])
        out.append((e, rng.choice(patterns)))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_collect(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.core.logic import false, true
    from sympy_extras.simplify.radsimp import collect, collect_dict, rcollect
    different = []
    for e, syms in _cases(seed):
        for exact, flag in [(False, false), (None, Symbol('None')), (True, true)]:
            checks = [
                ('collect', _outcome(lambda: str(_call('collect', e, Tuple(*syms), flag))),
                 _outcome(lambda: str(collect(e, syms, exact=exact)))),
                ('dict', _outcome(lambda: [(str(k), str(v)) for k, v in (p.args for p in _call('collect_dict', e, Tuple(*syms), flag).args)]),
                 _outcome(lambda: [(str(k), str(v)) for k, v in collect_dict(e, syms, exact=exact).items()]))]
            if exact is False:
                checks.append(('rcollect', _outcome(lambda: str(_call('rcollect', e, Tuple(*syms)))),
                               _outcome(lambda: str(rcollect(e, *syms)))))
            different += [(name, e, syms, exact, got, expected) for name, got, expected in checks if got != expected]
    assert not different, different[:3]


def _collectible(seed: int) -> list[tuple[Any, list[Any]]]:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.functions import exp, sin
    x, y, a, b, c = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b'), Symbol('c')
    rng = random.Random(seed)
    out = []
    for _ in range(40):
        base = rng.choice([x, exp(x), sin(x), x + 1, x*y])
        coefficient = [a, b, c, y, Integer(2), a*b, Integer(-3)]
        e = add(*[mul(rng.choice(coefficient), pow(base, Integer(rng.randint(0, 3))), rng.choice([Integer(1), y, a]))
                  for _ in range(rng.randint(2, 6))])
        out.append((e, [base] if rng.random() < 0.7 else [base, y]))
    return out


@pytest.mark.parametrize("seed", [4, 5])
def test_collectible(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.core.logic import false, true
    from sympy_extras.simplify.radsimp import collect, collect_dict
    different, changed = [], 0
    for e, syms in _collectible(seed):
        for exact, flag in [(False, false), (None, Symbol('None')), (True, true)]:
            got = _outcome(lambda: str(_call('collect', e, Tuple(*syms), flag)))
            expected = _outcome(lambda: str(collect(e, syms, exact=exact)))
            changed += expected[0] == 'ok' and expected[1] != str(e)
            got_d = _outcome(lambda: [(str(k), str(v)) for k, v in (p.args for p in _call('collect_dict', e, Tuple(*syms), flag).args)])
            expected_d = _outcome(lambda: [(str(k), str(v)) for k, v in collect_dict(e, syms, exact=exact).items()])
            if got != expected or got_d != expected_d:
                different.append((e, syms, exact, got, expected, got_d, expected_d))
    assert not different, different[:3]
    assert changed > 40
