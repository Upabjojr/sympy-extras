"""``floor``, ``ceiling`` and ``frac`` of the Rust core against
``sympy_extras.functions.elementary.integers``: random sums of rational
numbers, constants, radicals, imaginary terms and symbols with
assumptions, compared by structure, facts, values, derivatives and
conjugates; and the powers of powers whose branch needs ``floor``
(``Pow._eval_power`` with a half-integer exponent, and with a complex
exponent)."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close


def tree(e: Any) -> Any:
    args = getattr(e, 'args', None)
    if not args:
        return repr(e)
    return (type(e).__name__, tuple(tree(a) for a in args))


def outcome(f: Callable[[], Any]) -> Any:
    try:
        return tree(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def terms() -> list[Any]:
    from sympy_extras.core import E, I, Integer, Rational, Symbol, pi, sqrt
    x, n, p = Symbol('x', real=True), Symbol('n', integer=True), Symbol('p', positive=True)
    z = Symbol('z')
    return [Integer(3), Integer(-2), Rational(7, 3), Rational(-5, 2), Rational(1, 2), sqrt(2), -sqrt(3)/2, pi,
            pi/7, E, I, 2*I, I*Rational(3, 2), I*sqrt(2), x, x/3, n, p, I*x, z, 1/(1 + sqrt(2))]


def random_argument(rng: random.Random) -> Any:
    from sympy_extras.core import Integer
    total: Any = Integer(0)
    for t in rng.sample(terms(), rng.randint(1, 3)):
        total = total + t
    return total


@pytest.mark.parametrize("seed", range(40))
def test_rounding(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary import integers
    from sympy_extras.functions.elementary.complexes import conjugate
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("integers", seed))
    x = Symbol('x', real=True)
    for _ in range(10):
        name = rng.choice(['floor', 'ceiling', 'frac'])
        arg = random_argument(rng)
        expected = outcome(lambda: getattr(integers, name)(arg))
        got = outcome(lambda: getattr(rc, name)(arg))
        if got == 'SKIP':
            continue
        assert got == expected, (name, arg)
        if isinstance(expected, str) and expected.startswith('ERROR'):
            continue
        e = getattr(integers, name)(arg)
        r = getattr(rc, name)(arg)
        assert rc.facts(r) == dict(facts_of(e)), e
        if not e.free_symbols:
            assert close(rc.numeric_value(r, 53), numeric_value(e, 53), 53), e
        assert outcome(lambda: rc.diff(r, (x, 1))) == outcome(lambda: diff(e, x)), e
        assert outcome(lambda: rc.conjugate(r)) == outcome(lambda: conjugate(e)), e


def test_powers_of_powers(rust: Any) -> None:
    from sympy_extras.core import I, Rational, Symbol
    from sympy_extras.core.power import pow
    rc = load("sympy_extras_rs.core")
    x, n, z = Symbol('x', real=True), Symbol('n', negative=True), Symbol('z')
    bases: list[Any] = [x, n, z, 1 + I, -1 + 2*I, x - 1, z**2, x + I]
    exponents: list[Any] = [3, Rational(5, 2), -3, Rational(2, 3), 2, Rational(-7, 3), I, 1 + I, I/2, 2 - 3*I]
    for b in bases:
        for e in exponents:
            inner = pow(b, e)
            for expt in [Rational(1, 2), Rational(3, 2), Rational(-1, 2), Rational(1, 3)]:
                assert outcome(lambda: rc.pow(inner, expt)) == outcome(lambda: pow(inner, expt)), (inner, expt)
