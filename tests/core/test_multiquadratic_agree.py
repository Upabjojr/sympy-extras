"""``sympy_extras.polys.domains.multiquadratic`` against its Rust port:
random elements of ``QQ(sqrt(d_1), ..., sqrt(d_k))`` (``d_i`` among -1
and small primes) compared on the arithmetic, inverses, exact quotients,
conjugates, sizes and the printed form."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed


def to_core(a: dict[int, Fraction]) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.containers import Tuple
    return Tuple(*[Tuple(Integer(m), Rational(c.numerator, c.denominator)) for m, c in sorted(a.items())])


def result(f: Any) -> str:
    try:
        return str(f())
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


@pytest.mark.parametrize("seed", range(60))
def test_multiquadratic(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.polys.domains import QQ
    from sympy_extras.polys.domains.multiquadratic import MultiquadraticField
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("multiquadratic", seed))
    radicands = rng.sample([-1, 2, 3, 5, 7], rng.randint(1, 3))
    F = MultiquadraticField(QQ, radicands)
    k = len(radicands)

    def element() -> dict[int, Fraction]:
        a: dict[int, Fraction] = {}
        for mask in range(2**k):
            if rng.random() < 0.5:
                c = Fraction(rng.randint(-4, 4), rng.randint(1, 3))
                if c:
                    a[mask] = c
        return a

    def rust(op: str, *args: Any) -> str:
        return result(lambda: rc.logic_call('multiquadratic', [Tuple(*map(Integer, radicands)), Symbol(op), *args]))

    for _ in range(5):
        a, b = element(), element()
        assert rust('name', to_core(a)) == result(lambda: Symbol(F.name))
        assert rust('to_str', to_core(a)) == result(lambda: Symbol(F.to_str(a))), a
        assert rust('size', to_core(a)) == result(lambda: Integer(F.size(a)))
        assert rust('neg', to_core(a)) == result(lambda: to_core(F.neg(a)))
        assert rust('invert', to_core(a)) == result(lambda: to_core(F.invert(a))), a
        for op in ('add', 'sub', 'mul', 'exquo'):
            assert rust(op, to_core(a), to_core(b)) == result(lambda: to_core(getattr(F, op)(a, b))), (op, a, b)
        i = rng.randrange(k)
        assert rust('conjugate', to_core(a), Integer(i)) == result(lambda: to_core(F.conjugate(a, i)))
        inverse = F.invert(a) if a else None
        if inverse is not None:
            assert F.mul(a, inverse) == F.one


def test_invalid_radicands(rust: Any) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.polys.domains import QQ
    from sympy_extras.polys.domains.multiquadratic import MultiquadraticField
    rc = load("sympy_extras_rs.core")
    for radicands in ([2, 2], [0, 3], [1], [-1, 5, -1]):
        got = result(lambda: rc.logic_call('multiquadratic', [Tuple(*map(Integer, radicands)), Symbol('name'),
                                                            Tuple()]))
        assert got == result(lambda: MultiquadraticField(QQ, radicands)), radicands
