"""The bridge between the Python core (``sympy_extras.core``) and the Rust
core (``sympy_extras_rs.core``), and the parts of the Rust core it reaches
(the structural facts, ``is_number``, the numeric values, the ``compare``
order, equality, free symbols) against the Python core, on random
unevaluated trees of numbers, constants, symbols with assumptions and
dummies combined by ``Add``, ``Mul`` and ``Pow``."""

from __future__ import annotations

import importlib
import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed


def core() -> Any:
    return load("sympy_extras_rs.core")


def atoms() -> list[Any]:
    from sympy_extras.core.numbers import E, I, Integer, Rational, oo, pi, zoo
    from sympy_extras.core.symbol import Symbol
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(-7), Integer(10**30), Rational(1, 2),
            Rational(-3, 4), Rational(5, 3), pi, E, I, oo, zoo, Symbol('x'), Symbol('y', positive=True),
            Symbol('n', integer=True), Symbol('r', real=True, nonzero=True), _DUMMY]


def _dummy() -> Any:
    from sympy_extras.core.symbol import Dummy
    return Dummy('d', real=True)


_DUMMY: Any = None


def random_tree(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core.add import Add
    from sympy_extras.core.mul import Mul
    from sympy_extras.core.power import Pow
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(atoms())
    kind = rng.choice(["add", "mul", "pow"])
    if kind == "pow":
        return Pow(random_tree(rng, depth - 1), random_tree(rng, depth - 1))
    args = [random_tree(rng, depth - 1) for _ in range(rng.randint(2, 3))]
    return Add(*args) if kind == "add" else Mul(*args)


def count(e: Any) -> int:
    return 1 + sum(count(a) for a in e.args)


def close(rust_value: Any, python_value: Any, prec: int) -> bool:
    if rust_value is None or python_value is None:
        return rust_value is None and python_value is None
    re, im = rust_value
    tolerance = Fraction(1, 2**(prec - 12))
    for exact, approx in ((re, python_value.real), (im, python_value.imag)):
        sign, man, exp, _ = approx._mpf_
        value = Fraction(int(man) * 2**int(exp)) if exp >= 0 else Fraction(int(man), 2**int(-exp))
        if sign:
            value = -value
        scale: Fraction = abs(value) if abs(value) > 1 else Fraction(1)
        if abs(exact - value) > tolerance * scale:
            return False
    return True


@pytest.mark.parametrize("seed", range(80))
def test_bridge_agrees(rust: Any, seed: int) -> None:
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = _dummy()
    from sympy_extras.core.assumptions import facts_of, is_number
    evalf = importlib.import_module("sympy_extras.core.evalf")
    rc = core()
    rng = random.Random(stable_seed("bridge", seed))
    for _ in range(10):
        a, b = random_tree(rng, 3), random_tree(rng, 3)
        back = rc.roundtrip(a)
        assert back == a and repr(back) == repr(a), a
        assert dict(rc.facts(a)) == dict(facts_of(a)), a
        assert rc.is_number(a) == is_number(a), a
        assert rc.compare(a, b) == a.compare(b), (a, b)
        assert rc.equal(a, b) == (a == b), (a, b)
        assert rc.count_nodes(a) == count(a), a
        assert set(rc.free_symbols(a)) == set(a.free_symbols), a
        prec = rng.choice([53, 100])
        try:
            expected = evalf.numeric_value(a, prec)
        except (ZeroDivisionError, ValueError, ArithmeticError):
            continue
        if expected is not None and any(abs(int(part._mpf_[2])) > 2**60 for part in (expected.real, expected.imag)
                                        if part._mpf_[1]):
            # beyond the machine-word exponents of dashu's floats: no value in Rust
            assert rc.numeric_value(a, prec) is None, a
            continue
        assert close(rc.numeric_value(a, prec), expected, prec), (a, rc.numeric_value(a, prec), expected)
