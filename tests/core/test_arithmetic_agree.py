"""The evaluating functions of the Rust core (``add``, ``mul``, ``pow``,
``sub``, ``div``, ``neg``, ``number_pow``) against those of the Python
core, on random operands: numbers, constants, symbols with assumptions,
and the evaluated results of earlier steps, so that the operands are
canonical. The results are compared by their structure (classes included)
and their reprs; the cases which need a part of the Python core not
ported yet (``exp``, ``Abs``, ``Float``, ...) raise ``NotImplementedError``
in Rust and are counted, not compared."""

from __future__ import annotations

import random
from typing import Any, Callable, cast

import pytest

from ..conftest import load, stable_seed


def shape(e: Any) -> Any:
    """The class names and atoms of a tree (Python's srepr, roughly)."""
    if not e.args:
        return (type(e).__name__, repr(e), sorted(getattr(e, 'given_assumptions', {}).items())
                if hasattr(e, 'given_assumptions') and not callable(getattr(e, 'given_assumptions')) else None)
    return (type(e).__name__, tuple(shape(a) for a in e.args))


def atoms() -> list[Any]:
    from sympy_extras.core.numbers import E, I, Integer, Rational, oo, pi, zoo
    from sympy_extras.core.symbol import Symbol
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(3), Integer(-4), Integer(12), Integer(8),
            Rational(1, 2), Rational(-2, 3), Rational(3, 4), pi, I, oo, -oo, zoo, E,
            Symbol('x'), Symbol('y'), Symbol('p', positive=True), Symbol('n', integer=True),
            Symbol('r', real=True), Symbol('k', even=True), Symbol('c', commutative=False)]


def operations() -> list[tuple[str, int]]:
    return [("add", 2), ("add", 3), ("mul", 2), ("mul", 3), ("pow", 2), ("sub", 2), ("div", 2), ("neg", 1)]


def python_function(name: str) -> Callable[..., Any]:
    import importlib
    a = importlib.import_module("sympy_extras.core.add")
    m = importlib.import_module("sympy_extras.core.mul")
    p = importlib.import_module("sympy_extras.core.power")
    return {"add": a.add, "sub": a.sub, "mul": m.mul, "div": m.div, "neg": m.neg, "pow": p.pow}[name]


def convertible(e: Any) -> bool:
    try:
        load("sympy_extras_rs.core").roundtrip(e)
    except TypeError:
        return False
    return True


@pytest.mark.parametrize("seed", range(40))
def test_arithmetic_agrees(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("arithmetic", seed))
    pool = atoms()
    compared = skipped = 0
    for _ in range(30):
        name, arity = rng.choice(operations())
        args = [rng.choice(pool) for _ in range(arity)]
        if name == "pow" and rng.random() < 0.6:
            args[1] = rng.choice(atoms()[:12])     # mostly numerical exponents
        try:
            expected = python_function(name)(*args)
        except (ZeroDivisionError, ValueError, TypeError, ArithmeticError, NotImplementedError):
            continue
        try:
            got = getattr(rc, name)(*args)
        except NotImplementedError:
            skipped += 1
            continue
        compared += 1
        assert shape(got) == shape(expected) and repr(got) == repr(expected), (name, args, got, expected)
        if convertible(expected) and expected.count_ops() < 40:
            pool.append(expected)
    assert compared >= 15, (compared, skipped)


def test_number_pow_agrees(rust: Any) -> None:
    from sympy_extras.core.numbers import Integer, Rational, nan, oo, zoo
    from sympy_extras.core.power import number_pow
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("number pow"))
    values = [Integer(0), Integer(1), Integer(-1), oo, -oo, zoo, nan]
    for _ in range(400):
        b = rng.choice(values + [Rational(rng.randint(-50, 50), rng.randint(1, 12))])
        e = rng.choice(values + [Rational(rng.randint(-7, 7), rng.randint(1, 6))])
        try:
            expected = number_pow(cast(Any, b), cast(Any, e))
        except ZeroDivisionError:
            continue
        got = rc.number_pow(b, e)
        assert shape(got) == shape(expected) and repr(got) == repr(expected), (b, e, got, expected)
