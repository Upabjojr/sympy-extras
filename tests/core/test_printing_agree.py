"""The string printer of the Rust core (``sstr``, with the printing order
of the terms of a sum and the sort keys behind it) against ``str`` of the
Python core, on random evaluated expressions built by the Python
evaluating functions, and on unevaluated nodes."""

from __future__ import annotations

import importlib
import random
from typing import Any

import pytest

from ..conftest import load, stable_seed


def atoms() -> list[Any]:
    from sympy_extras.core.numbers import E, I, Integer, Rational, oo, pi, zoo
    from sympy_extras.core.numbers import Catalan, EulerGamma, GoldenRatio
    from sympy_extras.core.symbol import Dummy, Symbol, Wild
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(3), Integer(-4), Integer(12),
            Rational(1, 2), Rational(-2, 3), Rational(3, 4), pi, I, E, oo, -oo, zoo, EulerGamma, Catalan,
            GoldenRatio, Symbol('x'), Symbol('y'), Symbol('z'), Symbol('p', positive=True),
            Symbol('n', integer=True), Symbol('A', commutative=False), Symbol('B', commutative=False),
            Dummy('d'), Wild('w')]


def functions() -> dict[str, Any]:
    a = importlib.import_module("sympy_extras.core.add")
    m = importlib.import_module("sympy_extras.core.mul")
    p = importlib.import_module("sympy_extras.core.power")
    return {"add": a.add, "sub": a.sub, "mul": m.mul, "div": m.div, "neg": m.neg, "pow": p.pow}


def convertible(e: Any) -> bool:
    try:
        load("sympy_extras_rs.core").roundtrip(e)
    except (TypeError, NotImplementedError):
        return False
    return True


@pytest.mark.parametrize("seed", range(40))
def test_printing_agrees(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("printing", seed))
    pool = atoms()
    table = functions()
    arities = {"add": (2, 4), "sub": (2, 2), "mul": (2, 4), "div": (2, 2), "neg": (1, 1), "pow": (2, 2)}
    compared = 0
    for _ in range(40):
        name = rng.choice(sorted(table))
        low, high = arities[name]
        args = [rng.choice(pool) for _ in range(rng.randint(low, high))]
        if name == "pow" and rng.random() < 0.6:
            args[1] = rng.choice(atoms()[:10])
        try:
            e = table[name](*args)
        except (ZeroDivisionError, ValueError, TypeError, ArithmeticError, NotImplementedError):
            continue
        if not convertible(e):
            continue
        assert rc.sstr(e) == str(e), (name, args)
        compared += 1
        if e.count_ops() < 30:
            pool.append(e)
    assert compared >= 20


def test_unevaluated_nodes(rust: Any) -> None:
    from sympy_extras.core.add import Add
    from sympy_extras.core.mul import Mul
    from sympy_extras.core.numbers import Integer, Rational
    from sympy_extras.core.power import Pow
    from sympy_extras.core.symbol import Symbol
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    for e in [Add(y, x), Mul(y, x, Integer(2)), Pow(x, Rational(-1, 2)), Pow(Add(x, y), Integer(-1)),
              Mul(Integer(-1), Add(x, y)), Add(Integer(1), Mul(Integer(-1), x)), Pow(Integer(-2), Integer(3)),
              Mul(Rational(3, 1), x), Pow(Mul(Integer(2), x), Pow(y, Integer(2)))]:
        assert rc.sstr(e) == str(e), e
