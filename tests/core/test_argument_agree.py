"""``arg`` and ``atan2`` of the Rust core against
``sympy_extras.functions.elementary.complexes``: numbers, imaginary and
complex numbers, exponentials of comparable and symbolic angles, symbols
with signs, undefined functions, nested arguments, the infinities, and
random operations mixing them with the other functions (results by
structure, classes and reprs, facts, derivatives, conjugates, values)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, convertible, shape
from .test_bridge_agree import close
from .test_exponential_agree import angles
from .test_inverse_agree import special_arguments, table as inverse_table


def complexes() -> Any:
    import importlib
    return importlib.import_module("sympy_extras.functions.elementary.complexes")


def same(got: Any, expected: Any) -> bool:
    return bool(shape(got) == shape(expected) and repr(got) == repr(expected))


def arguments() -> list[Any]:
    from sympy_extras.core import Function, I, Integer, Rational, Symbol, oo, pi, sqrt, zoo
    from sympy_extras.functions import exp
    x, p, n, r = Symbol('x'), Symbol('p', positive=True), Symbol('n', negative=True), Symbol('r', real=True)
    f = Function('f')
    c = complexes()
    return [Integer(0), Integer(3), Integer(-3), Rational(-1, 2), I, -I, 2*I, 1 + I, -1 - I, 1 - sqrt(3)*I,
            -sqrt(2) + sqrt(2)*I, sqrt(2)*I, oo, -oo, zoo, pi, -pi, x, -x, p, n, r, p*I, n*I, -p*I, 2*x, x*I, p*x,
            n*x, f(x), p*f(x), exp(I*x), exp(I*pi/3), exp(7*I*pi/3), exp(-5*I*pi/4), exp(2 + 3*I), exp(p),
            exp(I*p), c.Arg(x), c.Arg(c.Arg(x)), c.Arg(c.Arg(r)), x + I, p + I, p - n*I, (1 + I)*p, -2*(1 - I)*x]


def test_arg(rust: Any) -> None:
    rc = load("sympy_extras_rs.core")
    c = complexes()
    for a in arguments():
        expected = c.arg(a)
        try:
            got = rc.arg(a)
        except NotImplementedError:
            continue
        assert same(got, expected), (a, got, expected)


def test_atan2(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, sqrt
    rc = load("sympy_extras_rs.core")
    c = complexes()
    x, p, n, r = Symbol('x'), Symbol('p', positive=True), Symbol('n', negative=True), Symbol('r', real=True)
    values = [Integer(0), Integer(1), Integer(-1), Integer(3), Rational(-2, 3), sqrt(3), -sqrt(3), oo, -oo, I, -2*I,
              1 + I, x, p, n, r, -p, p*I, n + 1, r**2 + 1]
    compared = 0
    for y in values:
        for xx in values:
            try:
                expected = c.atan2(y, xx)
            except (TypeError, ValueError):
                continue
            try:
                got = rc.atan2(y, xx)
            except NotImplementedError:
                continue
            assert same(got, expected), (y, xx, got, expected)
            compared += 1
    assert compared > 300


def test_derivatives_conjugates_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    rc = load("sympy_extras_rs.core")
    c = complexes()
    x, r = Symbol('x'), Symbol('r', real=True)
    nodes = [c.Arg(x), c.Arg(r), c.Arg(x**2 + I*x), c.Atan2(x, r), c.Atan2(r, x**2), c.Atan2(x, x), c.Atan2(r, r + 1)]
    for e in nodes:
        for got, expected in ((lambda: rc.diff(e, (x, 1)), lambda: diff(e, x)),
                              (lambda: rc.diff(e, (r, 1)), lambda: diff(e, r)),
                              (lambda: rc.conjugate(e), lambda: c.conjugate(e))):
            try:
                value = got()
            except NotImplementedError:
                continue
            assert same(value, expected()), e
        assert rc.facts(e) == dict(facts_of(e)), e
    points = [Integer(2), Integer(-2), I, -I, 1 + I, -1 - 2*I, -sqrt(7), Rational(-1, 3)*I]
    for z in points:
        for e in [c.Arg(z)] + [c.Atan2(z, w) for w in points]:
            for prec in (53, 120):
                assert close(rc.numeric_value(e, prec), numeric_value(e, prec), prec), (e, prec)


@pytest.mark.parametrize("seed", range(30))
def test_argument_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core.assumptions import facts_of
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("argument", seed))
    pool = atoms() + angles() + arguments() + special_arguments()[:30]
    operations = inverse_table()
    operations["arg"] = (complexes().arg, 1)
    operations["atan2"] = (complexes().atan2, 2)
    compared = 0
    for _ in range(30):
        name = rng.choice(["arg", "atan2"]) if rng.random() < 0.6 else rng.choice(sorted(operations))
        function, arity = operations[name]
        args = [rng.choice(pool) for _ in range(arity)]
        try:
            expected = function(*args)
        except (ZeroDivisionError, ValueError, TypeError, ArithmeticError, NotImplementedError):
            continue
        try:
            got = rc.log(*args) if name == "log_base" else getattr(rc, name)(*args)
        except NotImplementedError:
            continue
        compared += 1
        expected_parts = expected if isinstance(expected, tuple) else (expected,)
        got_parts = got if isinstance(got, tuple) else (got,)
        for g, e in zip(got_parts, expected_parts):
            assert same(g, e), (name, args, got, expected)
        for e in expected_parts:
            if convertible(e):
                assert rc.facts(e) == dict(facts_of(e)), e
                if e.count_ops() < 40:
                    pool.append(e)
    assert compared >= 8
