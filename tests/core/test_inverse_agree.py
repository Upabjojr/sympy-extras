"""The inverse trigonometric (``asin``, ``acos``, ``atan``, ``acot``,
``asec``, ``acsc``) and inverse hyperbolic (``asinh``, ``acosh``,
``atanh``, ``acoth``, ``asech``, ``acsch``) functions of the Rust core
against ``sympy_extras.functions.elementary``: the numbers, the
infinities, the entries of SymPy's tables and their negatives, imaginary
arguments, the inverse of a direct function of a comparable angle, and
random operations mixing them with the direct functions (results by
structure, classes and reprs, facts, derivatives, conjugates and
numerical values on and off the branch cuts)."""

from __future__ import annotations

import importlib
import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, convertible, shape
from .test_bridge_agree import close
from .test_exponential_agree import angles
from .test_hyperbolic_agree import table as hyperbolic_table
from .test_trigonometric_agree import multiples_of_pi

TRIGONOMETRIC = ("asin", "acos", "atan", "acot", "asec", "acsc")
HYPERBOLIC = ("asinh", "acosh", "atanh", "acoth", "asech", "acsch")


def module(name: str) -> Any:
    return importlib.import_module("sympy_extras.functions.elementary.%s" % name)


def python_function(name: str) -> Callable[[Any], Any]:
    return getattr(module("trigonometric" if name in TRIGONOMETRIC else "hyperbolic"), name)


def table() -> dict[str, tuple[Callable[..., Any], int]]:
    result = hyperbolic_table()
    for name in TRIGONOMETRIC + HYPERBOLIC:
        result[name] = (python_function(name), 1)
    return result


def agree(rc: Any, name: str, a: Any) -> bool:
    try:
        expected = python_function(name)(a)
    except (TypeError, ValueError, NotImplementedError):
        return False
    try:
        got = getattr(rc, name)(a)
    except NotImplementedError:
        return False
    assert shape(got) == shape(expected) and repr(got) == repr(expected), (name, a, got, expected)
    return True


def special_arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, nan, oo, pi, sqrt, zoo
    tr, hy = module("trigonometric"), module("hyperbolic")
    x = Symbol('x')
    s2, s3, s5, s6 = sqrt(2), sqrt(3), sqrt(5), sqrt(6)
    values: list[Any] = [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(-2), Rational(1, 2), Rational(-3, 2),
                         oo, -oo, zoo, nan, I, -I, 2*I, I*oo, -I*oo, x, -x, I*x, -I*x, x + 1, s2, s3/2, s2/2, 1/s2,
                         s3, s3/3, 1/s3, s2 - 1, 1 - s2, 2 - s3, s6/4 + s2/4, (s5 - 1)/4, (1 - s5)/4, 2*s3/3,
                         s6 - s2, 1 + s5, 2/s3, (1 + s3)/(2*s2), I*(s2 + s6), I*s2, sqrt(2 - 2/s5), 1 + 2*I]
    values += [-v for v in values[17:]]
    for angle in [pi/5, 3*pi/4, 7*pi/6, -2*pi/3, Rational(1, 3), Integer(4), 11*pi/5]:
        values += [tr.sin(angle), tr.cos(angle), tr.tan(angle), tr.cot(angle), tr.sec(angle), tr.csc(angle),
                   -tr.cos(angle), -tr.sec(angle)]
    values += [hy.sinh(Integer(2)), hy.tanh(Rational(1, 2)), hy.sinh(x), hy.tanh(I)]
    return values


def test_special_arguments(rust: Any) -> None:
    rc = load("sympy_extras_rs.core")
    compared = 0
    for a in special_arguments():
        for name in TRIGONOMETRIC + HYPERBOLIC:
            compared += agree(rc, name, a)
    assert compared > 500


def test_derivatives_conjugates_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    rc = load("sympy_extras_rs.core")
    x, r = Symbol('x'), Symbol('r', real=True)
    compared = 0
    for name in TRIGONOMETRIC + HYPERBOLIC:
        f = python_function(name)
        for a in [x, r, x**2 + 1, 2*x]:
            e = f(a)
            for got, expected in ((lambda: rc.diff(e, (x, 1)), lambda: diff(e, x)),
                                  (lambda: rc.conjugate(e), lambda: conjugate(e))):
                try:
                    value = got()
                except NotImplementedError:
                    continue
                assert repr(value) == repr(expected()), (name, a)
                compared += 1
        points = [Integer(v) for v in (-3, -2, 2, 3)] + [Rational(v, 3) for v in (-2, -1, 1, 2)] + \
            [v*I for v in (Integer(-3), Integer(-2), Rational(-1, 2), Rational(1, 2), Integer(2), Integer(3))] + \
            [1 + 2*I, -1 + 2*I, 1 - 2*I, -1 - 2*I, sqrt(7)]
        for p in points:
            # the unevaluated node, so that the value is the function's own
            e = getattr(module("trigonometric" if name in TRIGONOMETRIC else "hyperbolic"), name.capitalize())(p)
            for prec in (53, 120):
                assert close(rc.numeric_value(e, prec), numeric_value(e, prec), prec), (name, p, prec)
    assert compared >= 40


@pytest.mark.parametrize("seed", range(40))
def test_inverse_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core.assumptions import facts_of
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("inverse", seed))
    pool = atoms() + angles() + multiples_of_pi() + special_arguments()[:40]
    operations = table()
    names = list(TRIGONOMETRIC + HYPERBOLIC)
    compared = 0
    for _ in range(30):
        name = rng.choice(names) if rng.random() < 0.6 else rng.choice(sorted(operations))
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
            assert shape(g) == shape(e) and repr(g) == repr(e), (name, args, got, expected)
        for e in expected_parts:
            if convertible(e):
                assert rc.facts(e) == dict(facts_of(e)), e
                if e.count_ops() < 40:
                    pool.append(e)
    assert compared >= 10
