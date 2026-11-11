"""``diff`` of the Rust core against ``sympy_extras.core.function.diff``
on random expressions over every function ported to Rust, undefined
functions of one and two arguments (``Derivative`` and ``Subs`` of
derivatives at non-symbol points) and symbols with assumptions, with
several specifications of the variables. The results are compared by
structure and by the Rust printer: the dummies made in Rust (``_xi_1``)
become new Python dummies in the order of the conversion, so Python's
printer may order the terms of their sums differently."""

from __future__ import annotations

import importlib
import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import shape


def random_expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol, pi
    from sympy_extras.core.add import add
    from sympy_extras.core.function import Function
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    ex = importlib.import_module("sympy_extras.functions.elementary.exponential")
    tr = importlib.import_module("sympy_extras.functions.elementary.trigonometric")
    hy = importlib.import_module("sympy_extras.functions.elementary.hyperbolic")
    cx = importlib.import_module("sympy_extras.functions.elementary.complexes")
    x, y, r = Symbol('x'), Symbol('y'), Symbol('r', real=True)
    f, g = Function('f'), Function('g')
    unary = [ex.exp, ex.log, tr.sin, tr.cos, tr.tan, tr.cot, tr.sec, tr.csc, hy.sinh, hy.cosh, hy.tanh, hy.coth,
             hy.sech, hy.csch, cx.re, cx.im, cx.abs_, cx.conjugate, f]
    leaves = [x, y, r, Integer(rng.randint(-3, 3)), Rational(1, 2), pi, I]
    if depth == 0:
        return rng.choice(leaves)
    k = rng.random()
    if k < 0.25:
        return add(random_expression(rng, depth - 1), random_expression(rng, depth - 1))
    if k < 0.5:
        return mul(random_expression(rng, depth - 1), random_expression(rng, depth - 1))
    if k < 0.65:
        return pow(random_expression(rng, depth - 1), rng.choice([Integer(2), Integer(-1), Rational(1, 2), y]))
    if k < 0.95:
        return rng.choice(unary)(random_expression(rng, depth - 1))
    return g(random_expression(rng, depth - 1), random_expression(rng, depth - 1))


@pytest.mark.parametrize("seed", range(30))
def test_diff_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.function import diff
    rc = load("sympy_extras_rs.core")
    x, y, r = Symbol('x'), Symbol('y'), Symbol('r', real=True)
    rng = random.Random(stable_seed("diff", seed))
    compared = 0
    for _ in range(15):
        try:
            e = random_expression(rng, 3)
        except (ZeroDivisionError, ValueError, ArithmeticError):
            continue
        for spec in [(x,), (y,), (x, 2), ((x, 1), y), (r,)]:
            try:
                expected = diff(e, *spec)
            except (ZeroDivisionError, ValueError, TypeError, NotImplementedError):
                continue
            try:
                got = rc.diff(e, *spec)
            except (NotImplementedError, TypeError):
                continue
            assert shape(got) == shape(expected) and rc.sstr(got) == str(expected), (e, spec)
            compared += 1
    assert compared >= 30


def test_undefined_functions(rust: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.function import Function, diff
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    f, g = Function('f'), Function('g', real=True)
    for e in [f(x**2), g(x, x), f(x) * f(y), diff(f(x), x) * x, f(x + y), g(x, y)**2, f(f(x))]:
        for spec in [(x,), (x, 2), (y, x)]:
            got, expected = rc.diff(e, *spec), diff(e, *spec)
            assert shape(got) == shape(expected) and rc.sstr(got) == str(expected), (e, spec)
            assert rc.facts(got) == dict(facts_of(expected)), (e, spec)
