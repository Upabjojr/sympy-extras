"""The floats of the Rust core (``core/float.rs``, ``Head::Float``) against
those of the Python core: the round trip, the string form, the compare
order, the facts, the arithmetic of ``add``, ``mul``, ``pow``, ``sub``,
``div``, ``neg`` and ``number_pow`` on random operands with floats of
several precisions, and the LaTeX and pretty forms with the settings of
the floats. The powers which go through mpmath in Python (a float
exponent, a fractional power of a float) raise ``NotImplementedError`` in
Rust and are counted, not compared."""

from __future__ import annotations

import importlib
import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import atoms, operations, python_function


def shape(e: Any) -> Any:
    """The class names and atoms of a tree, with the precision of the
    floats (which their equality ignores)."""
    from sympy_extras.core.numbers import Float
    if isinstance(e, Float):
        return ('Float', e.as_fraction(), e.precision)
    if not e.args:
        return (type(e).__name__, repr(e))
    return (type(e).__name__, tuple(shape(a) for a in e.args))


def random_float(rng: random.Random) -> Any:
    from sympy_extras.core.numbers import Float
    precision = rng.choice([53, 53, 53, 20, 100, 333])
    kind = rng.random()
    if kind < 0.3:
        value: Any = Fraction(rng.randint(-999, 999), rng.choice([1, 2, 4, 8, 10, 100, 3, 7]))
    elif kind < 0.6:
        value = Fraction(rng.randint(-10**6, 10**6), 10**rng.randint(0, 9))
    elif kind < 0.8:
        # far from unit magnitude: the exponent notation
        value = Fraction(rng.randint(1, 999)) * Fraction(10) ** rng.randint(-40, 40) * rng.choice([1, -1])
    else:
        value = Fraction(rng.random()) * rng.choice([Fraction(1), Fraction(-1), Fraction(1000), Fraction(1, 10**7)])
    from sympy_extras.core.numbers import number_from_fraction
    return Float(number_from_fraction(value), precision=precision)


@pytest.mark.parametrize("seed", range(4))
def test_float_atoms_agree(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("float atoms", seed))
    floats = [random_float(rng) for _ in range(60)]
    for f in floats:
        back = rc.roundtrip(f)
        assert shape(back) == shape(f), (f, back)
        assert rc.sstr(f) == str(f), (f.as_fraction(), f.precision)
        facts = rc.facts(f)
        for name, value in facts.items():
            assert getattr(f, 'is_' + name) == value, (f, name)
    for a, b in zip(floats, floats[1:] + floats[:1]):
        assert rc.compare(a, b) == a.compare(b), (a, b)
        assert rc.equal(a, b) == (a == b)


@pytest.mark.parametrize("seed", range(30))
def test_float_arithmetic_agrees(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("float arithmetic", seed))
    pool = atoms() + [random_float(rng) for _ in range(8)]
    compared = skipped = 0
    for _ in range(40):
        name, arity = rng.choice(operations())
        args = [rng.choice(pool) for _ in range(arity)]
        if name == "pow" and rng.random() < 0.6:
            args[1] = rng.choice(atoms()[:8])     # mostly integer exponents
        if name == "pow" and getattr(args[1], 'is_Float', False) and abs(args[1].as_fraction()) > 1000:
            continue    # (the Python core hangs on huge float powers)
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
        assert shape(got) == shape(expected) and str(got) == str(expected), (name, args, got, expected)
        assert rc.sstr(expected) == str(expected), expected
        if expected.count_ops() < 30:
            pool.append(expected)
    assert compared >= 15, (compared, skipped)


def test_float_number_pow_agrees(rust: Any) -> None:
    from sympy_extras.core.numbers import Integer, Rational, nan, oo, zoo
    from sympy_extras.core.power import number_pow
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("float number pow"))
    values = [Integer(0), Integer(1), Integer(-1), oo, -oo, zoo, nan]
    compared = 0
    for _ in range(400):
        b = rng.choice(values + [random_float(rng), Rational(rng.randint(-50, 50), rng.randint(1, 12))])
        e = rng.choice(values + [random_float(rng), Integer(rng.randint(-7, 7))])
        if getattr(e, 'is_Float', False) and abs(e.as_fraction()) > 1000:
            # (the Python core hangs on huge float powers: the exact value of 16790.0**9.72e37)
            continue
        try:
            expected = number_pow(b, e)
        except ZeroDivisionError:
            continue
        try:
            got = rc.number_pow(b, e)
        except NotImplementedError:
            continue
        compared += 1
        assert shape(got) == shape(expected), (b, e, got, expected)
    assert compared > 200


LATEX_SETTINGS: list[dict[str, Any]] = [
    {}, {'full_prec': True}, {'decimal_separator': 'comma'}, {'min': -2, 'max': 3}, {'mul_symbol': 'times'},
    {'min': -20, 'max': 20, 'full_prec': True},
]
PRETTY_SETTINGS: list[dict[str, Any]] = [
    {'use_unicode': True}, {'use_unicode': False}, {'use_unicode': True, 'full_prec': True},
    {'use_unicode': True, 'full_prec': False},
]


@pytest.mark.parametrize("seed", range(4))
def test_float_printing_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.printing.latex import latex
    pretty = importlib.import_module('sympy_extras.printing.pretty.pretty').pretty
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("float printing", seed))
    x, y = Symbol('x'), Symbol('y')
    cases: list[Any] = []
    for _ in range(25):
        f, g = random_float(rng), random_float(rng)
        cases += [f, add(f, x), mul(f, x), mul(f, x, y), pow(x, f), pow(f, x), add(mul(f, x), g),
                  mul(f, pow(x, Integer_(-2))), pow(add(x, f), Integer_(2))]
    for e in cases:
        for s in LATEX_SETTINGS:
            assert rc.latex(e, s) == latex(e, **s), (e, s)
        for s in PRETTY_SETTINGS:
            assert rc.pretty(e, dict(s, wrap_line=False)) == pretty(e, wrap_line=False, **s), (e, s)


def Integer_(n: int) -> Any:
    from sympy_extras.core.numbers import Integer
    return Integer(n)
