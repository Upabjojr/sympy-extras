"""``rewrite`` of the elementary functions against its Rust port, for the
targets ``exp``, ``cos``, ``sin``, ``log``, ``tan``, ``cot``, ``sec``, ``csc``,
``sincos`` and the hyperbolic functions: random expressions of the
trigonometric, hyperbolic and inverse functions, exponentials, powers
and ``E``, with and without patterns."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import E, I, Integer, Rational, Symbol, pi
    from sympy_extras.functions import (acos, acosh, acot, acoth, acsc, acsch, asec, asech, asin, asinh, atan, atan2,
                                        atanh, cos, cosh, cot, coth, csc, csch, exp, log, sec, sech, sin, sinh, tan,
                                        tanh)
    x, y = Symbol('x'), Symbol('y')
    r = Symbol('r', positive=True)
    if depth == 0:
        return rng.choice([x, y, r, 2*x, x + 1, Integer(3), Rational(1, 2), pi/3, I*x, E])
    a, b = expression(rng, depth - 1), expression(rng, depth - 1)
    fs = [sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch, asin, acos, atan, acot, asec, acsc,
          asinh, acosh, atanh, acoth, asech, acsch, exp, log]
    builders = [lambda: rng.choice(fs)(a), lambda: a + b, lambda: a*b, lambda: a**b, lambda: Integer(2)**a,
                lambda: E**a, lambda: r**Rational(1, 3), lambda: atan2(a, b)]
    while True:
        try:
            return rng.choice(builders)()
        except (ValueError, TypeError, ZeroDivisionError):
            continue


def all_targets() -> list[tuple[str, Any]]:
    from sympy_extras.functions.elementary import hyperbolic as h, trigonometric as t
    from sympy_extras.functions.elementary.exponential import Exp, Log
    return [('exp', Exp), ('cos', t.Cos), ('sin', t.Sin), ('log', Log), ('tan', t.Tan), ('cot', t.Cot),
            ('sec', t.Sec), ('csc', t.Csc), ('sincos', 'sincos'), ('sinh', h.Sinh), ('cosh', h.Cosh),
            ('tanh', h.Tanh), ('coth', h.Coth), ('csch', h.Csch), ('sech', h.Sech)]


@pytest.mark.parametrize("seed", range(80))
def test_rewrite_elementary(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import cos, cosh, exp, sin, tan
    from sympy_extras.functions.elementary.exponential import Exp, Log
    from sympy_extras.functions.elementary.hyperbolic import Cosh
    from sympy_extras.functions.elementary.trigonometric import Cos, Sin, Tan
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("rewriting elementary", seed))
    x = Symbol('x')
    targets = all_targets()
    samples = [(Sin, sin(x)), (Cos, cos(x)), (Tan, tan(x)), (Cosh, cosh(x)), (Exp, exp(x))]
    for _ in range(4):
        e = expression(rng, rng.randint(1, 2))
        name, cls = rng.choice(targets)
        chosen = rng.sample(samples, rng.randint(1, 2)) if rng.random() < 0.3 else []
        got = outcome(lambda: rc.logic_call('rewrite', [e, Symbol(name), Integer(1), *[s for _, s in chosen]]))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: e.rewrite(*[c for c, _ in chosen], cls)), (e, name, chosen)
