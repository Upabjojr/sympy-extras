"""The functions of ``polytools`` on expressions in the Rust port
(``cancel``, ``cancel_fraction``, ``gcd``, ``lcm``, ``gcd_list``,
``lcm_list``, ``invert``, ``resultant``, ``discriminant``, ``degree``,
``LC``) and ``signsimp`` against ``sympy_extras``, on random rational
functions and numbers, compared through ``str`` and the class of the
error. The Rust port raises ``NotImplementedError`` where the Python
goes through SymPy (``ZZ_I``, ``RR``, ``EX``); those cases are skipped."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed


def outcome(f: Callable[[], Any]) -> str:
    try:
        return str(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def random_polynomial(rng: random.Random, variables: list[Any]) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    terms = []
    for _ in range(rng.randint(1, 3)):
        c: Any = rng.choice([Integer(rng.randint(-4, 4)), Rational(rng.randint(-3, 3), rng.randint(1, 3))])
        terms.append(mul(c, *[pow(v, Integer(rng.randint(0, 2))) for v in variables if rng.random() < 0.6]))
    return add(*terms)


def compare(rc: Any, name: str, python: Callable[[], Any], args: list[Any], gens: list[Any]) -> bool:
    got = outcome(lambda: rc.polytools_call(name, args, gens))
    if got == 'SKIP':
        return False
    expected = outcome(python)
    assert got == expected, (name, args, gens)
    return True


@pytest.mark.parametrize("seed", range(40))
def test_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.core.numbers import Integer
    from sympy_extras.polys import polytools as pt
    from sympy_extras.simplify.signsimp import signsimp
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("polytools functions", seed))
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    compared = 0
    for _ in range(4):
        variables = rng.choice([[x], [x], [x, y], [x, a]])
        common = random_polynomial(rng, variables)
        f = mul(random_polynomial(rng, variables), common)
        g = mul(random_polynomial(rng, variables), common)
        h = mul(f, pow(g, Integer(-1)))
        gens = rng.choice([[], [x]])
        compared += compare(rc, "cancel", lambda: pt.cancel(h, *gens), [h], gens)
        compared += compare(rc, "cancel_fraction", lambda: pt.cancel_fraction(f, g), [f, g], [])
        compared += compare(rc, "gcd", lambda: pt.gcd(f, g, *gens), [f, g], gens)
        compared += compare(rc, "lcm", lambda: pt.lcm(f, g, *gens), [f, g], gens)
        compared += compare(rc, "gcd_list", lambda: pt.gcd_list([f, g, common], *gens), [f, g, common], gens)
        compared += compare(rc, "lcm_list", lambda: pt.lcm_list([f, g, common], *gens), [f, g, common], gens)
        compared += compare(rc, "invert", lambda: pt.invert(f, g), [f, g], [])
        compared += compare(rc, "resultant", lambda: pt.resultant(f, g, *gens), [f, g], gens)
        compared += compare(rc, "discriminant", lambda: pt.discriminant(f, *gens), [f], gens)
        compared += compare(rc, "LC", lambda: pt.LC(f, *gens), [f], gens)
        compared += compare(rc, "degree", lambda: pt.degree(f, *gens[:1]), [f], gens)
        s = mul(random_polynomial(rng, variables), pow(mul(Integer(-1), common), Integer(rng.choice([-1, 1, 2]))))
        compared += compare(rc, "signsimp", lambda: signsimp(mul(s, common)), [mul(s, common)], [])
        compared += compare(rc, "signsimp_unevaluated", lambda: signsimp(s, evaluate=False), [s], [])
    assert compared >= 20


def test_numbers(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    from sympy_extras.polys import polytools as pt
    rc = load("sympy_extras_rs.core")
    x = Symbol('x')
    pairs: list[tuple[Any, Any]] = [
        (Integer(3), Integer(7)), (Integer(-3), Integer(7)), (Integer(3), Integer(-7)), (Integer(4), Integer(6)),
        (Integer(5), Integer(1)), (Integer(5), Integer(0)), (Rational(1, 2), Integer(3)),
        (Rational(2, 3), Rational(4, 9)), (I, Integer(2)), (sqrt(2), Integer(2)), (x**2 + 1, x**3),
        (x**2 - 1, x + 1), (x, Integer(0)), (Integer(0), x),
    ]
    for f, g in pairs:
        compare(rc, "gcd", lambda: pt.gcd(f, g), [f, g], [])
        compare(rc, "lcm", lambda: pt.lcm(f, g), [f, g], [])
        compare(rc, "invert", lambda: pt.invert(f, g), [f, g], [])
        compare(rc, "cancel_fraction", lambda: pt.cancel_fraction(f, g), [f, g], [])
    for e in [Integer(3), x, (x**2 - 1)/(x - 1), (x*y_ if (y_ := Symbol('y')) else x)/(x*y_ + x),
              (2*x + 2)/(x**2 - 1), 1/x + 1/(x + 1), sqrt(2)*x/(sqrt(2)*x**2), (x - 1)/(1 - x)]:
        compare(rc, "cancel", lambda: pt.cancel(e), [e], [])
        compare(rc, "cancel", lambda: pt.cancel(e, x), [e], [x])
    assert compare(rc, "gcd_list", lambda: pt.gcd_list([]), [], [])
    assert compare(rc, "lcm_list", lambda: pt.lcm_list([]), [], [])


@pytest.mark.parametrize("seed", range(20))
def test_gaussian_cancel(rust: Any, seed: int) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.polys import polytools as pt
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("gaussian cancel", seed))
    x, y = Symbol('x'), Symbol('y')

    def gaussian_polynomial(variables: list[Any]) -> Any:
        terms = []
        for _ in range(rng.randint(1, 3)):
            c: Any = add(Integer(rng.randint(-3, 3)), mul(rng.choice([Integer(rng.randint(-3, 3)), Rational(1, 2)]), I))
            terms.append(mul(c, *[pow(v, Integer(rng.randint(0, 2))) for v in variables if rng.random() < 0.6]))
        return add(*terms)

    compared = 0
    for _ in range(4):
        variables = rng.choice([[x], [x], [x, y]])
        common = gaussian_polynomial(variables)
        e = mul(gaussian_polynomial(variables), common, pow(mul(gaussian_polynomial(variables), common), Integer(-1)))
        compared += compare(rc, "cancel", lambda: pt.cancel(e), [e], [])
    assert compared >= 2
