"""The CAD sample points of the Rust port against
``sympy_extras.polys.cad.samplepoints``: points built from real roots of
random univariate polynomials (rational and algebraic coordinates, the
field extended by primitive elements), the exact signs of polynomials at
them, the specializations in one more variable (degree, leading sign,
real roots with multiplicities, signs between them), and the standalone
functions (``floor_scaled``, ``rational_below``, ``rational_above``,
``rational_between``, ``compare_real``), compared through the reprs."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed


def python_report(coordinates: list[tuple[list[Fraction], int]], signs: list[Any], lifts: list[Any],
                  gens: list[Any]) -> dict[str, Any]:
    from sympy_extras.polys.cad import samplepoints as sp
    out: dict[str, Any] = {}
    point = sp.SamplePoint()
    roots: list[Any] = []
    for dense, index in coordinates:
        found = sp.real_roots_of(dense)
        out["roots%d" % len(roots)] = [repr(r) for r in found]
        r = found[index % max(len(found), 1)]
        roots.append(r)
        point = point.extend(r)
    out["point"] = repr(point)
    out["degree"] = point.degree
    n = len(point)

    def attempt(f: Any) -> str:
        try:
            return str(f())
        except Exception as exc:
            return 'ERROR ' + type(exc).__name__
    out["signs"] = [attempt(lambda: point.sign(f, gens[:n])) for f in signs]

    def lift(f: Any) -> str:
        s = point.specialization(f, gens[:n + 1])
        parts = ", ".join("(%r, %d)" % (r, m) for r, m in s.roots)
        before = ", ".join(str(s.sign_before(i)) for i in range(len(s.roots) + 1))
        return "%d %d [%s] [%s]" % (s.degree, s.leading_sign, parts, before)
    out["lifts"] = [attempt(lambda: lift(f)) for f in lifts]
    functions = []
    for r in roots:
        functions.append("%d %d %s %s %d" % (sp.floor_scaled(r, 0), sp.floor_scaled(r, 5), sp.rational_below(r),
                                             sp.rational_above(r), sp.compare_real(r, roots[0])))
    for a, b in zip(roots, roots[1:]):
        c = sp.compare_real(a, b)
        functions.append(str(sp.rational_between(a, b)) if c < 0 else
                         str(sp.rational_between(b, a)) if c > 0 else "equal")
    out["functions"] = functions
    return out


def random_dense(rng: random.Random) -> list[Fraction]:
    kind = rng.random()
    if kind < 0.3:
        return [Fraction(rng.randint(1, 3)), Fraction(rng.randint(-5, 5), rng.randint(1, 3))]
    if kind < 0.7:
        return [Fraction(1), Fraction(0), Fraction(-rng.choice([2, 3, 5, 6, 7]))]
    degree = rng.choice([2, 3])
    f = [Fraction(1)] + [Fraction(rng.randint(-4, 4)) for _ in range(degree)]
    return f


def random_polynomial(rng: random.Random, gens: list[Any]) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    terms = []
    for _ in range(rng.randint(1, 4)):
        c: Any = rng.choice([Integer(rng.randint(-3, 3)), Rational(rng.randint(-2, 2), 2)])
        terms.append(mul(c, *[pow(g, Integer(rng.randint(0, 2))) for g in gens if rng.random() < 0.6]))
    return add(*terms)


@pytest.mark.parametrize("seed", range(30))
def test_sample_points(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("cad samplepoints", seed))
    gens = [Symbol(c) for c in 'xyzw']
    coordinates: list[tuple[list[Fraction], int]] = []
    # two coordinates at most: the fields of three algebraic ones (degree
    # 12) make the norms of the specializations slow in the Python
    for _ in range(rng.randint(1, 2)):
        dense = random_dense(rng)
        while len(dense) > 1 and _no_real_roots(dense):
            dense = random_dense(rng)
        coordinates.append((dense, rng.randint(0, 3)))
    n = len(coordinates)
    signs = [random_polynomial(rng, gens[:n]) for _ in range(3)]
    lifts = [random_polynomial(rng, gens[:n + 1]) for _ in range(3)]
    expected = python_report(coordinates, signs, lifts, gens)
    pairs = [([(c.numerator, c.denominator) for c in dense], index) for dense, index in coordinates]
    got = dict(rc.samplepoint_report(pairs, signs, lifts, gens))
    for key, value in expected.items():
        assert got[key] == value, (key, coordinates, got[key], value)


def _no_real_roots(dense: list[Fraction]) -> bool:
    from sympy_extras.polys.cad import samplepoints as sp
    return not sp.real_roots_of(dense)
