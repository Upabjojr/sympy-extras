"""The Rust port of ``complexroots`` (complex root isolation and counting
by exact winding numbers, Routh-Hurwitz and Schur-Cohn counts) against
the Python module: the docstring examples on both implementations, the
isolating and refined rectangles of random products of quadratic and
linear factors (square-free and with multiplicities), and the counts in
random rectangles, half-planes and discs (roots on the boundary
included)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def cr(impl: str) -> Any:
    return load(impl + ".polys.complexroots")


def outcome(function: Any, *args: Any) -> Any:
    try:
        result = function(*args)
    except ValueError as e:
        return ('ValueError', str(e))
    return result


def show(value: Any) -> Any:
    """Rectangles by their repr (the classes differ)."""
    if isinstance(value, (list, tuple)):
        return type(value)(show(v) for v in value)
    if type(value).__name__ == 'Rectangle':
        return repr(value)
    return value


def test_known(impl: str) -> None:
    m = cr(impl)
    f = [F(1), F(0), F(0), F(-1)]
    assert repr(m.dup_isolate_complex_roots_sqf(f)) == '[Rectangle(-3, 3, 3/4, 3)]'
    assert repr(m.dup_isolate_complex_roots_sqf(f, F(1, 10))) == '[Rectangle(-33/64, -31/64, 109/128, 113/128)]'
    assert repr(m.dup_isolate_all_roots_sqf(f)) == '([(Fraction(0, 1), Fraction(2, 1))], [Rectangle(-3, 3, 3/4, 3)])'
    R = m.Rectangle
    assert m.dup_count_complex_roots(f, R(F(-2), F(2), F(-2), F(2))) == 3
    assert m.dup_count_complex_roots(f, R(F(-2), F(0), F(-2), F(2))) == 2
    r = R(F(-1), F(1, 2), F(1), F(3))
    assert (repr(r), r.width, r.height, repr(r.conjugate()), r.contains(F(0), F(2))) == (
        'Rectangle(-1, 1/2, 1, 3)', F(3, 2), 2, 'Rectangle(-1, 1/2, -3, -1)', True)
    assert r == R(F(-1), F(1, 2), F(1), F(3)) and hash(r) == hash(R(F(-1), F(1, 2), F(1), F(3)))
    with pytest.raises(ValueError):
        R(F(1), F(1), F(0), F(1))
    assert repr(m.dup_refine_complex_root([F(1), F(0), F(1)], R(F(-1), F(2), F(0), F(3)), F(1, 4))) == \
        'Rectangle(-1/64, 1/64, 63/64, 65/64)'
    assert repr(m.dup_isolate_complex_roots([F(c) for c in (1, 0, 3, 0, 3, 0, 1)])) == '[(Rectangle(-3, 3, 3/4, 3), 3)]'
    assert (m.dup_count_roots_right_of(f), m.dup_count_roots_right_of(f, F(-1)), m.dup_count_roots_right_of(f, F(2))) \
        == (1, 3, 0)
    g = [F(c) for c in (2, 0, 0, 0, 1)]
    assert (m.dup_count_roots_in_disc(g), m.dup_count_roots_in_disc(g, radius=F(1, 2))) == (4, 0)
    assert m.dup_count_roots_in_disc(g, (F(1), F(1)), F(1)) == 1
    assert m.dup_hurwitz_stable_p([F(c) for c in (1, 3, 3, 1)]) and not m.dup_hurwitz_stable_p([F(c) for c in (1, 1, 1, 6)])
    assert m.dup_schur_stable_p([F(c) for c in (4, 0, 1)]) and not m.dup_schur_stable_p([F(c) for c in (1, 0, 4)])
    with pytest.raises(ValueError):
        m.dup_isolate_complex_roots_sqf([F(1), F(0), F(1), F(0), F(0)])
    with pytest.raises(ValueError):
        m.dup_count_complex_roots([F(1), F(0), F(1)], R(F(-1), F(1), F(1), F(2)))   # i on the boundary


def multiply(f: list[Fraction], g: list[Fraction]) -> list[Fraction]:
    result = [F(0)]*(len(f) + len(g) - 1)
    for i, a in enumerate(f):
        for j, b in enumerate(g):
            result[i + j] += a*b
    return result


def random_polynomial(rng: random.Random, square_free: bool) -> list[Fraction]:
    """A product of quadratics with complex roots (x - a)**2 + b**2 and of
    linear factors, the coefficients small rationals."""
    f = [F(rng.randint(1, 3))]
    used: set[tuple[Fraction, ...]] = set()
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.7:
            a, b = F(rng.randint(-3, 3), rng.choice([1, 2])), F(rng.randint(1, 3), rng.choice([1, 2]))
            factor = [F(1), -2*a, a*a + b*b]
        else:
            factor = [F(1), F(rng.randint(-3, 3), rng.choice([1, 2]))]
        if square_free and tuple(factor) in used:
            continue
        used.add(tuple(factor))
        for _ in range(1 if square_free else rng.choice([1, 1, 2])):
            f = multiply(f, factor)
    return f


@pytest.mark.parametrize("seed", range(15))
def test_isolation_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("complex isolation", seed))
    for _ in range(3):
        f = random_polynomial(rng, True)
        g = random_polynomial(rng, False)
        eps = F(1, rng.choice([4, 10, 100]))
        results = []
        for impl in IMPLEMENTATIONS:
            m = cr(impl)
            results.append(show([outcome(m.dup_isolate_complex_roots_sqf, f), outcome(m.dup_isolate_complex_roots_sqf, f, eps),
                                 outcome(m.dup_isolate_all_roots_sqf, f, eps), outcome(m.dup_isolate_complex_roots, g),
                                 outcome(m.dup_isolate_complex_roots, g, eps),
                                 [outcome(m.dup_refine_complex_root, f, r, F(1, 1000))
                                  for r in m.dup_isolate_complex_roots_sqf(f)]]))
        assert results[0] == results[1], (f, g)


def small(rng: random.Random) -> Fraction:
    return F(rng.randint(-6, 6), rng.choice([1, 2, 3]))


@pytest.mark.parametrize("seed", range(15))
def test_counts_agree(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("complex counts", seed))
    for _ in range(4):
        f = random_polynomial(rng, rng.random() < 0.5)
        x0, y0 = small(rng), small(rng)
        rectangle = (x0, x0 + F(rng.randint(1, 6), rng.choice([1, 2])), y0, y0 + F(rng.randint(1, 6), rng.choice([1, 2])))
        a = small(rng)
        center, radius = (small(rng), small(rng)), F(rng.randint(1, 8), rng.choice([1, 2, 3]))
        results = []
        for impl in IMPLEMENTATIONS:
            m = cr(impl)
            R = m.Rectangle(*rectangle)
            count = outcome(m.dup_count_complex_roots, f, R)
            refined: Any = None
            if count == 1:
                refined = repr(outcome(m.dup_refine_complex_root, f, R, F(1, 50)))
            results.append((count, refined, outcome(m.dup_count_roots_right_of, f, a),
                            outcome(m.dup_count_roots_in_disc, f, center, radius), m.dup_hurwitz_stable_p(f),
                            m.dup_schur_stable_p(f)))
        assert results[0] == results[1], (f, rectangle, a, center, radius)
