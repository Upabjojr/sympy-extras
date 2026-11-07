"""The typed roots against SymPy (the oracle) on random products of
factors solvable without the cubic and quartic formulas, compared through
the printed dictionaries (the order of the roots included)."""
from __future__ import annotations

import random
from typing import Callable

import pytest
import sympy

from sympy_extras.core import to_sympy
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.polys.polyroots import roots

sx, sa = sympy.symbols('x a')


@pytest.mark.parametrize('seed', [1, 2])
def test_roots_match_sympy(seed: int) -> None:
    rng = random.Random(seed)
    pieces: list[Callable[[], sympy.Expr]] = [
        lambda: sx - rng.randint(-3, 3), lambda: 2 * sx + rng.choice([1, 3, -5]), lambda: sx**2 + rng.randint(1, 5),
        lambda: sx**2 - rng.choice([2, 3, 5]), lambda: sx**2 + sx + 1,
        lambda: sx**2 + rng.randint(-3, 3) * sx + rng.randint(-3, 3),
        lambda: sx**rng.choice([3, 4, 5, 6]) - rng.choice([1, 2, 8, -1, -16]), lambda: sx + sa, lambda: sx - sa,
        lambda: sx**2 - sa, lambda: sx**2 + sa * sx + 1, lambda: sx**4 + sx**3 + sx**2 + sx + 1,
        lambda: sx**4 - 2 * sx**2 - 1, lambda: (sx**2 + 1)**2 - 3]
    different: list[tuple[str, str, str]] = []
    for _ in range(50):
        e = sympy.expand(sympy.Mul(*[rng.choice(pieces)()**rng.choice([1, 1, 2]) for _ in range(rng.randint(1, 3))])
                         * rng.choice([1, 2, sympy.Rational(-1, 3)]))
        if e.is_number:
            continue
        try:
            ours = roots(expr_from_sympy(e), expr_from_sympy(sx))
        except NotImplementedError:
            continue  # an irreducible cubic or quartic factor (not ported yet)
        assert isinstance(ours, dict)
        printed = str({to_sympy(k, evaluate=False): v for k, v in ours.items()})
        if printed != str(sympy.roots(e, sx)):
            different.append((str(e), printed, str(sympy.roots(e, sx))))
    assert not different, different[:2]
