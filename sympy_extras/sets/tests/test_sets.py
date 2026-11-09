"""The sets against SymPy (the oracle): random unions, intersections and
complements of intervals and finite sets, with membership, infimum,
supremum and subsets. The cases where SymPy raises (its ``x in S`` on an
undecided membership) are skipped."""
from __future__ import annotations

import random
from typing import Union

import sympy
from sympy import Rational as R, oo

from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy
from sympy_extras import sets as N

Tree = tuple[object, ...]
x = sympy.Symbol('x')
p = sympy.Symbol('p', positive=True)
ENDS = [-oo, -2, -1, 0, R(1, 2), 1, 2, 3, oo]
ELEMENTS = [0, 1, 2, R(1, 2), -1, 5, x, p]


def _ours_expr(v: object) -> Expr:
    return expr_from_sympy(sympy.sympify(v))


def _random(rng: random.Random, depth: int) -> Tree:
    if depth == 0 or rng.random() < 0.35:
        if rng.random() < 0.65:
            a, b = sorted(rng.sample(range(len(ENDS)), 2))
            return ('I', ENDS[a], ENDS[b], rng.random() < 0.5, rng.random() < 0.5)
        return ('F', tuple(rng.choice(ELEMENTS) for _ in range(rng.randint(1, 3))))
    return (rng.choice('UNC'), _random(rng, depth - 1), _random(rng, depth - 1))


def _theirs(t: Tree) -> sympy.Set:
    if t[0] == 'I':
        return sympy.Interval(*t[1:])
    if t[0] == 'F':
        elements = t[1]
        assert isinstance(elements, tuple)
        return sympy.FiniteSet(*elements)
    op = {'U': sympy.Union, 'N': sympy.Intersection, 'C': sympy.Complement}[str(t[0])]
    left, right = t[1], t[2]
    assert isinstance(left, tuple) and isinstance(right, tuple)
    return op(_theirs(left), _theirs(right))


def _ours(t: Tree) -> N.Set:
    if t[0] == 'I':
        lo, ro = t[3], t[4]
        assert isinstance(lo, bool) and isinstance(ro, bool)
        return N.interval(_ours_expr(t[1]), _ours_expr(t[2]), lo, ro)
    if t[0] == 'F':
        elements = t[1]
        assert isinstance(elements, tuple)
        return N.finite_set(*[_ours_expr(v) for v in elements])
    left, right = t[1], t[2]
    assert isinstance(left, tuple) and isinstance(right, tuple)
    a, b = _ours(left), _ours(right)
    return N.union(a, b) if t[0] == 'U' else N.intersection(a, b) if t[0] == 'N' else N.complement(a, b)


def _str(s: Union[N.Set, Expr, object]) -> str:
    from sympy_extras.core.basic import Basic
    return str(to_sympy(s, evaluate=False)) if isinstance(s, Basic) else str(s)


def test_random_sets_match_sympy() -> None:
    rng = random.Random(1)
    checked = 0
    for _ in range(150):
        t = _random(rng, 3)
        try:
            theirs = _theirs(t)
        except TypeError:
            continue
        ours = _ours(t)
        assert _str(ours) == str(theirs), t
        for v in [x, R(1, 2), 0, 2, -oo, p]:
            try:
                want = str(theirs.contains(v))
            except TypeError:
                continue
            got = _str(ours.contains(_ours_expr(v)))
            if 'Ne(' in want and want.count('Eq(') and got == 'True':
                continue   # Eq(a, x) | Ne(a, x): our or_ decides it
            assert got == want, (t, v)
        for name in ('inf', 'sup'):
            try:
                want_bound = str(getattr(theirs, name))
            except NotImplementedError:
                want_bound = 'NotImplementedError'
            try:
                got_bound = _str(ours.inf if name == 'inf' else ours.sup)
            except NotImplementedError:
                got_bound = 'NotImplementedError'
            assert got_bound == want_bound, (t, name)
        u = _random(rng, 1)
        try:
            want_subset = theirs.is_subset(_theirs(u))
        except TypeError:
            continue
        assert ours.is_subset(_ours(u)) == want_subset, (t, u)
        checked += 1
    assert checked > 100


def test_comparisons_with_infinities() -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.logic import true
    from sympy_extras.core.numbers import oo as ooo
    q, r = Symbol('q', positive=True), Symbol('r', real=True)
    assert (q < ooo) is true and (-ooo < q) is true and (r < ooo) is true and (ooo > r) is true
    assert str(Symbol('y') < ooo) == 'y < oo'
