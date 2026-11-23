"""Different expressions never print as the same picture.

A reader of pictures is hard to write; instead all the trees of small
grammars of *unevaluated* nodes (and seeded random larger ones) are
printed and two trees with the same picture must have the same value at
random points. The values come from an evaluator written here, which
knows nothing of the printer. This is what catches a missing bracket:
``-(x**2)`` against ``(-x)**2``, ``(a/b)/c`` against ``a/(b/c)``,
``(2/3)**x`` against ``2/3**x``, ``¬(p ∧ q)`` against ``¬p ∧ q``.
"""
from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from sympy_extras.core import (Add, And, Equivalent, Implies, Integer, Mul, Not, Or, Pow, Rational, Symbol, Xor)
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean
from sympy_extras.core.mod import Mod
from sympy_extras.core.numbers import Float
from sympy_extras.core.relational import Eq, Lt
from sympy_extras.functions.combinatorial.factorials import Factorial
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.functions.elementary.trigonometric import Sin
from sympy_extras.printing.pretty import pretty
from sympy_extras.sets.sets import Complement, FiniteSet, Intersection, Interval, ProductSet, Set, Union

Point = Sequence[complex]
Value = Callable[[Point], complex]


@dataclass(frozen=True)
class Tree:
    """A node with the function computing its value."""

    node: Expr
    value: Value


def _constant(c: complex) -> Value:
    return lambda point: c


def _variable(i: int) -> Value:
    return lambda point: point[i]


def _add(a: Tree, b: Tree) -> Tree:
    return Tree(Add(a.node, b.node), lambda point: a.value(point) + b.value(point))


def _mul(a: Tree, b: Tree) -> Tree:
    return Tree(Mul(a.node, b.node), lambda point: a.value(point)*b.value(point))


def _pow(a: Tree, b: Tree) -> Tree:
    return Tree(Pow(a.node, b.node), lambda point: complex(a.value(point))**complex(b.value(point)))


def _sin(a: Tree) -> Tree:
    return Tree(Sin(a.node), lambda point: cmath.sin(a.value(point)))


def _exp(a: Tree) -> Tree:
    return Tree(Exp(a.node), lambda point: cmath.exp(a.value(point)))


def _factorial(a: Tree) -> Tree:
    def value(point: Point) -> complex:
        v = a.value(point)
        if abs(v.imag) > 1e-12 or v.real < 0 or abs(v.real - round(v.real)) > 1e-12 or v.real > 20:
            raise ValueError('not a small natural number')
        return complex(math.factorial(round(v.real)))
    return Tree(Factorial(a.node), value)


def _mod(a: Tree, b: Tree) -> Tree:
    def value(point: Point) -> complex:
        u, v = a.value(point), b.value(point)
        if abs(u.imag) > 1e-12 or abs(v.imag) > 1e-12:
            raise ValueError('not real')
        return complex(math.fmod(math.fmod(u.real, v.real) + v.real, v.real))
    return Tree(Mod(a.node, b.node), value)


x, y = Symbol('x'), Symbol('y')
LEAVES = [Tree(x, _variable(0)), Tree(Integer(2), _constant(2)), Tree(Integer(-1), _constant(-1)),
          Tree(Rational(1, 2), _constant(0.5))]
MORE_LEAVES = LEAVES + [Tree(y, _variable(1)), Tree(Rational(-2, 3), _constant(-2/3)),
                        Tree(Float(-2.5), _constant(-2.5)), Tree(Integer(3), _constant(3)),
                        Tree(Rational(-1, 2), _constant(-0.5))]
BINARY = [_add, _mul, _pow]
POINTS: list[Point] = [(1.3 + 0.4j, 0.6 - 0.8j), (0.37 - 1.1j, 2.2 + 0.3j), (-0.8 + 0.25j, 1.4 + 1.7j)]
REAL_POINTS: list[Point] = [(1.7 + 0j, 0.6 + 0j), (3.0 + 0j, 2.0 + 0j), (0.4 + 0j, 5.0 + 0j)]


def _values(tree: Tree, points: Sequence[Point]) -> list[Optional[complex]]:
    values: list[Optional[complex]] = []
    for point in points:
        try:
            v = complex(tree.value(point))
        except (ZeroDivisionError, OverflowError, ValueError):
            values.append(None)
            continue
        values.append(None if cmath.isnan(v) or cmath.isinf(v) or abs(v) > 1e12 else v)
    return values


def _same(a: Sequence[Optional[complex]], b: Sequence[Optional[complex]]) -> bool:
    for u, v in zip(a, b):
        if u is None or v is None:
            continue
        if abs(u - v) > 1e-7*(1 + abs(u) + abs(v)):
            return False
    return True


def _collisions(trees: Sequence[Tree], points: Sequence[Point], order: Optional[str] = None,
                modes: Sequence[bool] = (True, False)) -> list[str]:
    """The pairs of trees with the same picture (in Unicode, in ASCII)
    and different values."""
    seen: dict[str, tuple[Tree, list[Optional[complex]]]] = {}
    found: list[str] = []
    for tree in trees:
        for unicode in modes:
            picture = ('u' if unicode else 'a') + pretty(tree.node, use_unicode=unicode, wrap_line=False, order=order)
            values = _values(tree, points)
            if picture not in seen:
                seen[picture] = (tree, values)
            elif not _same(seen[picture][1], values):
                found.append('%r and %r both print as\n%s' % (seen[picture][0].node, tree.node, picture[1:]))
    return found


def test_all_small_trees_print_differently() -> None:
    # every tree of depth two over x, 2, -1, 1/2 with +, *, **: 8164 trees
    depth1 = LEAVES + [op(a, b) for op in BINARY for a in LEAVES for b in LEAVES]
    depth2 = depth1 + [op(a, b) for op in BINARY for a in depth1 for b in depth1]
    assert len(depth2) == 8164
    assert _collisions(depth2, POINTS, modes=(True,)) == []
    assert _collisions(depth2[::3], POINTS, modes=(False,)) == []


def test_small_trees_in_the_order_of_the_arguments() -> None:
    depth1 = LEAVES + [op(a, b) for op in BINARY for a in LEAVES for b in LEAVES]
    rng = random.Random(30844)
    depth2 = [rng.choice(BINARY)(rng.choice(depth1), rng.choice(depth1)) for _ in range(2500)]
    assert _collisions(depth1 + depth2, POINTS, order='none') == []


def _random_tree(rng: random.Random, depth: int, real: bool) -> Tree:
    if depth == 0 or rng.random() < 0.15:
        return rng.choice(MORE_LEAVES)
    kind = rng.random()
    if kind < 0.12:
        return _sin(_random_tree(rng, depth - 1, real))
    if kind < 0.2:
        return _exp(_random_tree(rng, depth - 1, real))
    if kind < 0.26:
        return _factorial(_random_tree(rng, depth - 1, real))
    if real and kind < 0.4:
        return _mod(_random_tree(rng, depth - 1, real), _random_tree(rng, depth - 1, real))
    return rng.choice(BINARY)(_random_tree(rng, depth - 1, real), _random_tree(rng, depth - 1, real))


def _regrouped(rng: random.Random, depth: int, real: bool) -> list[Tree]:
    """A random tree and the trees which a missing bracket would
    confuse with it: ``a op1 (b op2 c)`` and ``(a op1 b) op2 c``,
    ``f(a op b)`` and ``f(a) op b``."""
    a, b, c = (_random_tree(rng, depth, real) for _ in range(3))
    binary = BINARY + [_mod] if real else BINARY
    op1, op2 = rng.choice(binary), rng.choice(binary)
    unary = rng.choice([_sin, _exp, _factorial])
    return [op1(a, op2(b, c)), op2(op1(a, b), c), op2(a, op1(b, c)), op1(op2(a, b), c),
            unary(op1(a, b)), op1(unary(a), b), op1(a, unary(b))]


def test_random_trees_print_differently() -> None:
    rng = random.Random(20261123)
    trees = [_random_tree(rng, rng.choice((2, 3, 3, 4)), False) for _ in range(700)]
    for _ in range(250):
        trees.extend(_regrouped(rng, rng.choice((0, 1, 1, 2)), False))
    assert _collisions(trees, POINTS) == []


def test_random_trees_with_mod_print_differently() -> None:
    # Mod and the factorial at real points: x mod (y + z), (x mod y) + z, -(x mod y), (-x) mod y, (x!)!
    rng = random.Random(23912)
    trees = [_random_tree(rng, rng.choice((2, 2, 3)), True) for _ in range(700)]
    for _ in range(250):
        trees.extend(_regrouped(rng, rng.choice((0, 1, 1, 2)), True))
    assert _collisions(trees, REAL_POINTS) == []


def _div(a: Tree, b: Tree) -> Tree:
    return _mul(a, _pow(b, Tree(Integer(-1), _constant(-1))))


def test_all_small_quotients_print_differently() -> None:
    # fraction bars in fraction bars: (a/b)/c, a/(b/c), 1/(1/x), (a/b)/(c/d)
    leaves = [LEAVES[0], MORE_LEAVES[4], LEAVES[1], Tree(Integer(1), _constant(1))]
    depth1 = leaves + [op(a, b) for op in (_div, _mul) for a in leaves for b in leaves]
    depth2 = depth1 + [op(a, b) for op in (_div, _mul) for a in depth1 for b in depth1]
    assert _collisions(depth2, POINTS, modes=(True,)) == []
    assert _collisions(depth2[:1500], POINTS, order='none', modes=(False,)) == []


def _summed(body: Tree, product: bool) -> Tree:
    """The sum (the product) of ``body`` for k from 1 to 3; k is the third coordinate of a point."""
    def value(point: Point) -> complex:
        total: complex = 1 if product else 0
        for i in (1, 2, 3):
            term = body.value((point[0], point[1], complex(i)))
            total = total*term if product else total + term
        return total
    limits = (Symbol('k'), Integer(1), Integer(3))
    return Tree(Product(body.node, limits) if product else Sum(body.node, limits), value)


def test_sum_and_product_signs_do_not_capture_what_follows() -> None:
    # (sum of k)*x and the sum of k*x, (product of k) + 1 and the product of k + 1
    k = Tree(Symbol('k'), _variable(2))
    xs, two = LEAVES[0], LEAVES[1]
    bodies = [k, _mul(k, xs), _add(k, xs), _pow(k, two), _mul(two, k), _add(k, two), _mul(LEAVES[2], k)]
    trees: list[Tree] = []
    for product in (False, True):
        for body in bodies:
            s = _summed(body, product)
            trees.extend([s, _mul(s, xs), _mul(xs, s), _add(s, xs), _add(xs, s), _pow(s, two), _mul(s, two),
                          _add(s, two), _mul(LEAVES[2], s), _mul(s, s), _summed(s, product), _summed(s, not product),
                          _mul(s, _summed(k, not product)), _mul(_summed(k, not product), s)])
    points: list[Point] = [(1.3 + 0.4j, 0j, 0j), (0.37 - 1.1j, 0j, 0j)]
    assert _collisions(trees, points) == []
    assert _collisions(trees, points, order='none') == []


# ----------------------------------------------------------------------
# formulas

Assignment = tuple[bool, bool, float, float]
Truth = Callable[[Assignment], bool]


@dataclass(frozen=True)
class Formula:
    node: Boolean
    value: Truth


p, q = Symbol('p'), Symbol('q')


def _formula_leaves() -> list[Formula]:
    return [Formula(p, lambda a: a[0]), Formula(q, lambda a: a[1]),
            Formula(Lt(x, y), lambda a: a[2] < a[3]), Formula(Eq(x, y), lambda a: a[2] == a[3]),
            Formula(Lt(Add(x, y), Integer(1)), lambda a: a[2] + a[3] < 1)]


def _connectives(a: Formula, b: Formula) -> list[Formula]:
    return [Formula(And(a.node, b.node), lambda v: a.value(v) and b.value(v)),
            Formula(Or(a.node, b.node), lambda v: a.value(v) or b.value(v)),
            Formula(Implies(a.node, b.node), lambda v: (not a.value(v)) or b.value(v)),
            Formula(Xor(a.node, b.node), lambda v: a.value(v) != b.value(v)),
            Formula(Equivalent(a.node, b.node), lambda v: a.value(v) == b.value(v))]


def _negation(a: Formula) -> Formula:
    return Formula(Not(a.node), lambda v: not a.value(v))


def test_all_small_formulas_print_differently() -> None:
    leaves = _formula_leaves()
    depth1 = leaves + [_negation(a) for a in leaves] + [f for a in leaves for b in leaves for f in _connectives(a, b)]
    depth2 = depth1 + [_negation(a) for a in depth1] + \
        [f for a in depth1[:40] for b in depth1[:40] for f in _connectives(a, b)]
    assignments: list[Assignment] = [(a, b, u, v) for a in (False, True) for b in (False, True)
                                     for u, v in ((0.0, 0.0), (0.2, 0.5), (0.9, 0.3))]
    seen: dict[str, tuple[Formula, list[bool]]] = {}
    for formula in depth2:
        picture = pretty(formula.node, use_unicode=True, wrap_line=False)
        values = [formula.value(v) for v in assignments]
        if picture in seen:
            # ¬(x < y) and ¬x < y, ¬(p ∧ q) and ¬p ∧ q, (p → q) → p and p → (q → p)
            assert seen[picture][1] == values, (seen[picture][0].node, formula.node, picture)
        else:
            seen[picture] = (formula, values)


# ----------------------------------------------------------------------
# sets

@dataclass(frozen=True)
class SetTree:
    node: Set
    members: frozenset[float]


SAMPLES = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def _interval(a: int, b: int) -> SetTree:
    return SetTree(Interval(Integer(a), Integer(b)), frozenset(s for s in SAMPLES if a <= s <= b))


def test_all_small_set_expressions_print_differently() -> None:
    leaves = [_interval(0, 2), _interval(1, 3), SetTree(FiniteSet(Integer(2), Integer(4)), frozenset({2.0, 4.0}))]

    def combine(a: SetTree, b: SetTree) -> list[SetTree]:
        return [SetTree(Union(a.node, b.node), a.members | b.members),
                SetTree(Intersection(a.node, b.node), a.members & b.members),
                SetTree(Complement(a.node, b.node), a.members - b.members)]

    depth1 = leaves + [t for a in leaves for b in leaves for t in combine(a, b)]
    depth2 = depth1 + [t for a in depth1 for b in depth1 for t in combine(a, b)]
    seen: dict[str, SetTree] = {}
    for tree in depth2:
        for unicode in (True, False):
            picture = ('u' if unicode else 'a') + pretty(tree.node, use_unicode=unicode, wrap_line=False)
            if picture in seen:
                # X \ (Y \ Z) and (X \ Y) \ Z, X ∪ (Y ∩ Z) and (X ∪ Y) ∩ Z
                assert seen[picture].members == tree.members, (seen[picture].node, tree.node, picture)
            else:
                seen[picture] = tree


def test_products_of_sets_show_their_factors() -> None:
    # the product is not associative as a set of tuples: the factors of
    # a nested product, and a product of unions, keep their brackets
    a, b, c = Interval(Integer(0), Integer(1)), Interval(Integer(2), Integer(3)), FiniteSet(Integer(5))
    pictures = {pretty(s, use_unicode=True) for s in (
        ProductSet(a, b, c), ProductSet(ProductSet(a, b), c), ProductSet(a, ProductSet(b, c)),
        ProductSet(Union(a, b), c), Union(a, ProductSet(b, c)), ProductSet(Union(a, b), Union(a, b)),
        Union(a, ProductSet(b, b)))}
    assert len(pictures) == 7


def test_the_evaluator_is_independent_of_the_printer() -> None:
    # the oracle itself: the two readings of -x**2 differ at the points
    minus_square = _mul(LEAVES[2], _pow(LEAVES[0], LEAVES[1]))
    square_of_minus = _pow(_mul(LEAVES[2], LEAVES[0]), LEAVES[1])
    assert not _same(_values(minus_square, POINTS), _values(square_of_minus, POINTS))
    nodes: list[Basic] = [minus_square.node, square_of_minus.node]
    assert pretty(nodes[0]) != pretty(nodes[1])
