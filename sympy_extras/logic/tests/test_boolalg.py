"""The two-level minimisation of sympy_extras.logic.boolalg, checked
against the truth tables of the formulas (an independent oracle), against
hand computations and against the examples of SymPy's documentation of
``simplify_logic``, ``SOPform`` and ``POSform`` (facts about the
functions, which any correct minimiser reproduces up to the order)."""
from __future__ import annotations

import random
from itertools import product

from sympy_extras.core import Eq, Ne, Symbol
from sympy_extras.core.logic import And, Boolean, Or, and_, false, not_, or_, true
from sympy_extras.core.relational import Relational
from sympy_extras.logic.boolalg import POSform, SOPform, is_cnf, is_dnf, is_literal, simplify_logic
from sympy.testing.pytest import raises

a, b, c, d, w, x, y, z = (Symbol(n) for n in 'abcdwxyz')


def _evaluate(f: Boolean, env: dict[Boolean, Boolean]) -> Boolean:
    from sympy_extras.core.basic import Basic
    from sympy_extras.core.logic import to_boolean
    rule: dict[Basic, Basic] = {k: v for k, v in env.items()}
    return to_boolean(f._xreplace_evaluated(rule))


def _same_function(f: Boolean, g: Boolean, atoms: list[Boolean]) -> bool:
    """Whether ``f`` and ``g`` agree on every row of the truth table of
    ``atoms`` (a relation and its negation get opposite values)."""
    for values in product((true, false), repeat=len(atoms)):
        env: dict[Boolean, Boolean] = {}
        for atom, v in zip(atoms, values):
            env[atom] = v
            if isinstance(atom, Relational):
                env[atom.negated] = not_(v)
        if _evaluate(f, env) != _evaluate(g, env):
            return False
    return True


def test_documented_examples() -> None:
    assert simplify_logic((~x & ~y & ~z) | (~x & ~y & z)) == and_(not_(x), not_(y))
    assert simplify_logic(x | y, dontcare=y) == x
    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    assert SOPform([w, x, y, z], minterms, dontcares) == or_(and_(y, z), and_(not_(w), not_(x)))
    assert POSform([w, x, y, z], minterms, dontcares) == and_(z, or_(y, not_(w)))


def test_hand_computations() -> None:
    # a & b | a & ~b is a; the consensus term b & c of (a & b) | (~a & c) is redundant
    assert simplify_logic((a & b) | (a & ~b)) == a
    assert simplify_logic((a & b) | (~a & c) | (b & c), form='dnf') == or_(and_(a, b), and_(not_(a), c))
    assert simplify_logic(a & ~a) == false and simplify_logic(a | ~a) == true
    assert simplify_logic(Eq(a, 1) | (Ne(a, 1) & Eq(b, 2))) == or_(Eq(a, 1), Eq(b, 2))
    assert SOPform([a, b], []) == false and POSform([a, b], []) == false


def test_quick_exits() -> None:
    # a relation is returned as it is, and a formula already in the
    # requested form with literal arguments too
    assert simplify_logic(x > 0) == (x > 0)
    f = Or(And(a, b), c)
    assert simplify_logic(Or(a, b, c), form='dnf') == Or(a, b, c)
    assert is_dnf(f) and not is_cnf(f) and is_literal(~a) and not is_literal(a & b)
    # more than eight variables without force: unchanged
    many = [Symbol('p%d' % i) for i in range(9)]
    big = or_(*[and_(p, q) for p, q in zip(many, many[1:])])
    assert simplify_logic(big) == big
    raises(ValueError, lambda: simplify_logic(a | b, form='xnf'))
    raises(ValueError, lambda: SOPform([a, b], [[0, 1]], [[0, 1]]))
    raises(ValueError, lambda: SOPform([a, b], [[0, 1, 1]]))


def test_random_formulas_keep_their_truth_table() -> None:
    atoms: list[Boolean] = [a, b, c, x > 0, Eq(y, 1), y < 2]
    leaves: list[Boolean] = atoms + [x <= 0, Ne(y, 1)]
    rnd = random.Random(3)

    def gen(depth: int) -> Boolean:
        if depth == 0 or rnd.random() < 0.3:
            leaf = rnd.choice(leaves)
            return not_(leaf) if rnd.random() < 0.3 else leaf
        args = [gen(depth - 1) for _ in range(rnd.randint(2, 3))]
        return and_(*args) if rnd.random() < 0.5 else or_(*args)

    for _ in range(150):
        f = gen(3)
        for form in (None, 'cnf', 'dnf'):
            g = simplify_logic(f, form=form)
            assert _same_function(f, g, atoms), (f, form, g)
            if form == 'cnf':
                assert is_cnf(g), g
            elif form == 'dnf':
                assert is_dnf(g), g
