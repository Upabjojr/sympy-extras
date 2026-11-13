"""The algebraic normal form of sympy_extras.logic.anf, checked by truth
tables (an independent oracle: the normal form is unique, so the form
built from the truth table by ``ANFform`` must be the one ``to_anf``
finds) and against SymPy 1.14's ``to_anf`` (the code ported)."""
from __future__ import annotations

import random
from itertools import product

from sympy.core.cache import clear_cache
from sympy.logic.boolalg import to_anf as sympy_to_anf
from sympy.testing.pytest import raises

from sympy_extras.core import Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import (And, Boolean, BooleanAtom, Not, Xor, and_, equivalent, false, implies, ite,
                                     not_, or_, true, xor)
from sympy_extras.interop import to_sympy
from sympy_extras.logic.anf import (ANFform, and_evaluated, anf_coeffs, bool_monomial, distribute_xor_over_and,
                                    is_anf, to_anf, xor_evaluated)

a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
x = Symbol('x')
_VARIABLES = [a, b, c, d]


def _truth_table(f: Boolean, variables: list[Symbol]) -> list[int]:
    """The values of ``f`` on the assignments in the order of
    ``product((0, 1), repeat=n)``."""
    values: list[int] = []
    for bits in product((0, 1), repeat=len(variables)):
        rule: dict[Basic, Basic] = {s: true if bit else false for s, bit in zip(variables, bits)}
        v = f._xreplace_evaluated(rule)
        assert isinstance(v, BooleanAtom), (f, v)
        values.append(1 if v else 0)
    return values


def _random_formula(rnd: random.Random, depth: int) -> Boolean:
    if depth == 0 or rnd.random() < 0.25:
        return rnd.choice(_VARIABLES)
    kind = rnd.randrange(7)
    args = [_random_formula(rnd, depth - 1) for _ in range(rnd.randint(2, 3))]
    if kind == 0:
        return xor(*args)
    if kind == 1:
        return and_(*args)
    if kind == 2:
        return or_(*args)
    if kind == 3:
        return not_(args[0])
    if kind == 4:
        return implies(args[0], args[1])
    if kind == 5:
        return equivalent(args[0], args[1])
    return not_(xor(*args))


def test_known_normal_forms() -> None:
    # SymPy's documentation and hand computations
    assert str(to_anf(not_(a))) == 'a ^ True'
    assert str(to_anf(and_(or_(a, b), not_(c)))) == 'a ^ b ^ (a & b) ^ (a & c) ^ (b & c) ^ (a & b & c)'
    assert str(to_anf(implies(not_(a), equivalent(b, c)), deep=False)) == 'True ^ ~a ^ (~a & (Equivalent(b, c)))'
    assert str(to_anf(implies(a, b))) == 'a ^ True ^ (a & b)'
    assert str(to_anf(equivalent(a, b))) == 'a ^ b ^ True'
    assert str(to_anf(or_(a, b))) == 'a ^ b ^ (a & b)'
    assert str(to_anf(and_(xor(a, b), xor(c, d)))) == '(a & c) ^ (a & d) ^ (b & c) ^ (b & d)'
    assert to_anf(true) is true and to_anf(a) is a
    # a relation, or a relation with compound sides, is a variable (SymPy
    # raises AttributeError for x + 1 > 0)
    assert str(to_anf(xor(a, x + 1 > 0))) == 'a ^ (x + 1 > 0)'
    assert str(to_anf(not_(xor(a, x > 0)))) == 'a ^ True ^ (x > 0)'


def test_is_anf() -> None:
    assert is_anf(true) and is_anf(a) and is_anf(And(a, b, c)) and is_anf(Xor(a, true, And(a, b)))
    assert not is_anf(Xor(a, Not(b))) and not is_anf(or_(a, b)) and not is_anf(Not(a))
    assert not is_anf(And(a, x > 0))


def test_xor_evaluated() -> None:
    assert str(xor_evaluated(a, b, true)) == '~(a ^ b)'
    assert str(xor_evaluated(a, b, true, remove_true=False)) == 'a ^ b ^ True'
    assert xor_evaluated(a, a) is false and xor_evaluated(a, false) == a
    # the relations compared by their canonical forms, as SymPy's Xor
    assert xor_evaluated(x > 1, 1 < x, a) == a
    assert xor_evaluated(x > 1, x <= 1, a) == not_(a)
    assert xor_evaluated(x > 1, 1 >= x, a, remove_true=False) == Xor(a, true)
    assert and_evaluated(x > 1, 1 >= x) is false and and_evaluated(x > 1, 1 < x) == (x > 1)


def test_distribute_and_monomials() -> None:
    assert str(distribute_xor_over_and(And(Xor(not_(a), b), c))) == '(b & c) ^ (c & ~a)'
    assert anf_coeffs([0, 1, 1, 0, 0, 1, 0, 1]) == [0, 1, 1, 0, 0, 0, 1, 0]
    assert anf_coeffs([1]) == [1]
    raises(ValueError, lambda: anf_coeffs([0, 1, 1]))
    assert str(bool_monomial([1, 0, 1], [a, b, c])) == 'a & c'
    assert str(bool_monomial(6, [a, b, c])) == 'a & b'
    assert bool_monomial(0, [a, b, c]) is true
    raises(ValueError, lambda: bool_monomial(8, [a, b, c]))
    assert str(ANFform([a], [1, 0])) == 'a ^ True'
    assert str(ANFform([a, b], [0, 1, 1, 1])) == 'a ^ b ^ (a & b)'
    assert ANFform([a, b], [0, 0, 0, 0]) is false and ANFform([a, b], [1, 1, 1, 1]) is true
    raises(ValueError, lambda: ANFform([a, b], [0, 1]))


def test_normal_form_is_the_one_of_the_truth_table() -> None:
    # the normal form is unique: to_anf of a random formula is ANFform of
    # its truth table, and has the same truth table
    rnd = random.Random(3)
    for _ in range(150):
        f = _random_formula(rnd, 3)
        table = _truth_table(f, _VARIABLES)
        found = to_anf(f)
        assert is_anf(found), (f, found)
        assert _truth_table(found, _VARIABLES) == table, f
        assert found == ANFform(_VARIABLES, table), (f, found)


def test_agrees_with_sympy() -> None:
    rnd = random.Random(5)
    for _ in range(60):
        f = _random_formula(rnd, 3)
        clear_cache()
        expected = sympy_to_anf(to_sympy(f))
        assert str(to_anf(f)) == str(expected), f
    # the shallow form of an implication of compound formulas
    f = implies(ite(a, b, c), xor(a, d))
    assert str(to_anf(f, deep=False)) == str(sympy_to_anf(to_sympy(f), deep=False))
