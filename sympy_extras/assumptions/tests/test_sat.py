from __future__ import annotations

from sympy import (S, Q, Eq, Ne, And, Or, Implies, symbols, Rational, sqrt,
    sin, CRootOf, true)
from typing import Any, Mapping

from sympy import Basic, sympify
from sympy.logic.boolalg import Boolean
from sympy.abc import x, y, z

from sympy_extras.assumptions.sat import Model, Witness

from sympy_extras.assumptions import (satisfiable, tautology, find_instance,
    element, ForAll, Exists)

p, q, r = symbols('p q r')


def _holds(formula: Boolean, model: Mapping[Any, object]) -> bool:
    substituted: Basic = formula
    for key, value in model.items():
        substituted = substituted.subs(key, sympify(value))
    return substituted is true


def _model(result: object) -> Model:
    """The model of a satisfiable formula."""
    assert isinstance(result, dict)
    return result


def _instances(result: object) -> list[Witness]:
    assert isinstance(result, list)
    return result


def test_satisfiable_propositional() -> None:
    assert satisfiable(p & ~q) == {p: True, q: False}
    assert satisfiable(p & ~p) is False
    assert satisfiable(True) == {}
    assert satisfiable(False) is False
    assert satisfiable(S.true) == {}
    m = _model(satisfiable((p | q) & (~p | r) & (~q | r)))
    assert m[r] is True
    models = satisfiable(p | q, all_models=True)
    assert models is not None
    assert len(models) == 3 and all(m[p] or m[q] for m in models)
    assert satisfiable(p & ~p, all_models=True) == []
    assert satisfiable(Implies(p, q) & p & ~q) is False


def test_satisfiable_polynomial() -> None:
    f = (x**2 + y**2 < 1) & (x + y > 1)
    m = _model(satisfiable(f))
    assert set(m) == {x, y} and _holds(f, m)
    assert satisfiable((x**2 + y**2 < 1) & (x + y > 2)) is False
    assert satisfiable(x**2 < 0, domain=S.Reals) is False
    assert satisfiable(Eq(x**2, 2), domain=S.Reals) == {x: CRootOf(x**2 - 2, 0)}
    assert satisfiable(x > 0) == {x: 1}
    assert satisfiable((x > 0) & (x < 1)) == {x: Rational(1, 2)}
    assert _model(satisfiable(Or(x > 1, x < -1) & (x**2 < 4)))[x] in (Rational(3, 2), Rational(-3, 2))
    assert satisfiable(And(x > y, y > z, z > x)) is False
    assert satisfiable(And(x > y, y > z, z > x - 3))
    # the theory decides which propositional models survive
    m = _model(satisfiable((p | (x > 1)) & (~p | (x < 0)) & (x > 0)))
    assert m[p] is False and m[x] > 1
    assert satisfiable((p | (x > 1)) & (~p | (x < 0)) & (x > 0) & (x < 1)) is False
    models = satisfiable((x > 0) | (x < 0), all_models=True)
    assert models is not None
    assert len(models) == 2
    # with assumptions
    assert satisfiable(x > 1, assumptions=x < 0) is False
    assert _model(satisfiable(x > 1, assumptions=x < 3))[x] == 2
    assert satisfiable(x > 1, assumptions=[x < 0, x > 0]) is False


def test_satisfiable_reals_and_complex() -> None:
    # a variable is not real unless something says so
    assert satisfiable(Eq(x**2, -1)) is None
    assert satisfiable(Eq(x**2, -1), domain=S.Reals) is False
    assert satisfiable(Eq(x**2, -1), element(x, S.Reals)) is False
    assert satisfiable(Eq(x**2, -1) & (x > 0)) is False
    assert satisfiable(Ne(x, x)) is False
    # a real solution of an equation is a witness
    assert satisfiable(Eq(x, y)) == {x: 0, y: 0}
    assert satisfiable(Eq(x, y), domain=S.Reals) == {x: 0, y: 0}
    # an inequality makes its variables real, as in Mathematica's Reduce
    assert satisfiable(x**2 < 0) is False
    assert satisfiable((x > 1) & (x < 0)) is False
    assert satisfiable(Eq(x, y) & (x > 0)) == {x: 1, y: 1}


def test_satisfiable_predicates() -> None:
    assert satisfiable(element(x, S.Integers) & (x > 0)) == {x: 1}
    assert satisfiable(Q.positive(x) & Q.negative(x)) is False
    assert satisfiable(Q.positive(x) & (x < 0)) is False
    assert satisfiable(element(x, S.Integers) & ~element(x, S.Rationals)) is False
    assert satisfiable(element(x, S.Naturals) & element(x, S.Rationals)) == {}
    assert satisfiable(Q.prime(x) & Q.even(x)) == {}
    assert satisfiable(Q.prime(x) & (x > 2)) == {x: 3}
    assert satisfiable(~element(x, S.Reals) & (x > 0)) is False
    assert satisfiable(p & element(x, S.Reals)) == {p: True}


def test_satisfiable_integers() -> None:
    assert satisfiable((x > 1) & (x < 3), domain=S.Integers) == {x: 2}
    assert satisfiable((x > 1) & (x < 2), domain=S.Integers) is False
    assert satisfiable(Eq(x**2, 2), domain=S.Integers) is False
    assert satisfiable(Eq(x**2, 4) & (x < 0), domain=S.Integers) == {x: -2}
    assert _model(satisfiable(x**2 > 5, domain=S.Integers))[x]**2 > 5
    assert satisfiable((x > Rational(1, 3)) & (x < Rational(2, 3)), domain=S.Integers) is False
    assert satisfiable((x > -Rational(3, 2)) & (x < -1), element(x, S.Integers)) is False
    # coefficients must be rational for the decomposition
    assert satisfiable((x > sqrt(2) - 2) & (x < 0), element(x, S.Integers)) is None
    assert satisfiable((x**2 > 2) & (x**2 < 5), domain=S.Integers) == {x: -2}
    assert _model(satisfiable((x < -10) | (x > 10), domain=S.Integers))[x] in (-11, 11)
    assert satisfiable(2*x > 1, element(x, S.Integers)) == {x: 1}
    # an integer variable with a real one
    m = satisfiable(Eq(x**2, y) & (y > 2) & (y < 3), element(x, S.Integers) & element(y, S.Reals))
    assert m is False
    m = _model(satisfiable(Eq(x**2, y) & (y > 3) & (y < 5), element(x, S.Integers) & element(y, S.Reals)))
    assert m[x] in (2, -2) and m[y] == 4
    # several integer variables: found around the sample points or unknown
    m = _model(satisfiable((x > 0) & (y > 0) & (x + y < 4), domain=S.Integers))
    assert m[x] >= 1 and m[y] >= 1 and m[x] + m[y] < 4
    assert satisfiable((x**2 + y**2 < 1) & (x > 0) & (y > 0), domain=S.Integers) in (False, None)


def test_satisfiable_undecidable() -> None:
    assert satisfiable(sin(x) > 0) is None
    assert satisfiable((sin(x) > 0) & (x > 1)) is None
    assert satisfiable((sin(x) > 0) & (x > 1) & (x < 0)) is False
    assert satisfiable(sqrt(x) > 1, domain=S.Reals) is None
    assert satisfiable(sin(x) > 0, all_models=True) is None
    assert satisfiable(p & (sin(x) > 0), all_models=True) is None


def test_satisfiable_quantified() -> None:
    assert satisfiable(ForAll(x, x**2 >= 0)) == {}
    assert satisfiable(ForAll(x, x**2 > 0)) is False
    assert satisfiable(Exists(y, Eq(y**2, x)) & (x < 0)) is False
    m = _model(satisfiable(Exists(y, Eq(y**2, x)) & (x > 1)))
    assert m[x] > 1


def test_tautology() -> None:
    assert tautology(p | ~p) is True
    assert tautology(p | q) is False
    assert tautology(Implies(p & q, p)) is True
    assert tautology(True) is True
    assert tautology(False) is False
    assert tautology(x**2 + y**2 >= 2*x*y, domain=S.Reals) is True
    assert tautology((x > 0) | (x <= 0), domain=S.Reals) is True
    assert tautology(x**2 > 0, domain=S.Reals) is False
    # the negation x**2 < 0 is an inequality, hence over the reals
    assert tautology(x**2 >= 0) is True
    assert tautology(Eq(x**2, -1) | Ne(x**2, -1)) is True
    assert tautology(Ne(x**2, -1)) is None
    assert tautology(Implies(x > 2, x > 1), domain=S.Reals) is True
    assert tautology(Implies(x > 1, x > 2), domain=S.Reals) is False
    assert tautology(x > 1, assumptions=x > 2) is True
    assert tautology(sin(x) > -2) is None
    assert tautology(Implies(element(x, S.Naturals), x > 0)) is True
    assert tautology(Implies(element(x, S.Integers), x > 0)) is False


def test_find_instance() -> None:
    assert find_instance((x**2 + y**2 < 1) & (x > y), [x, y]) == [{x: 0, y: -Rational(1, 2)}]
    assert len(_instances(find_instance((x**2 + y**2 < 1) & (x > y), [x, y], count=10))) == 3
    assert find_instance(Eq(x**2, 2), x) == [{x: CRootOf(x**2 - 2, 0)}]
    assert find_instance(Eq(x**2, 2), [x], count=2) == [{x: CRootOf(x**2 - 2, 0)}, {x: CRootOf(x**2 - 2, 1)}]
    assert find_instance(x**2 < 0, [x]) == []
    assert find_instance(True, [x, y]) == [{x: 0, y: 0}]
    assert find_instance(x > 0, [x], assumptions=x < 2) == [{x: 1}]
    assert find_instance(x > 0, [x], assumptions=x < -1) == []
    # integers
    assert find_instance(Eq(x**2, 2), [x], S.Integers) == []
    assert find_instance((x**2 > 5) & (x < 0), [x], S.Integers) == [{x: -3}]
    assert find_instance((x > 0) & (x < 4), [x], S.Integers) == [{x: 1}]
    inst = find_instance((x > 0) & (y > 0) & (x + y < 3), [x, y], S.Integers)
    assert inst == [{x: 1, y: 1}]
    # with propositional variables and undecidable parts
    inst = _instances(find_instance((p | (x > 1)) & (x < 0), [x]))
    assert inst[0][x] < 0
    assert find_instance(sin(x) > 0, [x]) is None
    # quantifiers
    assert find_instance(Exists(y, Eq(y**2, x)) & (x > 1), [x]) == [{x: 2}]
    # the values must satisfy the formula
    for formula in [(x**2 + y**2 - 7 > 0) & (x + y - 2 > 0) & (x**2 + y - 10 > 0),
                    Or(x > 2, x < -2) & (x**2 < 9), Ne(x, 0) & (x**3 < 1)]:
        for witness in _instances(find_instance(formula, sorted(formula.free_symbols, key=str), count=20)):
            assert _holds(formula, witness)
