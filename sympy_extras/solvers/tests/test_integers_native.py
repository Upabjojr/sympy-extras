"""The native linear problems over the integers on objects of the core,
checked by substitution and brute force (the SymPy face is tested in
test_integers.py)."""
from __future__ import annotations

import itertools

from sympy_extras.core import Basic, Eq, Integer, Mod, Ne, Symbol, as_expr
from sympy_extras.core.logic import Boolean, Implies, Not, and_, false, or_, to_boolean, true
from sympy_extras.polys.virtual_substitution import to_nnf
from sympy_extras.solvers.integers import (_drop_covered, _linear_system, cooper, is_presburger,
    linear_diophantine_system, presburger_quantifier_elimination, with_sides)

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def test_linear_systems() -> None:
    assert _linear_system([x + 2*y - 3, 4*x - z], [x, y, z]) == ([[1, 2, 0], [4, 0, -1]], [3, 0])
    assert _linear_system([x**2 - 3], [x]) is None and _linear_system([x - Symbol('a')], [x]) is None
    found = linear_diophantine_system([6*x + 10*y + 15*z - 1], [x, y, z])
    assert found is not None
    values, parameters = found
    assert len(parameters) == 2 and all(p.is_integer for p in parameters)
    for a, b in itertools.product(range(-2, 3), repeat=2):
        point = [as_expr(v.subs({parameters[0]: Integer(a), parameters[1]: Integer(b)})) for v in values]
        assert 6*point[0] + 10*point[1] + 15*point[2] == 1
    assert linear_diophantine_system([Eq(2*x + 4*y, 3)], [x, y]) is None
    assert linear_diophantine_system([x + y - 3, x - y - 1], [x, y]) == ([Integer(2), Integer(1)], [])
    try:
        linear_diophantine_system([x*y - 1], [x, y])
    except ValueError:
        pass
    else:
        raise AssertionError('not linear')


def _brute_force(formula: Boolean, values: dict[Basic, Basic], bound: int = 12) -> bool:
    return any(to_boolean(formula.subs({**values, x: Integer(v)})) is true for v in range(-bound, bound + 1))


def test_cooper_against_brute_force() -> None:
    formulas: list[Boolean] = [
        and_(x > y, x < y + 2), and_(3*x > y, 3*x < y + 3), Eq(2*x, y), and_(x > y, x < z),
        and_(2*x >= y, 3*x <= z), Eq(2*x + 3*y, z), and_(Ne(x, y), x > y - 1, x < y + 2),
        and_(Eq(Mod(x + y, 3), 0), x > z, x < z + 2), and_(Not(Eq(Mod(x, 2), 0)), x >= y, x <= y + 1),
        and_(or_(Eq(x, y), Eq(x, z)), x > 0), and_(Implies(x > y, x > z), x <= y + 1),
        and_(Eq(Mod(x, 3), 1), x > y), or_(Eq(Mod(x, 3), 5), x > y)]
    for f in formulas:
        result = cooper(f, x)
        assert not result.has(x)
        for a, b in itertools.product(range(-4, 5), repeat=2):
            values: dict[Basic, Basic] = {y: Integer(a), z: Integer(b)}
            actual = to_boolean(result.subs(values))
            assert actual in (true, false), (f, values, actual)
            assert (actual is true) == _brute_force(f, values), (f, values, result)


def test_presburger() -> None:
    assert presburger_quantifier_elimination(x < y, [('forall', x), ('exists', y)]) is true
    assert presburger_quantifier_elimination(x < y, [('exists', y), ('forall', x)]) is false
    assert presburger_quantifier_elimination(or_(Eq(3*y, x), Eq(3*y, x + 1), Eq(3*y, x + 2)),
                                             [('forall', x), ('exists', y)]) is true
    assert is_presburger(and_(2*x > y, Eq(Mod(x, 3), 0)), {x, y}) and not is_presburger(x*y > 0, {x, y})
    assert is_presburger(Eq(Mod(x, 3), 1), {x}) and not is_presburger(x > y, {x})


def test_helpers() -> None:
    # SymPy's to_nnf: the negations reach the relations
    assert to_nnf(Not(and_(x > 0, Implies(y > 0, z > 0)))) == or_(x <= 0, and_(y > 0, z <= 0))
    assert str(with_sides(and_(x - y > 0, x + 1 < 0))) == '(x > y) & (x < -1)'
    xi = Symbol('x', integer=True)
    # sympy-extras#45 and #51: two disjuncts pinning the same point are not both dropped
    kept = _drop_covered(or_(and_(Eq(xi, 0), Ne(xi, 1)), and_(Eq(xi, 0), Ne(xi, 2))))
    assert to_boolean(kept.subs(xi, 0)) is true
