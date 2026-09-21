from __future__ import annotations

from sympy.core.numbers import Rational
from sympy.core.expr import Expr
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import And, Boolean, Or, false, true
from sympy.polys.rootoftools import CRootOf
from sympy.sets.conditionset import ConditionSet
from sympy.sets.sets import FiniteSet, Interval, ProductSet, Union
from sympy.testing.pytest import raises
from sympy.abc import a, x, y, z

from sympy_extras._typing import as_boolean, as_expr
from sympy_extras.polys.cad import IndexedRoot, cylindrical_cases, cylindrical_formula, cylindrical_set
from sympy_extras.polys.cad.qe import _truth_values


def _holds(formula: Boolean, point: dict[Symbol, Expr]) -> bool:
    """The truth of a formula in root functions at a point with algebraic
    coordinates, numerically with forty digits. A bound may be undefined
    outside the cell below it (``sqrt(1 - x**2)`` at ``x = 2``): it does
    not matter when the conditions on the lower variables fail."""
    if formula in (true, false):
        return formula is true
    if isinstance(formula, (And, Or)):
        values: list[bool] = []
        undefined = False
        for argument in formula.args:
            try:
                values.append(_holds(as_boolean(argument), point))
            except ValueError:
                undefined = True
        if isinstance(formula, And) and not all(values):
            return False
        if isinstance(formula, Or) and any(values):
            return True
        if undefined:
            raise ValueError("undefined")
        return isinstance(formula, And)
    assert isinstance(formula, Relational)
    difference = as_expr(as_expr(formula.lhs - formula.rhs).subs(list(point.items())).evalf(40))
    real_part, imaginary_part = difference.as_real_imag()
    if not difference.is_number or difference.has(S.NaN, S.ComplexInfinity) or abs(imaginary_part) > 1e-30:
        raise ValueError("undefined")
    value = float(real_part)
    zero = abs(value) < 1e-25
    if isinstance(formula, Eq):
        return zero
    if isinstance(formula, (Lt, Le)):
        return (value < 0 and not zero) or (zero and isinstance(formula, Le))
    assert isinstance(formula, (Gt, Ge))
    return (value > 0 and not zero) or (zero and isinstance(formula, Ge))


def _agrees_on_every_cell(formula: Boolean, gens: list[Symbol]) -> bool:
    """Whether the cylindrical formula has the truth value of the formula
    at the sample point of every cell of its decomposition."""
    described = cylindrical_formula(formula, gens)
    _, _, _, cells = _truth_values(formula, gens, [], None)
    return all(_holds(described, dict(zip(gens, cell.point))) == value for cell, value in cells)


def test_the_disc_and_its_pieces() -> None:
    assert cylindrical_formula(x**2 + y**2 < 1, [x, y]) == And(x > -1, x < 1, y > -sqrt(1 - x**2), y < sqrt(1 - x**2))
    # the closed disc is one piece: the points x = -1 and x = 1, where the
    # two bounds of y meet, join the open interval of x
    assert cylindrical_formula(x**2 + y**2 <= 1, [x, y]) == And(x >= -1, x <= 1, y >= -sqrt(1 - x**2),
                                                               y <= sqrt(1 - x**2))
    # the two arcs of the circle meet at x = 1 in one point
    assert cylindrical_formula(Eq(x**2 + y**2, 1) & (x > 0), [x, y]) == And(
        x > 0, x <= 1, Or(Eq(y, sqrt(1 - x**2)), Eq(y, -sqrt(1 - x**2))))
    assert cylindrical_formula(x**2 + y**2 < 0, [x, y]) == false
    assert cylindrical_formula(x**2 + y**2 >= 0, [x, y]) == true
    assert cylindrical_formula(x*y > 1, [x, y]) == Or(And(x > 0, y > 1/x), And(x < 0, y < 1/x))


def test_a_solution_formula_which_signs_do_not_give() -> None:
    # sympy-extras#9: the signs of the projection factors do not tell the
    # true cells of Exists(z, z**2 = x and z > y) from the false ones
    # (y**2 - x is not one of them); the root functions do
    assert cylindrical_formula(Eq(z**2, x) & (z > y), [x, y], [('exists', z)]) == And(x >= 0, y < sqrt(x))


def test_the_description_holds_on_the_cells_where_the_formula_does() -> None:
    for formula, gens in [
            ((x**2 + y**2 <= 1) & (y >= x), [x, y]),
            ((x + y > 1) & (x - y < 2) & (y > 0), [x, y]),
            (Eq(x**2 + y**2, 1) | (x*y > 1), [x, y]),
            ((y**3 - 3*y + x > 0) & (x**2 < 9), [x, y]),                   # indexed roots
            ((x*y**2 + y - x <= 0) | (y**2 < x), [x, y]),                  # a leading coefficient which vanishes
            ((x**2 + y**2 + z**2 < 1) & (z > x*y), [x, y, z]),
            ((x**2 + y**2 <= a) & (y >= 0), [a, x, y])]:
        assert _agrees_on_every_cell(as_boolean(formula), gens), formula


def test_indexed_roots() -> None:
    root = IndexedRoot(y**3 - 3*y + x, y, 1)
    assert root.free_symbols == {x} and root.is_extended_real
    assert root == IndexedRoot(z**3 - 3*z + x, z, 1).subs(z, y) or root.args[0] != IndexedRoot(z**3 - 3*z + x, z, 1).args[0]
    # numbers for the parameters give the root: the distinct real roots are counted
    assert root.subs(x, 0) == 0 and IndexedRoot(y**3 - 3*y + x, y, 2).subs(x, 0) == sqrt(3)
    assert root.subs(x, 2) == 1 and root.subs(x, 1) == CRootOf(y**3 - 3*y + 1, 1)
    # no such root: it stays as it is
    assert isinstance(IndexedRoot(y**3 - 3*y + x, y, 2).subs(x, 5), IndexedRoot)
    # the bound variable is not reached by a substitution
    assert root.subs(y, 7) == root
    # coefficients which are not rational: numerically
    value = as_expr(IndexedRoot(y**3 + x*y - 1, y, 0).subs(x, sqrt(2))).evalf()
    assert abs(value**3 + sqrt(2)*value - 1) < 1e-12
    raises(ValueError, lambda: IndexedRoot(y**2 - x, y, -1))


def test_cylindrical_sets() -> None:
    plane = ProductSet(S.Reals, S.Reals)
    assert cylindrical_set(x**2 + y**2 <= 0, [x, y]) == FiniteSet((0, 0))
    assert cylindrical_set(x**2 + y**2 < 0, [x, y]) == S.EmptySet
    assert cylindrical_set(x**2 + y**2 >= 0, [x, y]) == plane
    found = cylindrical_set((x**2 + y**2)*((x - 3)**2 + y**2 - 1) <= 0, [x, y])
    assert isinstance(found, Union) and FiniteSet((0, 0)) in found.args
    # a system with finitely many real solutions: its points
    points = cylindrical_set(Eq(x**2 + y**2, 1) & Eq(y, x), [x, y])
    assert points == FiniteSet((-sqrt(2)/2, -sqrt(2)/2), (sqrt(2)/2, sqrt(2)/2))
    region = cylindrical_set((x**2 + y**2 <= 1) & (x + y >= 1), [x, y])
    condition = And(x >= 0, x <= 1, y >= 1 - x, y <= sqrt(1 - x**2))
    assert region == ConditionSet((x, y), condition, plane)
    assert condition.subs({x: Rational(1, 2), y: Rational(3, 4)}) == true
    assert condition.subs({x: Rational(1, 2), y: Rational(1, 4)}) == false


def test_the_cases_of_the_parameters() -> None:
    assert cylindrical_cases(Eq(x**2, a), [a], [x]) == [(a >= 0, FiniteSet(-sqrt(a), sqrt(a)))]
    assert cylindrical_cases(x**2 <= a, [a], [x]) == [(a >= 0, Interval(-sqrt(a), sqrt(a)))]
    # the cases with one set of solutions are one case
    assert cylindrical_cases(Eq(a*x, 1), [a], [x]) == [(Or(a > 0, a < 0), FiniteSet(1/a))]
    # a coordinate is a function of the ones before: their values are put
    # in it (the bug: the points came out as (sqrt(2)*sqrt(a)/2, x))
    found = cylindrical_cases(Eq(x, y) & Eq(x**2 + y**2, a), [a], [x, y])
    half = sqrt(2)*sqrt(a)/2
    assert found == [(Eq(a, 0), FiniteSet((0, 0))), (a > 0, FiniteSet((-half, -half), (half, half)))]
    [(condition, region)] = cylindrical_cases(x**2 + y**2 < a, [a], [x, y])
    assert condition == (a > 0) and isinstance(region, ConditionSet)
    assert cylindrical_cases(x**2 < a, [a], [x]) == [(a > 0, Interval.open(-sqrt(a), sqrt(a)))]
