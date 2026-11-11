"""Tests of the native quantifier elimination, cylindrical descriptions and
indexed roots (:mod:`sympy_extras.polys.cad.qe`,
:mod:`sympy_extras.polys.cad.cylindrical`,
:mod:`sympy_extras.polys.cad.indexedroot`), on the objects of the core,
and of the conversions of the faces."""
from __future__ import annotations

from fractions import Fraction
from typing import Callable

from sympy_extras.core import Rational, oo, sqrt
from sympy_extras.core.logic import And, Boolean, Or, equivalent, false, to_boolean, true
from sympy_extras.core.numbers import Integer
from sympy_extras.core.relational import Eq, Ne
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.cad.cylindrical import cylindrical_cases, cylindrical_formula, cylindrical_set
from sympy_extras.polys.cad.indexedroot import IndexedRoot, algebraic_of, indexed_root, real_number, real_root
from sympy_extras.polys.cad.lifting import Lifting
from sympy_extras.polys.cad.projection import augmented_projection_sets, projection_sets
from sympy_extras.polys.cad.qe import (QuantifierSpec, _equational_constraint, _inherited_truth,
    _separating_refinement, _sign_formula, _truth_values, decide, quantifier_elimination, sample_points,
    solution_set, truth_tables)
from sympy_extras.polys.cad.samplepoints import real_roots_of
from sympy_extras.polys.polytools import Poly
from sympy_extras.sets.sets import EmptySet, Interval, Union, finite_set, interval
from sympy_extras._testing import untyped

a, b, x, y, z = Symbol('a'), Symbol('b'), Symbol('x'), Symbol('y'), Symbol('z')
qe = quantifier_elimination


def _raises(exception: type[BaseException], f: Callable[..., object], *args: object) -> bool:
    try:
        untyped(f)(*args)
    except exception:
        return True
    return False


def _equivalent(f: Boolean, g: Boolean, gens: list[Symbol]) -> bool:
    _, [table] = truth_tables([equivalent(f, g)], gens)
    return all(table)


def test_answers_are_formulas_and_sets_of_the_core() -> None:
    assert decide(x**2 + y**2 < 1, [('exists', [x, y])]) is True
    assert qe(x**2 + b*x + a > 0, [('forall', x)]) == (4*a - b**2 > 0)
    assert qe(x**2 + b*x + 1 > 0, [('forall', x)]) == And(b > -2, b < 2)
    assert qe(x**2 + y**2 < 1, [('forall', x), ('exists', y)]) is false
    assert qe(Eq(y, x**2), [('forall', x), ('exists', y)]) is true
    # the ends of the intervals are real roots with rational coefficients
    found = solution_set(x**2 > 2, x)
    low, high = real_root(real_roots_of([1, 0, -2])[0], 'x'), real_root(real_roots_of([1, 0, -2])[1], 'x')
    assert found == Union(Interval(-oo, low, True, True), Interval(high, oo, True, True))
    assert solution_set(Eq(x**2, 4), x) == finite_set(-2, 2)
    assert solution_set(x**2 < 0, x) is EmptySet
    points = sample_points((x**2 + y**2 < 1) & (x > y), [x, y])
    assert points[0] == {x: Integer(0), y: Rational(-1, 2)}
    assert isinstance(points[1][x], IndexedRoot) and abs(float(points[1][x].evalf(15)) - 0.5**0.5) < 1e-12


def test_partial_decomposition_lifts_fewer_cells() -> None:
    formula = (x**2 + y**2 < 1) >> And(z**2 > x, z < y)
    blocks: QuantifierSpec = [('forall', [x, y]), ('exists', z)]
    partial, _, value, _ = _truth_values(formula, [], blocks, None)
    full, _, value_, _ = _truth_values(formula, [], blocks, None, partial=False)
    assert value is value_ is True
    assert len(full.cells_at(3)) > 4*len(partial.cells_at(3)) and len(partial.cells_at(3)) > 0
    cad, _, _, cells = _truth_values(And(x > 0, x**2 + y**2 < 1), [x, y], [], None, True, 'false')
    assert [c.point[0].rational_value() for c in cad.cells_at(1)] == [Fraction(-2), Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2),
                                          Fraction(1), Fraction(2)]
    # x > 0 fails on the first four cells of the line: nothing is lifted
    # over them; over the others the sign of the circle is not known
    assert [(c.index, v) for c, v in cells] == [((1,), False), ((2,), False), ((3,), False), ((4,), False),
        ((5, 1), False), ((5, 2), False), ((5, 3), True), ((5, 4), False), ((5, 5), False),
        ((6, 1), False), ((6, 2), False), ((6, 3), False), ((7, 1), False)]


def test_equational_constraint_in_the_formula() -> None:
    assert _equational_constraint(Eq(x*y, 1), [x, y]) == Poly(x*y - 1, x, y)
    assert _equational_constraint(And(x > 0, Eq(y, x**2), Eq(x, 2)), [x, y]) == Poly(y - x**2, x, y)
    # not implied by the formula, or not of the last level
    assert _equational_constraint(Or(Eq(x*y, 1), x > 0), [x, y]) is None
    assert _equational_constraint(And(Eq(x, 2), y > 0), [x, y]) is None
    # the reduced projection is used when the last variable is quantified;
    # the answer is that of the full decomposition
    # (the formula evaluated: the order of its arguments is SymPy's, which
    # takes the line for the constraint)
    formula = Eq(x**2 + y**2, 1) & Eq(y, a*x + b)
    prefix: QuantifierSpec = [('exists', [x, y])]
    reduced, _, _, cells = _truth_values(formula, [a, b], prefix, None)
    full, _, _, cells_ = _truth_values(formula, [a, b], prefix, None, partial=False)
    assert len(reduced.projection[2]) < len(full.projection[2])
    assert len(reduced.projection[0]) < len(full.projection[0])
    assert 0 < len(cells) < len(cells_)
    for cell, value in cells:
        if all(c.is_rational for c in cell.point):
            values = {v: Rational(c.rational_value().numerator, c.rational_value().denominator)
                      for v, c in zip([a, b], cell.point)}
            assert decide(to_boolean(formula.subs(values)), prefix) is value
    assert qe(formula, prefix, free=[a, b]) == qe(formula, prefix, free=[a, b], partial=False) == \
        (a**2 - b**2 + 1 >= 0)
    plain, _, _, _ = _truth_values(formula, [a, b, x, y], [], None)
    assert plain.projection == full.projection


def test_solution_formula_by_the_augmented_projection() -> None:
    # sympy-extras#9: the signs of the projection factors do not tell the
    # true cells from the false ones; the augmented projection adds the
    # derivative y of y**2 - x
    formula = Eq(z**2, x) & (z > y)
    prefix: QuantifierSpec = [('exists', z)]
    answer = qe(formula, prefix, free=[x, y])
    assert answer == Or(And(x >= 0, y < 0), x - y**2 > 0)
    assert _equivalent(answer, And(x >= 0, Or(y < 0, y**2 < x)), [x, y])
    cad, _, _, cells = _truth_values(formula, [x, y], prefix, None)
    assert _sign_formula(cad, cells, 2) is None
    refined = _separating_refinement(cad, cells, 2)
    assert refined is not None
    cad_, cells_ = refined
    assert [f.as_expr() for f in cad_.projection[1]] == [x - y**2, y] and cad_.projection[0] == cad.projection[0]
    assert len(cells_) > len(cells)
    for cell, value in cells_:
        if all(c.is_rational for c in cell.point):
            values = {v: Rational(c.rational_value().numerator, c.rational_value().denominator)
                      for v, c in zip([x, y], cell.point)}
            assert decide(to_boolean(formula.subs(values)), prefix) is value


def test_inherited_truth_of_the_refined_cells() -> None:
    # the cells of the refinement are mapped to the cells of the coarser
    # decomposition by counting the sections of the old factors (the
    # refined stack over x = 0 has the double root y = 0 of x - y**2 and
    # the root of y, one section)
    projection = projection_sets([x - y**2, x*y], [x, y])
    lifting = Lifting(projection, [x, y], 'mccallum')
    levels = lifting.lift_all()
    truth = {cell.index: cell.point[1].sign() < 0 for cell in levels[1]}
    refined = augmented_projection_sets(projection, [x, y], [y - 1, x - 1])
    cells = _inherited_truth(Lifting(refined, [x, y], 'mccallum'), [len(level) for level in projection], truth, 2)
    assert len(cells) > len(levels[1])
    assert all(value is (cell.point[1].sign() < 0) for cell, value in cells)
    assert [str(cell.point[1].rational_value()) for cell, _ in cells if cell.point[0].is_zero] == \
        ['-1', '0', '1/2', '1', '2']


def test_cylindrical_descriptions() -> None:
    assert cylindrical_formula(x**2 + y**2 < 1, [x, y]) == And(x > -1, x < 1, y > -sqrt(1 - x**2),
                                                              y < sqrt(1 - x**2)).doit()
    assert cylindrical_formula(x**2 + y**2 < 0, [x, y]) is false
    assert cylindrical_formula(x**2 + y**2 >= 0, [x, y]) is true
    assert cylindrical_formula(Eq(z**2, x) & (z > y), [x, y], [('exists', z)]) == And(x >= 0, y < sqrt(x)).doit()
    found = cylindrical_set(Eq(x**2 + y**2, 1) & Eq(y, x), [x, y])
    assert str(found) == '{(-sqrt(2)/2, -sqrt(2)/2), (sqrt(2)/2, sqrt(2)/2)}'
    assert cylindrical_set(x**2 + y**2 < 0, [x, y]) is EmptySet
    assert cylindrical_cases(Eq(a*x, 1), [a], [x]) == [(Or(a > 0, a < 0).doit(), finite_set(1/a))]
    [(condition, solutions)] = cylindrical_cases(x**2 <= a, [a], [x])
    assert condition == (a >= 0) and solutions == interval(-sqrt(a), sqrt(a))
    # a cubic: the bounds are indexed roots
    described = cylindrical_formula((y**3 - 3*y + x > 0) & (x > 1), [x, y])
    assert described.has(IndexedRoot)


def test_indexed_roots() -> None:
    root = indexed_root(y**3 - 3*y + x, y, 1)
    assert root.free_symbols == {x} and root.is_extended_real is True and isinstance(root, IndexedRoot)
    # the bound variable is the dummy of its name: equal roots are equal
    assert root == indexed_root(z**3 - 3*z + x, z, 1).subs(z, y) or root != indexed_root(z**3 - 3*z + x, z, 1)
    assert root == indexed_root(y**3 - 3*y + x, y, 1) and root.variable != y
    # numbers for the parameters give the root: the distinct real roots are counted
    assert root.subs(x, 0) == 0 and indexed_root(y**3 - 3*y + x, y, 2).subs(x, 0) == sqrt(3)
    assert root.subs(x, 2) == 1
    at_one = root.subs(x, 1)
    assert isinstance(at_one, IndexedRoot) and not at_one.free_symbols
    assert abs(float(at_one.evalf(20)) - 0.3472963553338607) < 1e-14
    # no such root: it stays as it is
    assert isinstance(indexed_root(y**3 - 3*y + x, y, 2).subs(x, 5), IndexedRoot)
    # the bound variable is not reached by a substitution
    assert root.subs(y, 7) == root and root.xreplace({y: Integer(7)}) == root
    # coefficients which are not rational: numerically
    value = indexed_root(y**3 + x*y - 1, y, 0).subs(x, sqrt(2)).evalf(30)
    assert abs(value**3 + sqrt(2)*value - 1).evalf(20) < Rational(1, 10**12)
    assert _raises(ValueError, IndexedRoot, y**2 - x, y, -1)
    assert _raises(ValueError, indexed_root, y**2 - x, x + y, 0)


def test_real_algebraic_numbers() -> None:
    low, high = real_roots_of([1, 0, -2])
    r = real_root(high, 'x')
    assert isinstance(r, IndexedRoot)
    assert str(r) == 'IndexedRoot(x**2 - 2, x, 1)' and algebraic_of(r) == high
    assert r.is_positive is True and r.is_real is True and (r > 1) is true and (r < Rational(3, 2)) is true
    assert real_number(high, 'x') == sqrt(2) and real_number(low, 'x') == -sqrt(2)
    assert real_root(real_roots_of([2, -1])[0], 'x') == Rational(1, 2)
    cubic = real_root(real_roots_of([1, 0, -1, -1])[0], 'x')
    assert real_number(real_roots_of([1, 0, -1, -1])[0], 'x') == cubic
    assert abs(float(cubic.evalf(30)) - 1.324717957244746) < 1e-14
    # a root rebuilt from its polynomial is the same number
    rebuilt = algebraic_of(IndexedRoot(r.expr, r.variable, 1))
    assert rebuilt is not None and rebuilt == high


def test_interop_of_the_roots() -> None:
    import sympy
    from sympy_extras.interop import from_sympy, to_sympy
    from sympy_extras.polys.cad.indexedroot_sympy import IndexedRoot as SymPyIndexedRoot
    sx, sy = sympy.symbols('x y')
    parametric = SymPyIndexedRoot(sy**3 + sx*sy - 1, sy, 0)
    native = from_sympy(parametric)
    assert native == indexed_root(y**3 + x*y - 1, y, 0) and to_sympy(native) == parametric
    # a root with rational coefficients becomes SymPy's CRootOf
    number = real_root(real_roots_of([1, 0, -1, -1])[0], 'x')
    assert to_sympy(number) == sympy.CRootOf(sx**3 - sx - 1, 0)
    assert to_sympy(native.subs(x, 1)) == parametric.subs(sx, 1)


def test_native_errors() -> None:
    from sympy_extras.polys.polyerrors import PolynomialError
    assert _raises(ValueError, qe, x > 0, [('some', x)])
    assert _raises(ValueError, qe, x > 0, [('exists', x)], [x])
    assert _raises(PolynomialError, qe, sqrt(x) > 0, [('exists', x)])
    assert _raises(ValueError, qe, Ne(x, 0) ^ (y > 0), [('exists', [x, y])]) is False
