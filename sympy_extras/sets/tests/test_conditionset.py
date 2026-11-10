"""Condition sets, images of sets, Cartesian products and finite sets of
points against SymPy's (the oracle): the evaluation of the constructors,
the printing and the conversion both ways."""
from __future__ import annotations

import sympy


from sympy_extras.core import Dummy, I, Integer, Symbol, Tuple, sqrt
from sympy_extras.core.logic import false, true
from sympy_extras.interop import SymPySetNode, from_sympy, to_sympy
from sympy_extras.sets import (ConditionSet, Complexes, EmptySet, ImageSet, Integers, Intersection, PointSet,
    ProductSet, Reals, condition_set, finite_set, image_set, interval, point_set, product_set)

x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
sx, sy, sa = sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('a')


def test_condition_set_evaluates_as_sympy() -> None:
    cases = [
        (condition_set(x, x > 0, finite_set(-1, 2, a)), sympy.ConditionSet(sx, sx > 0, sympy.FiniteSet(-1, 2, sa))),
        (condition_set(x, x > 0, finite_set(-1, 2)), sympy.ConditionSet(sx, sx > 0, sympy.FiniteSet(-1, 2))),
        (condition_set(x, a > 0, Reals), sympy.ConditionSet(sx, sa > 0, sympy.S.Reals)),
        (condition_set(x, true, interval(0, 1)), sympy.ConditionSet(sx, sympy.true, sympy.Interval(0, 1))),
        (condition_set(x, x > 0, EmptySet), sympy.ConditionSet(sx, sx > 0, sympy.S.EmptySet)),
        (condition_set(x, x < 1, condition_set(x, x > 0, Reals)),
         sympy.ConditionSet(sx, sx < 1, sympy.ConditionSet(sx, sx > 0, sympy.S.Reals))),
        (condition_set(Tuple(x, y), x < y, product_set(Reals, Reals)),
         sympy.ConditionSet((sx, sy), sx < sy, sympy.ProductSet(sympy.S.Reals, sympy.S.Reals))),
    ]
    for ours, theirs in cases:
        assert str(ours) == str(theirs), (ours, theirs)
        assert to_sympy(ours) == theirs


def test_condition_set_membership() -> None:
    c = ConditionSet(x, x > 0, interval(-1, 1))
    half = Integer(1)/2
    assert c.contains(half) is true and c.contains(-half) is false and c.contains(Integer(2)) is false
    assert str(c.contains(a)) == str(sympy.ConditionSet(sx, sx > 0, sympy.Interval(-1, 1)).contains(sa))
    assert ConditionSet(Tuple(x, y), x < y, Reals).contains(a) is false
    assert c.free_symbols == set()
    assert ConditionSet(x, x > a, Reals).free_symbols == {a}
    assert ConditionSet(Tuple(x, y), x < y, Reals).symbols == (x, y)


def test_images_products_and_points() -> None:
    n = Dummy('n', integer=True)
    image = image_set(Tuple(n), 2*n, Integers)
    assert isinstance(image, ImageSet) and str(image) == 'ImageSet(Lambda(_n, 2*_n), Integers)'
    assert image_set(Tuple(n), n, Integers) == Integers
    assert image_set(Tuple(n), Integer(3), Integers) == finite_set(3)
    assert image_set(Tuple(n), Tuple(3, 4), Integers) == point_set(Tuple(3, 4))
    # SymPy's signature check: one base set for each variable
    m = Dummy('m', integer=True)
    try:
        image_set(Tuple(n, m), Tuple(n, m), product_set(Integers, Integers))
    except ValueError:
        pass
    else:
        raise AssertionError('the signature does not match the base sets')
    assert product_set(Reals, EmptySet) == EmptySet and isinstance(product_set(Reals, Complexes), ProductSet)
    points = point_set(Tuple(1, 2), Tuple(0, x), Tuple(1, 2))
    assert isinstance(points, PointSet) and len(points) == 2 and str(points) == '{(0, x), (1, 2)}'
    assert point_set() == EmptySet


def test_conversions_round_trip() -> None:
    k, t0 = sympy.Dummy('k', integer=True), sympy.Symbol('t0')
    for theirs in [sympy.ConditionSet(sx, sx > 0, sympy.FiniteSet(-sympy.sqrt(sa), sympy.sqrt(sa))),
                   sympy.ConditionSet((sx, sy), sx < sy, sympy.ProductSet(sympy.S.Reals, sympy.S.Reals)),
                   sympy.ImageSet(sympy.Lambda(k, 2*k*sympy.pi), sympy.S.Integers),
                   sympy.ImageSet(sympy.Lambda(t0, sympy.Tuple(44 - 5*t0, 3*t0 - 22)), sympy.S.Integers),
                   sympy.FiniteSet((1, 2), (sx, 3)), sympy.ProductSet(sympy.S.Reals, sympy.S.Integers),
                   sympy.Range(-3, 4)]:
        ours = from_sympy(theirs)
        assert to_sympy(ours) == theirs, theirs
    assert isinstance(from_sympy(sympy.Range(-3, 4)), SymPySetNode)
    # SymPy's Interval(-oo, oo) equals its Reals, and is not converted to it
    assert type(to_sympy(from_sympy(sympy.Interval(-sympy.oo, sympy.oo)))) is sympy.Interval
    assert to_sympy(from_sympy(sympy.S.Reals)) is sympy.S.Reals


def test_sets_of_complex_numbers_are_ordered() -> None:
    # the bug: the infimum of {-I, I} raised ValueError (Min of numbers
    # which are not comparable), which the key of the order of the sets of
    # an intersection did not catch (SymPy's does)
    found = Intersection(finite_set(-I, I, sqrt(2)), Reals)
    assert len(found.args) == 2
