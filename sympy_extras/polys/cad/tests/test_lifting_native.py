"""The native lifting phase, on the typed polynomials and ``Algebraic``
sample points: the cells of known decompositions, the signs checked by an
exact evaluation at the sample points, and the SymPy face (inputs and
outputs converted by interop) on random systems."""
from __future__ import annotations

import random
from fractions import Fraction

import sympy

from sympy_extras.core import Symbol
from sympy_extras.interop import expr_from_sympy as from_sympy
from sympy_extras.polys.cad import lifting_sympy
from sympy_extras.polys.cad.lifting import (CAD, KEPT, Lifting, NotWellOriented, _liftings, cylindrical_algebraic_decomposition,
    factor_signs, lifting_for)
from sympy_extras.polys.cad.projection import projection_sets, to_polys
from sympy_extras.polys.cad.samplepoints import compare_real
from sympy_extras.polys.cad.samplepoints_sympy import core_real
from sympy_extras.polys.numberfields.qqbar import Algebraic

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def _check_signs(cad: CAD) -> None:
    """The signs of the inputs, read off the roots of the stacks, against
    an exact evaluation at the sample points."""
    for cell in cad:
        assert tuple(cell.sample.sign(f, cad.gens) for f in cad.polys) == cell.signs, cell
        assert all(compare_real(a, b) == 0 for a, b in zip(cell.point, cell.sample.values))


def test_circle() -> None:
    cad = cylindrical_algebraic_decomposition([x**2 + y**2 - 1], [x, y])
    assert len(cad) == 13 and cad.gens == (x, y) and cad.method == 'mccallum'
    assert [c.index for c in cad.cells_at(1)] == [(1,), (2,), (3,), (4,), (5,)]
    assert [[c.rational_value() for c in cell.point] for cell in cad.cells_at(1)] == [[-2], [-1], [0], [1], [2]]
    assert [cell.signs for cell in cad.children(cad.cells_at(1)[2])] == [(1,), (0,), (-1,), (0,), (1,)]
    assert [c.dimension for c in cad] == [2, 1, 0, 1, 2, 1, 2, 1, 2, 1, 0, 1, 2]
    _check_signs(cad)


def test_algebraic_sample_points() -> None:
    # the sections over the sections x = +-1/sqrt(2) of the circle and the
    # line: sample points in QQ(sqrt(2))
    cad = cylindrical_algebraic_decomposition([x**2 + y**2 - 1, y - x], [x, y])
    sections = [c for c in cad.cells_at(1) if c.is_section and not c.point[0].is_rational]
    assert [c.point[0].minpoly for c in sections] == [[2, 0, -1], [2, 0, -1]]
    assert all(cell.sample.degree == 2 for c in sections for cell in cad.children(c))
    _check_signs(cad)
    cad = cylindrical_algebraic_decomposition([x**2 + y**2 + z**2 - 1, z - x*y], [x, y, z])
    _check_signs(cad)
    assert max(cell.sample.degree for cell in cad) == 2


def test_lifting_on_demand_and_kept() -> None:
    polys = [x**2 + y**2 + z**2 - 1, z - x*y]
    projection = projection_sets(polys, [x, y, z])
    lifting = Lifting(projection, [x, y, z], 'mccallum')
    line = lifting.stack(lifting.root)
    assert lifting.lifted == len(line) and lifting.is_lifted(lifting.root) and not lifting.is_lifted(line[0])
    assert lifting.stack(lifting.root) is line and lifting.levels() == [line, [], []]
    levels = lifting.lift_all()
    cad = cylindrical_algebraic_decomposition(polys, [x, y, z])
    assert [[c.index for c in level] for level in levels] == [[c.index for c in cad.cells_at(k)] for k in (1, 2, 3)]
    assert [[c._signs for c in level] for level in levels] == [[c._signs for c in cad.cells_at(k)] for k in (1, 2, 3)]
    assert lifting_for(projection, [x, y, z], 'mccallum') is lifting_for(projection, [x, y, z], 'mccallum')
    for k in range(KEPT + 2):
        lifting_for(projection_sets([x - k, y], [x, y]), [x, y], 'mccallum')
    assert len(_liftings) == KEPT


def test_not_well_oriented_and_hong() -> None:
    # x*w + y vanishes identically on the line x = y = 0 in (x, y, z):
    # McCallum's operator is given up for Hong's
    w = Symbol('w')
    try:
        cylindrical_algebraic_decomposition([x*w + y], [x, y, z, w], method='mccallum')
    except NotWellOriented:
        pass
    else:
        raise AssertionError("not well oriented")
    cad = cylindrical_algebraic_decomposition([x*w + y], [x, y, z, w])
    assert cad.method == 'hong' and len(cad) == 21
    zero = [[c.rational_value() for c in cell.point] for cell in cad if cell.signs == (0,)]
    assert zero == [[-1, -1, 0, -1], [-1, 0, 0, 0], [-1, 1, 0, 1], [0, 0, 0, 0], [1, -1, 0, 1], [1, 0, 0, 0],
                    [1, 1, 0, -1]]
    _check_signs(cad)


def test_factor_signs() -> None:
    polys = to_polys([1 - x**2, -(x**2 + y**2 - 1)**2*(x - y)], [x, y])
    projection = projection_sets(polys, [x, y])
    signs = factor_signs(polys, projection, [x, y])
    assert signs[0] == (-1, [(1, 0, 1), (1, 1, 1)])
    assert signs[1][0] == -1 and sorted(e for _, _, e in signs[1][1]) == [1, 2]


def _random(rng: random.Random, gens: list[sympy.Symbol]) -> sympy.Expr:
    return sympy.Add(*[rng.randint(-3, 3)*sympy.Mul(*[g**rng.randint(0, 2) for g in gens])
                       for _ in range(rng.randint(2, 3))])


def test_native_agrees_with_the_face() -> None:
    # the face, on SymPy polynomials, and the native lifting, on the same
    # polynomials converted by interop: the same cells, signs and points
    rng = random.Random(3)
    sx, sy = sympy.symbols('x y')
    for _ in range(8):
        polys = [e for e in (_random(rng, [sx, sy]) for _ in range(2)) if e.free_symbols]
        if not polys:
            continue
        face = lifting_sympy.cylindrical_algebraic_decomposition(polys, [sx, sy])
        native = cylindrical_algebraic_decomposition([from_sympy(p) for p in polys], [x, y])
        assert [c.index for c in face] == [c.index for c in native]
        assert [c.signs for c in face] == [c.signs for c in native]
        for a, b in zip(face, native):
            assert all(compare_real(core_real(u), v) == 0 for u, v in zip(a.point, b.point))
        assert native.method == face.method


def test_points_are_algebraic_numbers() -> None:
    cad = cylindrical_algebraic_decomposition([x**3 - 2], [x])
    assert [isinstance(c.point[0], Algebraic) for c in cad] == [True]*3
    assert cad.cells[1].point[0].minpoly == [1, 0, 0, -2]
    assert cad.cells[0].point[0].rational_value() == 0 and cad.cells[2].point[0].rational_value() == 2
    assert compare_real(cad.cells[1].point[0], Fraction(5, 4)) == 1
