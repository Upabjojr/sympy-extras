"""The native projection operators, on the typed ``Poly`` and symbols of the
core: hand computations, SymPy's resultants, discriminants and principal
subresultant coefficients as the oracle, and the SymPy face on the same
input."""
from __future__ import annotations

import random

import sympy

from sympy_extras.core import Integer, Symbol
from sympy_extras.functions import sqrt
from sympy_extras.interop import expr_from_sympy as from_sympy, to_sympy
from sympy_extras.polys.cad import projection_sympy
from sympy_extras.polys.cad.projection import (PROJECTIONS, _coefficients, _psc, _reducta,
    augmented_projection_sets, hong_projection, mccallum_projection, projection_sets, squarefree_basis, to_polys)
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.polytools import Poly

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def test_to_polys_and_squarefree_basis() -> None:
    assert to_polys([x/2 - y/3], [x, y]) == [Poly(3*x - 2*y, x, y)]
    assert squarefree_basis([(x**2 - 1)*(x + y)**2, 6], [x, y]) == [
        [Poly(x - 1, x, y), Poly(x + 1, x, y)], [Poly(x + y, x, y)]]
    assert squarefree_basis([-x + 1, 2*x - 2, x - 1], [x]) == [[Poly(x - 1, x)]]
    for bad in [x + z, sqrt(Integer(2))*x]:
        try:
            squarefree_basis([bad], [x, y])
        except PolynomialError:
            continue
        raise AssertionError("%s is not a polynomial in x, y with rational coefficients" % (bad,))


def test_coefficients_reducta_and_psc() -> None:
    f = Poly(x*y**2 + (x - 1)*y + 3, y, x)
    assert [Poly.from_dict(dict(c), x) for c in _coefficients(f)] == [Poly(x, x), Poly(x - 1, x), Poly(3, x)]
    assert _reducta(f) == [f, Poly((x - 1)*y + 3, y, x), Poly(3, y, x)]
    h = Poly(x**3 - 2*x, x)
    assert _reducta(h) == [h, Poly(-2*x, x)]
    # the principal subresultant coefficients of the circle and its
    # derivative, then of the circle and a line, by hand: psc_0 is the
    # resultant, psc_1 the leading coefficient of the line (constant, left
    # out)
    circle = Poly(x**2 + y**2 - 1, y, x)
    assert _psc(circle, circle.diff(0)) == [Poly(4*x**2 - 4, x)]
    assert _psc(circle, Poly(y - x, y, x)) == [Poly(2*x**2 - 1, x)]


def test_operators_by_hand() -> None:
    circle = Poly(x**2 + y**2 - 1, x, y)
    assert mccallum_projection([circle], y) == [Poly(x - 1, x), Poly(x + 1, x)]
    assert hong_projection([circle], y) == [Poly(x - 1, x), Poly(x + 1, x)]
    assert mccallum_projection([Poly(x*y - 1, x, y), Poly(y - x, x, y)], y,
                               equational=[Poly(y - x, x, y)]) == [Poly(x - 1, x), Poly(x + 1, x)]
    assert set(PROJECTIONS) == {'mccallum', 'hong'}
    assert projection_sets([x**2 + y**2 - 1, x - y], [x, y], equational=x - y) == [
        [Poly(2*x**2 - 1, x)], [Poly(x**2 + y**2 - 1, x, y), Poly(x - y, x, y)]]
    projection = projection_sets([x - y**2], [x, y])
    assert augmented_projection_sets(projection, [x, y], [y - 1]) == [
        [Poly(x, x), Poly(x - 1, x)], [Poly(x - y**2, x, y), Poly(y - 1, x, y)]]
    try:
        projection_sets([circle], [x, y], method='collins')
    except ValueError:
        pass
    else:
        raise AssertionError("unknown method")


def _random_polynomial(rng: random.Random, gens: list[sympy.Symbol]) -> sympy.Expr:
    terms = [rng.randint(-3, 3)*sympy.Mul(*[g**rng.randint(0, 2) for g in gens]) for _ in range(rng.randint(2, 4))]
    return sympy.Add(*terms)


def test_native_agrees_with_the_face_on_random_systems() -> None:
    # the face converts SymPy's polynomials to the native ones and back:
    # the native sets, converted by interop, are the face's, in the same
    # order (the order of the factors fixes the indices of the signs)
    rng = random.Random(7)
    sx, sy, sz = sympy.symbols('x y z')
    for _ in range(12):
        gens = [sx, sy, sz][:rng.randint(2, 3)]
        polys = [_random_polynomial(rng, gens) for _ in range(rng.randint(1, 3))]
        polys = [p for p in polys if p.free_symbols]
        if not polys:
            continue
        core_gens = [from_sympy(g) for g in gens]
        for method in ('mccallum', 'hong'):
            face = projection_sympy.projection_sets(polys, gens, method)
            native = projection_sets([from_sympy(p) for p in polys], core_gens, method)
            assert [[str(to_sympy(p.as_expr())) for p in level] for level in native] == \
                [[str(p.as_expr()) for p in level] for level in face], (polys, method)


def _sympy_factors(items: list[sympy.Expr], gens: list[sympy.Symbol]) -> set[sympy.Expr]:
    """The irreducible non-constant factors with a positive leading
    coefficient, by SymPy."""
    found: set[sympy.Expr] = set()
    for e in items:
        if e.is_number:
            continue
        for g, _ in sympy.Poly(e, *gens).factor_list()[1]:
            if g.is_ground:
                continue
            found.add((-g if g.LC() < 0 else g).as_expr())
    return found


def test_mccallum_and_hong_against_sympy() -> None:
    # the projection of the last level, from SymPy's coefficients,
    # discriminants and resultants (McCallum) and from the principal
    # subresultant coefficients of sympy_extras.polys.subresultants, on
    # SymPy's dense polynomials (Hong)
    from itertools import combinations
    from sympy_extras.polys.subresultants import psc
    rng = random.Random(11)
    sx, sy = sympy.symbols('x y')
    R, Ry, _ = sympy.ring("y,x", sympy.ZZ)
    for _ in range(15):
        polys = [_random_polynomial(rng, [sx, sy]) + sy**rng.randint(1, 3) for _ in range(rng.randint(1, 3))]
        basis = sorted(_sympy_factors(polys, [sx, sy]) - {g for g in _sympy_factors(polys, [sx, sy])
                                                          if not g.has(sy)}, key=str)
        if not basis:
            continue
        mains = [sympy.Poly(g, sy, sx) for g in basis]
        items: list[sympy.Expr] = []
        for f in mains:
            for c in f.all_coeffs() if f.is_univariate else sympy.Poly(f.as_expr(), sy).all_coeffs():
                if c.is_number:
                    if c != 0:
                        break
                    continue
                items.append(c)
            if f.degree(sy) >= 2:
                items.append(sympy.discriminant(f.as_expr(), sy))
        for f, g in combinations(mains, 2):
            items.append(sympy.resultant(f.as_expr(), g.as_expr(), sy))
        native = mccallum_projection([Poly(from_sympy(g), x, y) for g in basis], y)
        assert {to_sympy(p.as_expr()) for p in native} == _sympy_factors(items, [sx]), polys
        hong_items: list[sympy.Expr] = []
        for i, f in enumerate(mains):
            r = R(f.as_expr())
            reducta = []
            while r:
                reducta.append(r)
                top = r.degree(Ry)
                r = R({m: c for m, c in r.terms() if m[0] != top})
            for r in reducta:
                hong_items.append(sympy.Poly(r.as_expr(), sy).LC() if r.degree(Ry) > 0 else r.as_expr())
                if r.degree(Ry) >= 2:
                    hong_items.extend(c.as_expr() for c in psc(r, r.diff(Ry)))
            for g in mains[i + 1:]:
                for r in reducta:
                    if r.degree(Ry) == 0:
                        hong_items.append(r.as_expr())
                    else:
                        hong_items.extend(c.as_expr() for c in psc(r, R(g.as_expr())))
        native_hong = hong_projection([Poly(from_sympy(g), x, y) for g in basis], y)
        assert {to_sympy(p.as_expr()) for p in native_hong} == _sympy_factors(
            [sympy.sympify(e) for e in hong_items], [sx]), polys
