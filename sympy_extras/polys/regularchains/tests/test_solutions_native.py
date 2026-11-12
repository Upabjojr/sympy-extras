"""The native exact solutions of regular chains (no SymPy): the points are
checked by substitution, through the native minimal polynomial."""
from sympy_extras.core import Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import sqrt
from sympy_extras.polys.cad.indexedroot import IndexedRoot, indexed_root
from sympy_extras.polys.complexrootof import ComplexRootOf
from sympy_extras.polys.numberfields.minpoly import minimal_polynomial_coeffs
from sympy_extras.polys.regularchains import triangularize
from sympy_extras.polys.regularchains.solutions import exact_solutions

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def _vanishes(e: Expr) -> bool:
    return minimal_polynomial_coeffs(e) == [1, 0]


def test_points_satisfy_the_system() -> None:
    system = [x**2 + y**2 - 5, x*y - 1, z - x - y]
    found = []
    for chain in triangularize(system, x, y, z):
        found.extend(chain.solutions())
    assert len(found) == 4
    for point in found:
        assert all(_vanishes(p.xreplace(point)) for p in system)


def test_roots_without_radicals() -> None:
    points = exact_solutions([y**3 - y - 1, x**2 - y], [y, x])
    assert len(points) == 6
    for point in points:
        assert _vanishes((x**2 - y).xreplace(point)) and _vanishes((y**3 - y - 1).xreplace(point))
    # the real ones on IndexedRoot, the others on the native CRootOf
    assert isinstance(points[0][y], IndexedRoot) and isinstance(points[0][x], IndexedRoot)
    assert isinstance(points[2][y], ComplexRootOf) and isinstance(points[2][x], ComplexRootOf)
    # SymPy's order (nroots): the two values of x at a complex y by real part
    assert [p[x] for p in points[2:4]] == [ComplexRootOf(x**6 - x**2 - 1, 3), ComplexRootOf(x**6 - x**2 - 1, 4)]
    real = exact_solutions([y**3 - y - 1, x**2 - y], [y, x], real=True)
    assert real == points[:2]


def test_rational_function_of_a_root_is_a_root() -> None:
    # x = y**2 at a root of y**3 - y - 1 is the root of its own polynomial
    points = exact_solutions([y**3 - y - 1, x - y**2], [y, x])
    assert points[0][x] == indexed_root(x**3 - 2*x**2 + x - 1, x, 0)
    assert points[1][x] == ComplexRootOf(x**3 - 2*x**2 + x - 1, 2)


def test_radicals() -> None:
    assert exact_solutions([y**2 - 2, x*y - 1], [y, x]) == [{y: -sqrt(2), x: -sqrt(2)/2}, {y: sqrt(2), x: sqrt(2)/2}]
    assert exact_solutions([y**2 + 1, x - y], [y, x], real=True) == []
