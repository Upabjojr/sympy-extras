"""``sympy_extras.solvers.thue_equation`` against its Rust port: the
solutions of Thue equations (cubic and quartic forms, real and complex
fields, right-hand sides with several classes of elements), the errors on
the forms which are not irreducible, homogeneous, integral, and the
``NotImplementedError`` of a too large right-hand side; the unit groups
(the same torsion and regulator, the fundamental systems themselves may
differ by units of equal heights), the elements of a given norm and the
Fincke–Pohst enumeration."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _cases() -> list[tuple[Any, int]]:
    from sympy_extras.core import symbols
    x, y = symbols('x y')
    return [(x**3 - 2*y**3, 1), (x**3 - 2*y**3, -1), (x**3 + x**2*y - 2*x*y**2 - y**3, 1),
            (x**3 - 3*x*y**2 - y**3, 1), (x**3 - 2*y**3, 5), (x**3 - 2*y**3, 3), (x**3 + 2*y**3, 1),
            (x**3 - 3*x*y**2 - y**3, -1), (2*x**3 + 3*y**3, 5), (x**3 + x*y**2 + y**3, 1), (x**3 - 2*y**3, 6),
            (x**4 + y**4, 2), (x**4 + y**4, 3), (x**4 - 2*y**4, -1), (x**4 - 2*y**4, 1), (x**3 - x*y**2 + y**3, -3),
            (x**3 - x*y**2 + y**3, 7), (2*x**3 + 4*y**3, 2), (x**3 - 2*y**3, 0),
            # errors
            (x**2 - 2*y**2, 1), (x**3 - y**3, 1), (x**3 + x*y, 1), (x*y**2 + y**3, 1), (x**3/2 - y**3, 1)]


@pytest.mark.parametrize("index", range(24))
def test_thue(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, symbols
    from sympy_extras.solvers.thue_equation import thue
    rc = load("sympy_extras_rs.core")
    x, y = symbols('x y')
    F, m = _cases()[index]

    def got() -> Any:
        return [tuple(int(v) for v in s.args) for s in rc.logic_call('thue', [F, Integer(m), x, y]).args]
    assert outcome(got) == outcome(lambda: thue(F, m, x, y)), (F, m)


def test_thue_too_large(rust: Any) -> None:
    from sympy_extras.core import Integer, symbols
    from sympy_extras.solvers.thue_equation import thue
    rc = load("sympy_extras_rs.core")
    x, y = symbols('x y')
    F = x**3 - 2*y**3
    assert outcome(lambda: rc.logic_call('thue', [F, Integer(10**5), x, y])) == \
        outcome(lambda: thue(F, 10**5, x, y)) == 'SKIP'


def _tuple(values: Any) -> Any:
    from sympy_extras.core import Integer, Tuple
    if isinstance(values, (list, tuple)):
        return Tuple(*[_tuple(v) for v in values])
    return Integer(values)


def _regulator(g: list[int], units: list[tuple[int, ...]]) -> Any:
    import mpmath
    with mpmath.workdps(40):
        roots = mpmath.polyroots(g, maxsteps=200, extraprec=200)
        real = [r for r in roots if abs(mpmath.im(r)) < mpmath.mpf(10)**-20]
        upper = [r for r in roots if mpmath.im(r) > mpmath.mpf(10)**-20]
        places = [(r, 1) for r in real] + [(r, 2) for r in upper]
        rows = [[w*mpmath.log(abs(sum(c*r**i for i, c in enumerate(u)))) for r, w in places[:-1]] for u in units]
        return mpmath.nstr(abs(mpmath.det(mpmath.matrix(rows))) if rows else mpmath.mpf(1), 25)


@pytest.mark.parametrize("g", [[1, 0, 0, -2], [1, 0, -3, -1], [1, 1, -2, -1], [1, 0, 0, 0, -2], [1, 0, 0, 0, 1],
                               [1, 0, 1, 1], [1, 0, 0, 3], [1, -1, -1, -1]])
def test_units_of_order(rust: Any, g: list[int]) -> None:
    from sympy_extras.solvers.thue_equation import units_of_order
    rc = load("sympy_extras_rs.core")

    def got() -> Any:
        units, torsion = rc.logic_call('thue_units', [_tuple(g)]).args
        units_ = [tuple(int(c) for c in u.args) for u in units.args]
        return (len(units_), _regulator(g, units_), [tuple(int(c) for c in u.args) for u in torsion.args])

    def expected() -> Any:
        units, torsion = units_of_order(g)
        return (len(units), _regulator(g, units), torsion)
    assert outcome(got) == outcome(expected), g


@pytest.mark.parametrize("norm", [1, 2, -2, 3, 5, 6, 10])
def test_elements_of_norm(rust: Any, norm: int) -> None:
    from sympy_extras.solvers.thue_equation import elements_of_norm, units_of_order
    rc = load("sympy_extras_rs.core")
    g = [1, 0, 0, -2]
    units = units_of_order(g)[0]
    assert outcome(lambda: [tuple(int(c) for c in e.args) for e in
                            rc.logic_call('thue_norm', [_tuple(g), _tuple(norm), _tuple(units)]).args]) == \
        outcome(lambda: elements_of_norm(g, norm, units)), norm


@pytest.mark.parametrize("rows, radius", [([[2, 0], [0, 3]], (7, 2)), ([[1, 0, 3], [0, 1, 5], [0, 0, 7]], (9, 1)),
                                          ([[5, 1], [1, 4]], (13, 2)), ([[3, 1, 0], [1, 3, 1], [0, 1, 3]], (5, 1))])
def test_short_vectors(rust: Any, rows: list[list[int]], radius: tuple[int, int]) -> None:
    from sympy_extras.core import Integer
    from sympy_extras.solvers.thue_equation import short_vectors
    rc = load("sympy_extras_rs.core")
    r = Integer(radius[0])/radius[1]
    got = [[int(c) for c in v.args] for v in rc.logic_call('thue_short', [_tuple(rows), r]).args]
    assert got == short_vectors(rows, radius[0]/radius[1])
