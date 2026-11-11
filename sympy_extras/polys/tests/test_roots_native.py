"""The native radical forms of root objects (SymPy's ``CRootOf``, wrapped
in the core), checked numerically against mpmath's polynomial roots (the
SymPy face is tested in solvers/tests/test_transcendental.py)."""
from __future__ import annotations

import mpmath
import sympy

from sympy_extras.core import Expr, Integer, Symbol, sqrt
from sympy_extras.interop import from_sympy
from sympy_extras.polys.roots import in_radicals, polynomial_root, radical_form
from sympy_extras.sets import finite_set, interval

t = sympy.Symbol('t')


def _root(p: sympy.Expr, index: int) -> Expr:
    r = from_sympy(sympy.CRootOf(p, index))
    assert isinstance(r, Expr)
    return r


def _value(e: Expr) -> mpmath.mpc:
    return mpmath.mpc(complex(e.evalf(30)))


def test_polynomial_root() -> None:
    found = polynomial_root(_root(t**5 - t - 1, 0))
    assert found is not None and found[1] == 0 and found[0].degree() == 5
    assert polynomial_root(Symbol('x')) is None and polynomial_root(sqrt(2)) is None


def test_radical_forms() -> None:
    x = Symbol('x')
    assert radical_form(_root(t**2 - 2, 1)) == sqrt(2) and radical_form(_root(t**2 - 2, 0)) == -sqrt(2)
    assert radical_form(_root(t**3 - 2, 0)) == Integer(2)**(Integer(1)/3)
    # no radicals for an irreducible quintic, nor for the casus irreducibilis
    assert radical_form(_root(t**5 - t - 1, 0)) is None and radical_form(_root(t**3 - 3*t - 1, 0)) is None
    assert radical_form(x) is None
    # a decomposable sextic and quartic: the form is the root, numerically
    for p, index in [(t**6 - 2*t**3 - 1, 0), (t**6 - 2*t**3 - 1, 1), (t**4 - 3*t**2 + 1, 3), (t**8 - 2, 1)]:
        form = radical_form(_root(p, index))
        assert form is not None
        expected = sorted(mpmath.polyroots([int(c) for c in sympy.Poly(p, t).all_coeffs()]),
                          key=lambda r: (abs(mpmath.im(r)) > 1e-20, mpmath.re(r)))[index]
        assert abs(_value(form) - expected) < 1e-12, (p, index, form)


def test_in_radicals() -> None:
    s = in_radicals(finite_set(_root(t**2 - 2, 0), _root(t**5 - t - 1, 0)))
    assert s == finite_set(-sqrt(2), _root(t**5 - t - 1, 0))
    assert in_radicals(interval(_root(t**2 - 3, 0), Integer(5))) == interval(-sqrt(3), Integer(5))
    plain = finite_set(Integer(1), Integer(2))
    assert in_radicals(plain) is plain
