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


def test_roots_over_ex_are_native() -> None:
    # roots_of_core sent these to SymPy's roots: the components of a
    # decomposition over EX, and coefficients like pi; checked by substitution
    from sympy_extras.core import Symbol, pi, expand
    from sympy_extras.polys.polyroots import ex_roots, roots
    from sympy_extras.polys.polytools import Poly
    x = Symbol('x')
    for f in [x**2 - pi, x**2 + pi*x + 1, pi*x - 1, x**2 - (1 + pi)*x + pi]:
        found = ex_roots(f, x)
        assert len(found) == 2 or (len(found) == 1 and f == pi*x - 1), (f, found)
        for r in found:
            assert abs(complex(f.subs(x, r).evalf(30))) < 1e-20, (f, r)
    g = expand((x**2 + x)**3 - 2)
    listed = roots(Poly(g, x))
    assert isinstance(listed, dict) and sum(listed.values()) == 6
    for r in listed:
        assert abs(complex(g.subs(x, r).evalf(30))) < 1e-20, r
    # degree five over EX: nothing, as SymPy without quintics
    assert roots(Poly(x**10 - 2*x**6 - 2*x**5 + x**2 + 2*x - 1, x)) == {}
