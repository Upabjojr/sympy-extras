from __future__ import annotations

from sympy import QQ, GF, grevlex, grlex, lex, ring
from sympy.polys.domains.domain import Domain
from sympy.polys.rings import PolyElement

from sympy_extras._typing_sympy import OrderSpec
from sympy.polys.groebnertools import groebner, is_groebner, is_reduced
from sympy.testing.pytest import raises

from sympy_extras.polys.groebnerwalk_sympy import groebner_walk, extended_groebner, initial_form
from sympy_extras.polys.orderings_sympy import WeightOrder


def _check_walk(F: list[str], gens: str, source: OrderSpec, target: OrderSpec,
                domain: Domain = QQ) -> list[PolyElement]:
    R, *xs = ring(gens, domain, source)
    polys = [R.from_expr(f) if not isinstance(f, str) else eval(f, dict(zip(gens.split(","), xs))) for f in F]
    G = groebner(polys, R)
    W = groebner_walk(G, R, target, check=True)
    T = R.clone(order=target)
    expected = groebner([T.from_dict(dict(p)) for p in polys], T)
    assert W == expected, (W, expected)
    assert groebner_walk(G, R, target, lift='cofactors') == expected
    raises(ValueError, lambda: groebner_walk(G, R, target, lift='magic'))
    assert is_groebner(W, T) and is_reduced(W, T)
    return W


def test_extended_groebner() -> None:
    R, x, y = ring("x,y", QQ)
    F = [x**2 - y, x*y - 1]
    H, C = extended_groebner(F, R)
    assert is_groebner(H, R)
    assert all(h == sum(c*f for c, f in zip(cs, F)) for h, cs in zip(H, C))
    H, C = extended_groebner([R.zero, x], R)
    assert H == [x] and C == [[R.zero, R.one]]
    assert initial_form(x**2*y + x*y**2 + x, (1, 2)) == x*y**2
    assert initial_form(x**2*y + x*y**2 + x, (1, 1)) == x**2*y + x*y**2


def test_walk() -> None:
    _check_walk(["x*z - y**2", "x**2 - y*z"], "x,y,z", grevlex, lex)
    _check_walk(["x*z - y**2", "x**2 - y*z"], "x,y,z", lex, grevlex)
    _check_walk(["x*z - y**2", "x**2 - y*z"], "x,y,z", grlex, lex)
    _check_walk(["x*z - y**2", "x**2 - y*z"], "x,y,z", grevlex, grlex)
    _check_walk(["x**2*y - z**3", "x*y*z - w**2", "y**3 - x*w", "x**3 - y*z*w"], "x,y,z,w", grevlex, lex)
    _check_walk(["x**2 + y**2 + z**2 - 1", "x*y - z", "x - y**2"], "x,y,z", grevlex, lex)
    _check_walk(["x**2 + y**2 + z**2 - 1", "x*y - z", "x - y**2"], "x,y,z", lex, grevlex)
    _check_walk(["x**5 - y**3*z", "x*y - z**2", "y**4 - x**2*z"], "x,y,z", grevlex, lex)
    _check_walk(["x**3 - 2*x*y", "x**2*y - 2*y**2 + x"], "x,y", grevlex, lex)
    _check_walk(["x**2 + y**2 - 1", "x - y"], "x,y", grevlex, lex, domain=GF(7))
    _check_walk(["x*z - y**2", "x**2 - y*z"], "x,y,z", grevlex, WeightOrder((1, 2, 3), 'lex'))
    _check_walk(["x*z - y**2", "x**2 - y*z"], "x,y,z", WeightOrder((1, 2, 3), 'lex'), grevlex)
    # trivial cases
    R, x, y = ring("x,y", QQ, grevlex)
    assert groebner_walk([], R, lex) == []
    assert groebner_walk([R.one], R, lex) == [R.clone(order=lex).one]
    T = R.clone(order=lex)
    assert groebner_walk([x**2 + 1], R, lex) == [T.from_expr(x.as_expr()**2 + 1)]
    from sympy.polys.orderings import ilex
    raises(ValueError, lambda: groebner_walk([x**2 + 1], R.clone(order=ilex), lex))


def test_walk_over_rings_and_other_fields() -> None:
    from sympy import EX, ZZ, sqrt
    # over ZZ the walk returned [] for (2*x**2 - 3*y, x*y - 4) (and the
    # cofactor lifting failed): it now walks over QQ and clears the
    # denominators, which is what SymPy's groebner returns over ZZ
    R, x, y = ring("x,y", ZZ, grevlex)
    G = groebner([2*x**2 - 3*y, x*y - 4], R)
    T = R.clone(order=lex)
    expected = groebner([T.from_dict(dict(g)) for g in G], T)
    assert [str(g) for g in expected] == ['8*x - 3*y**2', '3*y**3 - 32']
    for lift in ('normal_form', 'cofactors'):
        assert groebner_walk(G, R, lex, lift=lift) == expected
    # an algebraic field
    A = QQ.algebraic_field(sqrt(2))
    S, u, v = ring("u,v", A, grevlex)
    F = [u**2 - A.from_sympy(sqrt(2))*v, u*v - 1]
    G2 = groebner(F, S)
    L = S.clone(order=lex)
    assert groebner_walk(G2, S, lex, check=True) == groebner([L.from_dict(dict(f)) for f in F], L)
    # EX is not converted to the typed layer
    E, p, q = ring("p,q", EX, grevlex)
    raises(NotImplementedError, lambda: groebner_walk([p**2 - q], E, lex))
