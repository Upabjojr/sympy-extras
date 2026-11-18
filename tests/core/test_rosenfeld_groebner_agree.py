"""``sympy_extras.polys.differential.rosenfeld_groebner`` against its
Rust port: the components of the radical of differential ideals (their
equations, inequations, leaders and the Gröbner bases of the saturations
`(A) : H^\\infty`), Ritt's remainders, and the membership by Rosenfeld's
lemma through the regular chains, on ordinary and partial systems,
singular solutions, elimination rankings and inconsistent systems."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _cases() -> list[tuple[list[Any], list[Any], Any, list[Any], list[Any]]]:
    from sympy_extras.core import Function, Symbol
    x, y, t, c, g = [Symbol(n) for n in 'xytcg']
    u, v, w, X, Y, L, f, a = [Function(n) for n in ('u', 'v', 'w', 'X', 'Y', 'L', 'f', 'a')]
    U = u(x)
    W, V = w(x, y), v(x, y)
    Xt, Yt, Lt = X(t), Y(t), L(t)
    pendulum = [Xt.diff(t, 2) + Lt*Xt, Yt.diff(t, 2) + Lt*Yt + g, Xt**2 + Yt**2 - 1]
    return [
        ([U.diff(x)**2 - 4*U], [U], None, [], [U.diff(x, 2) - 2, U.diff(x), U, U.diff(x)**3 - 4*U*U.diff(x)]),
        ([U - x*U.diff(x) - U.diff(x)**2], [U], None, [], [U.diff(x, 2), x**2 + 4*U, x + 2*U.diff(x)]),
        ([U.diff(x) - U, U.diff(x, 2) - U - 1], [U], None, [], [U]),
        ([U.diff(x)**2 + 1, U.diff(x, 2) - 1], [U], None, [], [U]),
        ([U.diff(x) - U], [U], None, [U.diff(x, 2) - U], [U]),
        ([U.diff(x) - U], [U], None, [], [U.diff(x, 2) - U, U.diff(x)]),
        ([U, U - 1], [U], None, [], [U.diff(x) - 7]),
        ([U.diff(x)**2 - 4*U], [U], None, [U], [U.diff(x, 2) - 2]),
        ([U.diff(x)*(U.diff(x) - 1)], [U], None, [U.diff(x)], [U.diff(x, 2), U.diff(x) - 1]),
        ([U.diff(x)*(U.diff(x) - 1)], [U], None, [], [U.diff(x, 2), U.diff(x) - 1]),
        (pendulum, [Lt, Xt, Yt], None, [], [Xt*Xt.diff(t) + Yt*Yt.diff(t), Lt - Xt.diff(t)**2 - Yt.diff(t)**2 + g*Yt,
                                             Lt.diff(t) + 3*g*Yt.diff(t), Xt, Lt + g*Yt]),
        (pendulum, [Xt, Yt, Lt], [[Xt, Yt], [Lt]], [], [Lt.diff(t) + 3*g*Yt.diff(t)]),
        ([W.diff(x)**2 - 4*W, W.diff(x, y)*V.diff(y) - W + 1, V.diff(x, 2) - W.diff(x)], [W, V], None, [],
         [W.diff(y)**2 - 2*W, W.diff(y)]),
        ([f(x, y).diff(x, 2) - y**2*f(x, y), f(x, y).diff(y) - x*f(x, y)], [f(x, y)], None, [],
         [f(x, y).diff(x, y) - f(x, y) - x*y*f(x, y)]),
        ([U.diff(x)**2 + U**2 - 1], [U], None, [], [U.diff(x, 2) + U]),
        ([v(x) - U.diff(x), v(x).diff(x) + U], [U, v(x)], None, [], [U.diff(x, 2) + U]),
        ([x*U.diff(x) - c*U], [U], None, [], [x**2*U.diff(x, 2) - c*(c - 1)*U]),
        ([W.diff(y) - a(x)*W, W.diff(x)], [W, a(x)], None, [], [a(x).diff(x)*W, W, a(x).diff(x)]),
        ([U.diff(x)**2 - U**3 + 2*U - 1], [U], None, [], [2*U.diff(x, 2) - 3*U**2 + 2]),
    ]


@pytest.mark.parametrize("index", range(19))
def test_rosenfeld_groebner(rust: Any, index: int) -> None:
    from sympy_extras.core import Tuple
    from sympy_extras.polys.differential import rosenfeld_groebner
    rc = load("sympy_extras_rs.core")
    equations, functions, ranking, inequations, probes = _cases()[index]

    def expected() -> Any:
        ideal = rosenfeld_groebner(equations, functions, ranking=ranking, inequations=inequations)
        components = [[[str(e) for e in c.equations], [str(e) for e in c.inequations], [str(e) for e in c.leaders],
                       sorted(str(e) for e in c.saturation())] for c in ideal.components]
        return (components, [ideal.contains(p) for p in probes],
                [[str(c.remainder(p)) for c in ideal.components] for p in probes])

    def got() -> Any:
        blocks = Tuple(*[Tuple(*b) for b in ranking]) if ranking else Tuple()
        r = rc.logic_call('rosenfeld', [Tuple(*equations), Tuple(*functions), Tuple(), blocks, Tuple(*inequations),
                                        Tuple(*probes)])
        components = [[[str(e) for e in part.args] for part in c.args[:3]] + [sorted(str(e) for e in c.args[3].args)]
                      for c in r.args[0].args]
        return (components, [bool(v) for v in r.args[1].args], [[str(v) for v in row.args] for row in r.args[2].args])
    assert outcome(got) == outcome(expected), equations
