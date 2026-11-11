"""``janet_basis`` of the Rust port against
``sympy_extras.polys.differential.janet``: Janet's example, an
inconsistent system, a compatibility condition by elimination, and random
linear systems with polynomial coefficients; the equations, leaders,
multiplicative variables, dimension, Hilbert function, series and
polynomial, parametric derivatives, normal forms and series solutions."""

from __future__ import annotations

import random
from typing import Any, Optional

import pytest

from ..conftest import load, stable_seed


def python_ops(equations: list[Any], functions: list[Any], t: Any, s: Any, probes: list[Any], order: int,
               ranking: Optional[list[list[Any]]] = None) -> dict[str, Any]:
    from sympy_extras.polys.differential.janet import janet_basis
    B = janet_basis(equations, functions, ranking=ranking)
    result: dict[str, Any] = {"equations": B.equations, "leaders": B.leaders,
                              "multiplicative_variables": B.multiplicative_variables,
                              "is_consistent": B.is_consistent}
    if B.is_consistent:
        result.update(dimension=B.dimension, hilbert_function=[B.hilbert_function(q) for q in range(order + 1)],
                      hilbert_series=B.hilbert_series(t), hilbert_polynomial=B.hilbert_polynomial(s),
                      parametric_derivatives=B.parametric_derivatives(order),
                      reduce=[B.reduce(p) for p in probes], contains=[B.contains(p) for p in probes])
        try:
            result["series_solution"] = B.series_solution(order)
        except NotImplementedError:
            pass
        except ValueError as error:
            result["series_solution_error"] = str(error)
    return result


def agree(got: dict[str, Any], expected: dict[str, Any]) -> None:
    if "series_solution" not in got:
        expected = {k: v for k, v in expected.items() if k != "series_solution"}
    assert set(got) == set(expected)
    for key in expected:
        assert repr(got[key]) == repr(expected[key]), key


def symbols() -> tuple[Any, ...]:
    from sympy_extras.core import Symbol
    return tuple(Symbol(c) for c in 'xyzts')


def test_known_systems(rust: Any) -> None:
    from sympy_extras.core import Eq, Function
    rc = load("sympy_extras_rs.core")
    x, y, z, t, s = symbols()
    u = Function('u')(x, y, z)
    janet = [u.diff(z, 2) + y * u.diff(x, 2), u.diff(y, 2)]
    agree(rc.janet_basis_ops(janet, [u], t, s, [u.diff(x, y, z), u.diff(y, 3)], 3),
          python_ops(janet, [u], t, s, [u.diff(x, y, z), u.diff(y, 3)], 3))
    w = Function('w')(x, y)
    inconsistent = [w.diff(x) - 1, w.diff(x) - 2]
    agree(rc.janet_basis_ops(inconsistent, [w], t, s, [], 2), python_ops(inconsistent, [w], t, s, [], 2))
    f, g = Function('f')(x, y), Function('g')(x, y)
    compatibility = [Eq(w.diff(x), f), Eq(w.diff(y), g)]
    ranking = [[w], [f, g]]
    agree(rc.janet_basis_ops(compatibility, [w, f, g], t, s, [w.diff(x, y)], 1, ranking=ranking),
          python_ops(compatibility, [w, f, g], t, s, [w.diff(x, y)], 1, ranking))
    v = Function('v')(x)
    ode = [v.diff(x, 4) + v]
    agree(rc.janet_basis_ops(ode, [v], t, s, [v.diff(x, 5)], 5), python_ops(ode, [v], t, s, [v.diff(x, 5)], 5))


@pytest.mark.parametrize("seed", range(25))
def test_random_systems(rust: Any, seed: int) -> None:
    from sympy_extras.core import Function, Integer
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    rc = load("sympy_extras_rs.core")
    x, y, z, t, s = symbols()
    u = Function('u')(x, y)
    rng = random.Random(stable_seed("janet", seed))
    derivatives = [u, u.diff(x), u.diff(y), u.diff(x, 2), u.diff(x, y), u.diff(y, 2)]
    coefficients = [Integer(1), Integer(-1), Integer(2), x, y, x + y, Integer(0)]
    equations = []
    for _ in range(rng.randint(1, 3)):
        equations.append(add(*[mul(rng.choice(coefficients), d) for d in rng.sample(derivatives, rng.randint(1, 3))]))
    equations = [e for e in equations if e != 0] or [u.diff(x, 2)]
    probes = [u.diff(x, 3), u.diff(x, y, 2) + u, x * u.diff(y)]
    agree(rc.janet_basis_ops(equations, [u], t, s, probes, 3), python_ops(equations, [u], t, s, probes, 3))
