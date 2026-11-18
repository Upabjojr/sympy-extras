"""``triangularize`` (``sympy_extras.polys.regularchains.regularchain``) and
``parametric_cases`` (``sympy_extras.solvers.parametric``) against their
Rust ports: the docstring systems and random small systems with
parameters (linear and quadratic, with inequations), the chains by their
polynomials, main variables and initials, the cases by their conditions,
solutions and equations."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def systems() -> list[tuple[list[Any], list[Any], list[Any]]]:
    from sympy_extras.core import symbols
    a, b, c, x, y, z = symbols('a b c x y z')
    return [([a*x - b], [x], []), ([a*x**2 + b*x + c], [x], []), ([a*x + y - 1, x + a*y - 1], [x, y], []),
            ([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], [x, y, z], []),
            ([x*y - z, x - y], [x, y, z], []), ([x**2 + y**2 - 5, x*y - 1], [x, y], []),
            ([x**2 - 1, y**2 - 1], [x, y], [x - y]), ([a*y**2 - 1, x*y - 1], [x, y], [y])]


def random_system(rng: random.Random) -> tuple[list[Any], list[Any], list[Any]]:
    from sympy_extras.core import Integer, symbols
    a, b, x, y = symbols('a b x y')
    term = lambda: rng.choice([a, b, Integer(1), Integer(-2), a + 1, a*b, Integer(0)])
    monomial = lambda: rng.choice([x, y, x*y, x**2, y**2, Integer(1)])
    poly = lambda: sum((term()*monomial() for _ in range(rng.randint(2, 3))), Integer(0))
    unknowns = rng.choice([[x], [x, y]])
    equations = [poly() for _ in range(len(unknowns))]
    inequations = [rng.choice([x, a, x - 1])] if rng.random() < 0.3 else []
    return equations, unknowns, inequations


def chains_py(eqs: list[Any], syms: list[Any], ineqs: list[Any], kalkbrener: bool) -> Any:
    from sympy_extras.polys.regularchains import triangularize
    chains = triangularize(eqs, *syms, inequations=ineqs, mode='kalkbrener' if kalkbrener else 'lazard')
    return [(str(c.polys), str(c.main_variables), str(c.initials)) for c in chains]


def chains_rs(rc: Any, eqs: list[Any], syms: list[Any], ineqs: list[Any], kalkbrener: bool) -> Any:
    from sympy_extras.core import Integer
    from sympy_extras.core.containers import Tuple
    r = rc.logic_call('triangularize', [Tuple(*eqs), Tuple(*syms), Tuple(*ineqs), Integer(int(kalkbrener))])
    return [tuple(str(list(part.args)) for part in c.args) for c in r.args]


def cases_py(eqs: list[Any], unknowns: list[Any], ineqs: list[Any]) -> Any:
    from sympy_extras.solvers.parametric import parametric_cases
    return [(str(c.condition), str(c.solutions) if c.solutions is not None else 'False', str(c.equations))
            for c in parametric_cases(eqs, unknowns, inequations=ineqs)]


def cases_rs(rc: Any, eqs: list[Any], unknowns: list[Any], ineqs: list[Any]) -> Any:
    from sympy_extras.core.containers import Tuple
    from sympy_extras.core.logic import false
    r = rc.logic_call('parametric_cases', [Tuple(*eqs), Tuple(*unknowns), false, Tuple(*ineqs)])
    return [(str(c.args[0]), str(c.args[1]), str(list(c.args[2].args))) for c in r.args]


@pytest.mark.parametrize("index", range(8))
def test_docstring_systems(rust: Any, index: int) -> None:
    rc = load("sympy_extras_rs.core")
    eqs, unknowns, ineqs = systems()[index]
    for kalkbrener in (False, True):
        assert outcome(lambda: chains_rs(rc, eqs, unknowns, ineqs, kalkbrener)) == \
            outcome(lambda: chains_py(eqs, unknowns, ineqs, kalkbrener)), (eqs, kalkbrener)
    assert outcome(lambda: cases_rs(rc, eqs, unknowns, ineqs)) == outcome(lambda: cases_py(eqs, unknowns, ineqs)), eqs


@pytest.mark.parametrize("seed", range(40))
def test_random_systems(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("parametric", seed))
    eqs, unknowns, ineqs = random_system(rng)
    assert outcome(lambda: chains_rs(rc, eqs, unknowns, ineqs, False)) == \
        outcome(lambda: chains_py(eqs, unknowns, ineqs, False)), eqs
    assert outcome(lambda: cases_rs(rc, eqs, unknowns, ineqs)) == outcome(lambda: cases_py(eqs, unknowns, ineqs)), eqs
