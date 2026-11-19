"""``sympy_extras.steps.systems`` against its Rust port
(``steps/systems.rs``): the trees printed by ``format_steps`` (at several
depths, with and without the minor steps), the solutions and the numbers
of steps, on the systems of its tests (read with the core's parser)."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from ..conftest import load
from .test_integers_agree import outcome

CASES: list[Any] = [
    [[['x + y', '3'], ['x - y', '1']], ['x', 'y'], True, False, 20],
    [[['x + y', '3'], ['x - y', '1']], ['x', 'y'], True, True, 20],
    [[['2*x + 3*y', '7'], ['4*x - y', '7']], ['x', 'y'], True, False, 20],
    [[['2*x + 3*y', '7'], ['4*x - y', '7']], ['x', 'y'], True, True, 20],
    [[['3*x + 2*y', '12'], ['5*x - 3*y', '1']], ['x', 'y'], True, False, 20],
    [[['3*x + 2*y', '12'], ['5*x - 3*y', '1']], ['x', 'y'], True, True, 20],
    [[['x/2 + y/3', '1'], ['x - y', '1/2']], ['x', 'y'], True, False, 20],
    [[['x/2 + y/3', '1'], ['x - y', '1/2']], ['x', 'y'], True, True, 20],
    [[['x + y + z', '6'], ['x - y', '-1'], ['y + z', '5']], ['x', 'y', 'z'], True, False, 20],
    [[['2*x - y + z', '3'], ['x + 2*y - z', '1'], ['3*x + y + 2*z', '11']], ['x', 'y', 'z'], True, False, 20],
    [[['x + y', '1'], ['2*x + 2*y', '3']], ['x', 'y'], True, False, 20],
    [[['x + y', '1'], ['2*x + 2*y', '3']], ['x', 'y'], True, True, 20],
    [[['x**2 + y**2', '5'], ['x + y', '1']], ['x', 'y'], True, False, 20],
    [[['x**2 + y**2', '5'], ['x + y', '1']], ['x', 'y'], True, True, 20],
    [[['y', 'x**2'], ['y', 'x + 2']], ['x', 'y'], True, False, 20],
    [[['y', 'x**2'], ['y', 'x + 2']], ['x', 'y'], True, True, 20],
    [[['x*y', '6'], ['x + y', '5']], ['x', 'y'], True, False, 20],
    [[['x*y', '6'], ['x + y', '5']], ['x', 'y'], True, True, 20],
    [[['x**2 + y**2', '1'], ['y', 'x + 5']], ['x', 'y'], True, False, 20],
    [[['x**2 + y**2', '1'], ['y', 'x + 5']], ['x', 'y'], True, True, 20],
    [[['x**2', 'y'], ['y**2', 'x']], ['x', 'y'], True, False, 20],
    [[['x**2', 'y'], ['y**2', 'x']], ['x', 'y'], True, True, 20],
    [[['x**2 - y', '1'], ['x - y', '-1']], ['x', 'y'], True, False, 20],
    [[['x**2 - y', '1'], ['x - y', '-1']], ['x', 'y'], True, True, 20],
    [[['x**2 + y**2', '4'], ['y', 'x']], ['x', 'y'], True, False, 20],
    [[['x**2 + y**2', '4'], ['y', 'x']], ['x', 'y'], True, True, 20],
    [[['x', '2'], ['x', '3']], ['x'], True, False, 20],
    [[['x', '2'], ['x', '3']], ['x'], True, True, 20],
    [[['x', '2'], ['x**2', '4']], ['x'], True, False, 20],
    [[['x', '2'], ['x**2', '4']], ['x'], True, True, 20],
    [[['x + y', '2'], ['x - y', '0'], ['x*y', '1']], ['x', 'y'], True, False, 20],
    [[['x + y', '2'], ['x - y', '0'], ['x*y', '1']], ['x', 'y'], True, True, 20],
    [[['x + y', '2'], ['x - y', '0'], ['x*y', '5']], ['x', 'y'], True, False, 20],
    [[['x + y', '2'], ['x - y', '0'], ['x*y', '5']], ['x', 'y'], True, True, 20],
    [[['y', '-1/2'], ['-x - y', '5/2']], ['x', 'y'], True, False, 20],
    [[['4*x - y', '21'], ['-2*x + 4*y', '-14']], ['x', 'y'], True, False, 20],
    [[['3*x - 2*y', '3'], ['-x + 4*y', '-1']], ['x', 'y'], True, False, 20],
    [[['-3*x - 3*y', '3/2'], ['4*x', '6']], ['x', 'y'], True, False, 20],
    [[['-x - 3*y', '-27/2'], ['-3*x - y', '-17/2']], ['x', 'y'], True, False, 20],
    [[['-3*x - y', '10'], ['x - 3*y', '10']], ['x', 'y'], True, False, 20],
    [[['3*y', '3'], ['4*x - 3*y', '-3']], ['x', 'y'], True, False, 20],
    [[['-x + 3*y', '-16'], ['2*x - 2*y', '12']], ['x', 'y'], True, False, 20],
    [[['3*x - 4*y + z', '-14'], ['2*x - 3*y', '-10'], ['4*x + 3*y - 4*z', '31']], ['x', 'y', 'z'], True, False, 20],
    [[['4*x - 3*y - 4*z', '11'], ['3*x + y + 2*z', '-18'], ['-4*x - 4*y + 3*z', '26']], ['x', 'y', 'z'], True, False, 20],
    [[['-3*x + 2*y + 3*z', '-12'], ['-3*x - 2*y - z', '20'], ['-y - 3*z', '18']], ['x', 'y', 'z'], True, False, 20],
    [[['4*x + 2*y + 2*z', '-7'], ['3*x + 4*y', '-7/2'], ['-3*x + 3*y + 2*z', '-4']], ['x', 'y', 'z'], True, False, 20],
    [[['3*x - 3*y - 4*z', '-1'], ['-4*x - 2*y - 4*z', '14'], ['-3*x - 3*y + 4*z', '-5']], ['x', 'y', 'z'], True, False, 20],
    [[['3*y + 3*z', '-3/2'], ['4*x + 2*y - z', '-15/2'], ['2*x', '-4']], ['x', 'y', 'z'], True, False, 20],
    [[['3*x + 4*y + 2*z', '19/2'], ['4*x + 4*y + 2*z', '12'], ['2*y + 4*z', '16']], ['x', 'y', 'z'], True, False, 20],
    [[['-2*x - 2*y + z', '-19'], ['-4*x - y + 4*z', '-25'], ['-4*x + 4*y - 4*z', '8']], ['x', 'y', 'z'], True, False, 20],
    [[['sqrt(x)', 'y'], ['y', 'x - 2']], ['x', 'y'], True, False, 20],
    [[['x + y', '1'], ['2*x + 2*y', '2']], ['x', 'y'], True, False, 20],
    [[['2*x + y', '5']], ['x', 'y'], True, False, 20],
    [[['x*exp(x) + y', '1'], ['x + exp(y)', '2']], ['x', 'y'], True, False, 20],
    [[['x**2 + y**2', '1'], ['y', 'x + 5']], ['x', 'y'], False, False, 20],
    [[['x**2 + y', '0'], ['x + y', '1']], ['x', 'y'], False, False, 20],
    [[['x*y', '1'], ['x + y', '1']], ['x', 'y'], False, False, 20],
    [[['x + y', '3'], ['x - y', '1']], ['x', 'y'], False, False, 20],
    [[['x**2', '-1']], ['x'], False, False, 20],
    [[['x**2', '-1']], ['x'], True, False, 20],
    [[['Abs(x)', 'y'], ['y', '2']], ['x', 'y'], True, False, 20],
    [[['Abs(x)', 'y'], ['y', '2']], ['x', 'y'], True, True, 20],
    [[['y + Abs(x)', 'y + 2'], ['y', '2']], ['x', 'y'], True, False, 20],
    [[['Abs(x)', 'y'], ['y', '2']], ['x', 'y'], False, False, 20],
]


@pytest.mark.parametrize("i", range(len(CASES)))
@pytest.mark.parametrize("view", [(None, False), (1, False), (None, True)])
def test_system_steps(rust: Any, i: int, view: tuple[Optional[int], bool]) -> None:
    from sympy_extras.core import Integer, Symbol, Tuple
    from sympy_extras.core.parsing import parse
    from sympy_extras.sets.fancysets import Complexes
    from sympy_extras.sets.sets import Reals
    from sympy_extras.steps.core import count_steps, format_steps
    from sympy_extras.steps.systems import system_steps
    rc = load("sympy_extras_rs.core")
    raw_equations, raw_variables, real, branch, max_depth = CASES[i]
    equations = [(parse(lhs), parse(rhs)) for lhs, rhs in raw_equations]
    variables = [v for v in (parse(name) for name in raw_variables) if isinstance(v, Symbol)]
    assert len(variables) == len(raw_variables)
    depth, minor = view

    def expected() -> Any:
        steps = system_steps(equations, variables, Reals if real else Complexes, branch=branch, max_depth=max_depth)
        found = '?' if steps.contains_dont_know() else str(Tuple(*[Tuple(*s) for s in steps.solutions()]))
        return (format_steps(steps, depth=depth, minor=minor), found, count_steps(steps))

    def got() -> Any:
        r = rc.logic_call('system_steps', [Tuple(*[Tuple(lhs, rhs) for lhs, rhs in equations]), Tuple(*variables),
                                           Integer(int(real)), Integer(int(branch)), Integer(-1 if depth is None else depth),
                                           Integer(int(minor)), Integer(-1 if max_depth is None else max_depth)])
        return (r.args[0].name, str(r.args[1]), int(r.args[2]))
    assert outcome(got) == outcome(expected), CASES[i]
