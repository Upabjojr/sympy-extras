"""``sympy_extras.steps.equations`` against its Rust port
(``steps/equations.rs``): the trees printed by ``format_steps`` (at
several depths, with and without the minor steps), the solutions and the
numbers of steps, on the equations of its tests (read with the core's
parser)."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from ..conftest import load
from .test_integers_agree import outcome

CASES: list[tuple[str, str, str, bool, bool, Optional[int]]] = [
    ('-11757312*y', '-11757312', 'y', True, False, 30),
    ('-12*x - 4*y - 60', '-12', 'x', True, False, 30),
    ('-1772928*y', '-5318784', 'y', True, False, 30),
    ('-191476224*x', '-382952448', 'x', True, False, 30),
    ('-1959552*y', '-1959552', 'y', True, False, 30),
    ('-2', '-3/2', 'x', True, False, 30),
    ('-2**x + 4**x - 2', '0', 'x', False, False, 30),
    ('-2**x + 4**x - 2', '0', 'x', True, False, 30),
    ('-2*y', '-2', 'y', True, False, 30),
    ('-228*y', '-684', 'y', True, False, 30),
    ('-3**x + 4**x', '1', 'x', True, False, 30),
    ('-31912704*x', '-63825408', 'x', True, False, 30),
    ('-38*y', '-114', 'y', True, False, 30),
    ('-4*y/3 - 2*z/3 + 38/3', '12', 'y', True, False, 30),
    ('-4104*x', '-8208', 'x', True, False, 30),
    ('-5*x/36', '-7/36', 'x', True, False, 30),
    ('-5*x/46656', '-7/46656', 'x', True, False, 30),
    ('-5*x/7776', '-7/7776', 'x', True, False, 30),
    ('-684*x', '-1368', 'x', True, False, 30),
    ('-70543872*y', '-70543872', 'y', True, False, 30),
    ('-8208*y', '-24624', 'y', True, False, 30),
    ('-886464*x', '-1772928', 'x', True, False, 30),
    ('-sqrt(4 - 4*x**2)/2', 'x + 5', 'x', True, False, 30),
    ('-x + 3*y', '-16', 'x', True, False, 30),
    ('-x - 3*y', '-27/2', 'x', True, False, 30),
    ('-x**2 + 5*x', '6', 'x', True, False, 30),
    ('-x**2 + x + 1', '-1', 'x', True, False, 30),
    ('-y + sqrt(4*y + 4)/2 - 1', '-2', 'y', True, False, 30),
    ('-y + sqrt(4*y + 4)/2', '-1', 'y', True, False, 30),
    ('-y**2 + 5*y', '6', 'y', True, False, 30),
    ('0', '-sqrt(y) + y - 2', 'y', True, False, 30),
    ('1/2 - x', '5/2', 'x', True, False, 30),
    ('10*x + 30', '10', 'x', True, False, 30),
    ('1176 - 336*x', '504', 'x', True, False, 30),
    ('12312*x - 49248', '-24624', 'x', True, False, 30),
    ('1306368*x - 4572288', '-1959552', 'x', True, False, 30),
    ('1306368*x', '2612736', 'x', True, False, 30),
    ('1368 - 228*y', '684', 'y', True, False, 30),
    ('14*x - 21', '7', 'x', True, False, 30),
    ('14*y', '14', 'y', True, False, 30),
    ('1524096 - 435456*x', '653184', 'x', True, False, 30),
    ('15956352*x', '31912704', 'x', True, False, 30),
    ('16416*y - 98496', '-49248', 'y', True, False, 30),
    ('18 - 10*x', '11', 'x', True, False, 30),
    ('19*x/3 - 2/3', '12', 'x', True, False, 30),
    ('2 - x', '4*x + 1', 'x', True, False, 30),
    ('2 - x', 'x - 4', 'x', True, False, 30),
    ('2', 'y', 'y', False, False, 30),
    ('2', 'y', 'y', True, False, 30),
    ('2**x + 4**x', '-1', 'x', True, False, 30),
    ('2*x + 2*y', '2', 'x', True, False, 30),
    ('2*x + y', '5', 'y', True, False, 30),
    ('2*x - 1', '3', 'x', True, False, 30),
    ('2*x - 3', '1', 'x', True, False, 30),
    ('20 - 19*y/3', '1', 'y', True, False, 30),
    ('2052*x', '4104', 'x', True, False, 30),
    ('21275136*y - 127650816', '-63825408', 'y', True, False, 30),
    ('217728*x', '435456', 'x', True, False, 30),
    ('27216*y - 63504', '-36288', 'y', True, False, 30),
    ('3*x - 1', 'x + 5', 'x', True, True, 30),
    ('3*x - 2', '0', 'x', True, False, 30),
    ('3*y', '3', 'y', True, False, 30),
    ('4*x - y', '7', 'x', True, False, 30),
    ('4*x - y', '7', 'y', True, False, 30),
    ('49248 - 8208*y', '24624', 'y', True, False, 30),
    ('5', '2*x - 1', 'x', True, False, 30),
    ('5*x + z - 6', '1', 'z', True, False, 30),
    ('5*x/15552 - 5/7776', '-1/5184', 'x', True, False, 30),
    ('5*x/2 - 3', '1/2', 'x', True, False, 30),
    ('5*x/20155392 - 5/10077696', '-1/6718464', 'x', True, False, 30),
    ('5*x/72 - 5/36', '-1/24', 'x', True, False, 30),
    ('5*x/93312 - 5/46656', '-1/31104', 'x', True, False, 30),
    ('5*y/1296', '1/288', 'y', True, False, 30),
    ('5*y/2519424 - 5/839808', '-7/1679616', 'y', True, False, 30),
    ('5*y/54 - 5/18', '-7/36', 'y', True, False, 30),
    ('5*y/60466176', '1/13436928', 'y', True, False, 30),
    ('5/3359232 - 5*x/6718464', '1/2239488', 'x', True, False, 30),
    ('5/46656 - 5*y/139968', '7/93312', 'y', True, False, 30),
    ('6 - 2*y', '4', 'y', True, False, 30),
    ('7*y/2 + 7/2', '7', 'y', True, False, 30),
    ('7838208*x - 27433728', '-11757312', 'x', True, False, 30),
    ('7838208*x', '15676416', 'x', True, False, 30),
    ('9144576 - 2612736*x', '3919104', 'x', True, False, 30),
    ('Abs(2*x + 1)', 'x + 4', 'x', True, False, 30),
    ('Abs(x)', 'I', 'x', False, False, 30),
    ('Abs(x)', 'y', 'x', True, False, 30),
    ('Abs(x)', 'y', 'y', False, False, 30),
    ('Abs(x**2 + 1)', '0', 'x', False, False, 30),
    ('I*x + 1', '0', 'x', False, True, 30),
    ('I*x + 1', '0', 'x', True, False, 30),
    ('cos(x)', '-3', 'x', True, False, 30),
    ('exp(2*x) - 3*exp(x) + 2', '0', 'x', False, False, 30),
    ('exp(2*x)', '5', 'x', False, False, 30),
    ('exp(x)', '2', 'x', False, False, 30),
    ('log(x)**2 - 3*log(x) + 2', '0', 'x', True, False, 30),
    ('sin(x)', '1/2', 'x', False, False, 30),
    ('sin(x)', '1/2', 'x', True, False, 30),
    ('sin(x)', 'x', 'x', True, False, 30),
    ('sqrt(x + 1)', 'x - 1', 'x', False, True, 30),
    ('tan(x)', '1', 'x', True, False, 30),
    ('tan(x)', 'I', 'x', False, False, 30),
    ('x + 1', 'x + 1', 'x', True, False, 30),
    ('x + 1/x', '1', 'x', False, False, 30),
    ('x + 4', '0', 'x', True, False, 30),
    ('x + 6', '0', 'x', True, False, 30),
    ('x + log(x)', '0', 'x', True, False, 30),
    ('x + y', '1', 'x', False, False, 30),
    ('x + y', '5', 'x', True, False, 30),
    ('x - y', '-1', 'y', True, False, 30),
    ('x - y', '0', 'x', True, False, 30),
    ('x', '0', 'x', True, False, 30),
    ('x**2 + 1', '0', 'x', False, False, 30),
    ('x**2 + x + 1', '0', 'x', False, True, 30),
    ('x**2 + x', 'x**2 + 3', 'x', True, False, 30),
    ('x**2 + y**2', '1', 'x', True, False, 30),
    ('x**2 + y**2', '4', 'y', True, False, 30),
    ('x**2 + y**2', '5', 'y', True, False, 30),
    ('x**2 - 2*x + 5', '0', 'x', False, False, 30),
    ('x**2 - 2*x - 1', '0', 'x', True, True, 30),
    ('x**2 - 5*x + 6', '0', 'x', False, True, 30),
    ('x**2 - 7', '0', 'x', True, False, 30),
    ('x**2 - x - 1', '1', 'x', True, False, 30),
    ('x**2 - y', '1', 'x', True, False, 30),
    ('x**2 - y', '1', 'y', True, False, 30),
    ('x**2', '4', 'x', True, False, 30),
    ('x**2', 'y', 'y', True, False, 30),
    ('x**3 + x + 1', '0', 'x', False, False, 30),
    ('x**3 - x', '0', 'x', True, False, 30),
    ('x**3', '8', 'x', False, False, 30),
    ('x**4 + 2*x**2 + 1', '0', 'x', False, True, 30),
    ('x**4 + 3*x**2 + 1', '0', 'x', False, False, 30),
    ('x**4 + x**2 + 1', '0', 'x', False, True, 30),
    ('x**4 + x**2 + 1', '0', 'x', True, False, 30),
    ('x**4 - x - 1', '0', 'x', True, False, 30),
    ('x**4', 'x', 'x', True, False, 30),
    ('x**5', '2', 'x', False, True, 30),
    ('x**6 - 9*x**3 + 8', '0', 'x', False, False, 30),
    ('x**6', '1', 'x', False, True, 30),
    ('x*y', '5', 'x', True, False, 30),
    ('x*y', '6', 'x', True, False, 30),
    ('x/2 + 3/10', '1', 'x', True, False, 30),
    ('y + 2', '3', 'y', True, False, 30),
    ('y + Abs(y)', 'y + 2', 'y', False, False, 30),
    ('y + Abs(y)', 'y + 2', 'y', True, False, 30),
    ('y - exp(2)*exp(y)*exp(-exp(y)) + 2*exp(2)*exp(-exp(y))', '1', 'y', True, False, 30),
    ('y - sqrt(20 - 4*y**2)/2', '1', 'y', True, False, 30),
    ('y', 'sqrt(y) + 2', 'y', True, False, 30),
    ('y', 'x + 5', 'x', True, False, 30),
    ('y', 'x**2', 'x', True, False, 30),
    ('y', 'x**2', 'y', True, False, 30),
]


@pytest.mark.parametrize("i", range(len(CASES)))
@pytest.mark.parametrize("view", [(None, False), (1, False), (None, True)])
def test_equation_steps(rust: Any, i: int, view: tuple[Optional[int], bool]) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.parsing import parse
    from sympy_extras.sets.fancysets import Complexes
    from sympy_extras.sets.sets import Reals
    from sympy_extras.steps.core import count_steps, format_steps
    from sympy_extras.steps.equations import solve_steps
    rc = load("sympy_extras_rs.core")
    lhs_text, rhs_text, name, real, branch, max_depth = CASES[i]
    lhs, rhs, x = parse(lhs_text), parse(rhs_text), parse(name)
    assert isinstance(x, Symbol)
    depth, minor = view

    def expected() -> Any:
        steps = solve_steps(lhs, rhs, x, Reals if real else Complexes, branch=branch, max_depth=max_depth)
        found = '?' if steps.contains_dont_know() else str(steps.solutions())
        return (format_steps(steps, depth=depth, minor=minor), found, count_steps(steps))

    def got() -> Any:
        r = rc.logic_call('equation_steps', [lhs, rhs, x, Integer(int(real)), Integer(int(branch)),
                                             Integer(-1 if depth is None else depth), Integer(int(minor)),
                                             Integer(-1 if max_depth is None else max_depth)])
        return (r.args[0].name, str(r.args[1]), int(r.args[2]))
    assert outcome(got) == outcome(expected), CASES[i]
