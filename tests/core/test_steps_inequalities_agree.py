"""``sympy_extras.steps.inequalities`` against its Rust port
(``steps/inequalities.rs``): the trees printed by ``format_steps`` (at
several depths, with and without the minor steps), the solutions and the
numbers of steps, on the inequalities of its tests (read with the core's
parser)."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from ..conftest import load
from .test_integers_agree import outcome

CASES: list[Any] = [
    ['Lt', ['3 - 2*x', '7'], 'x', False, 30],
    ['Le', ['2*x + 1', '3*x - 4'], 'x', False, 30],
    ['Gt', ['x/3 - 1', '1/2'], 'x', False, 30],
    ['Ge', ['-x', '4'], 'x', False, 30],
    ['Gt', ['x', 'x + 1'], 'x', False, 30],
    ['Le', ['x + 1', 'x + 1'], 'x', False, 30],
    ['Gt', ['x**2 - 5*x + 6', '0'], 'x', False, 30],
    ['Lt', ['x**2', '4'], 'x', False, 30],
    ['Gt', ['x**2 + 1', '0'], 'x', False, 30],
    ['Lt', ['x**2 + 1', '0'], 'x', False, 30],
    ['Le', ['x**2', '0'], 'x', False, 30],
    ['Ge', ['x**2 - 2', '0'], 'x', False, 30],
    ['Gt', ['-x**2 + 3*x', '2'], 'x', False, 30],
    ['Ge', ['x**3 - x', '0'], 'x', False, 30],
    ['Gt', ['(x - 1)**2*(x + 3)', '0'], 'x', False, 30],
    ['Le', ['(x - 1)**2*(x + 3)', '0'], 'x', False, 30],
    ['Lt', ['x**4 - 5*x**2 + 4', '0'], 'x', False, 30],
    ['Lt', ['x**3 - 3*x + 1', '0'], 'x', False, 30],
    ['Le', ['(x - 1)/(x + 2)', '0'], 'x', False, 30],
    ['Lt', ['1/x', '2'], 'x', False, 30],
    ['Ge', ['(x + 1)/(x - 1)', '2'], 'x', False, 30],
    ['Gt', ['(x**2 - 1)/(x - 1)', '0'], 'x', False, 30],
    ['Ge', ['x/(x**2 - 4)', '0'], 'x', False, 30],
    ['Gt', ['x**(-2)', '1'], 'x', False, 30],
    ['Lt', ['(x - 3)*(x + 1)/(x*(x - 2))', '0'], 'x', False, 30],
    ['Le', ['1/(x - 1)', '1/(x + 1)'], 'x', False, 30],
    ['Lt', ['Abs(x - 1)', '3'], 'x', False, 30],
    ['Ge', ['Abs(2*x + 1)', '3'], 'x', False, 30],
    ['Le', ['Abs(x)', '0'], 'x', False, 30],
    ['Gt', ['Abs(x - 2)', '-1'], 'x', False, 30],
    ['Gt', ['Abs(x - 2)', '1'], 'x', False, 30],
    ['Lt', ['Abs(x**2 - 4)', '3'], 'x', False, 30],
    ['Le', ['x + Abs(x - 1)', '5'], 'x', False, 30],
    ['Le', ['(x - 3)**4*(x + 1/2)/(x - 1/3)', '0'], 'x', False, 30],
    ['Gt', ['(x - 5)*(2*x + 12)/(x + 5)', '0'], 'x', False, 30],
    ['Gt', ['(x - 3)**3*(x - 1)**2', '0'], 'x', False, 30],
    ['Le', ['(x + 3)**2*(2*x - 2)/(x + 2)**2', '0'], 'x', False, 30],
    ['Lt', ['(5 - x)/(x - 4/3)', '0'], 'x', False, 30],
    ['Lt', ['2*(x - 3)*(x + 1)**2/x', '0'], 'x', False, 30],
    ['Lt', ['(x - 2)*(x + 3)**2/(x - 1)', '0'], 'x', False, 30],
    ['Lt', ['(x - 1/2)**2*(x + 1/2)*(x + 6)**2', '0'], 'x', False, 30],
    ['Lt', ['(x - 5)*(x + 3)**2/((x - 2)*(x + 1))', '0'], 'x', False, 30],
    ['Le', ['-x*(x - 3)*(x + 4)**2', '0'], 'x', False, 30],
    ['Gt', ['(2*x - 4)/((x - 6)*(x - 1))', '0'], 'x', False, 30],
    ['Le', ['(1 - x)*(x - 6)*(x - 3)/(x - 2/3)', '0'], 'x', False, 30],
    ['Le', ['2*(x - 6)**2*(x - 1)**2*(x + 2)/(x + 5)', '0'], 'x', False, 30],
    ['Le', ['2*(x + 2)**2/(x - 6)', '0'], 'x', False, 30],
    ['Ge', ['(-x - 3)*(x - 2)**2/(x - 3)', '0'], 'x', False, 30],
    ['Ge', ['(x - 2)**2/((x - 5/3)*(x + 4))', '0'], 'x', False, 30],
    ['Lt', ['-(x - 3)**2*(x - 5/2)*(x + 1/2)**2', '0'], 'x', False, 30],
    ['Le', ['(-x - 6)*(x - 1)', '0'], 'x', False, 30],
    ['Lt', ['(x - 1/2)/(x + 3)', '0'], 'x', False, 30],
    ['Gt', ['(x - 2)**2*(x - 1/2)**2*(x + 3/2)**2/(x*(x - 5/3))', '0'], 'x', False, 30],
    ['Gt', ['(-x - 3)/((x - 1)*(x + 1))', '0'], 'x', False, 30],
    ['Gt', ['-x', '0'], 'x', False, 30],
    ['Lt', ['(x - 3)**2*(x + 1)**2*(x + 3)/(x - 5)', '0'], 'x', False, 30],
    ['Ge', ['(x - 4)*(x - 1)**2*(x + 3)**2', '0'], 'x', False, 30],
    ['Le', ['2*(x - 3/2)**2/(x*(x + 2))', '0'], 'x', False, 30],
    ['Lt', ['2*x', '6'], 'x', False, 30],
    ['Gt', ['(2*x - 3)*(x**2 - 3)*(x**2 - 2)', '0'], 'x', False, 30],
    ['Gt', ['exp(x)', 'x'], 'x', False, 30],
    ['Gt', ['a*x**2', '1'], 'x', False, 30],
    ['Gt', ['exp(x)', '0'], 'x', False, 30],
]


@pytest.mark.parametrize("i", range(len(CASES)))
@pytest.mark.parametrize("view", [(None, False), (1, False), (None, True)])
def test_inequality_steps(rust: Any, i: int, view: tuple[Optional[int], bool]) -> None:
    from sympy_extras.core import Ge, Gt, Integer, Le, Lt, Symbol
    from sympy_extras.core.parsing import parse
    from sympy_extras.steps.core import count_steps, format_steps
    from sympy_extras.steps.inequalities import inequality_steps
    rc = load("sympy_extras_rs.core")
    kind, (lhs_text, rhs_text), name, branch, max_depth = CASES[i]
    build = {'Lt': Lt, 'Le': Le, 'Gt': Gt, 'Ge': Ge}[kind]
    inequality = build(parse(lhs_text), parse(rhs_text))
    x = parse(name)
    assert isinstance(x, Symbol)
    depth, minor = view

    def expected() -> Any:
        steps = inequality_steps(inequality, x, branch=branch, max_depth=max_depth)
        found = '?' if steps.contains_dont_know() else str(steps.solutions())
        return (format_steps(steps, depth=depth, minor=minor), found, count_steps(steps))

    def got() -> Any:
        r = rc.logic_call('inequality_steps', [inequality, x, Integer(int(branch)), Integer(-1 if depth is None else depth),
                                               Integer(int(minor)), Integer(-1 if max_depth is None else max_depth)])
        return (r.args[0].name, str(r.args[1]), int(r.args[2]))
    assert outcome(got) == outcome(expected), CASES[i]
