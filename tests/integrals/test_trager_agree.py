"""The Rust port of Trager's algorithm (``integrals/trager.py``) against
the Python one, on the calls of the Python tests of trager, of the curves
and of the indefinite integration (read with the core's parser): the
antiderivatives, the reductions, the Hermite reductions and the proofs of
non-elementarity."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Symbol
from sympy_extras.core.parsing import parse
import sympy_extras.integrals.trager as trager

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    s = 'None' if str(r) == 'NONE' else str(r)
    return re.sub(r'\b_?([a-zA-Z]+)_?\d+\b', r'\1', s)


PYTHON: dict[str, Callable[..., object]] = {
    'antiderivative': trager.trager_antiderivative,
    'reduce': trager.trager_reduce,
    'hermite': trager.hermite_reduce_algebraic,
    'nonelementary': trager.is_nonelementary_algebraic,
}

CASES: list[tuple[str, str, str]] = [
    ('hermite', 'x**2/sqrt(x**2 + 1)', 'x'),
    ('hermite', '1/(x**2*sqrt(x**3 + 1))', 'x'),
    ('hermite', 'sqrt(x**3 + 1)/(x - 1)**2', 'x'),
    ('hermite', '1/sqrt(x) + x**(-3/2)', 'x'),
    ('hermite', '1/sqrt(x**2 + 1) + (x - 1)**(-2)', 'x'),
    ('hermite', 'sqrt(sqrt(x) + 1)', 'x'),
    ('hermite', 'x**2/sqrt(a + x**2)', 'x'),
    ('antiderivative', '(b**2/(4*c) + b*x + c*x**2)**(-3/2)', 'x'),
    ('antiderivative', 'sqrt(x + 1)*sqrt(x + 2)/x', 'x'),
    ('antiderivative', 'x**2*(a + x**3)**(1/3)', 'x'),
    ('antiderivative', '1/(x*(x**3 + 1)**(1/3))', 'x'),
    ('antiderivative', '(x**3 + 1)**(-1/3)', 'x'),
    ('antiderivative', '1/(x*sqrt(x**2 + 1))', 'x'),
    ('antiderivative', 'x/sqrt(x**4 + 1)', 'x'),
    ('antiderivative', '1/sqrt(x**2 - 1)', 'x'),
    ('antiderivative', '1/(x*sqrt(x**3 + 1))', 'x'),
    ('antiderivative', '(x**2 - 1)/((x**2 + 1)*sqrt(x**4 + 1))', 'x'),
    ('antiderivative', '1/((x**2 + 1)*sqrt(x**2 + 2))', 'x'),
    ('reduce', '1/sqrt(x**3 + 1)', 'x'),
    ('reduce', '(2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1))', 'x'),
    ('nonelementary', '(2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1))', 'x'),
    ('nonelementary', 'x/sqrt(x**4 + 1)', 'x'),
    ('nonelementary', '1/((x - 1)*sqrt(x**3 + 1))', 'x'),
    ('antiderivative', '1/sqrt(a + x**2)', 'x'),
    ('antiderivative', '1/sqrt(p + x**2)', 'x'),
    ('hermite', '1/sqrt(p + x**2)', 'x'),
    ('antiderivative', 'sqrt(a + b*x**2 + c*x**4)/(a - c*x**4)', 'x'),
    ('antiderivative', 'x**2*(x**3 + 1)**(1/3)', 'x'),
    ('antiderivative', '1/(x*(x**4 + 1)**(1/4))', 'x'),
    ('antiderivative', 'x/(x**3 + 1)**(1/3)', 'x'),
    ('antiderivative', 'sqrt(x) + sqrt(x + 1)', 'x'),
    ('antiderivative', 'sqrt(x)*sqrt(x + 1)', 'x'),
    ('antiderivative', 'sqrt(x**3 + 1)', 'x'),
    ('antiderivative', 'x**x', 'x'),
    ('antiderivative', 'besselk(7, x)', 'x'),
    ('antiderivative', 'x**2*asin(sqrt(1 - x**2)/(3*sqrt(1/9 - x**2/9)))', 'x'),
    ('antiderivative', '1/(x**(1/3) + sqrt(x))', 'x'),
    ('antiderivative', '2**x*cosh(x)', 'x'),
    ('antiderivative', 'sqrt(a + b*x + c*x**2)/x', 'x'),
    ('antiderivative', 'sqrt(a + b*x)/x', 'x'),
    ('antiderivative', 'sqrt(x**7/(1 - 5*x**2))', 'x'),
    ('antiderivative', 'sqrt(x)/(x**(1/3) + 1)', 'x'),
    ('antiderivative', 'sqrt(x**7/(a - b*x**2))', 'x'),
    ('antiderivative', 'sqrt(-x**5/b)', 'x'),
    ('antiderivative', 'x**(1/3)*sqrt(a + x**2)', 'x'),
    ('antiderivative', 'x**(1/3)*sqrt(x**2)', 'x'),
]


@pytest.mark.parametrize('k', range(len(CASES)))
def test_agrees(k: int) -> None:
    kind, f_text, x_text = CASES[k]
    f, x = parse(f_text), Symbol(x_text)
    py = outcome(lambda: PYTHON[kind](f, x))
    rs = outcome(lambda: rc.logic_call('trager', [Symbol(kind), f, x]))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
