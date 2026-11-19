"""The Rust port of ``integrals/mellin.py`` against the Python one: the
decompositions of integrands into Mellin kernels and the transforms, on
the integrands of the Mellin, Marichev and Slater tests (read with the
core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Symbol, Tuple
from sympy_extras.core.parsing import parse
import sympy_extras.integrals.mellin as ML

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


def python(f, x, third):
    if third in ('None', "'lower'", "'upper'"):
        cutoff = None if third == 'None' else third.strip("'")
        p = ML.decompose_integrand(f, x, cutoff)
        if p is None:
            return None
        return Tuple(p.constant, p.alpha, p.log_power, p.one_minus_power,
                     Tuple(*[Tuple(Symbol(m.kernel.name), m.beta, m.gamma, m.constant) for m in p.matches]))
    t = ML.mellin_transform(f, x, Symbol(third))
    return None if t is None else Tuple(t.transform, t.strip[0], t.strip[1], t.condition)


def rust(f, x, third):
    if third in ('None', "'lower'", "'upper'"):
        cutoff = Symbol('NONE') if third == 'None' else Symbol(third.strip("'"))
        return rc.logic_call('mellin', [Symbol('decompose'), f, x, cutoff])
    return rc.logic_call('mellin', [Symbol('transform'), f, x, Symbol(third)])


CASES: list[tuple[str, str, str]] = [
    ('(-log(x))**k', 'x', "'lower'"),
    ('(1 - x)**a', 'x', 'None'),
    ('(x + 1)**(-7/3)', 'x', 's'),
    ('(x + 1)**(-a)', 'x', 's'),
    ('(x**2 - 1)**a', 'x', "'upper'"),
    ('-1 + exp(-x)', 'x', 's'),
    ('1/(exp(x) + 1)', 'x', 's'),
    ('1/(sqrt(x)*(x + 3))', 'x', 'None'),
    ('1/cosh(x)', 'x', 's'),
    ('3*x**a*exp(-x)*log(x)**2*sin(2*x)', 'x', 'None'),
    ('airyai(x)', 'x', 's'),
    ('besseli(1, x**2)*besselk(0, x**2)', 'x', 'None'),
    ('besselj(0, 3*x)/(x**2 + 4)', 'x', 's'),
    ('besselj(0, x)*besselj(0, 2*x)', 'x', 'None'),
    ('besselj(0, x)*bessely(1, x)', 'x', 'None'),
    ('besselj(nu, x)**2', 'x', 'None'),
    ('besselj(nu, x)', 'x', 's'),
    ('besselk(1/3, x)**2', 'x', 's'),
    ('erfc(x)', 'x', 's'),
    ('exp(-2*x)*sin(3*x)', 'x', 'None'),
    ('exp(-2*x)', 'x', 's'),
    ('exp(-2*x**3)', 'x', 's'),
    ('exp(-3*x)*besseli(nu, 2*x)', 'x', 'None'),
    ('exp(-a*x)*besselj(0, x)**2', 'x', 'None'),
    ('exp(-a*x)*sin(b*x)', 'x', 'None'),
    ('exp(-x)*besselj(0, 2*x)*besselj(1, 2*x)', 'x', 'None'),
    ('exp(-x)*besselj(0, x)', 'x', 'None'),
    ('exp(-x)*cos(2*x)*besselj(0, x)', 'x', 'None'),
    ('exp(-x)*log(x + 2)/(x + 1)', 'x', 'None'),
    ('exp(-x)*log(x)**2', 'x', 's'),
    ('exp(-x)*log(x)', 'x', 's'),
    ('exp(-x)*sin(x)*cos(x)', 'x', 'None'),
    ('exp(-x)', 'x', "'lower'"),
    ('exp(-x)/x**(3/2)', 'x', 'None'),
    ('exp(-x*(1 + 2*I))*besselj(0, x)/2', 'x', 'None'),
    ('exp(-x**2)*log(x)', 'x', 'None'),
    ('exp(-x**2)/sqrt(x)', 'x', 's'),
    ('exp(4*x**2)*erfc(2*x)', 'x', 'None'),
    ('exp(x*(-1 + 2*I))*besselj(0, x)/2', 'x', 'None'),
    ('exp(x**2/2)*erfc(x)/x', 'x', 'None'),
    ('log(1 - x**2)', 'x', "'lower'"),
    ('log(x)**2', 'x', "'lower'"),
    ('sin(x)**2', 'x', 'None'),
    ('sin(x)/x', 'x', 's'),
    ('sqrt(x)/(x + 3)', 'x', 'None'),
    ('x**(1/3)/(x + 1)', 'x', "'lower'"),
    ('x**2*(1 - x)**a', 'x', "'lower'"),
    ('x**2*exp(-3*x)', 'x', 'None'),
    ('x**2*exp(-x)', 'x', 'None'),
    ('x**2*exp(-x)', 'x', 's'),
    ('x**a*cos(x + 1)', 'x', 's'),
    ('x**a*log(1 - x)**2', 'x', "'lower'"),
    ('x**k/(x + 3)', 'x', 'None'),
    ('x*besselk(0, x)**2', 'x', 'None'),
    ('x*exp(-x)', 'x', "'upper'"),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_mellin(i: int) -> None:
    text, variable, third = CASES[i]
    x = Symbol(variable)
    f = parse(text)
    got = outcome(lambda: rust(f, x, third))
    if got != 'SKIP':
        assert got == outcome(lambda: python(f, x, third)), (text, third)
