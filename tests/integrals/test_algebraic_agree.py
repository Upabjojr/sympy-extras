"""The Rust port of ``integrals/algebraic.py`` (Euler's substitutions,
the roots of a Möbius function, the binomial differentials and their
antiderivatives by ``2F1``) against the Python one on the calls of the
tests (read with the core's parser: plain symbols; the numbers of the
dummies are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Integer, Rational, Symbol, Tuple
from sympy_extras.core.power import Pow
from sympy_extras.core.expr import Expr
from sympy_extras.core.parsing import parse
from sympy_extras.integrals.algebraic import (Rationalisation, binomial_differential, binomial_exponents,
                                              binomial_hypergeometric_antiderivative, euler_substitutions,
                                              moebius_root)

from ..conftest import load

rc = load("sympy_extras_rs.core")


def normal(s: str) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', s)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else normal(str(r))


def radical_bases(f: Expr, x: Symbol) -> set[Expr]:
    return {p.base for p in f.atoms(Pow) if isinstance(p, Pow) and p.has(x) and isinstance(p.exp, Rational)
            and not isinstance(p.exp, Integer)}


def shown(s: Rationalisation) -> Tuple:
    return Tuple(s.t, s.x_of_t, s.t_of_x, Tuple(*[Tuple(k, v) for k, v in s.radicals.items()]))


#: (operation, integrand, variable)
CASES: list[list[str]] = [
    ['euler', '1/sqrt(x**2 + 1)', 'x'],
    ['euler', '1/sqrt(x**2 - 1)', 'x'],
    ['euler', 'sqrt(x*(1 - x))', 'x'],
    ['euler', 'sqrt(2*x + 3)', 'x'],
    ['euler', '1/sqrt(x**3 + 1)', 'x'],
    ['euler', '(x**2 + 1)**(1/3)', 'x'],
    ['euler', 'sqrt(x + 1)*sqrt(x + 2)', 'x'],
    ['moebius', '(x + 1)**(1/3)/x', 'x'],
    ['moebius', 'sqrt((1 - x)/(x + 1))', 'x'],
    ['moebius', '(x + 1)**(1/6)', 'x'],
    ['moebius', 'sqrt(x**2 + 1)', 'x'],
    ['moebius', 'sqrt(x + 1)*sqrt(x + 2)', 'x'],
    ['binomial', '3*x**3*sqrt(x**2 + 1)', 'x'],
    ['binomial', 'x**(1/3)/(x + 1)', 'x'],
    ['binomial', 'x**2*exp(x)', 'x'],
    ['binomial', 'sqrt(x**2 + x + 1)', 'x'],
    ['exponents', 'sqrt(x**7/(1 - 5*x**2))', 'x'],
    ['exponents', 'sqrt(x**7/(5*x**2 - 1))', 'x'],
    ['exponents', 'sqrt(-x**3 + x)', 'x'],
    ['exponents', '(x**2*(x**3 + 1))**(1/3)/x', 'x'],
    ['exponents', 'sqrt(1 + x**(-2))', 'x'],
    ['exponents', 'sqrt(x)/(x**(1/3) + 1)', 'x'],
    ['exponents', 'sqrt(1 - x**2)*sqrt(x**2 + 1)', 'x'],
    ['exponents', 'sqrt(x**4 + x**2 + 1)', 'x'],
    ['exponents', 'x**2*exp(x)', 'x'],
    ['exponents', 'sqrt(x**2 + 1)/sqrt(2*x**2 + 2)', 'x'],
    ['exponents', 'sqrt(x**7/(p - q*x**2))', 'x'],
    ['exponents', 'sqrt(a + x**3)', 'x'],
    ['exponents', 'x**3*sqrt(x**2 + 1)', 'x'],
    ['exponents', '1/(x*sqrt(x**3 + 1))', 'x'],
    ['exponents', '1/(x**2*sqrt(x**2 + 1))', 'x'],
    ['exponents', 'sqrt(x**6/(x**2 + 1))', 'x'],
    ['exponents', 'sqrt(x**3 + 1)', 'x'],
    ['exponents', 'x**(1/3)*sqrt(x**2 + 1)', 'x'],
    ['exponents', 'x**2/(1 - x**4)**(2/3)', 'x'],
    ['hyper', 'sqrt(x**7/(1 - 5*x**2))', 'x'],
    ['hyper', 'sqrt(x**3/(5*x**2 - 1))', 'x'],
    ['hyper', 'sqrt(x**3 + 1)', 'x'],
    ['hyper', 'x**(1/3)*sqrt(x**2 + 1)', 'x'],
    ['hyper', 'sqrt(-x**3 + x)', 'x'],
    ['hyper', '1/(x**2*sqrt(x**3 + 1))', 'x'],
    ['hyper', 'x**2/(1 - x**4)**(2/3)', 'x'],
    ['hyper', '(x**2 - 4)**(-5/4)', 'x'],
    ['hyper', '(x*(1 - x**2)**2)**(1/4)', 'x'],
    ['hyper', 'x**3*sqrt(x**2 + 1)', 'x'],
    ['hyper', '1/(x*sqrt(x**3 + 1))', 'x'],
    ['hyper', 'sqrt(x)/(x**(1/3) + 1)', 'x'],
    ['hyper', 'sqrt(x**7/(p - q*x**2))', 'x'],
    ['hyper', 'sqrt(c*x**3 + 1)', 'x'],
    ['hyper', 'x**(1/3)*sqrt(p + x**2)', 'x'],
    ['hyper', 'x**(1/3)*sqrt(p + q*sqrt(x))', 'x'],
    ['hyper', '(p + q*x**3)**(-2/3)', 'x'],
    ['hyper', 'sqrt(x**7/(u + v*x**2))', 'x'],
    ['moebius', 'x**(1/3)/(x + 1)', 'x'],
    ['moebius', 'x**3*sqrt(x**2 + 1)', 'x'],
    ['euler', 'x**3*sqrt(x**2 + 1)', 'x'],
    ['moebius', '1/(x**2*sqrt(x**2 + 1))', 'x'],
    ['euler', '1/(x**2*sqrt(x**2 + 1))', 'x'],
    ['moebius', 'sqrt(x**3 + 1)', 'x'],
    ['euler', 'sqrt(x**3 + 1)', 'x'],
    ['binomial', 'sqrt(x**3 + 1)', 'x'],
    ['moebius', '1/sqrt(x**2 + 1)', 'x'],
    ['moebius', '1/(x*sqrt(x**2 - 1))', 'x'],
    ['euler', '1/(x*sqrt(x**2 - 1))', 'x'],
    ['moebius', 'sqrt(x)/(x + 1)**2', 'x'],
    ['moebius', 'x**2*sqrt(1 - x**2)', 'x'],
    ['euler', 'x**2*sqrt(1 - x**2)', 'x'],
    ['moebius', 'sqrt(t + x)/x', 'x'],
    ['moebius', '((x + 1)/(1 - x))**(1/3) + sqrt((1 - x)/(x + 1))', 'x'],
    ['moebius', 'exp(x)', 'x'],
    ['binomial', 'x**2*(x + 1)**2', 'x'],
    ['exponents', '1/sqrt(1 - x**4)', 'x'],
    ['exponents', 'x/sqrt(x**4 + 1)', 'x'],
    ['exponents', 'x**(1/3)*(x + 1)**2', 'x'],
    ['hyper', '1/sqrt(1 - x**4)', 'x'],
    ['moebius', '1/sqrt(a**2 + x**2)', 'x'],
    ['euler', '1/sqrt(a**2 + x**2)', 'x'],
    ['moebius', 'sqrt(x)/(x + 1)', 'x'],
    ['hyper', 'besselk(7, x)', 'x'],
    ['binomial', 'besselk(7, x)', 'x'],
    ['hyper', 'x**2*asin(sqrt(1 - x**2)/(3*sqrt(1/9 - x**2/9)))', 'x'],
    ['binomial', '1/(x**(1/3) + sqrt(x))', 'x'],
    ['hyper', '2**x*cosh(x)', 'x'],
    ['hyper', 'sqrt(x**7/(a - b*x**2))', 'x'],
    ['hyper', 'x**(1/3)*sqrt(a + x**2)', 'x'],
    ['hyper', 'x**(1/3)*sqrt(x**2)', 'x'],
    ['binomial', 'x**(1/3)*sqrt(x**2)', 'x'],
    ['exponents', 'x/sqrt(x**3 + 1)', 'x'],
    ['exponents', '1/(x**2*sqrt(x**3 + 1))', 'x'],
    ['exponents', '(2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1))', 'x'],
    ['exponents', '1/((x - 1)*sqrt(x**3 + 1))', 'x'],
    ['exponents', '1/sqrt(a + x**2)', 'x'],
]


@pytest.mark.parametrize('case', CASES)
def test_agrees(case: list[str]) -> None:
    kind = case[0]
    f, x = parse(case[1]), parse(case[2])
    assert isinstance(f, Expr) and isinstance(x, Symbol)
    if kind in ('euler', 'moebius') and len(radical_bases(f, x)) > 1:
        # Python takes the radicals from a set (the first base depends on
        # the hash seed); Rust takes them in the order of the tree
        pytest.skip('the order of the radicals of the Python port is not deterministic')
    if kind == 'euler':
        py = outcome(lambda: Tuple(*[shown(s) for s in euler_substitutions(f, x, 0, 1)]))
        rs = outcome(lambda: rc.logic_call('algebraic', [Symbol('euler'), f, x, Tuple()]))
    elif kind == 'moebius':
        found = moebius_root(f, x)
        py = 'None' if found is None else normal(str(Tuple(shown(found))))
        rs = outcome(lambda: rc.logic_call('algebraic', [Symbol('moebius'), f, x]))
    elif kind == 'binomial':
        py = outcome(lambda: None if (b := binomial_differential(f, x)) is None else Tuple(*b))
        rs = outcome(lambda: rc.logic_call('algebraic', [Symbol('binomial'), f, x]))
    elif kind == 'exponents':
        py = outcome(lambda: None if (d := binomial_exponents(f, x)) is None else Tuple(d.m, d.a, d.b, d.n, d.p))
        rs = outcome(lambda: rc.logic_call('algebraic', [Symbol('exponents'), f, x]))
    else:
        py = outcome(lambda: binomial_hypergeometric_antiderivative(f, x))
        rs = outcome(lambda: rc.logic_call('algebraic', [Symbol('hyper'), f, x]))
    if rs == 'SKIP':
        pytest.skip('not ported')
    assert py == rs
