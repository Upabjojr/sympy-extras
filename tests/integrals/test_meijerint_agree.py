"""The Rust port of ``integrals/meijerint.py`` against the Python one, on
the calls of the Python integration tests (read with the core's parser:
plain symbols; the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Symbol, Tuple
from sympy_extras.core.parsing import parse
import sympy_extras.integrals.meijerint as mj

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    if r is None or str(r) == 'NONE':
        return 'None'
    if isinstance(r, tuple):
        r = Tuple(*[Tuple(*[Tuple(*t) for t in v]) if isinstance(v, list) else v for v in r])
    return re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


CASES: list[tuple[str, ...]] = [
    ('rewrite_single', 'exp(-x)*sin(x)', 'x', 'True'),
    ('meijerint_indefinite', 'exp(-x)*sin(x)', 'x'),
    ('meijerint_definite', 'exp(-(-a + x)**2)*exp(-(-b + x)**2)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(c*x)*exp(-(-a + x)**2)*exp(-(-b + x)**2)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-(-a - b - c + x)**2)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-(a + b + c + x)**2)', 'x', '0', 'oo'),
    ('meijerint_definite', 'x', 'x', '0', '0'),
    ('meijerint_definite', 'x', 'x', 'oo', 'oo'),
    ('meijerint_indefinite', 'exp(x)', 'x'),
    ('meijerint_definite', 'x**a', 'x', '0', 'b'),
    ('meijerint_definite', '(x + 1)**3*exp(-x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-(-mu + x)**2/(4*sigma**2))', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-mu*x)*exp(sigma*x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(x)', 'x', '-oo', '2'),
    ('meijerint_definite', 'exp(x)', 'x', '0', 'I'),
    ('meijerint_definite', 'exp(-x**2)', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'exp(-Abs(x))', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'exp(-(2*x - 3)**2)', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'exp(-Abs(2*x - 3))', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'sqrt(2)*exp(-(-mu + x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'sinc(x)**2', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'exp(-x)*sin(x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'x*exp(-x)*sin(x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'x**2*exp(-x)*sin(x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'x**3*exp(-x)*sin(x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-x)*sin(a + x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'sin(x**a)*sin(x**b)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-x**2)*log(x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'x**k*exp(-a*x**2)', 'x', '-oo', 'oo'),
    ('meijer_antiderivative', 'x*besselj(0, x)', 'x'),
    ('meijer_antiderivative', 'x*besseli(0, x)', 'x'),
    ('meijer_antiderivative', 'besselj(1, x)', 'x'),
    ('meijer_antiderivative', 'besselj(1, x)**2/x', 'x'),
    ('meijer_antiderivative', 'besselj(0, x)**2/x**2', 'x'),
    ('meijer_antiderivative', 'besselj(0, x)*besselj(1, x)', 'x'),
    ('meijer_antiderivative', 'x**2*besselj(0, x)*besselj(1, x)', 'x'),
    ('meijer_antiderivative', 'besselj(0, x)*besselj(1, x)/x', 'x'),
    ('meijer_antiderivative', 'x*besselj(1, x**2)', 'x'),
    ('meijerint_definite', 'besselj(a, z)*besselj(b, z)/z', 'z', '0', 'oo'),
    ('meijerint_indefinite', 'erf(x**3)', 'x'),
    ('meijerint_indefinite', 'sin(x - 1)', 'x'),
    ('meijerint_indefinite', 'besselj(1, x - 1)', 'x'),
    ('meijerint_definite', 'sqrt(2)*exp(-(-mu1 + x)**2/(2*sigma1**2))/(2*sqrt(pi)*sigma1)', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'sqrt(2)*x*exp(-(-mu1 + x)**2/(2*sigma1**2))/(2*sqrt(pi)*sigma1)', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'sqrt(2)*x**2*exp(-(-mu1 + x)**2/(2*sigma1**2))/(2*sqrt(pi)*sigma1)', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'sqrt(2)*x**3*exp(-(-mu1 + x)**2/(2*sigma1**2))/(2*sqrt(pi)*sigma1)', 'x', '-oo', 'oo'),
    ('meijerint_definite', 'lambda*exp(-lambda*x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'lambda*x*exp(-lambda*x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'lambda*x**2*exp(-lambda*x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'x**(alpha - 1)*(x + 1)**(-alpha - beta)*gamma(alpha + beta)/(gamma(alpha)*gamma(beta))', 'x', '0', 'oo'),
    ('meijerint_definite', 'x*x**(alpha - 1)*(x + 1)**(-alpha - beta)*gamma(alpha + beta)/(gamma(alpha)*gamma(beta))', 'x', '0', 'oo'),
    ('meijerint_definite', 'x*x**(a - 1)*(1 - x)**(b - 1)*gamma(a + b)/(gamma(a)*gamma(b))', 'x', '0', '1'),
    ('meijerint_definite', '2**(1 - k/2)*x**2*x**(k - 1)*exp(-x**2/2)/gamma(k/2)', 'x', '0', 'oo'),
    ('meijerint_definite', 'sqrt(2)*sqrt(lamda)*exp(-lamda*(-mu + x)**2/(2*mu**2*x))/(2*sqrt(pi)*sqrt(x))', 'x', '0', 'oo'),
    ('meijerint_definite', 'x*exp((-nu**2/2 - x**2/2)/sigma**2)*besseli(0, nu*x/sigma**2)/sigma**2', 'x', '0', 'oo'),
    ('meijerint_definite', 'x**2*exp(-Abs(mu - x)/b)/(2*b)', 'x', '-oo', 'oo'),
    ('meijerint_indefinite', 'Si(x)', 'x'),
    ('meijerint_indefinite', 'Shi(x)', 'x'),
    ('meijerint_indefinite', 'expint(1, x)', 'x'),
    ('meijerint_indefinite', 'expint(2, x)', 'x'),
    ('meijerint_indefinite', 'Ci(x)', 'x'),
    ('meijerint_indefinite', 'Chi(x)', 'x'),
    ('meijerint_indefinite', 'exp(-x)/x', 'x'),
    ('meijerint_indefinite', 'exp(-x)/x**2', 'x'),
    ('meijerint_indefinite', 'cos(x)/x', 'x'),
    ('meijerint_indefinite', 'cosh(x)/x', 'x'),
    ('meijerint_definite', 'exp(-x)*Si(x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'sin(x)*expint(1, x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'besselj(0, x)*expint(1, x)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-I*x**2)', 'x', '-oo', 'oo'),
    ('meijerint_indefinite', 'sin(pi*x**2/2)', 'x'),
    ('meijerint_indefinite', 'cos(pi*x**2/2)', 'x'),
    ('meijerint_indefinite', 'x*sqrt(2*x + 3)', 'x'),
    ('meijerint_indefinite', '(b + t)**(-a)', 't'),
    ('meijerint_indefinite', 'x**(x**x)', 'x'),
    ('meijerint_indefinite', 'cos(x**y)/x**y', 'x'),
    ('meijerint_inversion', '1/x', 'x', 't'),
    ('meijerint_inversion', 'x**(-2)', 'x', 't'),
    ('meijerint_inversion', 'x**(-3)', 'x', 't'),
    ('meijerint_inversion', '1/(a + x)', 'x', 't'),
    ('meijerint_inversion', '(a + x)**(-2)', 'x', 't'),
    ('meijerint_inversion', '1/(x**2 + 1)', 'x', 't'),
    ('meijerint_inversion', 'x/(x**2 + 1)', 'x', 't'),
    ('meijerint_inversion', 'a/(a**2 + x**2)', 'x', 't'),
    ('meijerint_inversion', 'x/(a**2 + x**2)', 'x', 't'),
    ('meijerint_inversion', '1/(-a**2 + x**2)', 'x', 't'),
    ('meijerint_inversion', 'x/(-a**2 + x**2)', 'x', 't'),
    ('meijerint_inversion', '1/(x*(x + 1))', 'x', 't'),
    ('meijerint_inversion', '1/(x**2*(a + x))', 'x', 't'),
    ('meijerint_inversion', '1/(x*(x**2 + 1))', 'x', 't'),
    ('meijerint_inversion', '(a**2 + x**2)**(-2)', 'x', 't'),
    ('meijerint_inversion', 'x/(a**2 + x**2)**2', 'x', 't'),
    ('meijerint_inversion', '1/(x**2*(a**2 + x**2))', 'x', 't'),
    ('meijerint_inversion', '1/(x*(a + x)**2)', 'x', 't'),
    ('meijerint_inversion', 'x**(-3/2)', 'x', 't'),
    ('meijerint_inversion', '(a + x)**(-5/2)', 'x', 't'),
    ('meijerint_inversion', '1/sqrt(x**2 + 1)', 'x', 't'),
    ('meijerint_inversion', 'exp(-sqrt(x))', 'x', 't'),
    ('meijerint_inversion', 'exp(-a*sqrt(x))/sqrt(x)', 'x', 't'),
    ('meijerint_inversion', 'log(x)/x', 'x', 't'),
    ('meijerint_inversion', '1/(x*sqrt(x + 1))', 'x', 't'),
    ('meijerint_inversion', 'exp(-a/x)/x', 'x', 't'),
    ('meijerint_inversion', 'exp(-a*x)/x', 'x', 't'),
    ('meijerint_inversion', 'exp(-a*x)/x**2', 'x', 't'),
    ('meijerint_inversion', 'exp(-a*x)/(b + x)', 'x', 't'),
    ('meijerint_inversion', 'exp(-2*x)/(x**2 + 1)', 'x', 't'),
    ('meijerint_inversion', 'exp(-x)/(x*(a + x))', 'x', 't'),
    ('meijerint_inversion', 'x**(-a - 1)*exp(-b*x)', 'x', 't'),
    ('meijerint_inversion', 'gamma(a)/x**a', 'x', 't'),
    ('meijerint_inversion', 'x**(-a)', 'x', 't'),
    ('meijerint_inversion', '(a + x)**(-b)', 'x', 't'),
    ('meijerint_inversion', 'exp(-a*x)*gamma(b)/x**b', 'x', 't'),
    ('meijerint_inversion', 'exp(-a*x)', 'x', 't'),
    ('meijerint_inversion', '3*exp(-a*x - 1)', 'x', 't'),
    ('meijerint_inversion', '1/sqrt(x)', 'x', 't'),
    ('meijerint_inversion', '1/sqrt(a + x)', 'x', 't'),
    ('meijerint_inversion', 'exp(-a/x)/sqrt(x)', 'x', 't'),
    ('meijerint_inversion', '1/(x**2 + 2*x + 2)', 'x', 't'),
    ('meijerint_inversion', 'log(a/x + 1)', 'x', 't'),
    ('meijerint_inversion', '1/(2**x*x)', 'x', 't'),
    ('meijerint_inversion', 'atan(1/x)', 'x', 't'),
    ('meijerint_inversion', '1/((a + x)*(b + x))', 'x', 't'),
    ('meijerint_inversion', '1/((x + 1)*(x + 2))', 'x', 't'),
    ('meijerint_inversion', 'exp(-a*x)/((x + 1)*(x + 2))', 'x', 't'),
    ('meijerint_inversion', 'x/((a + x)*(b + x))', 'x', 't'),
    ('meijerint_inversion', '1/((x + 1)*(x + 2)*(x + 3))', 'x', 't'),
    ('meijerint_inversion', 'Heaviside(x - 1)/x', 'x', 't'),
    ('meijer_antiderivative', 'sin(x)', 'x'),
    ('meijer_antiderivative', 'x**x', 'x'),
    ('meijerint_definite', 'x**k/(x + 3)', 'x', '0', 'oo'),
    ('meijerint_definite', 'exp(-x**2*y)', 'x', '0', 'oo'),
    ('meijerint_definite', '1/(x*(-q + x))', 'x', '0', '1'),
    ('meijer_antiderivative', 'Abs(x)', 'x'),
    ('meijerint_definite', 'Abs(x)', 'x', 'p', 'q'),
    ('meijerint_definite', 'exp(-a*x)*besselj(nu, b*x)', 'x', '0', 'oo'),
]


def python_call(case: tuple[str, ...]) -> object:
    name, args = case[0], [parse(a) for a in case[1:]]
    if name == 'rewrite_single':
        return mj.rewrite_single(args[0], args[1], case[3] == 'True')
    return getattr(mj, name)(*args)


RUST_NAMES = {'meijerint_indefinite': 'indefinite', 'meijerint_definite': 'definite',
              'meijerint_inversion': 'inversion', 'meijer_antiderivative': 'antiderivative',
              'rewrite_single': 'rewrite'}


def rust_call(case: tuple[str, ...]) -> object:
    name = case[0]
    if name == 'rewrite_single':
        args = [parse(case[1]), parse(case[2]), Symbol(case[3])]
    else:
        args = [parse(a) for a in case[1:]]
    return rc.logic_call('meijerint', [Symbol(RUST_NAMES[name])] + args)


@pytest.mark.parametrize('case', CASES, ids=[' '.join(c)[:80] for c in CASES])
def test_agree(case: tuple[str, ...]) -> None:
    if any('sinc' in a for a in case):
        pytest.skip('sinc has no head in the Rust core')
    rust = outcome(lambda: rust_call(case))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: python_call(case))
