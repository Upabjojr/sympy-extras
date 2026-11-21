"""The Rust port of ``integrals/indefinite.py`` against the Python one, on
the calls of the Python tests without assumptions and on the integrands of
the manualintegrate tests (read with the core's parser: plain symbols; the
numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Symbol, Tuple
from sympy_extras.core.parsing import parse
import sympy_extras.integrals.indefinite as ind

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
        r = Tuple(r[0], Symbol(r[1].replace(' ', '_')))
    return re.sub(r'\b_?([a-zA-Z]+)(_indefinite)?_?\d+\b', r'\1', str(r))


CASES: list[tuple[str, ...]] = [
    ('verified', 'sqrt(1 - x**2)', 'x'),
    ('check', 'x*sqrt(1 - x**2)/2 + asin(x)/2', 'sqrt(1 - x**2)', 'x'),
    ('verified', 'x/(x**2 + 1)', 'x'),
    ('check', 'log(x**2 + 1)/2', 'x/(x**2 + 1)', 'x'),
    ('verified', 'x*exp(x)', 'x'),
    ('check', 'x*exp(x) - exp(x)', 'x*exp(x)', 'x'),
    ('verified', 'tan(x)**3', 'x'),
    ('check', 'log(cos(x)) + tan(x)**2/2', 'tan(x)**3', 'x'),
    ('verified', '1/(x*(log(x)**2 + 1))', 'x'),
    ('check', 'atan(log(x))', '1/(x*(log(x)**2 + 1))', 'x'),
    ('verified', '1/(x**3 + 1)', 'x'),
    ('check', 'log(x + 1)/3 - log(x**2 - x + 1)/6 + sqrt(3)*atan(2*sqrt(3)*x/3 - sqrt(3)/3)/3', '1/(x**3 + 1)', 'x'),
    ('verified', 'x/sqrt(x**4 + 1)', 'x'),
    ('check', 'log(x**2 + sqrt(x**4 + 1))/2', 'x/sqrt(x**4 + 1)', 'x'),
    ('verified', 'exp(x)*sin(x)', 'x'),
    ('check', '(sin(x) - cos(x))*exp(x)/2', 'exp(x)*sin(x)', 'x'),
    ('verified', 'log(x)**2', 'x'),
    ('check', 'x*log(x)**2 - 2*x*log(x) + 2*x', 'log(x)**2', 'x'),
    ('verified', 'sqrt(x**3 + 1)', 'x'),
    ('integral', 'sqrt(1 - x**2)', 'x', 'True'),
    ('integral', 'x**x', 'x', 'True'),
    ('integral', 'sin(x)', 'x', 'True'),
    ('verified', 'sin(x)', 'x'),
    ('check', 'sin(x)', 'sin(x)', 'x'),
    ('check', 'Integral(exp(-x**2), x)', 'exp(-x**2)', 'x'),
    ('check', 'x**2*sign(x)/2', 'x*sign(x)', 'x'),
    ('check', '-asinh(1/x)', '1/(x*sqrt(x**2 + 1))', 'x'),
    ('conjugate', '-I*log(x - I)/2 + I*log(x + I)/2', 'x'),
    ('real_form', '-I*log(x - I)/2 + I*log(x + I)/2', '1/(x**2 + 1)', 'x'),
    ('real_form', 'I*log(x - I)/2 - I*log(x + I)/2', '1/(x**2 + 1)', 'x'),
    ('integral', '1/(x**4 + 1)', 'x', 'True'),
    ('conjugate', 'log(x) - I*log(x - I)/2 + I*log(x + I)/2', 'x'),
    ('conjugate', 'I*log(x) - I*log(x - I)/2 + I*log(x + I)/2', 'x'),
    ('integral', 'exp(k*x)', 'x', 'True'),
    ('integral', 'exp(k*x)', 'x', 'False'),
    ('check', '6*x**(1/6) - 3*x**(1/3) + 2*sqrt(x) - 6*log(x**(1/6) + 1)', '1/(x**(1/3) + sqrt(x))', 'x'),
    ('check', '-c*log(1 - x/c) + x*log(1 - x/c) + x*log(Abs(c)) - x', 'log(Abs(c - x))', 'x'),
    ('check', 'exp(c*x)/c', 'exp(c*x)', 'x'),
    ('check', 'x*log(c*x) - x', 'log(c*x)', 'x'),
    ('check', '-besselj(0, x)**2/2 - besselj(1, x)**2/2', 'besselj(1, x)**2/x', 'x'),
    ('check', '-2*x*besselj(0, x)**2 - 2*x*besselj(1, x)**2 + 2*besselj(0, x)*besselj(1, x) - besselj(0, x)**2/x', 'besselj(0, x)**2/x**2', 'x'),
    ('check', '-besselj(0, x)**2/2', 'besselj(0, x)*besselj(1, x)', 'x'),
    ('check', 'x**2*besselj(1, x)**2/2', 'x**2*besselj(0, x)*besselj(1, x)', 'x'),
    ('check', 'x*besselj(0, x)**2 + x*besselj(1, x)**2 - besselj(0, x)*besselj(1, x)', 'besselj(0, x)*besselj(1, x)/x', 'x'),
    ('check', '-besselj(0, x**2)/2', 'x*besselj(1, x**2)', 'x'),
    ('check', '2*x*erf(x**3)*gamma(2/3)/(3*gamma(5/3)) - 2*gamma(2/3)*lowergamma(2/3, x**6)/(3*sqrt(pi)*gamma(5/3))', 'erf(x**3)', 'x'),
    ('check', '3*fresnels(x)*gamma(3/4)/(4*gamma(7/4))', 'sin(pi*x**2/2)', 'x'),
    ('check', 'fresnelc(x)*gamma(1/4)/(4*gamma(5/4))', 'cos(pi*x**2/2)', 'x'),
    ('check', '2*x**2*sqrt(2*x + 3)/5 + x*sqrt(2*x + 3)/5 - 3*sqrt(2*x + 3)/5', 'x*sqrt(2*x + 3)', 'x'),
    ('check', '-b*(1 + t/b)**(1 - a)/(a*b**a - b**a)', '(b + t)**(-a)', 't'),
    ('verified', 'x', 'x'),
    ('integral', 'y', 'x', 'True'),
    ('integral', 'exp(2)', 'x', 'True'),
    ('integral', 'x**2', 'x', 'True'),
    ('integral', '4*x**3 + 3*x**2', 'x', 'True'),
    ('integral', '(x + 2)**3', 'x', 'True'),
    ('integral', '(3*x + 4)**2', 'x', 'True'),
    ('integral', 'exp(2*x)', 'x', 'True'),
    ('integral', '2**x', 'x', 'True'),
    ('integral', '1/x', 'x', 'True'),
    ('integral', '1/(2*x + 3)', 'x', 'True'),
    ('integral', 'log(x)**2/x', 'x', 'True'),
    ('integral', '1/sqrt(1 - exp(2*x))', 'x', 'True'),
    ('integral', 'x**x*(log(x) + 1)', 'x', 'True'),
    ('integral', 'exp(x)*sin(x)', 'x', 'True'),
    ('integral', '2*x*cos(x)', 'x', 'True'),
    ('integral', 'x*log(x)', 'x', 'True'),
    ('integral', 'log(x)', 'x', 'True'),
    ('integral', '(3*x**2 + 5)*exp(x)', 'x', 'True'),
    ('integral', 'atan(x)', 'x', 'True'),
    ('integral', 't**2*cos(3*t - 1)', 't', 'True'),
    ('integral', 'x**2*exp(-x)*sin(2*x)', 'x', 'True'),
    ('integral', 'tan(x)', 'x', 'True'),
    ('integral', 'sec(x)', 'x', 'True'),
    ('integral', 'csc(x)', 'x', 'True'),
    ('integral', 'sin(x)*cos(x)', 'x', 'True'),
    ('integral', '-tan(x)*sec(x)', 'x', 'True'),
    ('integral', 'cot(x)*csc(x)', 'x', 'True'),
    ('integral', 'sec(x)**2', 'x', 'True'),
    ('integral', 'csc(x)**2', 'x', 'True'),
    ('integral', 'x*sec(x**2)', 'x', 'True'),
    ('integral', 'cos(x)*csc(sin(x))', 'x', 'True'),
    ('integral', 'cos(3*x)*sec(x)', 'x', 'True'),
    ('integral', 'sin(3*x)*sec(x)', 'x', 'True'),
    ('integral', 'sinh(2*x)', 'x', 'True'),
    ('integral', 'x*cosh(x**2)', 'x', 'True'),
    ('integral', 'tanh(x)', 'x', 'True'),
    ('integral', 'coth(x)', 'x', 'True'),
    ('integral', 'sech(x)', 'x', 'True'),
    ('integral', 'csch(x)', 'x', 'True'),
    ('integral', 'sin(x)**2*cos(x)', 'x', 'True'),
    ('integral', 'sin(x)**2*cos(x)**2', 'x', 'True'),
    ('integral', 'sin(x)*cos(x)**3', 'x', 'True'),
    ('integral', 'sin(x)**3*cos(x)**2', 'x', 'True'),
    ('integral', 'tan(x)**3*sec(x)', 'x', 'True'),
    ('integral', 'tan(x)*sec(x)**2', 'x', 'True'),
    ('integral', 'cot(x)**5*csc(x)', 'x', 'True'),
    ('integral', 'cot(x)**2*csc(x)**6', 'x', 'True'),
    ('integral', 'exp(x)/(exp(2*x) + 1)', 'x', 'True'),
    ('integral', '1/(9*x**2 + 4)', 'x', 'True'),
    ('integral', '1/(16*x**2 + 16)', 'x', 'True'),
    ('integral', '1/(x**2 + 4)', 'x', 'True'),
    ('integral', '1/(4*x**2 + 1)', 'x', 'True'),
    ('integral', '1/(4*x**2 + 4)', 'x', 'True'),
    ('integral', '1/sqrt(1 - x**2)', 'x', 'True'),
    ('integral', '1/sqrt(4 - 4*x**2)', 'x', 'True'),
    ('integral', '3/sqrt(1 - 9*x**2)', 'x', 'True'),
    ('integral', '1/sqrt(4 - 9*x**2)', 'x', 'True'),
    ('integral', '1/sqrt(x**2 + 1)', 'x', 'True'),
    ('integral', '1/sqrt(x**2 + 4)', 'x', 'True'),
    ('integral', '1/sqrt(4*x**2 + 4)', 'x', 'True'),
    ('integral', '1/sqrt(4*x**2 + 1)', 'x', 'True'),
    ('integral', '1/sqrt(x**2 - 1)', 'x', 'True'),
    ('integral', 'asin(x)', 'x', 'True'),
    ('integral', 'acos(x)', 'x', 'True'),
    ('integral', 'acot(x)', 'x', 'True'),
    ('integral', '1/(a + b*x**2)', 'x', 'True'),
    ('integral', 'x*asin(a*x)', 'x', 'True'),
    ('integral', 'x*atan(a*x)', 'x', 'True'),
    ('integral', '1/sqrt(a - b*x**2)', 'x', 'True'),
    ('integral', 'sqrt(16*x**2 - 9)/x', 'x', 'True'),
    ('integral', '1/(x**4*sqrt(25 - x**2))', 'x', 'True'),
    ('integral', 'x**7/(49*x**2 + 1)**(3/2)', 'x', 'True'),
    ('integral', '(x**2 + 1)**(-2)', 'x', 'True'),
    ('integral', 'x**2*sqrt(4 - x**2)', 'x', 'True'),
    ('integral', '1/(x**2*sqrt(x**2 + 4))', 'x', 'True'),
    ('integral', '1/(4 - x**2)', 'x', 'True'),
    ('integral', '1/(x**2 - 1)', 'x', 'True'),
    ('integral', '(x - 3)/(x**2 - 2*x + 2)', 'x', 'True'),
    ('integral', '(exp(x) - exp(-x))/x', 'x', 'True'),
    ('integral', 'exp(2*x)/x', 'x', 'True'),
    ('integral', '3*exp(4*x**2)', 'x', 'True'),
    ('integral', '4*exp(-x**2/3)', 'x', 'True'),
    ('integral', 'cosh(x/2)/x', 'x', 'True'),
    ('integral', 'cos(x**2)/x', 'x', 'True'),
    ('integral', '1/log(2*x + 1)', 'x', 'True'),
    ('integral', 'polylog(2, 5*x)/x', 'x', 'True'),
    ('integral', 'sin(3*x + 2)/x', 'x', 'True'),
    ('integral', 'sinh(3*x - 2)/x', 'x', 'True'),
    ('integral', 'x**(1/3)*exp(-x/8)', 'x', 'True'),
    ('integral', 'exp(-x**2 + 2*x + 1)', 'x', 'True'),
    ('integral', 'sin(x**2 + 4*x + 1)', 'x', 'True'),
    ('integral', 'cos(4*x**2)', 'x', 'True'),
    ('integral', '5*cos(2*x - 3)/x', 'x', 'True'),
    ('integral', '5/sqrt(3 - 2*sin(x)**2)', 'x', 'True'),
    ('integral', 'sqrt(9*sin(x)**2 + 4)', 'x', 'True'),
    ('integral', 'pi*Derivative(x**2 + 2*x + 3, x)', 'x', 'True'),
    ('integral', 'DiracDelta(3*x + 2)', 'x', 'True'),
    ('integral', 'Heaviside(x)', 'x', 'True'),
    ('integral', 'x', 'x', 'True'),
    ('integral', '0', 'x', 'True'),
    ('integral', 'x*Heaviside(x)', 'x', 'True'),
    ('integral', 'Heaviside(2*x + 4)', 'x', 'True'),
    ('integral', 'sin(x + 7)*Heaviside(3*x - 7)', 'x', 'True'),
    ('integral', 'jacobi(n, 7, 5/3, x)', 'x', 'True'),
    ('integral', 'gegenbauer(n, 7, x)', 'x', 'True'),
    ('integral', 'chebyshevt(n, x)', 'x', 'True'),
    ('integral', 'chebyshevu(n, x)', 'x', 'True'),
    ('integral', 'legendre(n, x)', 'x', 'True'),
    ('integral', 'hermite(n, x)', 'x', 'True'),
    ('integral', 'laguerre(n, x)', 'x', 'True'),
    ('integral', 'assoc_laguerre(n, 7, x)', 'x', 'True'),
    ('integral', 'x**y', 'x', 'True'),
    ('integral', 'y**x', 'x', 'True'),
    ('integral', 'exp(n*x)', 'x', 'True'),
    ('integral', '(y + 1)**x', 'x', 'True'),
    ('integral', 'exp(x**3)', 'x', 'True'),
    ('integral', '(4*x**4 + 4*x**3 + 16*x**2 + 12*x + 8)/(x**6 + 2*x**5 + 3*x**4 + 4*x**3 + 3*x**2 + 2*x + 1)', 'x', 'True'),
    ('integral', '1/(4*x + 2*(x - 1)**2 - 2)', 'x', 'True'),
    ('integral', 'exp(-a*x)', 'x', 'True'),
    ('integral', 'exp(-x**2)*log(x + 2)', 'x', 'True'),
    ('integral', 'sin(pi*t/4 + pi/4)**3', 't', 'True'),
    ('integral', 'sin(x)/x**3', 'x', 'True'),
    ('integral', '_x_61**3*exp(-_x_61)', '_x_61', 'True'),
    ('integral', '(-_x_61**3 - 3*_x_61**2 - 6*_x_61 - 6)*exp(-_x_61)/_x_61', '_x_61', 'True'),
    ('integral', '-sinh(_w_63 - u)', '_w_63', 'True'),
    ('integral', '-exp(-_w_63**2 - _w_63 + u)/2', '_w_63', 'True'),
    ('integral', '-exp(-_w_63**2 + _w_63 - u)/2', '_w_63', 'True'),
    ('integral', '-sinh(_w_66 - u)', '_w_66', 'True'),
    ('integral', '-exp(-_w_66**2 - _w_66 + u)/2', '_w_66', 'True'),
    ('integral', '-exp(-_w_66**2 + _w_66 - u)/2', '_w_66', 'True'),
    ('integral', 'sin(b*x)**2', 'x', 'True'),
    ('integral', 'x**k/(x + 3)', 'x', 'True'),
    ('integral', 'x**k*log(x)/(x + 3)', 'x', 'True'),
    ('integral', 'x**_j*x**k', 'x', 'True'),
    ('integral', 'sin(x/2)', 'x', 'True'),
    ('integral', 'exp(_t146)', '_t146', 'True'),
    ('integral', 'sin(x)**2', 'x', 'True'),
    ('integral', 'cos(x)', 'x', 'True'),
    ('integral', 'log(1 - x)/x', 'x', 'True'),
    ('integral', 'exp(-x)', 'x', 'True'),
    ('integral', 'sin(m*x)**2', 'x', 'True'),
    ('integral', 'sin(_t242*m)**2', '_t242', 'True'),
    ('integral', 'sin(_t242*m + pi*m/2)**2', '_t242', 'True'),
    ('integral', '(-a + u)**n*exp(-u*v)*Heaviside(-a + u, 0)', 'u', 'True'),
    ('integral', '_t295**n', '_t295', 'True'),
    ('integral', '(-a + u)**n', 'u', 'True'),
    ('integral', '_t301**n*log(_t301)', '_t301', 'True'),
    ('integral', '(z**(I/2))**I', 'z', 'True'),
    ('integral', 'sin(_t0)', '_t0', 'True'),
    ('integral', '_t0**(2*_j + 1)', '_t0', 'True'),
]


def python_call(case: tuple[str, ...]) -> object:
    kind, args = case[0], [parse(a) for a in case[1:] if a not in ('True', 'False')]
    if kind == 'verified':
        return ind.verified_antiderivative(args[0], args[1])
    if kind == 'integral':
        return ind.indefinite_integral(args[0], args[1], special_values=case[3] == 'True')
    if kind == 'check':
        return ind.is_antiderivative(args[0], args[1], args[2])
    if kind == 'real_form':
        return ind.real_form(args[0], args[1], args[2])
    return ind.conjugate_logarithms(args[0], args[1])


def rust_call(case: tuple[str, ...]) -> object:
    args = [Symbol(a) if a in ('True', 'False') else parse(a) for a in case[1:]]
    return rc.logic_call('indefinite', [Symbol(case[0])] + args)


@pytest.mark.parametrize('case', CASES, ids=[' '.join(c)[:80] for c in CASES])
def test_agree(case: tuple[str, ...]) -> None:
    rust = outcome(lambda: rust_call(case))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: python_call(case))
