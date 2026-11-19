"""The Rust port of ``simplify/normalform.py`` against the Python one,
on the expressions it receives in the test suite (read with the core's
parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Symbol
from sympy_extras.core.parsing import parse
from sympy_extras.simplify.normalform import normal_form

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


CASES: list[str] = [
    '(-(-x + cos(y))**2 + 1/3 - 1/(y + 2))/(x**(3/2)/3 - (-x + sqrt(x + 1) - 1)**2 + 3)',
    '(-2*m**2 + m*r)*sin(theta)**2/r**2',
    '(-a*b**2*cos(t) - a*(a**2*sin(t)**2 + a**2*cos(t)**2)*cos(t))/(a*(a**2 + b**2))',
    '(-rho*sin(phi) + 1)/rho',
    '(-sqrt(u**2 + v**2)*sin(v) + 1)/sqrt(u**2 + v**2)',
    '(-v**2*exp(4*u) - exp(2*u))/(v**4*exp(4*u) + 2*v**2*exp(4*u) + 2*v**2*exp(2*u) + exp(4*u) + 2*exp(2*u) + 1)',
    '(-y**2/9 + sinh(y) - 1/(y + 2))/(3*cos(x)*cos(y)*tan(x)/10 + 3)',
    '(1 - cosh(u)**2)/(cosh(u)**2 - 1)',
    '(2*m**2 - m*r)*(2*m**2*sin(theta)**2 - m*r*sin(theta)**2)/r**8',
    '(2*sinh(u)*cosh(u)/(sinh(u)**2 + 1) - 2*sinh(u)**3*cosh(u)/(sinh(u)**2 + 1)**2)*(sinh(u)**2 + 1)/(2*sinh(u)**2)',
    '(4*m**3 - 2*m**2*r)/(r**9*sin(theta)**2)',
    '(4*m*r**2/(a**2*cos(theta)**2 + r**2)**2 - 2*m/(a**2*cos(theta)**2 + r**2))*(a**2 - 2*m*r + r**2)/(2*(a**2*cos(theta)**2 + r**2))',
    '(a**2*cos(theta)**2 + r**2)*(a**3*m*sin(theta)*cos(theta)**3 - 3*a*m*r**2*sin(theta)*cos(theta))/(a**6*cos(theta)**6 + 3*a**4*r**2*cos(theta)**4 + 3*a**2*r**4*cos(theta)**2 + r**6)',
    '(a*b*sin(t)**2 + a*b*cos(t)**2)**2/(a**2*sin(t)**2 + b**2*cos(t)**2)**3',
    '(phi**2*r + theta)*cos(theta)/(r*sin(theta))',
    '(phi*rho*exp(z) + 3*rho*sin(phi) + z)/rho',
    '(r**2*sin(theta)**2 + 2*r*(phi + 2*r*sin(theta))*sin(theta) + theta*exp(phi)/sin(theta) + (r**2*cos(theta) + exp(phi))*cos(theta))/(r**2*sin(theta))',
    '(sin(x) + sinh(y))*(sin(x) - tan(x)**2*cosh(y)**2 - 1/9)/(cos(y)/3 + 3)',
    '(x**(3/2) + 1/sqrt(y + 2))*(cosh(y) - 4)/(x*y*(x + 1/3) + 3)',
    '-(-a**3*m*sin(theta)*cos(theta)**3 + 3*a*m*r**2*sin(theta)*cos(theta))/(a**6*cos(theta)**6 + 3*a**4*r**2*cos(theta)**4 + 3*a**2*r**4*cos(theta)**2 + r**6)',
    '-(-exp(4*u) - exp(2*u))/(v**4*exp(4*u) + 2*v**2*exp(4*u) + 2*v**2*exp(2*u) + exp(4*u) + 2*exp(2*u) + 1)',
    '-115/32',
    '-2*(a**2*cos(theta)**2 + r**2)*(2*a**3*m*r*sin(theta)*cos(theta) + 2*a*m*r**3*sin(theta)*cos(theta))/(a**6*cos(theta)**6 + 3*a**4*r**2*cos(theta)**4 + 3*a**2*r**4*cos(theta)**2 + r**6)',
    '-2*m*(-2*m**2*sin(theta)**2 + m*r*sin(theta)**2)/(r**9*sin(theta)**2)',
    '-2*m*(2*m**2 - m*r)*sin(theta)**2/r**5',
    '-2/cosh(u)**4',
    '-24/25',
    '-3*log(2) + log(8/9) + 2*log(3)',
    '-3040/1309',
    '-37/6',
    '-5 - 2*sqrt(6) + (sqrt(2) + sqrt(3))**2',
    '-73/42',
    '-75/7',
    '-86219/8',
    '-9/14',
    '-H**2*r**2*sin(theta)**2',
    '-a*b*sin(t)**2/(a**2 + b**2) - a*b*cos(t)**2/(a**2 + b**2) + a*b/(a**2 + b**2)',
    '-a*cos(t)',
    '-log(3) + log(2)',
    '-m*(-2*m**2 + m*r)/(r**2*(2*m*r**4*sin(theta)**2 - r**5*sin(theta)**2))',
    '-m*(2*m**2 - m*r)/(r**2*(2*m*r**4*sin(theta)**2 - r**5*sin(theta)**2))',
    '-m**2*sin(theta)**2/((2*m - r)*(2*m*r**4 - r**5))',
    '-phi**2 + (-phi**2*theta*cos(theta) + phi**2*sin(theta))/sin(theta)',
    '-phi**2*theta*cos(theta)/sin(theta) + (phi**2*theta*cos(theta) + theta**2*sin(theta)**3*cos(theta) + sin(theta)**2)/sin(theta)',
    '-phi**2*theta*cos(theta)/sin(theta) + phi**2',
    '-phi*theta**3*cos(theta)/sin(theta) + (phi*theta**3*cos(theta) - phi*sin(theta)*cos(theta))/sin(theta)',
    '-r*sin(theta)**2 + (-2*m*sin(theta)**2 + r*sin(theta)**2)/(-2*m/r + 1)',
    '-sinh(r)*cosh(r)',
    '-u*v/(u**2 + v**2 + 1)**(3/2) - sqrt((u**2 + 1)*(v**2 + 1))/(u**2 + v**2 + 1)**(3/2)',
    '-x**2 + y*sin(a)**2',
    '-y**2/(exp(x) + 3)**2 + (sin(x) + cosh(y))*tan(x)/(cos(x) + 3) + cos(y) - 1/(y + 2)',
    '1 - sin(x)**2',
    '1287/8',
    '131279/343',
    '1369/310',
    '149/49',
    '152300281/1944',
    '19/40',
    '2*m*sin(theta)**2 - r*sin(theta)**2 + (-2*m + r)*sin(theta)**2',
    '2*phi*cos(theta)',
    '2209/49',
    '3/91',
    '3718/735',
    '4*a**2*m*r*(a**2*cos(theta)**2 + r**2)*sin(theta)*cos(theta)/(a**6*cos(theta)**6 + 3*a**4*r**2*cos(theta)**4 + 3*a**2*r**4*cos(theta)**2 + r**6)',
    '4*m**2/(r**10*sin(theta)**2)',
    '4*sqrt(x)*sqrt(x + 1)*cos(x)*cos(y)*sinh(y) - 4*x*sqrt(x + 1)*cos(x)*cos(y)*sinh(y) - 4*(sqrt(x) - x)*sqrt(x + 1)*cos(x)*cos(y)*sinh(y)',
    '53/27',
    '634463856/12005',
    'a**(-2)',
    'a**2*sin(theta)*cos(theta)/((a**2*cos(theta)**2 + r**2)*(a**2 - 2*m*r + r**2))',
    'c*cos(t)',
    'cos(chi)**2 - 1',
    'cos(u)/(R*r + r**2*cos(u)) - cos(u)/(r*(R + r*cos(u)))',
    'cosh(r)/sinh(r)',
    'cosh(u)**(-2)',
    'exp(u)',
    'exp(x)/((3 + cos(x)/(cos(x) + 3))*((x**(3/2) + exp(x))/(sin(x) + 3 + 1/sqrt(y + 2)) + 3)*(cosh(y) + 3))',
    'log(8)/log(2)',
    'm*(-2*m**2 + m*r)/(r**2*(2*m*r**4*sin(theta)**2 - r**5*sin(theta)**2))',
    'm**2*sin(theta)**2/r**6',
    'm*sin(theta)**2/(r*(-2*m/r + 1))',
    'phi*exp(z) + 2*sin(phi) + (-rho**2*sin(phi) + rho*(2*rho*sin(phi) + z))/rho**2',
    'phi*theta**3 + phi*sin(theta)',
    'phi*theta**3*sin(theta)*cos(theta) + phi*theta',
    'r**(-2)',
    'r**2*a(t)*sin(theta)**2*Derivative(a(t), t)',
    'r**2*sin(theta)*cos(phi)*cos(theta)',
    'r*sin(phi)**2*sin(theta)*cos(theta) + r*sin(theta)*cos(phi)**2*cos(theta) - r*sin(theta)*cos(theta)',
    'sin(u)**2 + cos(u)**2 + 1',
    'sin(x)**2 + cos(x)**2 - 1',
    'sinh(r)**(-2)',
    'sinh(u)**2/((sinh(u)**2 + 1)*cosh(u)**2)',
    'sqrt(x) + sqrt(y)',
    'v*(exp(2*u) + 1)*exp(2*u)/(v**2*exp(2*u) + exp(2*u) + 1) - v*exp(4*u)/(v**2*exp(2*u) + exp(2*u) + 1)',
    'v**2*exp(2*u) + 1',
    'v*exp(4*u)/(v**4*exp(4*u) + 2*v**2*exp(4*u) + 2*v**2*exp(2*u) + exp(4*u) + 2*exp(2*u) + 1)',
    'v/sqrt(u**2 + v**2) + v*(u**2 + z*sin(v))/(u**2 + v**2)**(3/2)',
    'x*(-x**2 + cos(y))*(tan(x) + 1/sqrt(y + 2))*sinh(y)/(sqrt(x + 1) + sin(x) + 3)',
    'x**(3/2) + 2*(y + exp(x))*cos(x) + cos(x) - 4*cos(y)**2',
    'y**2*sqrt(x + 1)/(5*(-(sqrt(x) + 1/sqrt(y + 2))**2 + 3 + 2/(sqrt(x + 1) + 3)))',
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_normal_form(i: int) -> None:
    e = parse(CASES[i])
    got = outcome(lambda: rc.logic_call('calculus', [Symbol('normal_form'), e]))
    if got != 'SKIP':
        assert got == outcome(lambda: normal_form(e)), CASES[i]
