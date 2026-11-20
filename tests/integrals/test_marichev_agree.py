"""The Rust port of ``integrals/marichev.py`` against the Python one on
the integrands of the tests (read with the core's parser: plain symbols;
the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Symbol, Tuple
from sympy_extras.core.logic import false
from sympy_extras.core.parsing import parse
from sympy_extras.integrals.marichev import mellin_integrate

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


CASES: list[tuple[str, str]] = [
    ('x**2*exp(-3*x)', 'x'),
    ('x**k/(x + 3)', 'x'),
    ('sin(x)/x', 'x'),
    ('exp(-a*x)*sin(b*x)', 'x'),
    ('exp(-x)*besselj(0, x)', 'x'),
    ('exp(-x)*log(x)', 'x'),
    ('log(x)**2/(x**2 + 1)', 'x'),
    ('x/(exp(x) + 1)', 'x'),
    ('x**2/(exp(x) - 1)', 'x'),
    ('sin(x)/(exp(x) + 1)', 'x'),
    ('(1 - cos(x))/x**2', 'x'),
    ('exp(-a*x)*cos(k*x)', 'x'),
    ('exp(x)', 'x'),
    ('sin(x + 1)', 'x'),
    ('exp(-x)*besselj(0, x)*erf(x)', 'x'),
    ('x**k', 'x'),
    ('exp(-x)*sin(x)*cos(x)', 'x'),
    ('exp(-x)/x**(3/2)', 'x'),
    ('airyai(x)**2', 'x'),
    ('airyai(x)', 'x'),
    ('exp(-a*x)*airyai(x)', 'x'),
    ('_t2*besselk(0, _t2)**2', '_t2'),
    ('exp(-_t3*a)*besselj(0, _t3)**2', '_t3'),
    ('exp(-_t4*s)*erf(sqrt(_t4))', '_t4'),
    ('_t7**(n - 1)*exp(-_t7*s - a/_t7)', '_t7'),
    ('exp(-_t11*s - k**2/(4*_t11))/_t11', '_t11'),
    ('exp(-_t19**2)', '_t19'),
    ('exp(-_t22**2)', '_t22'),
    ('exp(-_t28)', '_t28'),
    ('log(_t32)**2/(_t32**2 + 1)', '_t32'),
    ('exp(-_t35*a)', '_t35'),
    ('_t37**k/(_t37 + 3)', '_t37'),
    ('exp(-_t38**2*y)', '_t38'),
    ('exp(-_t39**2*y)', '_t39'),
    ('exp(-_t43*a)', '_t43'),
    ('exp(-_t46**2)', '_t46'),
    ('exp(-_t49**2)', '_t49'),
    ('exp(-_t70)', '_t70'),
    ('exp(-_t71)', '_t71'),
    ('_t73**k/(_t73 + 3)', '_t73'),
    ('_t78**k/(_t78 + 3)', '_t78'),
    ('_t94**k/(_t94 + 3)', '_t94'),
    ('_t96**k/(_t96 + 3)', '_t96'),
    ('1/((-_t101 - 2)*(-_t101 - 1))', '_t101'),
    ('1/(_t101**2 + 3*_t101 + 2)', '_t101'),
    ('exp(-_t116)', '_t116'),
    ('exp(-_t117)', '_t117'),
    ('exp(_t120)', '_t120'),
    ('exp(-_t126)', '_t126'),
    ('4/(9*_u128**2 + 4)', '_u128'),
    ('-2*log(_t135 + 1) + log((_t135 + 1)**2 + 1)', '_t135'),
    ('-2*log(_t135 + 1) + log(_t135**2 + 2*_t135 + 2)', '_t135'),
    ('-2*log(_t140 + 1) + log(_t140**2 + 2*_t140 + 2)', '_t140'),
    ('exp(-_t172**2)*log(_t172)', '_t172'),
    ('exp(-_t174*s)*sin(_t174*a)/_t174', '_t174'),
    ('exp(-_t176*a)*cos(_t176*k)', '_t176'),
    ('exp(-_t195*a)*sin(_t195)*cos(_t195)', '_t195'),
    ('exp(-_t196*a)*sin(_t196)**2', '_t196'),
    ('-exp(-_t196*a)*cos(2*_t196)/2 + exp(-_t196*a)/2', '_t196'),
    ('exp(-_t198)*cosh(_t198/2)', '_t198'),
    ('(_t199**2/2 - 2*_t199 + 1)*exp(-_t199)', '_t199'),
    ('_t199**2*exp(-_t199)/2 - 2*_t199*exp(-_t199) + exp(-_t199)', '_t199'),
    ('_t266**n*exp(-v*(_t266 + a))', '_t266'),
    ('_t266**n*exp(-_t266*v - a*v)', '_t266'),
    ('_t270**n', '_t270'),
    ('_t276**n*log(_t276)', '_t276'),
    ('sin(_t282)', '_t282'),
    ('_t289*sin(_t289)', '_t289'),
    ('sin(_t298)**2', '_t298'),
    ('1/2 - cos(2*_t298)/2', '_t298'),
    ('exp(-_t302)', '_t302'),
    ('airyai(_t303)**2', '_t303'),
    ('exp(_t487)', '_t487'),
    ('1/(_t491 + 1)', '_t491'),
    ('_t494/(_t494**2 + 1)', '_t494'),
    ('_t497*Shi(_t497)', '_t497'),
    ('cos(_t508)', '_t508'),
    ('sin(_t514) + cos(_t514)', '_t514'),
    ('cos(_t518)', '_t518'),
    ('(exp(-_t522*b) - exp(-_t522*a))*exp(-_t522*s)/_t522**(3/2)', '_t522'),
    ('-exp(-_t522*a - _t522*s)/_t522**(3/2) + exp(-_t522*b - _t522*s)/_t522**(3/2)', '_t522'),
    ('(exp(_t522*a) - exp(_t522*b))*exp(-_t522*a)*exp(-_t522*b)*exp(-_t522*s)/_t522**(3/2)', '_t522'),
    ('(2*_t560 - 1)*exp(-_t560*s)/sqrt(_t560)', '_t560'),
    ('2*sqrt(_t560)*exp(-_t560*s) - exp(-_t560*s)/sqrt(_t560)', '_t560'),
    ('(4*_t562**2 - 12*_t562 + 3)*exp(-_t562*s)/sqrt(_t562)', '_t562'),
    ('4*_t562**(3/2)*exp(-_t562*s) - 12*sqrt(_t562)*exp(-_t562*s) + 3*exp(-_t562*s)/sqrt(_t562)', '_t562'),
    ('sqrt(_t565)*(2*_t565 - 3)*exp(-_t565*s)', '_t565'),
    ('2*_t565**(3/2)*exp(-_t565*s) - 3*sqrt(_t565)*exp(-_t565*s)', '_t565'),
    ('exp(-_t639*v - a*exp(-_t639))', '_t639'),
    ('exp(-_t639*v)*exp(-a*exp(-_t639))', '_t639'),
    ('log(_t735)/(_t735**2 + 1)', '_t735'),
    ('1/(_t736**4 + 2*_t736**2*cosh(2*a) + 1)', '_t736'),
    ('1/(_t739**4 + 2*_t739**2*(_y738/2 + 1/(2*_y738)) + 1)', '_t739'),
    ('1/(_t739**4 + _t739**2*_y738 + _t739**2/_y738 + 1)', '_t739'),
    ('1/(_t741**4 + 2*_t741**2*cosh(2*c) + 1)', '_t741'),
    ('1/(_t744**4 + 2*_t744**2*(_y743/2 + 1/(2*_y743)) + 1)', '_t744'),
    ('1/(_t744**4 + _t744**2*_y743 + _t744**2/_y743 + 1)', '_t744'),
    ('(-log(_t751) + 1/(1 - exp(-_t751)) - 1/_t751)*exp(-_t751)', '_t751'),
    ('(-_t751*exp(_t751)*log(_t751) + _t751*exp(_t751) + _t751*log(_t751) - exp(_t751) + 1)/(_t751*exp(2*_t751) - _t751*exp(_t751))', '_t751'),
    ('(_t757*exp(_t757) - exp(_t757) + 1)/(_t757*exp(2*_t757) - _t757*exp(_t757))', '_t757'),
    ('exp(-(_t1176 - mu)**2)', '_t1176'),
    ('exp(-_t1176**2 + 2*_t1176*mu - mu**2)', '_t1176'),
    ('exp(-_t1176**2)*exp(-mu**2)*exp(2*_t1176*mu)', '_t1176'),
    ('exp(-(-_t1181 - mu)**2)', '_t1181'),
    ('exp(-_t1181**2 - 2*_t1181*mu - mu**2)', '_t1181'),
    ('exp(-_t1181**2)*exp(-mu**2)*exp(-2*_t1181*mu)', '_t1181'),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_mellin_integrate(i: int) -> None:
    text, variable = CASES[i]
    x = Symbol(variable)
    f = parse(text)

    def python() -> object:
        found = mellin_integrate(f, x)
        return None if found is None else Tuple(found.value, found.condition)
    got = outcome(lambda: rc.logic_call('marichev', [Symbol('integrate'), f, x, Tuple(), Symbol('NONE'), false]))
    if got != 'SKIP':
        assert got == outcome(python), text
