"""The Rust port of ``integrals/antiderivative.py`` against the Python
one, on the calls of the Python tests without assumptions and on a list
of definite integrals (read with the core's parser: plain symbols; the
numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Symbol, Tuple
from sympy_extras.core.logic import false, true
from sympy_extras.core.parsing import parse
import sympy_extras.integrals.antiderivative as ad

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
    if r.__class__.__name__ == 'ConditionalValue':
        r = r.value  # type: ignore[attr-defined]
    if isinstance(r, bool):
        r = true if r else false
    if isinstance(r, (list, tuple)):
        r = Tuple(*[true if v is True else false if v is False else v for v in r])
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


CASES: list[tuple[str, ...]] = [
    ('singularities', '1/(x**2 - 1)', 'x'),
    ('singularities', 'sqrt(x) + log(x + 2)', 'x'),
    ('discontinuities', 'log(x - 1)', 'x', '0', '3'),
    ('discontinuities', 'atan(x)', 'x', '-oo', 'oo'),
    ('discontinuities', 'atan(1/(x - 2))', 'x', '0', '3'),
    ('discontinuities', '2*atan(tan(x/2))', 'x', '0', '2*pi'),
    ('limit', 'atan(x)', 'x', 'oo', '-'),
    ('limit', '-1/x', 'x', '0', '+'),
    ('limit', 'sin(x)', 'x', 'oo', '-'),
    ('shape', 'sin(pi*t/4 + pi/4)**3', 't'),
    ('shape', 't**2*exp(2*t)*cos(t)', 't'),
    ('shape', 'sin(t**2)', 't'),
    ('shape', 'sin(t)/t', 't'),
    ('shape', 'x**3*exp(-x)*log(x)', 'x'),
    ('shape', 'x*log(x)', 'x'),
    ('shape', 'log(x)*cos(x)', 'x'),
    ('real_logs', 'log(x - 1)', 'x'),
    ('real_logs', 'log(x + I)', 'x'),
    ('antiderivative', 'exp(-a*x)', 'x'),
    ('integral', '1/(cos(x) + 2)', 'x', '0', '2*pi'),
    ('shape', 'sin(b*x)**2', 'x'),
    ('antiderivative', 'sin(b*x)**2', 'x'),
    ('singularities', 'sin(b*x)**2', 'x'),
    ('antiderivative', 'exp(-x**2)*log(x + 2)', 'x'),
    ('limit', '2*atan(tan(x/2))', 'x', 'pi', '-'),
    ('limit', '2*atan(tan(x/2))', 'x', 'pi', '+'),
    ('integral', '1/(5 - 4*cos(x))', 'x', '0', '2*pi'),
    ('integral', 'x**(-2)', 'x', '-1', '1'),
    ('integral', '-1/x**2', 'x', '-1', '1'),
    ('integral', '1/x', 'x', '-1', '2'),
    ('integral', '1/(x**2 + 1)', 'x', '-oo', 'oo'),
    ('fp', 'x**(-2)', 'x', '-1', '1'),
    ('fp', '1/x', 'x', '-1', '2'),
    ('fp', '(x - 1)**(-3)', 'x', '0', '3'),
    ('shape', 't**2*exp(2*t)*cos(t) + sinh(3*t - 1)', 't'),
    ('shape', 'log(t)*cos(t)', 't'),
    ('antiderivative', 'sin(pi*t/4 + pi/4)**3', 't'),
    ('integral', 'sin(x)/x**3', 'x', '0', '1'),
    ('integral', 'sin(x)**2*cot(x - 1)', 'x', '0', '1'),
    ('integral', '1/x', 'x', '-3', '-1'),
    ('antiderivative', 'x**3*exp(-x)*log(x)', 'x'),
    ('antiderivative', 'sinh(u - w)*erf(w)', 'w'),
    ('shape', 'sinh(u - w)*erf(w)', 'w'),
    ('generic', '-2*(-sqrt(pi)*exp(1/4 - u)*erf(w - 1/2)/4 - sqrt(pi)*exp(u + 1/4)*erf(w + 1/2)/4)/sqrt(pi) - cosh(u - w)*erf(w)', 'w'),
    ('integral', 'sinh(u - w)*erf(w)', 'w', '0', 'u'),
    ('antiderivative', 'log(1 - sqrt(x)) - acoth(sqrt(x))', 'x'),
    ('antiderivative', 'acosh(x)', 'x'),
    ('integral', '1/(2 + cos(x))', 'x', '0', '2*pi'),
    ('pv', '1/(2 + cos(x))', 'x', '0', '2*pi'),
    ('fp', '1/(2 + cos(x))', 'x', '0', '2*pi'),
    ('integral', '1/x**2', 'x', '-1', '1'),
    ('pv', '1/x**2', 'x', '-1', '1'),
    ('fp', '1/x**2', 'x', '-1', '1'),
    ('pv', '1/(5 - 4*cos(x))', 'x', '0', '2*pi'),
    ('fp', '1/(5 - 4*cos(x))', 'x', '0', '2*pi'),
    ('integral', '1/(1 + x**2)', 'x', '-oo', 'oo'),
    ('pv', '1/(1 + x**2)', 'x', '-oo', 'oo'),
    ('fp', '1/(1 + x**2)', 'x', '-oo', 'oo'),
    ('integral', '1/x', 'x', '0', '1'),
    ('pv', '1/x', 'x', '0', '1'),
    ('fp', '1/x', 'x', '0', '1'),
    ('pv', '1/x', 'x', '-1', '2'),
    ('integral', 'log(x)', 'x', '0', '1'),
    ('pv', 'log(x)', 'x', '0', '1'),
    ('fp', 'log(x)', 'x', '0', '1'),
    ('integral', 'x*exp(-x)', 'x', '0', 'oo'),
    ('pv', 'x*exp(-x)', 'x', '0', 'oo'),
    ('fp', 'x*exp(-x)', 'x', '0', 'oo'),
    ('integral', '1/sqrt(1 - x**2)', 'x', '-1', '1'),
    ('pv', '1/sqrt(1 - x**2)', 'x', '-1', '1'),
    ('fp', '1/sqrt(1 - x**2)', 'x', '-1', '1'),
    ('integral', 'exp(-x)*sin(x)', 'x', '0', 'oo'),
    ('pv', 'exp(-x)*sin(x)', 'x', '0', 'oo'),
    ('fp', 'exp(-x)*sin(x)', 'x', '0', 'oo'),
    ('integral', '1/(x**2 - 4)', 'x', '0', '1'),
    ('pv', '1/(x**2 - 4)', 'x', '0', '1'),
    ('fp', '1/(x**2 - 4)', 'x', '0', '1'),
    ('integral', 'tan(x)', 'x', '0', 'pi/4'),
    ('pv', 'tan(x)', 'x', '0', 'pi/4'),
    ('fp', 'tan(x)', 'x', '0', 'pi/4'),
    ('integral', '1/(x - 1)', 'x', '0', '2'),
    ('pv', '1/(x - 1)', 'x', '0', '2'),
    ('fp', '1/(x - 1)', 'x', '0', '2'),
    ('integral', 'sqrt(x)', 'x', '0', '4'),
    ('pv', 'sqrt(x)', 'x', '0', '4'),
    ('fp', 'sqrt(x)', 'x', '0', '4'),
    ('integral', 'x*log(x)', 'x', '0', '1'),
    ('pv', 'x*log(x)', 'x', '0', '1'),
    ('fp', 'x*log(x)', 'x', '0', '1'),
    ('integral', '1/(1 + x**4)', 'x', '0', 'oo'),
    ('pv', '1/(1 + x**4)', 'x', '0', 'oo'),
    ('fp', '1/(1 + x**4)', 'x', '0', 'oo'),
    ('integral', 'sin(x)**2', 'x', '0', 'pi'),
    ('pv', 'sin(x)**2', 'x', '0', 'pi'),
    ('fp', 'sin(x)**2', 'x', '0', 'pi'),
    ('integral', 'atan(x)', 'x', '0', '1'),
    ('pv', 'atan(x)', 'x', '0', '1'),
    ('fp', 'atan(x)', 'x', '0', '1'),
    ('integral', 'x**3*exp(-x)*log(x)', 'x', '0', 'oo'),
    ('pv', 'x**3*exp(-x)*log(x)', 'x', '0', 'oo'),
    ('fp', 'x**3*exp(-x)*log(x)', 'x', '0', 'oo'),
    ('integral', 'abs(x - 1)', 'x', '0', '3'),
    ('pv', 'abs(x - 1)', 'x', '0', '3'),
    ('fp', 'abs(x - 1)', 'x', '0', '3'),
]


def args_of(case: tuple[str, ...]) -> list[object]:
    return [a if a in ('+', '-') else parse(a) for a in case[1:]]


def python_call(case: tuple[str, ...]) -> object:
    kind, args = case[0], args_of(case)
    if kind == 'integral':
        return ad.antiderivative_integral(*args)  # type: ignore[arg-type]
    if kind == 'pv':
        return ad.principal_value_integral(*args)  # type: ignore[arg-type]
    if kind == 'fp':
        return ad.finite_part_integral(*args)  # type: ignore[arg-type]
    if kind == 'shape':
        f, x = args
        return (ad.elementary_polynomial(f, x), ad.quick_shape(f, x), ad.by_parts_shape(f, x))  # type: ignore[arg-type]
    functions: dict[str, Callable[..., object]] = {
        'antiderivative': ad.antiderivative, 'singularities': ad.singularities, 'discontinuities': ad.discontinuities,
        'select': ad.select_branch, 'generic': ad.generic_antiderivative, 'continuous': ad.continuous_at,
        'limit': ad.one_sided_limit, 'cancelled': ad.cancelled_logarithms, 'real_logs': ad.real_logarithms}
    return functions[kind](*args)


def rust_call(case: tuple[str, ...]) -> object:
    args = [Symbol(a) if a in ('+', '-') else parse(a) for a in case[1:]]
    return rc.logic_call('antiderivative', [Symbol(case[0])] + args)


@pytest.mark.parametrize('case', CASES, ids=[' '.join(c)[:80] for c in CASES])
def test_agree(case: tuple[str, ...]) -> None:
    rust = outcome(lambda: rust_call(case))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: python_call(case))
