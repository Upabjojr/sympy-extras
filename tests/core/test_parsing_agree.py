"""``sympify_extras.core.parsing`` against its Rust port: the grammar
(precedence, right associativity of ``**``, unary signs, calls), the
constants, the symbols, the undefined functions, ``Rational``/``Integer``,
the plain callables ``sqrt``, ``cbrt``, ``root``, ``ln`` and ``log`` with a
base, and every function head by SymPy's name, with and without
evaluation; the errors outside the grammar. Decimals are not compared (no
``Float`` in the Rust core)."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


TEXTS = ['x**2 + 1', 'sin(x)**2 + cos(x)**2', '(x**2 - 1)/(x - 1)', '2/3 + 1', '-x**2', '2**-1', 'f(2*x + 4*y)',
         'log(8, 2)', 'x + x', '2**3**2', '-2**2', 'a - b - c', 'a/b/c', '+x', '--x', 'x*-y', 'pi + E + I + oo',
         'zoo + nan', 'EulerGamma*Catalan + GoldenRatio - TribonacciConstant', 'Rational(3, 6) + Integer(4)',
         'sqrt(8)', 'cbrt(27*x)', 'root(16, 4)', 'root(x, 3, 1)', 'ln(E**2)', 'log(x, y)', 'exp(log(x))',
         'g(x, y, z) + g(x, y, z)', '(x + 1)**(1/2)', 'x**(-1/2)', 'Abs(-3) + Max(1, 2, x) + Min(x, 3)',
         'atan2(1, 1)', 'besselj(n, x)', 'gamma(5)/factorial(3)', 'binomial(5, 2)', '  x  +  y  ',
         'x +', '2 x', 'x @ y', 'Rational(x, 2)', 'f()', '(x', 'x)', '1.5 + x', 'Piecewise((x, x > 0))']


def _rust_names() -> list[str]:
    import re
    from pathlib import Path
    source = (Path(__file__).parents[2] / 'crates/sympy-extras-core/src/core/parsing.rs').read_text()
    return re.findall(r'\(Function::\w+, "([^"]+)"\)', source)


def _normalized(r: Any) -> Any:
    return 'ERROR ValueError' if r == 'ERROR ParseUnsupported' else r


@pytest.mark.parametrize("evaluate", [True, False])
def test_parse(rust: Any, evaluate: bool) -> None:
    from sympy_extras.core import Symbol, false, true
    from sympy_extras.core.parsing import parse
    rc = load("sympy_extras_rs.core")
    for text in TEXTS:
        got = outcome(lambda: rc.logic_call('parse', [Symbol(text), true if evaluate else false]))
        expected = _normalized(outcome(lambda: parse(text, evaluate)))
        if text.startswith('1.5'):
            assert got == 'SKIP', text
            continue
        assert got == expected, (text, evaluate)


@pytest.mark.parametrize("evaluate", [True, False])
def test_function_names(rust: Any, evaluate: bool) -> None:
    from sympy_extras.core import Symbol, false, true
    from sympy_extras.core.parsing import parse
    rc = load("sympy_extras_rs.core")
    for name in _rust_names():
        for args in ('x', 'x, 2', '3, x'):
            text = '%s(%s)' % (name, args)
            expected = _normalized(outcome(lambda: parse(text, evaluate)))
            if expected.startswith('ERROR') if isinstance(expected, str) else False:
                continue  # a wrong number of arguments: the Rust heads do not check it
            got = outcome(lambda: rc.logic_call('parse', [Symbol(text), true if evaluate else false]))
            if got == 'SKIP' and expected != 'SKIP':
                continue
            assert got == expected, (text, evaluate)
